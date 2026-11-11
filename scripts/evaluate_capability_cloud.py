#!/usr/bin/env python3
"""Paired sampled-input / restored-cloud readout with frozen geometry, offline.

Only source-camera reprojection is used. The same crop, samples, visibility
rule and 1 mm quantization apply to both clouds; sparse pixels are not filled.
Target labels are opened after all estimators for each paired cloud record.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from neck_localization.active_stereo import quantize_depth
from neck_localization.depth_profile import DepthFrame
from neck_localization.geometry import Camera, Estimate
from scripts import evaluate_capability_depth as shared


def load_rgb_frames(folder, names, tracked):
    folder = Path(folder)
    path = folder/"camera_true_parameters.json"
    tracked[str(path.resolve())] = shared.file_hash(path)
    metadata = shared._json(path)
    items = {item["name"]:item for item in metadata["cameras"]}
    if len(items) != len(metadata["cameras"]) or set(items) != set(names):
        raise ValueError("camera_name_metadata_mismatch")
    frames = []
    for name in names:
        item = items[name]
        path = folder/(name+"_rgb.png")
        tracked[str(path.resolve())] = shared.file_hash(path)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (item["height"], item["width"]):
            raise ValueError("ordinary_RGB_missing_or_shape_mismatch")
        camera = Camera(name, np.asarray(item["K"], float), np.asarray(item["T_world_from_camera"], float),
                        item["width"], item["height"])
        frames.append(DepthFrame(camera, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None))
    return frames


def source_camera_zbuffers(points, source_ids, frames):
    """Project each observed point only to the view from which it originated."""
    points, source_ids = np.asarray(points), np.asarray(source_ids)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("nonfinite_or_invalid_cloud")
    if source_ids.shape != (len(points),) or not np.issubdtype(source_ids.dtype, np.integer):
        raise ValueError("invalid_source_camera_ids")
    if np.any(source_ids < 0) or np.any(source_ids >= len(frames)):
        raise ValueError("source_camera_id_out_of_range")
    output, details = [], []
    for index, frame in enumerate(frames):
        selected = points[source_ids == index]
        local = ((selected-frame.camera.T_world_from_camera[:3, 3]) @
                 frame.camera.T_world_from_camera[:3, :3])
        front = np.isfinite(local).all(axis=1) & (local[:, 2] > 0)
        projected = local[front] @ frame.camera.K.T
        uv = np.rint(projected[:, :2]/projected[:, 2:3]).astype(np.int64)
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < frame.camera.width) &
                  (uv[:, 1] >= 0) & (uv[:, 1] < frame.camera.height))
        uv, z = uv[inside], local[front, 2][inside]
        depth = np.full((frame.camera.height, frame.camera.width), np.inf)
        np.minimum.at(depth, (uv[:, 1], uv[:, 0]), z)
        depth[~np.isfinite(depth)] = 0
        depth = quantize_depth(depth, .001).astype(np.float32)*.001
        output.append(DepthFrame(frame.camera, frame.rgb, depth))
        details.append(dict(camera=frame.camera.name, source_points=len(selected),
                            behind_camera=int((~front).sum()), projected_inside=len(z),
                            occupied_pixels=int((depth > 0).sum()),
                            pixel_collisions=len(z)-int((depth > 0).sum())))
    return output, details


def read_cloud(row, tracked):
    if row["status"] != "inference_completed":
        raise ValueError("restoration_not_completed:"+row["status"])
    for filename, expected in row["input_files_sha256"].items():
        actual = shared.file_hash(filename)
        tracked[str(Path(filename).resolve())] = actual
        if actual != expected:
            raise ValueError("cloud_input_sha256_mismatch:"+filename)
    path = Path(row["pointcloud"])
    tracked[str(path.resolve())] = shared.file_hash(path)
    if tracked[str(path.resolve())] != row["pointcloud_sha256"]:
        raise ValueError("cloud_prediction_sha256_mismatch")
    with np.load(path, allow_pickle=False) as data:
        original = data["input_points_world_m"].copy()
        restored = data["points_world_m"].copy()
        ids = data["source_camera_ids"].copy()
        uv = data["source_pixel_uv"].copy()
        updates = data["prediction_update_count"].copy()
    if len(original) == 0 or original.shape != restored.shape or uv.shape != (len(original), 2) or updates.shape != (len(original),):
        raise ValueError("paired_cloud_shape_mismatch")
    if original.ndim != 2 or original.shape[1] != 3 or not np.isfinite(original).all() or not np.isfinite(restored).all():
        raise ValueError("nonfinite_or_invalid_paired_cloud")
    if not np.issubdtype(updates.dtype, np.integer) or np.any(updates < 0):
        raise ValueError("negative_prediction_update_count")
    return original, restored, ids, dict(points=len(original), updated_points=int((updates > 0).sum()),
        untouched_points=int((updates == 0).sum()),
        displacement_mm_median=float(np.median(np.linalg.norm(restored-original, axis=1))*1000),
        displacement_mm_max=float(np.linalg.norm(restored-original, axis=1).max()*1000))


def run(predictions, output):
    predictions, output = Path(predictions), Path(output)
    if output.exists():
        raise FileExistsError("Output directory must be new")
    manifest_path, provenance_path = predictions/"manifest.json", predictions/"provenance.json"
    inputs, provenance = shared._json(manifest_path), shared._json(provenance_path)
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("expected_nonempty_cloud_manifest_list")
    model_camera_set = provenance.get("camera_set", "all")
    if model_camera_set != "all":
        raise ValueError("this_paired_audit_requires_all_three_model_input_views")
    tracked = {str(p.resolve()):shared.file_hash(p) for p in (manifest_path, provenance_path)}
    sources, settings = shared.source_hashes(), shared.frozen_settings()
    sources[str(Path(__file__).relative_to(ROOT))] = shared.file_hash(__file__)
    output.mkdir(parents=True)
    records, checks = [], []
    seen = set()
    for row in inputs:
        key = row["layout"], row["case"], row["condition"]
        if key in seen:
            raise ValueError("duplicate_cloud_manifest_case_condition")
        seen.add(key)
        pending, projected, info = [], {}, dict(zip(("layout", "case", "condition"), key))
        error = None
        try:
            if tuple(row["camera_names"]) != shared.ALL_CAMERAS:
                raise ValueError("expected_all_three_model_input_cameras_in_declared_order")
            frames = load_rgb_frames(row["observations_directory"], row["camera_names"], tracked)
            original, restored, ids, stats = read_cloud(row, tracked)
            info.update(status="ok", cloud=stats)
            for variant, points in (("sampled_input_reprojected", original), ("restored_reprojected", restored)):
                projected[variant], details = source_camera_zbuffers(points, ids, frames)
                info[variant] = details
        except Exception as exception:
            error = f"input_error:{type(exception).__name__}:{exception}"
            info.update(status="input_error", error=error, traceback=traceback.format_exc())
        for variant in ("sampled_input_reprojected", "restored_reprojected"):
            for backend, infer in shared.BACKENDS.items():
                status, estimate = "ok", Estimate()
                if error is not None:
                    status, estimate.reasons = "input_error", [error]
                else:
                    started = perf_counter()
                    try:
                        estimate = infer(projected[variant])
                        shared._validate_estimate(estimate)
                    except Exception as exception:
                        status = "backend_error"
                        estimate = Estimate(reasons=[f"backend_exception:{type(exception).__name__}:{exception}"],
                                            diagnostics=dict(traceback=traceback.format_exc()))
                    estimate.elapsed_ms = (perf_counter()-started)*1000
                pending.append((variant, backend, status, estimate))
        # No target, profile, ideal depth or mesh was opened by preparation/inference.
        truth_pair = dict(render_root=provenance["render_root"], case=row["case"])
        truth, _ = shared._load_truth(truth_pair, tracked)
        for variant, backend, status, estimate in pending:
            metrics = shared.score(estimate, truth)
            metrics["fit_condition"] = metrics.pop("condition")
            result = dict(zip(("layout", "case", "condition"), key),
                camera_set="all", model_input_camera_set="all", variant=variant, backend=backend,
                split=provenance["split"], status=status, **metrics)
            result["evaluation_role"] = ("oracle_measurement_diagnostic_not_deployed_input"
                                         if row["condition"] == "ideal_rgb" else "method_capability_audit")
            result["band"], result["band_evaluation"] = None, None
            if estimate.center is not None and "near_min_interval_from_axis_origin_m" in estimate.diagnostics:
                try:
                    band = shared.plan_band(result["estimate"], estimate.diagnostics)
                    result["band"] = band
                    result["band_evaluation"] = {label:shared.evaluate_band(band, truth, truth[field])
                        for label, field in (("near_min_proxy", "near_min_interval_s_m"),
                                             ("full_annotated_neck", "search_interval_s_m"))}
                except Exception as exception:
                    result["band_evaluation_error"] = f"{type(exception).__name__}:{exception}"
            result["restoration_record_runtime_s"] = row.get("runtime_s") if variant == "restored_reprojected" else None
            records.append(result)
            with (output/"records.jsonl").open("a") as stream:
                stream.write(json.dumps(result, allow_nan=False)+"\n")
        checks.append(info)
        print(*key, "backends", [(v, b, e.accepted) for v, b, _, e in pending], flush=True)
    summaries = []
    keys = sorted({(r["layout"], r["condition"], r["variant"], r["backend"]) for r in records})
    for key in keys:
        rows = [r for r in records if (r["layout"], r["condition"], r["variant"], r["backend"]) == key]
        summaries.append(dict(zip(("layout", "condition", "variant", "backend"), key),
                              evaluation_role=rows[0]["evaluation_role"], **shared.aggregate(rows)))
    shared._write_json(output/"summary.json", summaries)
    shared._write_json(output/"projection_checks.json", checks)
    (output/"records.json.gz").write_bytes(gzip.compress(json.dumps(records, allow_nan=False).encode(), mtime=0))
    final_sources = shared.source_hashes()
    final_sources[str(Path(__file__).relative_to(ROOT))] = shared.file_hash(__file__)
    changed = [p for p, h in tracked.items() if not Path(p).exists() or shared.file_hash(p) != h]
    valid = not changed and sources == final_sources and settings == shared.frozen_settings()
    shared._write_json(output/"validation.json", dict(passed=valid, changed_inputs=changed,
        frozen_sources_unchanged=sources == final_sources, settings_unchanged=settings == shared.frozen_settings(),
        paired_cloud_count=len(inputs), backend_record_count=len(records)))
    shared._write_json(output/"provenance.json", dict(schema=1, source_prediction_root=str(predictions.resolve()),
        source_sha256=sources, input_sha256=tracked, settings=settings, no_hardware=True, no_gpu_inference=True,
        model_input_camera_set="all", readout_camera_set="all",
        projection="source-camera-only nearest rounded pixel; nearest positive RGB optical Z; 0.001m Z16; no filling",
        paired_control="identical observed crop, sampled points, camera IDs and reprojection for original/restored",
        inference_truth_access="target/profile first opened after both variants and both frozen backends per record",
        acceptance="original frozen gates unchanged; candidate geometry and accepted correct outputs separate",
        limitations=["previously viewed synthetic cases; no unseen-test or physical-device claim",
            "sampled sparse reprojection is an adapter; paired input controls its losses",
            "three-view joint restoration does not establish single-view ability",
            "ideal_rgb is an oracle-measurement diagnostic, not a deployed input",
            "known incorrect portfolio silhouette gate is retained"]))
    if not valid:
        raise RuntimeError("Input/source/settings changed during cloud evaluation")
    return records, summaries


if __name__ == "__main__":
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.predictions, args.output)
