#!/usr/bin/env python3
"""Evaluate saved stereo predictions with unchanged geometry backends, offline.

The estimator receives ordinary RGB, supplied cameras and equally quantized /
registered predicted depth. Target/profile/ideal depth are first read AFTER all
backend calls for a case. Rejections, input errors and exceptions stay in every
denominator. No silhouette gate is removed and no ground-truth candidate is
selected. This is a capability audit on previously opened simulation cases.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from neck_localization.active_stereo import OfflineSpatialFilter, StereoSettings
from neck_localization.band_geometry import evaluate_band, plan_band
from neck_localization.depth_profile import DepthFrame
from neck_localization.evaluation import score
from neck_localization.geometry import Camera, Estimate
from neck_localization.joint_profile import JointSettings
from neck_localization.reliable_output import ReliabilitySettings
from neck_localization.stereo_robust import (
    BlueScreenSettings, blue_screen_foreground, estimate_stereo_robust,
)
from neck_localization.structure_portfolio import PortfolioSettings, estimate_portfolio
from neck_localization.task_validated_output import TaskValidationSettings

ALL_CAMERAS = ("cam_a", "cam_b", "cam_c")
BACKENDS = {"portfolio": estimate_portfolio, "stereo_robust": estimate_stereo_robust}
FROZEN_MODULES = (
    "__init__", "active_stereo", "band_geometry", "depth_profile", "evaluation",
    "geometry", "joint_profile", "reliable_output", "silhouette_profile",
    "stereo_robust", "structural_localization", "structure_portfolio",
    "support_feedback", "task_validated_output",
)


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    # Freeze the actually imported backend dependency closure. Other adapters
    # can be developed concurrently without changing these frozen estimators.
    package = (ROOT / "neck_localization").resolve()
    files = sorted([package/(name+".py") for name in FROZEN_MODULES] + [Path(__file__).resolve()])
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files}


def frozen_settings():
    return {name: asdict(cls()) for name, cls in (
        ("portfolio", PortfolioSettings), ("task_validation", TaskValidationSettings),
        ("reliability", ReliabilitySettings), ("joint", JointSettings),
        ("foreground", BlueScreenSettings), ("registration", StereoSettings))}


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_prediction_index(directory):
    result = {}
    path = Path(directory) / "predictions.jsonl"
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        pair_id = row["pair_id"]
        if pair_id in result:
            raise ValueError(f"duplicate_prediction_status:{pair_id}:line={line_number}")
        result[pair_id] = row
    return result


def read_prediction(pair, directory, index, tracked):
    """Validate exact file inputs and return depth with declared invalids zeroed."""
    pair_id = pair["pair_id"]
    for key, expected in zip(("left_path", "right_path"), pair["input_sha256"]):
        path = Path(pair[key])
        actual = file_hash(path)
        tracked[str(path.resolve())] = actual
        if actual != expected:
            raise ValueError(f"input_sha256_mismatch:{key}:{pair_id}")
    status = index.get(pair_id)
    if status is None or status.get("status") != "ok":
        raise ValueError(f"prediction_not_ok:{pair_id}:{None if status is None else status.get('status')}")
    if status.get("input_sha256") != pair["input_sha256"]:
        raise ValueError(f"prediction_input_sha256_mismatch:{pair_id}")
    pair_path = Path(pair_id)
    if pair_path.is_absolute() or ".." in pair_path.parts:
        raise ValueError("unsafe_pair_id")
    path = Path(directory) / "predictions" / pair_path.with_suffix(".npz")
    tracked[str(path.resolve())] = file_hash(path)
    if status.get("prediction_sha256") != tracked[str(path.resolve())]:
        raise ValueError(f"prediction_sha256_mismatch:{pair_id}")
    with np.load(path, allow_pickle=False) as data:
        depth = np.array(data["native_float_z"], dtype=np.float32, copy=True)
        disparity = np.array(data["disparity_px"], dtype=np.float32, copy=True)
        raw_valid = np.array(data["valid_mask"], copy=True)
        if raw_valid.dtype != bool and not np.isin(raw_valid, (0, 1)).all():
            raise ValueError("invalid_valid_mask_values")
        valid = raw_valid.astype(bool)
    left = cv2.imread(str(pair["left_path"]), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(pair["right_path"]), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None or left.shape != right.shape:
        raise ValueError("missing_or_mismatched_IR_pair")
    if depth.shape != left.shape or valid.shape != depth.shape or disparity.shape != depth.shape:
        raise ValueError("prediction_shape_differs_from_input_IR")
    if depth.shape != (pair["height"], pair["width"]):
        raise ValueError("manifest_dimensions_differ_from_input_IR")
    if not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError("nonfinite_or_negative_prediction_depth")
    if np.any(valid & (~np.isfinite(disparity) | (disparity <= 0) | (depth <= 0))):
        raise ValueError("declared_valid_prediction_has_invalid_disparity_or_depth")
    focal_baseline = float(pair["K_ir"][0][0]) * float(pair["baseline_m"])
    if not np.isfinite(focal_baseline) or focal_baseline <= 0:
        raise ValueError("invalid_IR_focal_baseline")
    # Detect RGB focal length / millimetre-baseline mistakes before registration.
    if not np.allclose(depth[valid] * disparity[valid], focal_baseline,
                       rtol=2e-5, atol=1e-6):
        raise ValueError("native_depth_disparity_IR_focal_baseline_mismatch")
    outside = valid & ((depth < .2) | (depth >= 10.))
    invalidated = int(outside.sum())
    valid &= (depth >= .2) & (depth < 10.)
    depth[~valid] = 0
    return depth, dict(prediction_status=status, prediction_path=str(path.resolve()),
                       prediction_sha256=tracked[str(path.resolve())],
                       range_invalidated_pixels=invalidated,
                       valid_pixels=int(valid.sum()), total_pixels=int(valid.size))


def prepare_frame(pair, depth, tracked):
    """Same 1 mm Z16 and SDK raw registration for every supplied prediction."""
    sensor_path = Path(pair["sensor_path"])
    tracked[str(sensor_path.resolve())] = file_hash(sensor_path)
    sensor = _json(sensor_path)
    matches = [c for c in sensor["cameras"] if c["name"] == pair["camera"]]
    if len(matches) != 1:
        raise ValueError("sensor_camera_missing_or_duplicated")
    item = matches[0]
    for key, sensor_key in (("K_ir", "K_ir"), ("T_world_from_left_ir", "T_world_from_left_ir")):
        np.testing.assert_allclose(pair[key], item[sensor_key], rtol=0, atol=1e-10)
    if not np.isclose(pair["baseline_m"], item["baseline_m"], rtol=0, atol=1e-10):
        raise ValueError("manifest_sensor_baseline_mismatch")
    rgb_path = Path(pair["scene_dir"]) / "observations" / (pair["camera"] + "_rgb.png")
    tracked[str(rgb_path.resolve())] = file_hash(rgb_path)
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("ordinary_RGB_missing")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_camera = Camera(pair["camera"], np.array(item["K_rgb"]),
                        np.array(item["T_world_from_rgb"]), rgb.shape[1], rgb.shape[0])
    native_camera = Camera(pair["camera"], np.array(pair["K_ir"]),
                           np.array(pair["T_world_from_left_ir"]), depth.shape[1], depth.shape[0])
    processor = OfflineSpatialFilter(native_camera, pair["baseline_m"], StereoSettings(),
                                     rgb_camera=rgb_camera)
    try:
        quantized, _unused_filtered = processor.process(depth)
        aligned = processor.register(rgb, filtered=False)
    finally:
        processor.close()
    if not np.isfinite(aligned).all() or (aligned < 0).any():
        raise ValueError("invalid_registered_depth")
    info = dict(native_float_valid=int((depth > 0).sum()),
                native_quantized_valid=int((quantized > 0).sum()),
                aligned_valid=int((aligned > 0).sum()),
                foreground_sha256=sha256(blue_screen_foreground(rgb).astype(np.uint8).tobytes()).hexdigest(),
                ordinary_RGB_sha256=tracked[str(rgb_path.resolve())])
    return DepthFrame(rgb_camera, rgb, aligned), native_camera, info


def _load_truth(pair, tracked):
    # Called only after inference for this case/condition, never by an estimator.
    folder = Path(pair["render_root"]) / "geometry" / pair["case"] / "evaluation_only"
    target_path, profile_path = folder / "target.json", folder / "profile.npz"
    tracked[str(target_path.resolve())] = file_hash(target_path)
    tracked[str(profile_path.resolve())] = file_hash(profile_path)
    target = _json(target_path)
    with np.load(profile_path, allow_pickle=False) as data:
        profile = {k: data[k].copy() for k in ("s_m", "radius_m")}
    return target, profile


def region_metrics(predicted, ideal, region):
    """Fixed evaluation pixels: invalid predictions remain in coverage denominator."""
    region = np.asarray(region, bool) & np.isfinite(ideal) & (ideal > 0)
    valid = region & np.isfinite(predicted) & (predicted > 0)
    error = (np.asarray(predicted)[valid] - np.asarray(ideal)[valid]) * 1000
    count, available = int(region.sum()), int(valid.sum())
    return dict(reference_pixels=count, valid_prediction_pixels=available,
                missing_prediction_pixels=count-available,
                valid_coverage=available/count if count else None,
                signed_mean_mm=float(np.mean(error)) if available else None,
                median_abs_mm=float(np.median(abs(error))) if available else None,
                rmse_mm=float(np.sqrt(np.mean(error**2))) if available else None,
                p95_abs_mm=float(np.percentile(abs(error), 95)) if available else None,
                max_abs_mm=float(np.max(abs(error))) if available else None,
                absolute_error_le_2mm_fraction_all_reference=(int((abs(error) <= 2).sum())/count if count else None))


def measurement_metrics(pair, predicted, frame, native_camera, truth, profile, tracked,
                        common_predictions=None, common_errors=None, common_members=()):
    """Reference-defined regions; target labels are evaluation-only."""
    path = Path(pair["render_root"]) / "sensor_diagnostics" / pair["case"] / (pair["camera"] + "_native.npz")
    tracked[str(path.resolve())] = file_hash(path)
    with np.load(path, allow_pickle=False) as data:
        ideal = data["ideal_left_z"].copy()
    if ideal.shape != predicted.shape:
        raise ValueError("ideal_and_predicted_native_shapes_differ")
    yy, xx = np.nonzero(np.isfinite(ideal) & (ideal > 0))
    world = native_camera.unproject(np.c_[xx, yy], ideal[yy, xx])
    local_rgb = (world-frame.camera.T_world_from_camera[:3, 3]) @ frame.camera.T_world_from_camera[:3, :3]
    front = local_rgb[:, 2] > 0
    uv = np.zeros((len(world), 2), dtype=np.int64)
    uv[front] = np.rint(frame.camera.project(world[front])).astype(np.int64)
    inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < frame.camera.width) & (uv[:, 1] >= 0) & (uv[:, 1] < frame.camera.height)
    foreground = np.zeros(len(world), bool)
    image_mask = blue_screen_foreground(frame.rgb)
    foreground[inside] = image_mask[uv[inside, 1], uv[inside, 0]]
    # Same reference rays for all predictors; NOT measured-depth-dependent masks.
    full = np.zeros(ideal.shape, bool)
    full[yy[foreground], xx[foreground]] = True
    upper = np.zeros(ideal.shape, bool)
    station = foreground & (world[:, 2] >= .40) & (world[:, 2] <= .76) & (np.linalg.norm(world[:, :2], axis=1) < .28)
    upper[yy[station], xx[station]] = True
    n = np.asarray(truth["normal"], float)
    delta = world-np.asarray(truth["center_m"], float)
    axial = delta @ n
    profile_s = axial + float(truth["s_m"])
    radial = np.linalg.norm(delta-axial[:, None]*n, axis=1)
    lo, hi = truth["search_interval_s_m"]
    neck_selection = ((profile_s >= lo) & (profile_s <= hi) &
                      (abs(radial-np.interp(profile_s, profile["s_m"], profile["radius_m"])) < .001))
    neck = np.zeros(ideal.shape, bool)
    neck[yy[neck_selection], xx[neck_selection]] = True
    results = {}
    for label, mask in (
        ("ordinary_RGB_foreground_reference_projection", full),
        ("fixed_upper_station_reference_projection", upper),
        ("GT_visible_outer_neck_evaluation_only", neck)):
        metrics = region_metrics(predicted, ideal, mask)
        if common_predictions is not None:
            metrics["common_valid_comparison"] = common_region_metrics(
                predicted, ideal, mask, common_predictions, common_members, common_errors)
        results[label] = metrics
    return results


def common_region_metrics(predicted, ideal, region, predictions, members, errors=None):
    """Identical intersection pixels for each member, alongside its own coverage.

    A missing comparator invalidates the common-support comparison instead of
    silently producing a more favourable intersection from fewer methods.
    """
    if errors:
        return dict(status="comparison_input_error", members=list(members), errors=errors)
    reference = np.asarray(region, bool) & np.isfinite(ideal) & (ideal > 0)
    common = reference.copy()
    for depth in (predicted, *predictions):
        if np.asarray(depth).shape != ideal.shape:
            raise ValueError("common_prediction_shape_mismatch")
        common &= np.isfinite(depth) & (depth > 0)
    result = region_metrics(predicted, ideal, common)
    # The nested error denominator is the explicit common support, not all
    # reference pixels. Coverage against the fixed region is reported separately.
    result.update(status="ok", members=list(members),
                  common_valid_pixels=int(common.sum()),
                  fixed_reference_pixels=int(reference.sum()),
                  common_coverage_of_fixed_reference=(float(common.sum()/reference.sum())
                                                       if reference.any() else None))
    return result


def aggregate(rows):
    n = len(rows)
    out = dict(n=n, candidates=sum(r.get("estimate") is not None for r in rows),
               candidate_geometry_pass=sum(bool(r["truth_pass"]) for r in rows),
               accepted=sum(bool(r["accepted"]) for r in rows),
               correct_output=sum(bool(r["accepted"] and r["truth_pass"]) for r in rows),
               wrong_accepted=sum(bool(r["accepted"] and not r["truth_pass"]) for r in rows),
               rejected=sum(not r["accepted"] for r in rows),
               input_or_backend_errors=sum(r["status"] != "ok" for r in rows))
    out.update(candidate_geometry_rate=out["candidate_geometry_pass"]/n if n else None,
               correct_output_rate=out["correct_output"]/n if n else None,
               wrong_accepted_rate=out["wrong_accepted"]/n if n else None)
    out["reasons"] = {reason: sum(reason in r["reasons"] for r in rows)
                      for reason in sorted({s for r in rows for s in r["reasons"]})}
    for metric in ("center_mm", "normal_deg", "diameter_mm", "runtime_ms", "frontend_inference_ms"):
        values = [float(r[metric]) for r in rows if r.get(metric) is not None]
        for suffix, p in (("median", 50), ("p95", 95), ("max", 100)):
            out[metric+"_"+suffix] = float(np.percentile(values, p)) if values else None
    return out


def _validate_estimate(estimate):
    if estimate.center is None:
        if estimate.accepted:
            raise ValueError("backend_accepted_without_candidate")
        return
    if (np.asarray(estimate.center).shape != (3,) or np.asarray(estimate.normal).shape != (3,)
        or not np.isfinite(estimate.center).all() or not np.isfinite(estimate.normal).all()
        or estimate.radius is None or not np.isfinite(estimate.radius) or estimate.radius <= 0
        or not np.isclose(np.linalg.norm(estimate.normal), 1, atol=1e-5)):
        raise ValueError("backend_nonfinite_or_invalid_circle")


def run(manifest_path, predictions, output, camera_sets=("all", "cam_a"), common_with=(), layouts=(),
        measurement_only=False):
    manifest_path, predictions, output = map(Path, (manifest_path, predictions, output))
    if output.exists():
        raise FileExistsError("Output directory must be new")
    manifest = _json(manifest_path)
    if manifest.get("schema") != 1 or not manifest.get("pairs"):
        raise ValueError("Expected schema=1 and nonempty pairs")
    evaluation_role = manifest.get("evaluation_role", "method_capability_audit")
    groups = defaultdict(list)
    pair_ids = set()
    for pair in manifest["pairs"]:
        if pair["pair_id"] in pair_ids:
            raise ValueError("duplicate_manifest_pair_id")
        pair_ids.add(pair["pair_id"])
        if layouts and pair["layout"] not in layouts:
            continue
        groups[(pair["layout"], pair["case"], pair["condition"])].append(pair)
    if not groups:
        raise ValueError("No groups match requested layouts")
    sets = {name: ALL_CAMERAS if name == "all" else tuple(name.split(",")) for name in camera_sets}
    if any(not names or len(set(names)) != len(names) or not set(names) <= set(ALL_CAMERAS) for names in sets.values()):
        raise ValueError("Unknown/duplicate requested camera")
    index = load_prediction_index(predictions)
    sources, settings = source_hashes(), frozen_settings()
    tracked = {str(manifest_path.resolve()): file_hash(manifest_path),
               str((predictions/"predictions.jsonl").resolve()): file_hash(predictions/"predictions.jsonl")}
    common_roots = tuple(Path(p).resolve() for p in common_with)
    if len(set(common_roots)) != len(common_roots) or predictions.resolve() in common_roots:
        raise ValueError("duplicate_common_prediction_directory")
    common_indexes = {p: load_prediction_index(p) for p in common_roots}
    for p in common_roots:
        tracked[str(p/"predictions.jsonl")] = file_hash(p/"predictions.jsonl")
    common_members = [str(predictions.resolve()), *(str(p) for p in common_roots)]
    output.mkdir(parents=True)
    records, measurements, pair_checks = [], [], []
    for (layout, case, condition), pairs in sorted(groups.items()):
        names = [p["camera"] for p in pairs]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate camera within layout/case/condition")
        frames, depths, native_cameras, errors = {}, {}, {}, {}
        for pair in pairs:
            name = pair["camera"]
            info = dict(pair_id=pair["pair_id"], layout=layout, case=case, condition=condition, camera=name)
            try:
                depth, prediction_info = read_prediction(pair, predictions, index, tracked)
                frame, native_camera, registration_info = prepare_frame(pair, depth, tracked)
                frames[name], depths[name], native_cameras[name] = frame, depth, native_camera
                info.update(status="ok", **prediction_info, registration=registration_info)
            except Exception as error:
                errors[name] = f"input_error:{type(error).__name__}:{error}"
                info.update(status="error", error=errors[name], traceback=traceback.format_exc())
            pair_checks.append(info)
        pending = []
        # No evaluation_only, ideal depth, mesh or true dimensions opened above.
        for camera_set, requested in (() if measurement_only else sets.items()):
            absent = [n for n in requested if n not in frames]
            for backend, infer in BACKENDS.items():
                status, estimate = "ok", Estimate()
                if absent:
                    status = "input_error"
                    estimate.reasons = [errors.get(n, "missing_camera_prediction:"+n) for n in absent]
                else:
                    started = perf_counter()
                    try:
                        estimate = infer([frames[n] for n in requested])
                        _validate_estimate(estimate)
                    except Exception as error:
                        status = "backend_error"
                        estimate = Estimate(reasons=[f"backend_exception:{type(error).__name__}:{error}"],
                                            diagnostics=dict(traceback=traceback.format_exc()))
                    estimate.elapsed_ms = (perf_counter()-started)*1000
                pending.append((camera_set, requested, backend, status, estimate))
        # Target/profile are opened only AFTER every backend/cameraset above.
        truth, profile = _load_truth(pairs[0], tracked)
        for camera_set, requested, backend, status, estimate in pending:
            metrics = score(estimate, truth)
            metrics["fit_condition"] = metrics.pop("condition")
            row = dict(layout=layout, case=case, condition=condition, camera_set=camera_set,
                       backend=backend, status=status, split=pairs[0]["split"],
                       evaluation_role=evaluation_role,
                       **metrics)
            times = [index[p["pair_id"]].get("inference_ms") for p in pairs
                     if p["camera"] in requested and p["pair_id"] in index]
            row["frontend_inference_ms"] = (sum(float(t) for t in times)
                if len(times) == len(requested) and all(t is not None for t in times) else None)
            row["band"], row["band_evaluation"] = None, None
            if estimate.center is not None and "near_min_interval_from_axis_origin_m" in estimate.diagnostics:
                try:
                    band = plan_band(row["estimate"], estimate.diagnostics)
                    row["band"] = band
                    row["band_evaluation"] = {label: evaluate_band(band, truth, truth[field]) for label, field in (
                        ("near_min_proxy", "near_min_interval_s_m"), ("full_annotated_neck", "search_interval_s_m"))}
                except Exception as error:
                    row["band_evaluation_error"] = f"{type(error).__name__}:{error}"
            records.append(row)
            with (output/"records.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
        for pair in pairs:
            name = pair["camera"]
            row = dict(pair_id=pair["pair_id"], layout=layout, case=case, condition=condition, camera=name)
            if name in errors:
                row.update(status="input_error", error=errors[name])
            else:
                try:
                    common_depths, common_errors = [], {}
                    for directory, comparison_index in common_indexes.items():
                        try:
                            comparison_depth, _ = read_prediction(pair, directory, comparison_index, tracked)
                            common_depths.append(comparison_depth)
                        except Exception as error:
                            common_errors[str(directory)] = f"{type(error).__name__}:{error}"
                    row.update(status="ok", regions=measurement_metrics(pair, depths[name], frames[name],
                        native_cameras[name], truth, profile, tracked,
                        common_predictions=common_depths if common_roots else None,
                        common_errors=common_errors, common_members=common_members))
                except Exception as error:
                    row.update(status="evaluation_error", error=f"{type(error).__name__}:{error}")
            measurements.append(row)
        print(layout, case, condition, "backends", [(b, c, e.accepted) for c, _, b, _, e in pending], flush=True)
    summaries = []
    keys = sorted({(r["layout"], r["condition"], r["camera_set"], r["backend"]) for r in records})
    for key in keys:
        chosen = [r for r in records if (r["layout"], r["condition"], r["camera_set"], r["backend"]) == key]
        summaries.append(dict(zip(("layout", "condition", "camera_set", "backend"), key),
                              evaluation_role=evaluation_role, **aggregate(chosen)))
    _write_json(output/"summary.json", summaries)
    _write_json(output/"measurement_metrics.json", measurements)
    _write_json(output/"pair_checks.json", pair_checks)
    (output/"records.json.gz").write_bytes(gzip.compress(json.dumps(records, allow_nan=False).encode(), mtime=0))
    changed = [p for p, h in tracked.items() if not Path(p).exists() or file_hash(p) != h]
    valid = not changed and sources == source_hashes() and settings == frozen_settings()
    _write_json(output/"validation.json", dict(passed=valid, changed_inputs=changed,
        frozen_sources_unchanged=sources == source_hashes(), settings_unchanged=settings == frozen_settings(),
        group_count=len(groups), backend_record_count=len(records),
        pair_count=sum(len(p) for p in groups.values()), manifest_pair_count=len(pair_ids)))
    _write_json(output/"provenance.json", dict(schema=1, manifest=str(manifest_path.resolve()),
        evaluation_role=evaluation_role,
        predictions=str(predictions.resolve()), input_sha256=tracked, source_sha256=sources, settings=settings,
        camerasets=list(camera_sets), layouts=list(layouts),
        mode="measurements_only" if measurement_only else "geometry_and_measurements",
        backends=[] if measurement_only else list(BACKENDS),
        common_valid_comparison_members=common_members if common_roots else [],
        no_hardware=True, no_gpu_inference=True,
        registration="same native-Z16 quantization at 0.001 m; offline SDK align(raw); explicit left-Z to RGB-Z",
        foreground="ordinary_RGB_only_blue_key_largest_filled_contour",
        inference_truth_access="target/profile/ideal depth first read after all backend calls per case/condition",
        evaluation_thresholds=dict(center_mm=4, normal_deg=2, diameter_mm=2),
        acceptance="original frozen gates retained; no gate-removal counterfactual or GT candidate selection",
        measurement_regions="fixed ideal-left rays; reference projection onto ordinary RGB mask; fixed .40-.76m upper station; GT outer neck separately evaluation-only",
        limitations=["previously opened synthetic cases; no unseen-test or real-device accuracy claim",
                     "paired layouts share jar geometry; not independent additional jars",
                     "valid-only error statistics must be read with all-reference coverage and missing count",
                     "no calibrated confidence; no bootstrap coverage claim for shared calibration errors",
                     "known incorrect portfolio silhouette gate remains visible in accepted-output results"] ))
    if not valid:
        raise RuntimeError("Input/source/settings changed during evaluation")
    return records, measurements, summaries


if __name__ == "__main__":
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-sets", "--camerasets", nargs="+", default=["all", "cam_a"])
    parser.add_argument("--common-with", type=Path, nargs="*", default=[],
                        help="Other prediction roots; report the identical valid-pixel intersection")
    parser.add_argument("--layouts", nargs="*", default=[])
    parser.add_argument("--measurements-only", action="store_true",
                        help="Only depth/common-support metrics; do not repeat geometry estimation")
    args = parser.parse_args()
    run(args.manifest, args.predictions, args.output, args.camera_sets, args.common_with, args.layouts,
        args.measurements_only)
