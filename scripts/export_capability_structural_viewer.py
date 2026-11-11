#!/usr/bin/env python3
"""Export saved capability results into the existing MuJoCo viewer contract.

This is post-processing only. Depth is re-created with the exact evaluator's
quantization and RGB registration; RGB, meshes, targets and fitted points remain
their original saved sources. No model or geometry estimator is called.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from neck_localization.stereo_robust import extract_stereo_masks
from scripts.evaluate_capability_depth import (
    file_hash, load_prediction_index, prepare_frame, read_prediction,
)


def read_records(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()))


def viewer_record(row, frontend, source):
    """Add viewer routing fields without modifying saved scientific results."""
    value = copy.deepcopy(row)
    value.update(method=row["backend"], evidence="rgb", frontend=frontend,
                 source_records=str(Path(source).resolve()))
    return value


def check_fit_subset(row, frames, tolerance_m=2e-6, masks=None):
    """Saved fitted measurements must match these actual depth observations."""
    d = row.get("diagnostics", {})
    if "fit_point_positions_m" not in d:
        return dict(points=0, status="no_saved_fit_subset")
    points = np.asarray(d["fit_point_positions_m"], float).reshape(-1, 3)
    ids = np.asarray(d["fit_camera_ids"])
    weights = np.asarray(d["fit_point_weights"])
    names = d["fit_camera_names"]
    if len(ids) != len(points) or len(weights) != len(points):
        raise ValueError("saved_fit_subset_length_mismatch")
    if (len(names) != len(set(names)) or ids.ndim != 1 or not np.isin(ids, range(len(names))).all()
            or not np.isfinite(points).all() or not np.isfinite(weights).all()):
        raise ValueError("saved_fit_subset_has_invalid_camera_ids_or_values")
    maximum = 0.
    by_name = {f.camera.name: f for f in frames}
    for index, name in enumerate(names):
        subset = points[ids == index]
        if not len(subset):
            continue
        f = by_name[name]
        uv = np.rint(f.camera.project(subset)).astype(int)
        if (np.any(uv < 0) or np.any(uv[:, 0] >= f.camera.width)
                or np.any(uv[:, 1] >= f.camera.height)):
            raise ValueError("saved_fit_points_outside_image")
        depth = f.depth_z_m[uv[:, 1], uv[:, 0]]
        if masks is not None:
            foreground = masks[name]
            if not np.all(foreground[uv[:, 1], uv[:, 0]]):
                raise ValueError("saved_fit_points_outside_exported_foreground:"+name)
        restored = f.camera.unproject(uv, depth)
        error = np.linalg.norm(restored-subset, axis=1)
        if np.any(depth <= 0) or np.max(error) > tolerance_m:
            raise ValueError(f"saved_fit_points_differ_from_registered_depth:{name}:{np.max(error)}")
        maximum = max(maximum, float(np.max(error)))
    return dict(points=len(points), max_observation_difference_m=maximum, status="verified")


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def _source_spec(value):
    name, separator, directory = value.partition("=")
    if not separator or not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("Use --evaluation frontend_name=/absolute/evaluation/directory")
    return name, Path(directory).resolve()


def preferred_method(methods):
    methods = sorted(set(methods))
    return next((m for m in ("visible_profile", "standard_sor", "portfolio") if m in methods), methods[0])


def load_foreground_cache(directory, tracked):
    """Read an explicit RGB segmentation cache, without generating new masks."""
    directory = Path(directory).resolve()
    path = directory if directory.is_file() else directory/"manifest.json"
    tracked[str(path)] = file_hash(path)
    value = json.loads(path.read_text())
    result = {}
    for entry in value["entries"]:
        key = (entry["layout"], entry["case"], entry["camera"])
        if key in result:
            raise ValueError("duplicate_foreground_cache_entry:"+str(key))
        result[key] = entry
    return result


def exported_masks(frames, pairs, cache, tracked):
    if cache is None:
        return extract_stereo_masks(frames)
    pair_index = {p["camera"]: p for p in pairs}
    masks, details = [], []
    for frame in frames:
        name = frame.camera.name
        pair = pair_index[name]
        key = (pair["layout"], pair["case"], name)
        if key not in cache:
            raise ValueError("foreground_cache_missing_camera:"+str(key))
        entry = cache[key]
        rgb_path = Path(pair["scene_dir"])/"observations"/(name+"_rgb.png")
        if file_hash(rgb_path) != entry["rgb_sha256"]:
            raise ValueError("foreground_cache_RGB_sha256_mismatch:"+str(key))
        mask_path = Path(entry["mask_path"])
        if not mask_path.is_absolute():
            raise ValueError("foreground_cache_mask_path_must_be_absolute")
        tracked[str(mask_path)] = file_hash(mask_path)
        if tracked[str(mask_path)] != entry["mask_sha256"]:
            raise ValueError("foreground_cache_mask_file_sha256_mismatch:"+str(key))
        image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != frame.depth_z_m.shape or not np.isin(image, (0, 255)).all():
            raise ValueError("foreground_cache_mask_shape_or_values_invalid:"+str(key))
        mask = image.astype(bool)
        pixels_hash = sha256(mask.astype(np.uint8).tobytes()).hexdigest()
        if pixels_hash != entry["mask_pixels_sha256"]:
            raise ValueError("foreground_cache_mask_pixel_sha256_mismatch:"+str(key))
        masks.append(mask)
        details.append(dict(camera=name, pixels=int(mask.sum()), mask_sha256=pixels_hash,
            mask_file_sha256=tracked[str(mask_path)], rgb_file_sha256=entry["rgb_sha256"]))
    return masks, dict(method="explicit_saved_RGB_segmentation_cache", cameras=details)


def run(evaluations, output, layouts=("close", "far")):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    grouped = defaultdict(list)
    inputs = {}
    for frontend, directory in evaluations:
        directory = Path(directory).resolve()
        provenance_path = directory/"provenance.json"
        records_path = directory/"records.json.gz"
        provenance = json.loads(provenance_path.read_text())
        manifest_path = Path(provenance["manifest"])
        prediction_root = Path(provenance["predictions"])
        cache_root = (Path(provenance["foreground_cache"]).resolve()
                      if provenance.get("foreground_cache") else None)
        manifest = json.loads(manifest_path.read_text())
        inputs.update({str(p): file_hash(p) for p in (provenance_path, records_path, manifest_path)})
        by_group = defaultdict(list)
        for pair in manifest["pairs"]:
            by_group[(pair["layout"], pair["case"], pair["condition"])].append(pair)
        for row in read_records(records_path):
            if row["layout"] not in layouts:
                continue
            condition = f"{row['layout']}/{frontend}/{row['condition']}"
            grouped[condition].append((viewer_record(row, frontend, records_path),
                by_group[(row["layout"], row["case"], row["condition"])], prediction_root, cache_root))
    if not grouped:
        raise ValueError("No matching saved results")
    output.mkdir(parents=True)
    datasets, scene_checks, fit_checks = [], [], []
    indexes, caches = {}, {}
    for condition, entries in sorted(grouped.items()):
        folder = output/condition
        folder.mkdir(parents=True)
        rows = [e[0] for e in entries]
        identities = [(r["case"], r["camera_set"], r["method"], r["evidence"]) for r in rows]
        if len(identities) != len(set(identities)):
            raise ValueError(f"duplicate_viewer_records:{condition}")
        for case in sorted({r["case"] for r in rows}):
            chosen = [e for e in entries if e[0]["case"] == case]
            _row, pairs, prediction_root, cache_root = chosen[0]
            # All merged readouts must have been evaluated on identical frames.
            if any(p != pairs or root != prediction_root for _, p, root, _ in chosen):
                raise ValueError("merged_readouts_have_different_prediction_inputs")
            if any(cache != cache_root for _, _, _, cache in chosen):
                raise ValueError("different_foreground_versions_require_separate_frontend_routes")
            if {p["camera"] for p in pairs} != {"cam_a", "cam_b", "cam_c"} or len(pairs) != 3:
                raise ValueError("three_unique_saved_camera_predictions_required")
            source = Path(pairs[0]["scene_dir"])
            destination = folder/"scenes"/case
            observations = destination/"observations"
            observations.mkdir(parents=True)
            for child in ("assets", "evaluation_only", "model_manifest.json"):
                (destination/child).symlink_to((source/child).resolve(), target_is_directory=(source/child).is_dir())
            shutil.copy2(source/"scene.xml", destination/"scene.xml")
            metadata_path = source/"observations/camera_true_parameters.json"
            (observations/metadata_path.name).symlink_to(metadata_path.resolve())
            inputs[str(metadata_path.resolve())] = file_hash(metadata_path)
            if prediction_root not in indexes:
                indexes[prediction_root] = load_prediction_index(prediction_root)
                status_path = prediction_root/"predictions.jsonl"
                inputs[str(status_path)] = file_hash(status_path)
            frames = []
            for pair in sorted(pairs, key=lambda p: p["camera"]):
                native, prediction = read_prediction(pair, prediction_root, indexes[prediction_root], inputs)
                frame, _, registration = prepare_frame(pair, native, inputs)
                frames.append(frame)
                name = pair["camera"]
                rgb_path = source/"observations"/(name+"_rgb.png")
                (observations/rgb_path.name).symlink_to(rgb_path.resolve())
                depth_path = observations/(name+"_depth_z_m.npz")
                np.savez_compressed(depth_path, depth_z_m=frame.depth_z_m)
                with np.load(depth_path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["depth_z_m"], frame.depth_z_m)
                scene_checks.append(dict(condition=condition, case=case, camera=name,
                    pair_id=pair["pair_id"], depth_file_sha256=file_hash(depth_path),
                    prediction_sha256=prediction["prediction_sha256"], registration=registration))
            if cache_root is not None and cache_root not in caches:
                caches[cache_root] = load_foreground_cache(cache_root, inputs)
            masks, foreground = exported_masks(frames, pairs, caches.get(cache_root), inputs)
            (destination/"foreground_masks").mkdir()
            for f, mask, info in zip(frames, masks, foreground["cameras"]):
                mask_path = destination/"foreground_masks"/(f.camera.name+"_foreground.png")
                cv2.imwrite(str(mask_path), mask.astype(np.uint8)*255)
                for row, _, _, _ in chosen:
                    expected = row.get("diagnostics", {}).get("foreground", {}).get("cameras", [])
                    expected_camera = [c for c in expected if c["camera"] == f.camera.name]
                    if expected_camera:
                        expected_hash = expected_camera[0]["mask_sha256"]
                        if expected_hash != info["mask_sha256"]:
                            raise ValueError("foreground_mask_differs_from_saved_inference")
            for row, _, _, _ in chosen:
                fit_checks.append(dict(condition=condition, case=case, method=row["method"],
                    camera_set=row["camera_set"], foreground_cache=str(cache_root) if cache_root else None,
                    **check_fit_subset(row, frames, masks={f.camera.name: m for f, m in zip(frames, masks)})))
            print("exported", condition, case, flush=True)
        (folder/"records.json.gz").write_bytes(gzip.compress(
            json.dumps(rows, allow_nan=False).encode(), mtime=0))
        layout, frontend, noise = condition.split("/")
        frontend_label = {"foundation_stereo":"FoundationStereo", "simsense":"SimSense",
            "foundation_stereo_sam":"FS+SAM2诊断", "simsense_sam":"传统双目+SAM2诊断",
            "foundation_stereo_confirm_sam":"FS+SAM2新8", "simsense_confirm_sam":"传统双目+SAM2新8",
            "foundation_stereo_confirm":"FS新8坛", "simsense_confirm":"传统双目新8坛",
            "ideal_native":"理想深度"}.get(frontend, frontend)
        full_label = ("近" if layout == "close" else "远")+"·"+frontend_label+("净" if noise == "clean" else "噪")
        label = (full_label.replace("FoundationStereo", "FS").replace("SimSense", "传统双目")
                 .replace("传统双目新8坛", "双目新8")
                 .replace("FS+SAM2诊断", "FS+SAM2")
                 .replace("传统双目+SAM2诊断", "双目+SAM2")
                 .replace("FS+SAM2新8", "FS8+SAM2")
                 .replace("传统双目+SAM2新8", "双目8+SAM2"))
        _write_json(folder/"settings.json", dict(preprocessing=dict(kind="saved_capability_predictions", condition=full_label),
            depth_display_range_m=[.18, .85] if layout == "close" else [.45, 1.35],
            observation_semantics="same evaluator prepare_frame: native 1mm Z16, SDK raw RGB alignment, explicit RGB optical Z"))
        methods = sorted({r["method"] for r in rows})
        datasets.append(dict(condition=condition, label=label, scenes=str(folder/"scenes"),
            records=str(folder/"records.json.gz"), method=preferred_method(methods)))
    changed = [p for p, h in inputs.items() if file_hash(p) != h]
    if changed:
        raise RuntimeError(f"source_files_changed:{changed}")
    _write_json(output/"datasets.json", datasets)
    _write_json(output/"validation.json", dict(passed=True, postprocessing_only=True,
        no_model_inference=True, no_estimator_calls=True, input_sha256=inputs,
        scene_checks=scene_checks, fit_checks=fit_checks, changed_sources=changed,
        saved_rows_preserved_except_viewer_routing_fields=True,
        foreground_semantics="same saved inference mask: ordinary RGB blue-key or explicit segmentation cache; never mesh/GT mask"))
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", required=True, type=_source_spec,
                        help="frontend_name=/path/to/evaluation; repeat for readouts/frontends")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layouts", nargs="+", default=["close", "far"])
    args = parser.parse_args()
    cv2.setNumThreads(1)
    run(args.evaluation, args.output, args.layouts)
