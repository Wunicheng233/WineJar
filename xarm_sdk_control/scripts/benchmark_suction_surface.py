#!/usr/bin/env python3
"""Compare a trusted local source snapshot with current suction search, offline."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from xarm_sdk_control import suction_surface as current


def synthetic_cases():
    profile = current.SurfaceProfile(5.0, 15, .10, 1.0, 15.0, .90)
    matrix = np.array([[600., 0, 160.], [0, 600., 120.], [0, 0, 1.]])
    for name in ("flat", "crack", "depth_hole", "no_feasible", "non_default", "image_edges"):
        color = np.full((240, 320, 3), (120, 180, 160), np.uint8)
        depth = np.full((240, 320), .4, np.float32)
        mask = np.zeros((240, 320), np.uint8)
        cv2.rectangle(mask, (30, 30), (290, 210), 255, -1)
        options = {}
        if name in ("crack", "non_default"):
            cv2.line(color, (160, 40), (160, 200), (10, 10, 10), 5)
        if name == "depth_hole":
            depth[90:150, 135:185] = 0
        if name == "no_feasible":
            options = dict(cup_spacing_mm=1000.)
        if name == "non_default":
            options = dict(grid_step_px=9, angle_step_deg=22.5, preferred_angle_deg=137.)
        if name == "image_edges":
            mask[:] = 255
        yield name, (color, depth, mask, matrix, profile), options


def real_cases(folder):
    metadata = json.loads((folder / "frame.json").read_text())
    color = cv2.imread(str(folder / metadata["files"]["color_bgr"]))
    raw = cv2.imread(str(folder / metadata["files"]["depth_raw"]), -1)
    if color is None or raw is None:
        raise ValueError(f"Cannot load RGB-D from {folder}")
    depth = raw.astype(np.float32) * metadata["stream"]["depth_scale_m_per_unit"]
    matrix = np.asarray(metadata["intrinsics"]["camera_matrix"])
    for name, mask_name in (("lotus", "mask_00_lotus_leaf.png"), ("ceramic", "mask_01_ceramic_disk.png")):
        mask = cv2.imread(str(folder / mask_name), 0)
        if mask is None:
            raise ValueError(f"Cannot load {mask_name}")
        saved = json.loads((folder / f"suction_surface_result_{name}.json").read_text())
        profile = current.SurfaceProfile(**saved["profile"])
        yield name, (color, depth, mask, matrix, profile), {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Trusted Python source snapshot to import.")
    parser.add_argument("--frame-dir", type=Path, help="Optional 20260903 suction evaluation frame directory.")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.output.exists():
        parser.error("--output must be a new file")
    spec = importlib.util.spec_from_file_location("suction_baseline", args.baseline.resolve())
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    cv2.setNumThreads(1)
    cases = list(synthetic_cases())
    if args.frame_dir:
        cases.extend(real_cases(args.frame_dir.resolve()))
    report = {
        "python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(), "repeats": args.repeats,
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "current_sha256": hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
        "frame_dir": str(args.frame_dir.resolve()) if args.frame_dir else None,
        "cases": [],
    }
    for name, inputs, options in cases:
        # Full search is for equivalence; timings below use the public default top 20.
        before = baseline.analyze_suction_surface(*inputs, max_candidates=1000000, **options)
        after = current.analyze_suction_surface(*inputs, max_candidates=1000000, **options)
        assert [baseline.candidate_to_dict(c) for c in before.candidates] == [
            current.candidate_to_dict(c) for c in after.candidates
        ], f"Candidate mismatch: {name}"
        for field in ("object_mask", "defect_mask", "roughness_mm"):
            np.testing.assert_array_equal(getattr(before, field), getattr(after, field))
        for field in ("cup_radius_px", "cup_spacing_px", "reference_depth_m"):
            assert getattr(before, field) == getattr(after, field)
        samples = {"before": [], "after": []}
        for repeat in range(args.repeats):
            order = [("before", baseline), ("after", current)]
            for label, module in (order if repeat % 2 == 0 else order[::-1]):
                start = time.perf_counter()
                result = module.analyze_suction_surface(*inputs, **options)
                samples[label].append(time.perf_counter() - start)
                assert [module.candidate_to_dict(c) for c in result.candidates] == [
                    baseline.candidate_to_dict(c) for c in before.candidates[:20]
                ]
        medians = {label: statistics.median(values) for label, values in samples.items()}
        item = {
            "name": name, "options": options, "candidate_count": len(after.candidates),
            "exact_match": True, "seconds": samples, "median_seconds": medians,
            "reduction_percent": 100 * (1 - medians["after"] / medians["before"]),
            "input_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in inputs[:4]],
            "profile": vars(inputs[4]),
        }
        report["cases"].append(item)
        print(f"{name}: {medians['before']:.6f} -> {medians['after']:.6f} s "
              f"({item['reduction_percent']:.1f}% less), {len(after.candidates)} candidates match", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
