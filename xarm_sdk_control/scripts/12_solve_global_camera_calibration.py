#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.calibration_utils import (  # noqa: E402
    matrix_to_list,
    residuals_m,
    solve_rigid_transform,
    transform_points,
)


SAMPLES_ROOT = ROOT / "data" / "global_camera_samples"
OUTPUT_ROOT = ROOT / "calibration"


def load_samples(samples_root: Path, excludes: set[str]):
    samples = []
    skipped = []
    for path in sorted(samples_root.glob("*/sample.json")):
        if path.parent.name in excludes or str(path) in excludes:
            skipped.append((path, "explicitly excluded"))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("type") != "global_camera_point_pair":
            skipped.append((path, "not a global camera point-pair sample"))
            continue
        pair = data.get("point_pair", {})
        camera_point = pair.get("camera_point_m")
        base_point = pair.get("base_point_m")
        if camera_point is None or base_point is None:
            skipped.append((path, "missing camera_point_m or base_point_m"))
            continue
        samples.append(
            {
                "path": path,
                "timestamp": data.get("timestamp", path.parent.name),
                "point_name": data.get("point_name"),
                "camera_point_m": np.array(camera_point, dtype=np.float64),
                "base_point_m": np.array(base_point, dtype=np.float64),
                "median_depth_m": data.get("selection", {}).get("median_depth_m"),
            }
        )
    return samples, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Solve fixed global-camera calibration: base_point = base_T_camera * camera_point."
    )
    parser.add_argument("--samples-root", type=Path, default=SAMPLES_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--exclude", action="append", default=[], help="Sample timestamp folder or sample.json path to exclude. Can be repeated.")
    args = parser.parse_args()

    samples, skipped = load_samples(args.samples_root, set(args.exclude))
    print(f"Loaded valid global-camera samples: {len(samples)}")
    if skipped:
        print(f"Skipped samples: {len(skipped)}")
        for path, reason in skipped:
            print(f"  - {path}: {reason}")
    if len(samples) < 4:
        print("Need at least 4 samples; 8-12 spread-out points is preferred.")
        return 2

    camera_points = np.stack([sample["camera_point_m"] for sample in samples], axis=0)
    base_points = np.stack([sample["base_point_m"] for sample in samples], axis=0)
    try:
        base_t_camera = solve_rigid_transform(camera_points, base_points)
    except ValueError as exc:
        print(f"Failed to solve calibration: {exc}")
        return 3

    residuals = residuals_m(base_t_camera, camera_points, base_points)
    predicted = transform_points(base_t_camera, camera_points)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or (OUTPUT_ROOT / f"global_camera_{timestamp}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Residual mean/max/std: {np.mean(residuals) * 1000.0:.2f}/"
          f"{np.max(residuals) * 1000.0:.2f}/{np.std(residuals) * 1000.0:.2f} mm")
    print(f"base_T_camera translation [mm]: {(base_t_camera[:3, 3] * 1000.0).round(3).tolist()}")
    for sample, err_m, pred in zip(samples, residuals, predicted):
        name = sample["point_name"] or sample["timestamp"]
        print(
            f"  {name}: residual={err_m * 1000.0:.2f} mm, "
            f"pred_base_mm={(pred * 1000.0).round(2).tolist()}, "
            f"measured_base_mm={(sample['base_point_m'] * 1000.0).round(2).tolist()}"
        )

    output = {
        "timestamp": timestamp,
        "description": "Eye-to-hand/global camera calibration. Transform maps fixed RealSense camera coordinates into xArm base coordinates.",
        "sample_count": len(samples),
        "sample_paths": [str(sample["path"]) for sample in samples],
        "base_T_camera": matrix_to_list(base_t_camera),
        "base_T_camera_translation_m": base_t_camera[:3, 3].tolist(),
        "quality": {
            "residual_mean_mm": float(np.mean(residuals) * 1000.0),
            "residual_max_mm": float(np.max(residuals) * 1000.0),
            "residual_std_mm": float(np.std(residuals) * 1000.0),
        },
        "per_sample": [
            {
                "path": str(sample["path"]),
                "timestamp": sample["timestamp"],
                "point_name": sample["point_name"],
                "camera_point_m": sample["camera_point_m"].tolist(),
                "measured_base_point_m": sample["base_point_m"].tolist(),
                "predicted_base_point_m": pred.tolist(),
                "residual_mm": float(err_m * 1000.0),
                "median_depth_m": sample["median_depth_m"],
            }
            for sample, err_m, pred in zip(samples, residuals, predicted)
        ],
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved global camera calibration: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
