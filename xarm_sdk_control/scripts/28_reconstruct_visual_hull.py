#!/usr/bin/env python3
"""Reconstruct a local visual hull from one three-camera segmentation result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.multiview_rig import load_fixed_camera_rig  # noqa: E402
from xarm_sdk_control.run_output import new_timestamped_run_dir  # noqa: E402
from xarm_sdk_control.visual_hull import (  # noqa: E402
    AxisAlignedBounds,
    derive_bounds_from_reference_mask,
    reconstruct_visual_hull,
    write_ascii_ply,
)


DEFAULT_OUTPUT_ROOT = ROOT / "data" / "multiview_neck" / "visual_hulls"


def parse_bounds(value: str) -> AxisAlignedBounds:
    """Parse x_min,x_max,y_min,y_max,z_min,z_max in reference-frame metres."""
    try:
        x_min, x_max, y_min, y_max, z_min, z_max = (float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Bounds must be x_min,x_max,y_min,y_max,z_min,z_max") from exc
    return AxisAlignedBounds(np.array([x_min, y_min, z_min]), np.array([x_max, y_max, z_max]))


def read_json(path: Path) -> dict:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True, help="Directory containing segmentation.json.")
    parser.add_argument("--voxel-size-mm", type=float, default=4.0)
    parser.add_argument("--bounds", type=parse_bounds, help="Optional reference-frame bounds in metres.")
    parser.add_argument(
        "--max-auto-depth-span-mm",
        type=float,
        default=180.0,
        help="Maximum camera-depth span for automatic local-neck bounds; avoids transparent-object background depth.",
    )
    parser.add_argument("--max-voxels", type=int, default=800_000)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.voxel_size_mm <= 0 or args.max_voxels <= 0 or args.max_auto_depth_span_mm <= 0:
        parser.error("--voxel-size-mm, --max-voxels and --max-auto-depth-span-mm must be positive")
    try:
        rig = load_fixed_camera_rig(args.rig)
        segmentation_dir = args.segmentation.expanduser().resolve()
        report = read_json(segmentation_dir / "segmentation.json")
        if report.get("type") != "fixed_multiview_foreground_segmentation":
            raise ValueError("segmentation.json has an unexpected type")
        if set(report.get("cameras", {})) != set(rig.cameras):
            raise ValueError("Segmentation cameras do not match rig cameras")
        silhouettes = {}
        for camera_id, record in report["cameras"].items():
            path = segmentation_dir / record["files"]["silhouette_mask"]
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"{camera_id} silhouette mask is unreadable: {path}")
            silhouettes[camera_id] = mask
        reference = rig.camera(rig.reference_camera_id)
        foreground_capture_dir = Path(report["foreground_capture_dir"])
        reference_metadata = read_json(foreground_capture_dir / reference.camera_id / "frame.json")
        depth_raw = cv2.imread(str(foreground_capture_dir / reference.camera_id / "depth_raw.png"), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise ValueError("Reference depth image is unreadable")
        bounds = args.bounds or derive_bounds_from_reference_mask(
            silhouettes[reference.camera_id],
            depth_raw,
            reference_metadata["stream"]["depth_scale_m_per_unit"],
            reference.camera_matrix,
            reference.distortion_coefficients,
            max_depth_span_m=args.max_auto_depth_span_mm / 1000.0,
        )
        hull = reconstruct_visual_hull(
            rig,
            silhouettes,
            bounds,
            args.voxel_size_mm / 1000.0,
            max_voxels=args.max_voxels,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: visual-hull reconstruction failed: {exc}")
        return 2
    run_dir = new_timestamped_run_dir(args.output_root.expanduser().resolve())
    np.savez_compressed(run_dir / "hull_points_reference_m.npz", points_reference_m=hull.occupied_points_reference_m)
    write_ascii_ply(run_dir / "hull_points_reference_m.ply", hull.occupied_points_reference_m)
    payload = {
        "schema_version": 1,
        "type": "fixed_multiview_visual_hull",
        "rig_path": str(args.rig.expanduser().resolve()),
        "segmentation_dir": str(segmentation_dir),
        "reference_camera_id": rig.reference_camera_id,
        "voxel_size_mm": args.voxel_size_mm,
        "max_auto_depth_span_mm": args.max_auto_depth_span_mm,
        "bounds_reference_m": {
            "minimum": hull.bounds.minimum_m.tolist(),
            "maximum": hull.bounds.maximum_m.tolist(),
            "extent": hull.bounds.extent_m.tolist(),
        },
        "candidate_voxel_count": hull.candidate_voxel_count,
        "occupied_voxel_count": int(len(hull.occupied_points_reference_m)),
        "occupancy_fraction": float(len(hull.occupied_points_reference_m) / hull.candidate_voxel_count),
        "files": {
            "points_npz": "hull_points_reference_m.npz",
            "points_ply": "hull_points_reference_m.ply",
        },
    }
    (run_dir / "visual_hull.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved visual hull: {run_dir}")
    print(
        f"bounds_m min={payload['bounds_reference_m']['minimum']} max={payload['bounds_reference_m']['maximum']}; "
        f"occupied={payload['occupied_voxel_count']}/{payload['candidate_voxel_count']}"
    )
    print("Robot connection/motion: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
