#!/usr/bin/env python3
"""Calibrate the workcell support plane from a fixed multiview RGB-D capture."""

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
from xarm_sdk_control.workcell_frame import (  # noqa: E402
    build_workcell_frame,
    deproject_depth_roi_to_reference,
    fit_plane_ransac,
)


DEFAULT_OUTPUT_ROOT = ROOT / "calibration" / "multiview_neck" / "workcell_frames"


def parse_roi(value: str) -> tuple[int, int, int, int]:
    try:
        roi = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height") from exc
    if len(roi) != 4:
        raise argparse.ArgumentTypeError("ROI must contain x,y,width,height")
    return roi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True, help="Directory containing capture.json and camera folders.")
    parser.add_argument("--camera-id", help="RGB-D camera used for plane fitting; defaults to rig reference camera.")
    parser.add_argument("--roi", type=parse_roi, required=True, help="Interior support-plane ROI: x,y,width,height.")
    parser.add_argument("--distance-threshold-mm", type=float, default=2.0)
    parser.add_argument("--ransac-iterations", type=int, default=600)
    parser.add_argument("--minimum-inlier-ratio", type=float, default=0.85)
    parser.add_argument("--maximum-rms-mm", type=float, default=1.5)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if (
        args.distance_threshold_mm <= 0
        or args.ransac_iterations < 1
        or not 0 < args.minimum_inlier_ratio <= 1
        or args.maximum_rms_mm <= 0
    ):
        parser.error("Plane thresholds and iterations are invalid")
    capture_dir = args.capture.expanduser().resolve()
    try:
        rig = load_fixed_camera_rig(args.rig)
        camera_id = args.camera_id or rig.reference_camera_id
        camera = rig.camera(camera_id)
        camera_dir = capture_dir / camera_id
        frame_metadata = json.loads((camera_dir / "frame.json").read_text(encoding="utf-8"))
        capture_metadata = json.loads((capture_dir / "capture.json").read_text(encoding="utf-8"))
        if capture_metadata.get("type") != "fixed_multiview_rgbd_capture":
            raise ValueError("capture.json has an unexpected type")
        if frame_metadata.get("camera", {}).get("serial") != camera.serial:
            raise ValueError("Capture camera serial does not match the rig")
        color = cv2.imread(str(camera_dir / "color.png"))
        depth = cv2.imread(str(camera_dir / "depth_raw.png"), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None:
            raise ValueError(f"Unreadable color/depth images in {camera_dir}")
        points, pixels, valid_depth_fraction = deproject_depth_roi_to_reference(
            depth,
            frame_metadata["stream"]["depth_scale_m_per_unit"],
            camera.camera_matrix,
            camera.distortion_coefficients,
            camera.transform_reference_from_camera,
            args.roi,
        )
        camera_position_reference = camera.transform_reference_from_camera[:3, 3]
        plane = fit_plane_ransac(
            points,
            distance_threshold_m=args.distance_threshold_mm / 1000.0,
            iterations=args.ransac_iterations,
            minimum_inlier_ratio=args.minimum_inlier_ratio,
            toward_point_m=camera_position_reference,
        )
        if plane.rms_residual_m * 1000.0 > args.maximum_rms_mm:
            raise ValueError(
                f"Plane RMS {plane.rms_residual_m * 1000.0:.3f} mm exceeds {args.maximum_rms_mm:.3f} mm"
            )
        frame = build_workcell_frame(
            plane,
            camera.transform_reference_from_camera[:3, 0],
            x_axis_source=f"{camera_id}_camera_x_projected_provisional",
        )
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as exc:
        print(f"ERROR: workcell-plane calibration failed: {exc}")
        return 2

    run_dir = new_timestamped_run_dir(args.output_root.expanduser().resolve())
    overlay = color.copy()
    x, y, width, height = args.roi
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 0), 2)
    inlier_pixels = pixels[plane.inlier_mask]
    for column, row in inlier_pixels[:: max(1, len(inlier_pixels) // 1500)]:
        cv2.circle(overlay, (int(column), int(row)), 1, (0, 220, 0), -1)
    cv2.putText(
        overlay,
        f"inliers={np.mean(plane.inlier_mask):.3f} rms={plane.rms_residual_m * 1000.0:.3f}mm",
        (max(8, x), max(24, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(run_dir / "workcell_plane_overlay.png"), overlay)
    payload = {
        "schema_version": 1,
        "type": "fixed_workcell_support_plane",
        "status": "ok",
        "rig_path": str(args.rig.expanduser().resolve()),
        "capture_dir": str(capture_dir),
        "reference_frame": rig.reference_camera_id,
        "fitting_camera": {"camera_id": camera_id, "serial": camera.serial},
        "coordinate_convention": {
            "units": "metres",
            "origin": "mean of fitted support-plane inliers inside ROI",
            "z_axis": "support-plane normal pointing toward fitting camera",
            "x_axis": frame.x_axis_source,
            "transform": "p_reference = T_reference_from_workcell @ p_workcell_homogeneous",
            "note": "X/Y orientation is provisional; only Z=0 and +Z are production-relevant for height estimation.",
        },
        "roi_xywh": list(args.roi),
        "plane_reference": {
            "point_m": plane.point_m.tolist(),
            "normal_toward_camera": plane.normal_toward_camera.tolist(),
            "offset_m": plane.offset_m,
            "equation": "normal dot p_reference + offset = 0",
        },
        "transform_reference_from_workcell": frame.transform_reference_from_workcell.tolist(),
        "transform_workcell_from_reference": frame.transform_workcell_from_reference.tolist(),
        "quality": {
            "valid_depth_fraction": valid_depth_fraction,
            "sample_count": int(len(points)),
            "inlier_count": int(np.count_nonzero(plane.inlier_mask)),
            "inlier_ratio": float(np.mean(plane.inlier_mask)),
            "rms_residual_mm": plane.rms_residual_m * 1000.0,
            "p95_residual_mm": plane.p95_residual_m * 1000.0,
            "maximum_inlier_residual_mm": plane.maximum_residual_m * 1000.0,
        },
        "parameters": {
            "distance_threshold_mm": args.distance_threshold_mm,
            "ransac_iterations": args.ransac_iterations,
            "minimum_inlier_ratio": args.minimum_inlier_ratio,
            "maximum_rms_mm": args.maximum_rms_mm,
        },
        "files": {"overlay": "workcell_plane_overlay.png"},
    }
    (run_dir / "workcell_frame.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved workcell frame: {run_dir}")
    print(
        f"valid_depth={valid_depth_fraction:.3f} inliers={payload['quality']['inlier_ratio']:.3f} "
        f"rms={payload['quality']['rms_residual_mm']:.3f}mm p95={payload['quality']['p95_residual_mm']:.3f}mm"
    )
    print(f"+Z reference normal: {plane.normal_toward_camera.tolist()}")
    print("Robot connection/motion: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
