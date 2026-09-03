#!/usr/bin/env python3
"""Estimate bottle-top or open-rim height directly from calibrated RGB-D views."""

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

from xarm_sdk_control.mouth_height import (  # noqa: E402
    estimate_camera_top_height,
    estimate_neck_band,
    fuse_camera_heights,
    transform_reference_points_to_workcell,
)
from xarm_sdk_control.multiview_rig import load_fixed_camera_rig  # noqa: E402
from xarm_sdk_control.run_output import new_timestamped_run_dir  # noqa: E402
from xarm_sdk_control.workcell_frame import deproject_depth_roi_to_reference  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT / "data" / "multiview_neck" / "mouth_heights"


def parse_camera_roi(value: str) -> tuple[str, tuple[int, int, int, int]]:
    try:
        camera_id, raw_roi = value.split(":", maxsplit=1)
        roi = tuple(int(part) for part in raw_roi.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be camera_id:x,y,width,height") from exc
    if not camera_id or len(roi) != 4:
        raise argparse.ArgumentTypeError("ROI must be camera_id:x,y,width,height")
    return camera_id, roi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--workcell-frame", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--roi", action="append", type=parse_camera_roi, required=True)
    parser.add_argument("--planar-radius-mm", type=float, default=75.0)
    parser.add_argument("--minimum-object-height-mm", type=float, default=30.0)
    parser.add_argument("--maximum-object-height-mm", type=float, default=500.0)
    parser.add_argument("--top-quantile", type=float, default=0.99)
    parser.add_argument("--top-band-mm", type=float, default=4.0)
    parser.add_argument("--agreement-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--minimum-camera-count", type=int, default=2)
    parser.add_argument("--neck-boundary-agreement-mm", type=float, default=8.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    rois = dict(args.roi)
    if len(rois) != len(args.roi):
        parser.error("Each camera may only have one --roi")
    if (
        args.planar_radius_mm <= 0
        or args.minimum_object_height_mm < 0
        or args.maximum_object_height_mm <= args.minimum_object_height_mm
        or not 0.5 < args.top_quantile < 1.0
        or args.top_band_mm <= 0
        or args.agreement_tolerance_mm <= 0
        or args.minimum_camera_count < 1
        or args.neck_boundary_agreement_mm <= 0
    ):
        parser.error("Height-estimation parameters are invalid")

    capture_dir = args.capture.expanduser().resolve()
    try:
        rig = load_fixed_camera_rig(args.rig)
        workcell_payload = json.loads(args.workcell_frame.expanduser().resolve().read_text(encoding="utf-8"))
        capture_payload = json.loads((capture_dir / "capture.json").read_text(encoding="utf-8"))
        if workcell_payload.get("type") != "fixed_workcell_support_plane" or workcell_payload.get("status") != "ok":
            raise ValueError("Workcell frame is not a successful support-plane calibration")
        if capture_payload.get("type") != "fixed_multiview_rgbd_capture":
            raise ValueError("Capture has an unexpected type")
        if set(rois) != set(rig.cameras):
            raise ValueError(f"ROIs must match rig cameras exactly: expected {sorted(rig.cameras)}")
        transform_workcell_from_reference = np.asarray(
            workcell_payload["transform_workcell_from_reference"], dtype=np.float64
        )
        estimates = {}
        diagnostics = {}
        images = {}
        point_sets = {}
        pixel_sets = {}
        for camera_id, camera in rig.cameras.items():
            camera_dir = capture_dir / camera_id
            metadata = json.loads((camera_dir / "frame.json").read_text(encoding="utf-8"))
            color = cv2.imread(str(camera_dir / "color.png"))
            depth = cv2.imread(str(camera_dir / "depth_raw.png"), cv2.IMREAD_UNCHANGED)
            if color is None or depth is None:
                raise ValueError(f"{camera_id}: color/depth image is unreadable")
            points_reference, pixels, valid_fraction = deproject_depth_roi_to_reference(
                depth,
                metadata["stream"]["depth_scale_m_per_unit"],
                camera.camera_matrix,
                camera.distortion_coefficients,
                camera.transform_reference_from_camera,
                rois[camera_id],
            )
            points_workcell = transform_reference_points_to_workcell(
                points_reference, transform_workcell_from_reference
            )
            estimate = estimate_camera_top_height(
                points_workcell,
                planar_radius_m=args.planar_radius_mm / 1000.0,
                minimum_object_height_m=args.minimum_object_height_mm / 1000.0,
                maximum_object_height_m=args.maximum_object_height_mm / 1000.0,
                top_quantile=args.top_quantile,
                top_band_m=args.top_band_mm / 1000.0,
            )
            estimates[camera_id] = estimate
            point_sets[camera_id] = points_workcell
            pixel_sets[camera_id] = pixels
            overlay = color.copy()
            x, y, width, height = rois[camera_id]
            cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 0), 2)
            candidate_pixels = pixels[estimate.candidate_mask]
            top_pixels = pixels[estimate.top_support_mask]
            for column, row in candidate_pixels[:: max(1, len(candidate_pixels) // 1200)]:
                cv2.circle(overlay, (int(column), int(row)), 1, (0, 180, 0), -1)
            for column, row in top_pixels[:: max(1, len(top_pixels) // 500)]:
                cv2.circle(overlay, (int(column), int(row)), 2, (0, 165, 255), -1)
            cv2.putText(
                overlay,
                f"height={estimate.height_m * 1000.0:.1f}mm top_n={estimate.top_support_count}",
                (max(8, x), max(24, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            images[camera_id] = overlay
            diagnostics[camera_id] = {
                "roi_xywh": list(rois[camera_id]),
                "valid_depth_fraction_in_roi": valid_fraction,
                "height_mm": estimate.height_m * 1000.0,
                "candidate_count": estimate.candidate_count,
                "top_support_count": estimate.top_support_count,
                "top_support_mad_mm": estimate.median_absolute_deviation_m * 1000.0,
            }
        fused = fuse_camera_heights(
            estimates,
            agreement_tolerance_m=args.agreement_tolerance_mm / 1000.0,
            minimum_camera_count=args.minimum_camera_count,
        )
        reference_top_points = point_sets[rig.reference_camera_id][
            estimates[rig.reference_camera_id].top_support_mask
        ]
        axis_center_xy = np.median(reference_top_points[:, :2], axis=0)
        neck_estimates = {
            camera_id: estimate_neck_band(
                point_sets[camera_id],
                axis_center_xy,
                fused.height_m,
            )
            for camera_id in rig.cameras
            if camera_id != rig.reference_camera_id
        }
        neck_lower_values = np.array(
            [estimate.lower_height_m for estimate in neck_estimates.values()], dtype=np.float64
        )
        neck_lower_spread = float(np.ptp(neck_lower_values))
        if neck_lower_spread > args.neck_boundary_agreement_mm / 1000.0:
            raise ValueError(
                f"Side-camera neck boundaries differ by {neck_lower_spread * 1000.0:.1f} mm"
            )
        neck_lower_height = float(np.median(neck_lower_values))
        for camera_id, overlay in images.items():
            points_workcell = point_sets[camera_id]
            pixels = pixel_sets[camera_id]
            in_neck = (
                (points_workcell[:, 2] >= neck_lower_height)
                & (points_workcell[:, 2] <= fused.height_m + args.top_band_mm / 1000.0)
                & (np.linalg.norm(points_workcell[:, :2] - axis_center_xy, axis=1) <= args.planar_radius_mm / 1000.0)
            )
            neck_pixels = pixels[in_neck]
            for column, row in neck_pixels[:: max(1, len(neck_pixels) // 700)]:
                cv2.circle(overlay, (int(column), int(row)), 1, (255, 80, 0), -1)
            x, y, _, _ = rois[camera_id]
            cv2.putText(
                overlay,
                f"neck={neck_lower_height * 1000.0:.1f}..{fused.height_m * 1000.0:.1f}mm",
                (max(8, x), max(46, y + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 80, 0),
                2,
                cv2.LINE_AA,
            )
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as exc:
        print(f"ERROR: mouth-height estimation failed: {exc}")
        return 2

    run_dir = new_timestamped_run_dir(args.output_root.expanduser().resolve())
    for camera_id, overlay in images.items():
        camera_dir = run_dir / camera_id
        camera_dir.mkdir()
        cv2.imwrite(str(camera_dir / "height_overlay.png"), overlay)
    payload = {
        "schema_version": 1,
        "type": "multiview_direct_mouth_height",
        "status": "ok",
        "measurement": "highest stable object surface; closed test bottle measures cap top",
        "height_above_workcell_mm": fused.height_m * 1000.0,
        "neck_height_range_mm": {
            "lower": neck_lower_height * 1000.0,
            "upper": fused.height_m * 1000.0,
            "length": (fused.height_m - neck_lower_height) * 1000.0,
            "side_camera_lower_boundary_spread": neck_lower_spread * 1000.0,
        },
        "axis_center_xy_workcell_mm": (axis_center_xy * 1000.0).tolist(),
        "accepted_camera_ids": list(fused.accepted_camera_ids),
        "rejected_camera_ids": list(fused.rejected_camera_ids),
        "accepted_camera_spread_mm": fused.camera_spread_m * 1000.0,
        "cameras": diagnostics,
        "neck_profiles": {
            camera_id: {
                "lower_height_mm": estimate.lower_height_m * 1000.0,
                "upper_height_mm": estimate.upper_height_m * 1000.0,
                "body_radius_mm": estimate.body_radius_m * 1000.0,
                "neck_radius_threshold_mm": estimate.neck_radius_threshold_m * 1000.0,
                "samples": [
                    {
                        "lower_height_mm": sample.lower_height_m * 1000.0,
                        "upper_height_mm": sample.upper_height_m * 1000.0,
                        "visible_radius_percentile_mm": sample.radius_m * 1000.0,
                        "point_count": sample.point_count,
                    }
                    for sample in estimate.profile
                ],
            }
            for camera_id, estimate in neck_estimates.items()
        },
        "rig_path": str(args.rig.expanduser().resolve()),
        "workcell_frame_path": str(args.workcell_frame.expanduser().resolve()),
        "capture_dir": str(capture_dir),
        "parameters": {
            "planar_radius_mm": args.planar_radius_mm,
            "minimum_object_height_mm": args.minimum_object_height_mm,
            "maximum_object_height_mm": args.maximum_object_height_mm,
            "top_quantile": args.top_quantile,
            "top_band_mm": args.top_band_mm,
            "agreement_tolerance_mm": args.agreement_tolerance_mm,
            "minimum_camera_count": args.minimum_camera_count,
            "neck_boundary_agreement_mm": args.neck_boundary_agreement_mm,
        },
    }
    (run_dir / "mouth_height.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved mouth-height result: {run_dir}")
    print(
        f"height={payload['height_above_workcell_mm']:.2f}mm "
        f"spread={payload['accepted_camera_spread_mm']:.2f}mm "
        f"accepted={payload['accepted_camera_ids']} rejected={payload['rejected_camera_ids']}"
    )
    print(
        f"neck range={payload['neck_height_range_mm']['lower']:.2f}.."
        f"{payload['neck_height_range_mm']['upper']:.2f}mm "
        f"length={payload['neck_height_range_mm']['length']:.2f}mm"
    )
    print("Robot connection/motion: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
