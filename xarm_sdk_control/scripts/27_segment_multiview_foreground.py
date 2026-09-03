#!/usr/bin/env python3
"""Extract three filled object silhouettes from a fixed-background RGB-D capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.multiview_rig import load_fixed_camera_rig  # noqa: E402
from xarm_sdk_control.run_output import new_timestamped_run_dir  # noqa: E402
from xarm_sdk_control.silhouette_geometry import (  # noqa: E402
    draw_silhouette_overlay,
    segment_fixed_background_foreground,
)


DEFAULT_OUTPUT_ROOT = ROOT / "data" / "multiview_neck" / "segmentations"


def parse_roi(value: str) -> tuple[str, tuple[int, int, int, int]]:
    """Parse ``camera_id:x,y,width,height`` into one camera-specific workspace ROI."""
    try:
        camera_id, values = value.split(":", maxsplit=1)
        x, y, width, height = (int(part) for part in values.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be camera_id:x,y,width,height") from exc
    if not camera_id:
        raise argparse.ArgumentTypeError("ROI camera_id must not be empty")
    return camera_id, (x, y, width, height)


def parse_camera_float(value: str) -> tuple[str, float]:
    """Parse ``camera_id:value`` for a camera-specific numeric override."""
    try:
        camera_id, raw_value = value.split(":", maxsplit=1)
        numeric_value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be camera_id:number") from exc
    if not camera_id or numeric_value <= 0:
        raise argparse.ArgumentTypeError("camera_id must be nonempty and number must be positive")
    return camera_id, numeric_value


def read_capture_metadata(capture_dir: Path) -> dict:
    path = capture_dir.expanduser().resolve() / "capture.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "fixed_multiview_rgbd_capture":
        raise ValueError(f"{path} is not a fixed multiview RGB-D capture")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--foreground", type=Path, required=True, help="Capture directory containing the object.")
    parser.add_argument("--background", type=Path, required=True, help="Capture directory with the unchanged empty scene.")
    parser.add_argument("--roi", action="append", default=[], type=parse_roi, help="Optional camera_id:x,y,width,height; repeat per camera.")
    parser.add_argument(
        "--camera-threshold-scale",
        action="append",
        default=[],
        type=parse_camera_float,
        help="Optional camera_id:positive_scale; multiplies that camera's Otsu threshold.",
    )
    parser.add_argument("--close-kernel", type=int, default=15)
    parser.add_argument("--min-component-area-px", type=int, default=250)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.min_component_area_px <= 0:
        parser.error("--min-component-area-px must be positive")
    rois = dict(args.roi)
    threshold_scales = dict(args.camera_threshold_scale)
    if len(rois) != len(args.roi):
        parser.error("Each camera_id may only have one --roi")
    if len(threshold_scales) != len(args.camera_threshold_scale):
        parser.error("Each camera_id may only have one --camera-threshold-scale")
    try:
        rig = load_fixed_camera_rig(args.rig)
        foreground_metadata = read_capture_metadata(args.foreground)
        background_metadata = read_capture_metadata(args.background)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: invalid input: {exc}")
        return 2
    expected_ids = set(rig.cameras)
    if set(foreground_metadata.get("camera_ids", [])) != expected_ids or set(background_metadata.get("camera_ids", [])) != expected_ids:
        print("ERROR: foreground/background captures do not contain exactly the rig cameras")
        return 2
    unknown_roi_cameras = set(rois) - expected_ids
    if unknown_roi_cameras:
        print(f"ERROR: ROI mentions cameras absent from rig: {sorted(unknown_roi_cameras)}")
        return 2
    unknown_threshold_cameras = set(threshold_scales) - expected_ids
    if unknown_threshold_cameras:
        print(f"ERROR: threshold scale mentions cameras absent from rig: {sorted(unknown_threshold_cameras)}")
        return 2
    run_dir = new_timestamped_run_dir(args.output_root.expanduser().resolve())
    records = {}
    try:
        for camera_id in rig.cameras:
            foreground = cv2.imread(str(args.foreground.expanduser().resolve() / camera_id / "color.png"))
            background = cv2.imread(str(args.background.expanduser().resolve() / camera_id / "color.png"))
            if foreground is None or background is None:
                raise ValueError(f"{camera_id}: color.png is unreadable")
            segmentation = segment_fixed_background_foreground(
                foreground,
                background,
                roi_xywh=rois.get(camera_id),
                close_kernel=args.close_kernel,
                min_component_area_px=args.min_component_area_px,
                threshold_scale=threshold_scales.get(camera_id, 1.0),
            )
            camera_dir = run_dir / camera_id
            camera_dir.mkdir()
            cv2.imwrite(str(camera_dir / "difference.png"), segmentation.difference_u8)
            cv2.imwrite(str(camera_dir / "raw_mask.png"), segmentation.raw_mask)
            cv2.imwrite(str(camera_dir / "silhouette_mask.png"), segmentation.silhouette_mask)
            cv2.imwrite(str(camera_dir / "overlay.png"), draw_silhouette_overlay(foreground, segmentation))
            records[camera_id] = {
                "roi_xywh": rois.get(camera_id),
                "threshold_u8": segmentation.threshold_u8,
                "threshold_scale": threshold_scales.get(camera_id, 1.0),
                "silhouette_area_px": segmentation.silhouette_area_px,
                "bounding_box_xywh": segmentation.bounding_box_xywh,
                "files": {
                    "difference": f"{camera_id}/difference.png",
                    "raw_mask": f"{camera_id}/raw_mask.png",
                    "silhouette_mask": f"{camera_id}/silhouette_mask.png",
                    "overlay": f"{camera_id}/overlay.png",
                },
            }
    except ValueError as exc:
        print(f"ERROR: segmentation failed: {exc}")
        return 2
    report = {
        "schema_version": 1,
        "type": "fixed_multiview_foreground_segmentation",
        "rig_path": str(args.rig.expanduser().resolve()),
        "foreground_capture_dir": str(args.foreground.expanduser().resolve()),
        "background_capture_dir": str(args.background.expanduser().resolve()),
        "parameters": {
            "close_kernel": args.close_kernel,
            "min_component_area_px": args.min_component_area_px,
            "camera_threshold_scales": threshold_scales,
        },
        "cameras": records,
    }
    (run_dir / "segmentation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved multiview silhouettes: {run_dir}")
    for camera_id, record in records.items():
        print(f"  {camera_id}: area={record['silhouette_area_px']}px bbox={record['bounding_box_xywh']}")
    print("Robot connection/motion: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
