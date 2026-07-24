#!/usr/bin/env python3
"""Click the desired suction center and save its offset from the leaf mask centroid."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "config" / "leaf_grasp_target.json"


def latest_frame_dir() -> Path | None:
    candidates = sorted(
        path.parent
        for path in (ROOT / "data" / "visual_vacuum_pick").glob("*/captures/*/vision_result.json")
    )
    return candidates[-1] if candidates else None


def parse_pixel(value: str) -> tuple[int, int]:
    values = value.split(",")
    if len(values) != 2:
        raise argparse.ArgumentTypeError("pixel must be formatted as U,V")
    return int(values[0]), int(values[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-dir", type=Path, default=latest_frame_dir())
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pixel", type=parse_pixel, help="Optional non-interactive U,V target pixel")
    return parser.parse_args()


def deproject_at_depth(u: float, v: float, depth_m: float, intrinsics: dict) -> np.ndarray:
    x_m = (u - float(intrinsics["ppx"])) * depth_m / float(intrinsics["fx"])
    y_m = (v - float(intrinsics["ppy"])) * depth_m / float(intrinsics["fy"])
    return np.array([x_m, y_m, depth_m], dtype=float)


def transform_point(transform: np.ndarray, point_m: np.ndarray) -> np.ndarray:
    return (transform @ np.r_[point_m, 1.0])[:3]


def choose_pixel(image: np.ndarray, centroid: tuple[int, int]) -> tuple[int, int] | None:
    selected: list[tuple[int, int] | None] = [None]
    window = "Select taped suction center | click target | S save | Q cancel"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected[0] = (x, y)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        preview = image.copy()
        cv2.drawMarker(preview, centroid, (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(preview, "yellow: mask centroid", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        if selected[0] is not None:
            cv2.drawMarker(preview, selected[0], (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
            cv2.putText(preview, "red: selected suction center", (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            selected[0] = None
            break
        if key == ord("s") and selected[0] is not None:
            break
    cv2.destroyWindow(window)
    return selected[0]


def main() -> int:
    args = parse_args()
    if args.frame_dir is None:
        raise RuntimeError("No visual-vacuum-pick frame is available")
    frame_dir = args.frame_dir.expanduser().resolve()
    metadata = json.loads((frame_dir / "frame.json").read_text(encoding="utf-8"))
    vision = json.loads((frame_dir / "vision_result.json").read_text(encoding="utf-8"))
    leaves = [item for item in vision["detections"] if "bamboo" in item["label"].lower()]
    if not leaves:
        raise RuntimeError("No bamboo leaf detection is available in the selected frame")
    leaf = max(leaves, key=lambda item: float(item["score"]))
    mask = cv2.imread(str(frame_dir / leaf["mask_path"]), cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(str(frame_dir / metadata["files"]["color_bgr"]), cv2.IMREAD_COLOR)
    if mask is None or color is None:
        raise RuntimeError("Failed to load the frame color image or leaf mask")

    calibration_path = args.calibration
    if calibration_path is None:
        candidates = sorted((ROOT / "calibration").glob("global_camera_*.json"))
        calibration_path = candidates[-1] if candidates else None
    if calibration_path is None:
        raise RuntimeError("Global camera calibration is missing")
    calibration_path = calibration_path.expanduser().resolve()
    base_t_camera = np.asarray(
        json.loads(calibration_path.read_text(encoding="utf-8"))["base_T_camera"],
        dtype=float,
    )

    centroid = tuple(int(round(value)) for value in leaf["centroid_px"])
    target = args.pixel or choose_pixel(color, centroid)
    if target is None:
        print("Cancelled; no grasp configuration was written.")
        return 1
    u, v = target
    if not (0 <= u < mask.shape[1] and 0 <= v < mask.shape[0]) or mask[v, u] == 0:
        raise RuntimeError(f"Selected pixel {target} is outside the segmented leaf mask")

    depth_m = float(leaf["median_depth_m"])
    intrinsics = metadata["intrinsics"]
    target_camera = deproject_at_depth(u, v, depth_m, intrinsics)
    target_base = transform_point(base_t_camera, target_camera)
    centroid_base = transform_point(base_t_camera, np.asarray(leaf["point_camera_m"], dtype=float))
    offset_base_mm = (target_base - centroid_base) * 1000.0

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "description": "Offset from SAM2 bamboo-leaf mask centroid to taped dual-suction center.",
        "source_frame": str(frame_dir),
        "calibration": str(calibration_path),
        "mask_centroid_pixel": list(centroid),
        "selected_pixel": [u, v],
        "leaf_median_depth_m": depth_m,
        "offset_base_mm": offset_base_mm.tolist(),
    }
    output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    preview = color.copy()
    cv2.drawMarker(preview, centroid, (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.drawMarker(preview, target, (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
    preview_path = output.with_suffix(".png")
    cv2.imwrite(str(preview_path), preview)
    print(f"Mask centroid pixel: {centroid}")
    print(f"Selected suction pixel: {target}")
    print(f"Base-frame grasp offset [mm]: {offset_base_mm.round(2).tolist()}")
    print(f"Saved grasp configuration: {output}")
    print(f"Saved preview: {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
