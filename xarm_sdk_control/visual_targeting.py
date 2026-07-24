"""Shared fixed-camera RGB-D targeting utilities for real xArm workflows."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CAPTURE_SCRIPT = ROOT / "scripts" / "13_capture_realsense_frame.py"
VISION_SCRIPT = REPO_ROOT / "vision" / "scripts" / "01_offline_grounded_sam2.py"


def latest_global_calibration() -> Path | None:
    candidates = sorted((ROOT / "calibration").glob("global_camera_*.json"))
    return candidates[-1] if candidates else None


def transform_point(transform: np.ndarray, point_m: list[float]) -> np.ndarray:
    point_h = np.array([point_m[0], point_m[1], point_m[2], 1.0], dtype=float)
    return (transform @ point_h)[:3]


def capture_rgbd(run_dir: Path, serial: str | None) -> Path:
    capture_root = run_dir / "captures"
    cmd = [sys.executable, str(CAPTURE_SCRIPT), "--output-root", str(capture_root)]
    if serial:
        cmd.extend(["--serial", serial])
    print("Capturing RGB-D:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    frame_dirs = sorted(path for path in capture_root.iterdir() if path.is_dir())
    if len(frame_dirs) != 1:
        raise RuntimeError(f"Expected one captured frame directory, found {len(frame_dirs)}")
    return frame_dirs[0]


def run_vision(frame_dir: Path, vision_env: str, min_score: float) -> dict[str, Any]:
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        vision_env,
        "python",
        str(VISION_SCRIPT),
        "--frame-dir",
        str(frame_dir),
        "--text-prompt",
        "bamboo leaf. ceramic disk.",
        "--box-threshold",
        str(min_score),
        "--output-min-score",
        str(min_score),
        "--text-threshold",
        "0.15",
    ]
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    print("Running vision:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
    return json.loads((frame_dir / "vision_result.json").read_text(encoding="utf-8"))


def detections_in_base(
    detections: list[dict[str, Any]],
    base_t_camera: np.ndarray,
    min_score: float,
) -> list[dict[str, Any]]:
    converted = []
    for detection in detections:
        score = float(detection.get("score", 0.0))
        point_camera = detection.get("point_camera_m")
        if score < min_score or point_camera is None:
            continue
        item = dict(detection)
        item["point_base_m"] = transform_point(base_t_camera, point_camera).tolist()
        converted.append(item)
    return converted


def select_ceramic(detections: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        item
        for item in detections
        if "ceramic" in str(item.get("label", "")).lower()
        and not bool(item.get("touches_image_edge", False))
    ]
    if not candidates:
        clipped = [
            item
            for item in detections
            if "ceramic" in str(item.get("label", "")).lower()
            and bool(item.get("touches_image_edge", False))
        ]
        if clipped:
            raise RuntimeError(
                "Ceramic disk was detected but its mask touches the image edge; "
                "move the disk fully into the calibrated camera view"
            )
        raise RuntimeError("No ceramic disk detection passed the confidence threshold")
    return max(candidates, key=lambda item: float(item["score"]))


def select_leaf_waypoints(
    detections: list[dict[str, Any]],
    distinct_distance_m: float,
    allow_any_clipping: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    def clipping_is_usable(item: dict[str, Any]) -> bool:
        if not bool(item.get("touches_image_edge", False)):
            return True
        box = item.get("box_xyxy")
        sides = set(item.get("edge_sides", []))
        if not isinstance(box, list) or len(box) != 4:
            return False
        width = max(float(box[2]) - float(box[0]), 1.0)
        height = max(float(box[3]) - float(box[1]), 1.0)
        vertical_axial_clip = height / width >= 2.0 and sides == {"top", "bottom"}
        horizontal_axial_clip = width / height >= 2.0 and sides == {"left", "right"}
        return vertical_axial_clip or horizontal_axial_clip

    candidates = sorted(
        [
            item
            for item in detections
            if "bamboo" in str(item.get("label", "")).lower()
            and (allow_any_clipping or clipping_is_usable(item))
        ],
        key=lambda item: float(item["score"]),
        reverse=True,
    )
    if not candidates:
        clipped = [
            item
            for item in detections
            if "bamboo" in str(item.get("label", "")).lower()
            and not clipping_is_usable(item)
        ]
        if clipped:
            raise RuntimeError(
                "Bamboo leaf was detected but its mask touches the image edge; "
                "reframe the camera so the full leaf is visible"
            )
        raise RuntimeError("No bamboo leaf detection passed the confidence threshold")
    selected = [candidates[0]]
    first_xy = np.asarray(candidates[0]["point_base_m"], dtype=float)[:2]
    for candidate in candidates[1:]:
        candidate_xy = np.asarray(candidate["point_base_m"], dtype=float)[:2]
        if np.linalg.norm(candidate_xy - first_xy) >= distinct_distance_m:
            selected.append(candidate)
            break
    stacked_or_occluded = len(selected) == 1
    if stacked_or_occluded:
        selected.append(selected[0])
    return selected, stacked_or_occluded


def make_waypoint(point_base_m: list[float], above_mm: float, orientation: list[float]) -> list[float]:
    point_mm = np.asarray(point_base_m, dtype=float) * 1000.0
    return [float(point_mm[0]), float(point_mm[1]), float(point_mm[2] + above_mm), *map(float, orientation)]


def safety_report(
    waypoints: list[list[float]],
    min_x_mm: float,
    max_x_mm: float,
    min_y_mm: float,
    max_y_mm: float,
    min_z_mm: float,
    max_z_mm: float,
) -> dict[str, Any]:
    checks = []
    for index, pose in enumerate(waypoints, start=1):
        x_mm, y_mm, z_mm = pose[:3]
        checks.append(
            {
                "waypoint": index,
                "pose_mm_deg": pose,
                "x_ok": min_x_mm <= x_mm <= max_x_mm,
                "y_ok": min_y_mm <= y_mm <= max_y_mm,
                "z_ok": min_z_mm <= z_mm <= max_z_mm,
            }
        )
    return {
        "limits_mm": {
            "x": [min_x_mm, max_x_mm],
            "y": [min_y_mm, max_y_mm],
            "z": [min_z_mm, max_z_mm],
        },
        "waypoints": checks,
        "all_ok": all(item[axis] for item in checks for axis in ("x_ok", "y_ok", "z_ok")),
    }
