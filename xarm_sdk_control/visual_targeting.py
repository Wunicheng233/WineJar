"""Shared fixed-camera RGB-D targeting utilities for real xArm workflows."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CAPTURE_SCRIPT = ROOT / "scripts" / "13_capture_realsense_frame.py"
VISION_SCRIPT = REPO_ROOT / "vision" / "scripts" / "01_offline_grounded_sam2.py"
DEFAULT_VISION_SOCKET = Path("/tmp/winejar_grounded_sam2.sock")


def latest_global_calibration() -> Path | None:
    candidates = sorted((ROOT / "calibration").glob("global_camera_*.json"))
    return candidates[-1] if candidates else None


def transform_point(transform: np.ndarray, point_m: list[float]) -> np.ndarray:
    point_h = np.array([point_m[0], point_m[1], point_m[2], 1.0], dtype=float)
    return (transform @ point_h)[:3]


def mask_principal_axis_deg(frame_dir: Path, detection: dict[str, Any]) -> tuple[float, float]:
    """Return the mask's undirected image-axis angle and PCA elongation ratio."""
    relative_path = detection.get("mask_path")
    if not relative_path:
        raise ValueError("Detection has no mask_path")
    mask = cv2.imread(str(frame_dir / str(relative_path)), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(frame_dir / str(relative_path))
    rows, columns = np.nonzero(mask > 127)
    if len(columns) < 100:
        raise ValueError("Detection mask is too small for a stable principal axis")
    points = np.column_stack([columns, rows]).astype(np.float64)
    covariance = np.cov(points - points.mean(axis=0), rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_index = int(np.argmax(eigenvalues))
    major = eigenvectors[:, major_index]
    angle_deg = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    minor_value = max(float(np.min(eigenvalues)), 1e-9)
    elongation_ratio = float(np.max(eigenvalues) / minor_value)
    return angle_deg, elongation_ratio


def undirected_axis_delta_deg(start_deg: float, end_deg: float) -> float:
    """Shortest signed difference between two axes whose direction is modulo 180 degrees."""
    return (float(end_deg) - float(start_deg) + 90.0) % 180.0 - 90.0


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


def run_vision(
    frame_dir: Path,
    vision_env: str,
    min_score: float,
    text_prompt: str = "bamboo leaf. ceramic disk.",
    service_socket: Path | None = DEFAULT_VISION_SOCKET,
    require_service: bool = False,
) -> dict[str, Any]:
    if service_socket is not None:
        socket_path = service_socket.expanduser().resolve()
        try:
            response = request_vision_service(
                socket_path=socket_path,
                frame_dir=frame_dir,
                text_prompt=text_prompt,
                min_score=min_score,
            )
            print(
                "Warm vision service timing:",
                response.get("timing_seconds", {}),
            )
            return json.loads((frame_dir / "vision_result.json").read_text(encoding="utf-8"))
        except Exception as exc:
            if require_service:
                raise RuntimeError(f"Required vision service is unavailable: {exc}") from exc
            print(f"Vision service unavailable ({exc}); falling back to one-shot inference.")

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
        text_prompt,
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


def request_vision_service(
    socket_path: Path,
    frame_dir: Path,
    text_prompt: str,
    min_score: float,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    request = {
        "command": "infer",
        "request_id": uuid.uuid4().hex,
        "frame_dir": str(frame_dir.expanduser().resolve()),
        "text_prompt": text_prompt,
        "box_threshold": min_score,
        "output_min_score": min_score,
        "text_threshold": 0.15,
    }
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout_seconds)
    try:
        client.connect(str(socket_path))
        client.sendall((json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8"))
        response_file = client.makefile("r", encoding="utf-8")
        line = response_file.readline()
        if not line:
            raise RuntimeError("Vision service closed the connection without a response")
        response = json.loads(line)
    finally:
        client.close()
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error", "unknown vision service error")))
    if response.get("request_id") != request["request_id"]:
        raise RuntimeError("Vision service response ID does not match the request")
    return response


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


def detection_object_kind(item: dict[str, Any]) -> str | None:
    """Map GroundingDINO text spans to one unambiguous task object class."""
    label = str(item.get("label", "")).lower()
    has_bamboo = "bamboo" in label
    has_ceramic = "ceramic" in label
    if has_bamboo and has_ceramic:
        box = item.get("box_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            return None
        width = max(float(box[2]) - float(box[0]), 1.0)
        height = max(float(box[3]) - float(box[1]), 1.0)
        elongation = max(width / height, height / width)
        return "bamboo" if elongation >= 1.8 else None
    if has_bamboo:
        return "bamboo"
    if has_ceramic:
        return "ceramic"
    return None


def select_ceramic(detections: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        item
        for item in detections
        if detection_object_kind(item) == "ceramic"
        and not bool(item.get("touches_image_edge", False))
    ]
    if not candidates:
        clipped = [
            item
            for item in detections
            if detection_object_kind(item) == "ceramic"
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
            if detection_object_kind(item) == "bamboo"
            and (allow_any_clipping or clipping_is_usable(item))
        ],
        key=lambda item: float(item["score"]),
        reverse=True,
    )
    if not candidates:
        clipped = [
            item
            for item in detections
            if detection_object_kind(item) == "bamboo"
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
