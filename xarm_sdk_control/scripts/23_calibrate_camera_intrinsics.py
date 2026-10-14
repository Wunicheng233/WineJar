#!/usr/bin/env python3
"""Interactively calibrate one fixed RealSense color camera with a full chessboard.

The camera must remain fixed. Move a rigid, flat chessboard through the camera
view, press S to keep a detected pose, then press C to solve and save the
intrinsic calibration. This script never connects to or moves the xArm.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time

from xarm_sdk_control.calibration_data import intrinsic_health, write_new_json
from xarm_sdk_control.calibration_sampling import StationaryBoardGate, duplicate_pose
from xarm_sdk_control.multiview_calibration import (
    BoardObservation,
    calibrate_chessboard_intrinsics,
    chessboard_object_points,
    find_chessboard_corners,
    normalized_board_area_ratio,
)
from xarm_sdk_control.realsense_capture import camera_info, choose_device

DEFAULT_DATA_ROOT = ROOT / "data" / "multiview_neck" / "intrinsics"
DEFAULT_CALIBRATION_ROOT = ROOT / "calibration" / "multiview_neck"
WINDOW_NAME = "WineJar chessboard intrinsic calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera-id",
        required=True,
        help="Stable rig name, for example overhead or side_a.",
    )
    parser.add_argument(
        "--serial", required=True, help="RealSense serial number to calibrate."
    )
    parser.add_argument(
        "--pattern-cols",
        type=int,
        required=True,
        help="Number of chessboard INNER corners along columns, not square count.",
    )
    parser.add_argument(
        "--pattern-rows",
        type=int,
        required=True,
        help="Number of chessboard INNER corners along rows, not square count.",
    )
    parser.add_argument(
        "--square-size-mm",
        type=float,
        required=True,
        help="Measured physical edge length of one chessboard square in millimetres.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument(
        "--model",
        choices=("brown4", "brown5"),
        default="brown4",
        help="brown4 fixes k3=0; neither model is accepted on training RMS alone",
    )
    parser.add_argument("--min-board-area-ratio", type=float, default=0.03)
    parser.add_argument("--max-board-area-ratio", type=float, default=0.75)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--calibration-root", type=Path, default=DEFAULT_CALIBRATION_ROOT
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.pattern_cols < 2 or args.pattern_rows < 2:
        raise ValueError("Chessboard must have at least 2 by 2 inner corners")
    if not np.isfinite(args.square_size_mm) or args.square_size_mm <= 0:
        raise ValueError("--square-size-mm must be positive")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.warmup < 0:
        raise ValueError(
            "Capture width, height, fps must be positive and warmup must be non-negative"
        )
    if not 4 <= args.min_samples <= args.max_samples:
        raise ValueError("Require 4 <= --min-samples <= --max-samples")
    if not 0 < args.min_board_area_ratio < args.max_board_area_ratio < 1:
        raise ValueError("Board-area ratios must satisfy 0 < min < max < 1")


def draw_overlay(
    color_bgr: np.ndarray,
    pattern_size: tuple[int, int],
    found: bool,
    corners: np.ndarray | None,
    area_ratio: float,
    saved_count: int,
    args: argparse.Namespace,
    message: str,
) -> np.ndarray:
    display = color_bgr.copy()
    if found and corners is not None:
        cv2.drawChessboardCorners(display, pattern_size, corners, True)
    quality = (
        found and args.min_board_area_ratio <= area_ratio <= args.max_board_area_ratio
    )
    color = (0, 220, 0) if quality else (0, 0, 255)
    lines = [
        f"camera={args.camera_id} serial={args.serial}",
        f"inner corners={pattern_size[0]}x{pattern_size[1]} saved={saved_count}/{args.max_samples}",
        "S: save good pose | C: calibrate and save | Q/Esc: quit",
        f"detected={found} board area={area_ratio:.1%} accepted={quality}",
        message,
    ]
    for index, text in enumerate(lines):
        cv2.putText(
            display,
            text,
            (12, 26 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            color if index in (3, 4) else (255, 255, 255),
            2,
        )
    return display


def save_sample(
    run_dir: Path,
    sample_index: int,
    color_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    corners: np.ndarray,
    area_ratio: float,
) -> dict[str, object]:
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    color_path = sample_dir / f"sample_{sample_index:02d}.png"
    overlay_path = sample_dir / f"sample_{sample_index:02d}_corners.png"
    if not cv2.imwrite(str(color_path), color_bgr):
        raise RuntimeError(f"Failed to save {color_path}")
    if not cv2.imwrite(str(overlay_path), overlay_bgr):
        raise RuntimeError(f"Failed to save {overlay_path}")
    return {
        "sample_index": sample_index,
        "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "color_path": str(color_path.relative_to(run_dir)),
        "corner_overlay_path": str(overlay_path.relative_to(run_dir)),
        "corner_points_px": np.asarray(corners, dtype=float).reshape(-1, 2).tolist(),
        "board_area_ratio": float(area_ratio),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    pattern_size = (args.pattern_cols, args.pattern_rows)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_dir = args.output_root.expanduser().resolve() / args.camera_id / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    context = rs.context()
    device = choose_device(context, args.serial)
    info = camera_info(device)
    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(info["serial"])
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    profile = None
    samples: list[dict[str, object]] = []
    saved_observations = []
    stability = StationaryBoardGate()
    message = "Move the fixed camera's chessboard through varied distance and tilt."
    calibrate_requested = False
    try:
        profile = pipeline.start(config)
        for _ in range(args.warmup):
            pipeline.wait_for_frames()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        image_size = (int(intrinsics.width), int(intrinsics.height))
        print(
            f"Calibrating {info['name']} serial={info['serial']} at "
            f"{image_size[0]}x{image_size[1]} @{args.fps} fps"
        )
        print(
            f"Chessboard inner corners={args.pattern_cols}x{args.pattern_rows}, "
            f"square={args.square_size_mm:.4f} mm."
        )
        print("Robot connection/motion: none")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            frames = pipeline.wait_for_frames(2000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                message = "No color frame received"
                continue
            color = np.asanyarray(color_frame.get_data()).copy()
            found, corners = find_chessboard_corners(color, pattern_size)
            observation = (
                None
                if not found
                else BoardObservation(
                    corners,
                    np.arange(len(corners)),
                    chessboard_object_points(*pattern_size, args.square_size_mm / 1000),
                    0,
                )
            )
            still = stability.update((observation,), time.monotonic())
            area_ratio = (
                normalized_board_area_ratio(corners, image_size)
                if found and corners is not None
                else 0.0
            )
            overlay = draw_overlay(
                color,
                pattern_size,
                found,
                corners,
                area_ratio,
                len(samples),
                args,
                message,
            )
            cv2.imshow(WINDOW_NAME, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("Cancelled; no calibration was solved.")
                return 1
            if key in (ord("s"), ord("S")):
                if not found or corners is None:
                    message = "Not saved: complete chessboard was not detected"
                    continue
                if area_ratio < args.min_board_area_ratio:
                    message = "Not saved: chessboard is too small in the image"
                    continue
                if area_ratio > args.max_board_area_ratio:
                    message = "Not saved: chessboard is too close to the image edge"
                    continue
                if not still or duplicate_pose((observation,), saved_observations):
                    message = "Not saved: hold still, or change the repeated board pose"
                    continue
                record = save_sample(
                    run_dir,
                    len(samples) + 1,
                    color,
                    overlay,
                    corners,
                    area_ratio,
                )
                samples.append(record)
                saved_observations.append((observation,))
                write_new_json(run_dir / f"sample_{len(samples):02d}.json", record)
                message = f"Saved sample {len(samples)}; vary board position and tilt before the next one"
                print(
                    f"Saved sample {len(samples):02d}: board area={area_ratio:.1%}, "
                    f"corners={len(corners)}"
                )
            if key in (ord("c"), ord("C")):
                if len(samples) < args.min_samples:
                    message = (
                        f"Need at least {args.min_samples} samples before calibration"
                    )
                    continue
                calibrate_requested = True
                break
            if len(samples) >= args.max_samples:
                print(f"Reached --max-samples={args.max_samples}; solving calibration.")
                calibrate_requested = True
                break
    finally:
        cv2.destroyAllWindows()
        if profile is not None:
            pipeline.stop()

    if not calibrate_requested:
        return 1
    corner_sets = [
        np.asarray(record["corner_points_px"], dtype=np.float32) for record in samples
    ]
    calibration = calibrate_chessboard_intrinsics(
        corner_sets,
        image_size,
        args.pattern_cols,
        args.pattern_rows,
        args.square_size_mm / 1000.0,
        model=args.model,
    )
    args.calibration_root.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    calibration_path = (
        args.calibration_root.expanduser().resolve()
        / f"{args.camera_id}_intrinsics_{timestamp}.json"
    )
    output = {
        "schema_version": 2,
        "calibration_status": "candidate",
        "intrinsic_model": args.model,
        "type": "fixed_camera_chessboard_intrinsics",
        "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "camera_id": args.camera_id,
        "camera": info,
        "capture": {
            "width": image_size[0],
            "height": image_size[1],
            "fps": args.fps,
            "color_format": "BGR8",
        },
        "chessboard": {
            "inner_corners_cols": args.pattern_cols,
            "inner_corners_rows": args.pattern_rows,
            "square_size_mm": args.square_size_mm,
        },
        "sample_run_dir": str(run_dir),
        "sample_count": len(samples),
        "camera_matrix": calibration.camera_matrix.tolist(),
        "distortion_coefficients": calibration.distortion_coefficients.tolist(),
        "quality": {
            "numerical_health": intrinsic_health(
                calibration.camera_matrix,
                calibration.distortion_coefficients,
                image_size,
            ),
            "absolute_accuracy_validated": False,
            "rms_reprojection_error_px": calibration.rms_reprojection_error_px,
            "per_view_reprojection_error_px": calibration.per_view_reprojection_error_px,
            "per_view_board_area_ratio": [
                record["board_area_ratio"] for record in samples
            ],
        },
        "samples": samples,
    }
    write_new_json(calibration_path, output)
    print(
        f"Calibration RMS reprojection error: {calibration.rms_reprojection_error_px:.4f} px"
    )
    print(
        "Per-view reprojection error [px]: "
        + ", ".join(
            f"{value:.3f}" for value in calibration.per_view_reprojection_error_px
        )
    )
    print(f"Saved chessboard samples: {run_dir}")
    print(f"Saved intrinsic calibration: {calibration_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
