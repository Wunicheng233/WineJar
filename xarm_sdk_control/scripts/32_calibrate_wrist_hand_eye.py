#!/usr/bin/env python3
"""Calibrate a wrist camera to the xArm flange with a fixed chessboard.

Fix the chessboard in the workspace. Move the robot with UFactory Studio, stop
at each varied camera pose, and press S. This program only reads robot state; it
never enables or commands robot motion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.end_effector_config import load_end_effector_config  # noqa: E402
from xarm_sdk_control.hand_eye_calibration import (  # noqa: E402
    calibrate_eye_in_hand,
    flange_transform_from_tcp_pose,
    pose_set_diversity,
)
from xarm_sdk_control.multiview_calibration import (  # noqa: E402
    chessboard_object_points,
    find_chessboard_corners,
    normalized_board_area_ratio,
    reprojection_error_px,
    transform_from_rvec_tvec,
    transform_rotation_angle_deg,
)
from xarm_sdk_control.realsense_capture import camera_info, choose_device  # noqa: E402
from xarm_sdk_control.robot_motion import load_robot_ip, require_code  # noqa: E402


DEFAULT_INTRINSICS = ROOT / "calibration" / "multiview_neck" / "wrist_intrinsics_20260903_135209_313.json"
DEFAULT_DATA_ROOT = ROOT / "data" / "hand_eye"
DEFAULT_CALIBRATION_ROOT = ROOT / "calibration" / "hand_eye"
WINDOW_NAME = "WineJar wrist hand-eye calibration (read-only robot)"


def parse_args() -> argparse.Namespace:
    end_effector = load_end_effector_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default=load_robot_ip())
    parser.add_argument("--serial", default="011422070172")
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--pattern-cols", type=int, default=11)
    parser.add_argument("--pattern-rows", type=int, default=8)
    parser.add_argument("--square-size-mm", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--min-board-area-ratio", type=float, default=0.03)
    parser.add_argument("--max-board-area-ratio", type=float, default=0.75)
    parser.add_argument("--max-pnp-error-px", type=float, default=0.8)
    parser.add_argument("--stationary-wait-seconds", type=float, default=0.12)
    parser.add_argument("--max-stationary-translation-mm", type=float, default=0.5)
    parser.add_argument("--max-stationary-rotation-deg", type=float, default=0.15)
    parser.add_argument(
        "--tcp-offset-mm-deg",
        nargs=6,
        type=float,
        default=list(end_effector.tcp_offset_mm_deg),
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Expected active TCP offset for configuration checking; the controller-reported value is used.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_CALIBRATION_ROOT)
    return parser.parse_args()


def load_intrinsics(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    path = args.intrinsics.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("camera", {}).get("serial") != args.serial:
        raise ValueError(f"{path} does not belong to wrist camera serial {args.serial}")
    capture = payload.get("capture", {})
    if (capture.get("width"), capture.get("height")) != (args.width, args.height):
        raise ValueError(f"{path} was not calibrated at {args.width}x{args.height}")
    board = payload.get("chessboard", {})
    expected = (args.pattern_cols, args.pattern_rows, args.square_size_mm)
    actual = (
        board.get("inner_corners_cols"),
        board.get("inner_corners_rows"),
        board.get("square_size_mm"),
    )
    if actual[:2] != expected[:2] or not np.isclose(actual[2], expected[2]):
        raise ValueError(f"Chessboard mismatch: intrinsics used {actual}, command requests {expected}")
    return (
        np.asarray(payload["camera_matrix"], dtype=np.float64),
        np.asarray(payload["distortion_coefficients"], dtype=np.float64),
        payload,
    )


def read_robot_sample(arm, args: argparse.Namespace) -> tuple[list[float], list[float], np.ndarray]:
    if arm.get_is_moving():
        raise RuntimeError("robot is still moving; stop jogging before saving")
    first_code, first_pose = arm.get_position(is_radian=False)
    require_code("read first TCP pose", (first_code, first_pose))
    time.sleep(args.stationary_wait_seconds)
    if arm.get_is_moving():
        raise RuntimeError("robot moved during synchronization window")
    pose_code, tcp_pose = arm.get_position(is_radian=False)
    joint_code, joints = arm.get_servo_angle(is_radian=False)
    require_code("read synchronized TCP pose", (pose_code, tcp_pose))
    require_code("read synchronized joint angles", (joint_code, joints))
    first_transform = flange_transform_from_tcp_pose(first_pose, args.active_tcp_offset_mm_deg)
    flange_transform = flange_transform_from_tcp_pose(tcp_pose, args.active_tcp_offset_mm_deg)
    translation_mm = float(np.linalg.norm(first_transform[:3, 3] - flange_transform[:3, 3]) * 1000.0)
    rotation_deg = transform_rotation_angle_deg(first_transform, flange_transform)
    if translation_mm > args.max_stationary_translation_mm or rotation_deg > args.max_stationary_rotation_deg:
        raise RuntimeError(
            f"robot was not stationary (drift {translation_mm:.2f} mm, {rotation_deg:.3f} deg)"
        )
    return list(map(float, tcp_pose)), list(map(float, joints)), flange_transform


def write_manifest(run_dir: Path, metadata: dict, samples: list[dict]) -> None:
    payload = {**metadata, "sample_count": len(samples), "samples": samples}
    (run_dir / "samples.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def draw_overlay(
    color: np.ndarray,
    pattern_size: tuple[int, int],
    corners: np.ndarray | None,
    area_ratio: float,
    pnp_error: float | None,
    sample_count: int,
    message: str,
) -> np.ndarray:
    display = color.copy()
    if corners is not None:
        cv2.drawChessboardCorners(display, pattern_size, corners, True)
    lines = [
        "FIX BOARD; jog robot/camera, then STOP",
        f"samples={sample_count} | S save | C solve | Q quit",
        f"board area={area_ratio:.1%} | PnP={pnp_error:.3f}px" if pnp_error is not None else "board not detected",
        message,
    ]
    for index, line in enumerate(lines):
        cv2.putText(display, line, (10, 25 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return display


def main() -> int:
    args = parse_args()
    if not 8 <= args.min_samples <= args.max_samples:
        raise ValueError("Require 8 <= --min-samples <= --max-samples")
    camera_matrix, distortion, intrinsic_payload = load_intrinsics(args)
    object_points = chessboard_object_points(
        args.pattern_cols, args.pattern_rows, args.square_size_mm / 1000.0
    ).astype(np.float64)
    pattern_size = (args.pattern_cols, args.pattern_rows)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_dir = args.output_root.expanduser().resolve() / timestamp
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
        "type": "eye_in_hand_chessboard_samples",
        "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "robot_ip": args.robot_ip,
        "camera_serial": args.serial,
        "intrinsic_calibration": str(args.intrinsics.expanduser().resolve()),
        "expected_tcp_offset_mm_deg": list(map(float, args.tcp_offset_mm_deg)),
        "pose_convention": "base_T_flange = base_T_tcp @ inverse(flange_T_tcp)",
        "chessboard_fixed_in_robot_base": True,
    }
    samples: list[dict] = []

    from xarm.wrapper import XArmAPI

    arm = XArmAPI(do_not_open=True, is_radian=False, report_type="rich")
    pipeline = None
    try:
        require_code("connect read-only robot client", arm.connect(port=args.robot_ip))
        print("SAFETY: this script sends no robot motion, enable, mode, state, TCP, or I/O commands.")
        active_tcp_offset = np.asarray(arm.tcp_offset, dtype=np.float64)
        if active_tcp_offset.shape != (6,) or not np.all(np.isfinite(active_tcp_offset)):
            raise RuntimeError("Controller did not report a valid active TCP offset")
        args.active_tcp_offset_mm_deg = active_tcp_offset.tolist()
        metadata["active_tcp_offset_mm_deg"] = args.active_tcp_offset_mm_deg
        print(f"Controller TCP offset used for flange recovery: {args.active_tcp_offset_mm_deg}")
        if not np.allclose(active_tcp_offset, args.tcp_offset_mm_deg, atol=0.1):
            print(
                "WARNING: controller TCP offset differs from end_effector.yaml; "
                "this is safe for read-only calibration because the reported active value is used."
            )
        context = rs.context()
        device = choose_device(context, args.serial)
        metadata["camera"] = camera_info(device)
        config = rs.config()
        config.enable_device(args.serial)
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        pipeline = rs.pipeline(context)
        pipeline.start(config)
        for _ in range(args.warmup):
            pipeline.wait_for_frames()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        message = "Move to a varied pose; keep the full board visible."

        while True:
            frame = pipeline.wait_for_frames(2000).get_color_frame()
            if not frame:
                continue
            color = np.asanyarray(frame.get_data()).copy()
            found, corners = find_chessboard_corners(color, pattern_size)
            area = normalized_board_area_ratio(corners, (args.width, args.height)) if found else 0.0
            camera_from_board = None
            pnp_error = None
            if found and corners is not None:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points, corners, camera_matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if ok:
                    pnp_error = reprojection_error_px(
                        object_points, corners, camera_matrix, distortion, rvec, tvec
                    )
                    camera_from_board = transform_from_rvec_tvec(rvec, tvec)
            overlay = draw_overlay(color, pattern_size, corners, area, pnp_error, len(samples), message)
            cv2.imshow(WINDOW_NAME, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print(f"Stopped. Recoverable samples remain in {run_dir}")
                return 1
            if key in (ord("s"), ord("S")):
                try:
                    if camera_from_board is None or corners is None or pnp_error is None:
                        raise RuntimeError("complete chessboard was not detected")
                    if not args.min_board_area_ratio <= area <= args.max_board_area_ratio:
                        raise RuntimeError(f"board area {area:.1%} is outside the accepted range")
                    if pnp_error > args.max_pnp_error_px:
                        raise RuntimeError(f"PnP reprojection error {pnp_error:.3f} px is too high")
                    tcp_pose, joints, base_from_flange = read_robot_sample(arm, args)
                    if samples:
                        prior = [np.asarray(sample["transform_base_from_flange"], dtype=np.float64) for sample in samples]
                        nearest_translation = min(
                            np.linalg.norm(base_from_flange[:3, 3] - pose[:3, 3]) * 1000.0 for pose in prior
                        )
                        nearest_rotation = min(transform_rotation_angle_deg(base_from_flange, pose) for pose in prior)
                        if nearest_translation < 8.0 and nearest_rotation < 2.0:
                            raise RuntimeError("pose is too similar to an existing sample")
                    index = len(samples) + 1
                    color_path = sample_dir / f"sample_{index:02d}.png"
                    overlay_path = sample_dir / f"sample_{index:02d}_corners.png"
                    if not cv2.imwrite(str(color_path), color) or not cv2.imwrite(str(overlay_path), overlay):
                        raise RuntimeError("failed to save sample images")
                    samples.append(
                        {
                            "sample_index": index,
                            "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                            "color_path": str(color_path.relative_to(run_dir)),
                            "corner_overlay_path": str(overlay_path.relative_to(run_dir)),
                            "corner_points_px": corners.reshape(-1, 2).astype(float).tolist(),
                            "board_area_ratio": float(area),
                            "pnp_reprojection_error_px": float(pnp_error),
                            "tcp_pose_mm_deg": tcp_pose,
                            "joint_angles_deg": joints,
                            "transform_base_from_flange": base_from_flange.tolist(),
                            "transform_camera_from_board": camera_from_board.tolist(),
                        }
                    )
                    write_manifest(run_dir, metadata, samples)
                    diversity = pose_set_diversity(
                        [np.asarray(sample["transform_base_from_flange"]) for sample in samples]
                    )
                    message = (
                        f"Saved {index}; spread {diversity['max_translation_m']*1000:.0f}mm/"
                        f"{diversity['max_rotation_deg']:.1f}deg"
                    )
                    print(message)
                except RuntimeError as exc:
                    message = f"NOT SAVED: {exc}"
                    print(message)
            if key in (ord("c"), ord("C")) or len(samples) >= args.max_samples:
                if len(samples) < args.min_samples:
                    message = f"Need at least {args.min_samples} samples"
                    continue
                diversity = pose_set_diversity(
                    [np.asarray(sample["transform_base_from_flange"]) for sample in samples]
                )
                if diversity["max_rotation_deg"] < 20.0:
                    message = "Need more wrist-orientation diversity (at least 20 deg spread)"
                    continue
                break

        robot_poses = [np.asarray(sample["transform_base_from_flange"], dtype=np.float64) for sample in samples]
        board_poses = [np.asarray(sample["transform_camera_from_board"], dtype=np.float64) for sample in samples]
        result = calibrate_eye_in_hand(robot_poses, board_poses)
        args.calibration_root.expanduser().resolve().mkdir(parents=True, exist_ok=True)
        calibration_path = args.calibration_root.expanduser().resolve() / f"wrist_hand_eye_{timestamp}.json"
        translation_mm = np.asarray(result.board_translation_residuals_m) * 1000.0
        rotation_deg = np.asarray(result.board_rotation_residuals_deg)
        output = {
            "schema_version": 1,
            "type": "eye_in_hand_chessboard_calibration",
            "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "camera": metadata["camera"],
            "robot_ip": args.robot_ip,
            "sample_manifest": str(run_dir / "samples.json"),
            "intrinsic_calibration": str(args.intrinsics.expanduser().resolve()),
            "intrinsic_rms_reprojection_error_px": intrinsic_payload["quality"]["rms_reprojection_error_px"],
            "active_tcp_offset_mm_deg": list(map(float, args.active_tcp_offset_mm_deg)),
            "expected_tcp_offset_mm_deg": list(map(float, args.tcp_offset_mm_deg)),
            "transform_definition": "p_flange = transform_flange_from_camera @ p_camera",
            "transform_flange_from_camera": result.transform_flange_from_camera.tolist(),
            "estimated_transform_base_from_board": result.transform_base_from_board.tolist(),
            "method": result.method,
            "accepted_sample_indices": [index + 1 for index in result.accepted_indices],
            "rejected_sample_indices": [index + 1 for index in result.rejected_indices],
            "quality": {
                "accepted_count": len(result.accepted_indices),
                "board_translation_residual_mm": {
                    "mean": float(np.mean(translation_mm)),
                    "median": float(np.median(translation_mm)),
                    "max": float(np.max(translation_mm)),
                },
                "board_rotation_residual_deg": {
                    "mean": float(np.mean(rotation_deg)),
                    "median": float(np.median(rotation_deg)),
                    "max": float(np.max(rotation_deg)),
                },
                "robot_pose_diversity": pose_set_diversity(robot_poses),
            },
        }
        calibration_path.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(f"Method: {result.method}")
        print(f"Accepted/rejected samples: {len(result.accepted_indices)}/{len(result.rejected_indices)}")
        print(
            f"Fixed-board residual: median={np.median(translation_mm):.2f} mm, "
            f"max={np.max(translation_mm):.2f} mm; median rotation={np.median(rotation_deg):.3f} deg"
        )
        print(f"Saved hand-eye calibration: {calibration_path}")
        return 0
    finally:
        cv2.destroyAllWindows()
        if pipeline is not None:
            pipeline.stop()
        arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
