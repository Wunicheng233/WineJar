#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from xarm.wrapper import XArmAPI


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "robot.yaml"
DEFAULT_CALIBRATION = ROOT / "calibration" / "handeye_20260703_200745.json"
OUTPUT_ROOT = ROOT / "data" / "handeye_static_verification"


def parse_pattern(text: str) -> tuple[int, int]:
    try:
        cols_text, rows_text = text.lower().replace("x", ",").split(",", maxsplit=1)
        cols = int(cols_text.strip())
        rows = int(rows_text.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Pattern must look like 8x11 or 11,8") from exc
    if cols < 2 or rows < 2:
        raise argparse.ArgumentTypeError("Pattern dimensions must both be >= 2")
    return cols, rows


def rx(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def ry(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rz(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def tcp_pose_to_base_t_tcp(pose_mm_deg: list[float]) -> np.ndarray:
    x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg = pose_mm_deg
    rotation = (
        rz(math.radians(yaw_deg))
        @ ry(math.radians(pitch_deg))
        @ rx(math.radians(roll_deg))
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0
    return transform


def rt_to_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def find_chessboard(gray: np.ndarray, pattern: tuple[int, int]):
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags)
    method = "findChessboardCornersSB"
    if found:
        return corners, method

    legacy_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, pattern, legacy_flags)
    method = "findChessboardCorners"
    if not found:
        return None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria), method


def make_object_points(pattern: tuple[int, int], square_size_m: float) -> np.ndarray:
    cols, rows = pattern
    points = np.zeros((rows * cols, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= square_size_m
    return points


def camera_matrix_from_intrinsics(intrinsics: rs.intrinsics) -> np.ndarray:
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, 0.0 + intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def read_robot_pose(robot_ip: str) -> dict:
    arm = XArmAPI(do_not_open=True, is_radian=False)
    try:
        connect_code = arm.connect(port=robot_ip)
        position_code, position = arm.get_position(is_radian=False)
        state = arm.get_state()
        err_warn = arm.get_err_warn_code()
        return {
            "connect_code": connect_code,
            "connected": bool(arm.connected),
            "state": state,
            "err_warn": err_warn,
            "tcp_pose_code": position_code,
            "tcp_pose_mm_deg": position,
        }
    finally:
        arm.disconnect()


def matrix_to_list(transform: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in transform]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Static handeye verification: detect chessboard and compute its base-frame pose. "
            "This script reads xArm pose only and never commands robot motion."
        )
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--pattern", type=parse_pattern, default=(8, 11))
    parser.add_argument("--square-size-m", type=float, default=0.020)
    args = parser.parse_args()

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        robot_ip = yaml.safe_load(f)["robot_ip"]
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    tcp_t_camera = np.array(calibration["tcp_T_camera"], dtype=np.float64)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Static handeye verification: no motion_enable, no set_mode, no set_state, no motion command.")
    print(f"Using calibration: {args.calibration}")
    print(f"Reading xArm TCP pose from {robot_ip}")
    robot = read_robot_pose(robot_ip)
    print(f"TCP pose [x,y,z,roll,pitch,yaw] mm/deg: {robot['tcp_pose_mm_deg']}")
    base_t_tcp = tcp_pose_to_base_t_tcp(robot["tcp_pose_mm_deg"])

    context = rs.context()
    devices = context.query_devices()
    print(f"RealSense device count: {len(devices)}")
    if len(devices) == 0:
        print("No RealSense camera detected.")
        return 2

    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)

    profile = pipeline.start(config)
    try:
        for _ in range(args.warmup):
            pipeline.wait_for_frames()
        frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            print("Failed to receive synchronized color/depth frames.")
            return 3

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        corners, method = find_chessboard(gray, args.pattern)
        color_path = output_dir / "color.png"
        depth_path = output_dir / "depth_raw.png"
        annotated_path = output_dir / "annotated.png"
        result_path = output_dir / "result.json"

        if corners is None:
            cv2.imwrite(str(color_path), color)
            cv2.imwrite(str(depth_path), depth)
            print("Chessboard not detected; saved raw images for diagnosis.")
            print(f"Saved result folder: {output_dir}")
            return 1

        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        camera_matrix = camera_matrix_from_intrinsics(intrinsics)
        dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64).reshape(-1, 1)
        object_points = make_object_points(args.pattern, args.square_size_m)
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            corners.reshape(-1, 2),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            print("Chessboard detected, but solvePnP failed.")
            return 4

        rotation_camera_target, _ = cv2.Rodrigues(rvec)
        camera_t_target = rt_to_transform(rotation_camera_target, tvec.reshape(3))
        base_t_target = base_t_tcp @ tcp_t_camera @ camera_t_target

        cols, rows = args.pattern
        target_center_in_target = np.array(
            [
                (cols - 1) * args.square_size_m / 2.0,
                (rows - 1) * args.square_size_m / 2.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )
        target_origin_base = base_t_target[:3, 3]
        target_center_base = (base_t_target @ target_center_in_target)[:3]

        reprojected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        reprojection_error_px = float(
            np.mean(np.linalg.norm(corners.reshape(-1, 2) - reprojected.reshape(-1, 2), axis=1))
        )

        annotated = color.copy()
        cv2.drawChessboardCorners(annotated, args.pattern, corners, True)
        cv2.drawFrameAxes(annotated, camera_matrix, dist_coeffs, rvec, tvec, args.square_size_m * 2.0)
        cv2.imwrite(str(color_path), color)
        cv2.imwrite(str(depth_path), depth)
        cv2.imwrite(str(annotated_path), annotated)

        result = {
            "timestamp": timestamp,
            "safety": {
                "robot_motion_commanded": False,
                "note": "Static verification only; no xArm movement commands are sent.",
            },
            "calibration_path": str(args.calibration),
            "robot": robot,
            "target": {
                "type": "chessboard",
                "pattern_inner_corners": list(args.pattern),
                "square_size_m": args.square_size_m,
                "corner_count": int(len(corners)),
                "detection_method": method,
                "mean_reprojection_error_px": reprojection_error_px,
                "camera_T_target_rvec": rvec.reshape(-1).tolist(),
                "camera_T_target_tvec_m": tvec.reshape(-1).tolist(),
                "base_T_target": matrix_to_list(base_t_target),
                "target_origin_base_m": target_origin_base.tolist(),
                "target_center_base_m": target_center_base.tolist(),
                "target_center_base_mm": (target_center_base * 1000.0).tolist(),
            },
            "outputs": {
                "color": str(color_path),
                "depth_raw": str(depth_path),
                "annotated": str(annotated_path),
                "result": str(result_path),
            },
            "camera": {
                "depth_scale_m_per_unit": profile.get_device()
                .first_depth_sensor()
                .get_depth_scale()
            },
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        print(f"Chessboard detected: corners={len(corners)}, method={method}")
        print(f"mean reprojection error: {reprojection_error_px:.4f} px")
        print(
            "Target origin in base frame [mm]: "
            f"{(target_origin_base * 1000.0).round(3).tolist()}"
        )
        print(
            "Target center in base frame [mm]: "
            f"{(target_center_base * 1000.0).round(3).tolist()}"
        )
        print(f"Saved annotated image: {annotated_path}")
        print(f"Saved result: {result_path}")
        return 0
    finally:
        pipeline.stop()
        print("RealSense pipeline stopped.")


if __name__ == "__main__":
    raise SystemExit(main())
