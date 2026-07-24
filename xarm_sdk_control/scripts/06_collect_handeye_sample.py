#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from xarm.wrapper import XArmAPI


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "robot.yaml"
OUTPUT_ROOT = ROOT / "data" / "handeye_samples"


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
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined, method


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
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def read_robot_pose(robot_ip: str) -> dict:
    arm = XArmAPI(do_not_open=True, is_radian=False)
    try:
        connect_code = arm.connect(port=robot_ip)
        state = arm.get_state()
        err_warn = arm.get_err_warn_code()
        position_code, position = arm.get_position(is_radian=False)
        angle_code, angles = arm.get_servo_angle(is_radian=False)
        return {
            "connect_code": connect_code,
            "connected": bool(arm.connected),
            "state": state,
            "err_warn": err_warn,
            "tcp_pose_code": position_code,
            "tcp_pose_mm_deg": position,
            "joint_angles_code": angle_code,
            "joint_angles_deg": angles,
        }
    finally:
        arm.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect one eye-in-hand calibration sample. "
            "This script reads xArm pose only; it never commands robot motion."
        )
    )
    parser.add_argument("--width", type=int, default=640, help="Stream width in pixels.")
    parser.add_argument("--height", type=int, default=480, help="Stream height in pixels.")
    parser.add_argument("--fps", type=int, default=30, help="Stream frame rate.")
    parser.add_argument("--warmup", type=int, default=30, help="Frames to discard first.")
    parser.add_argument("--pattern", type=parse_pattern, default=(8, 11), help="Inner corners.")
    parser.add_argument(
        "--square-size-m",
        type=float,
        default=0.020,
        help="Chessboard square size in meters.",
    )
    args = parser.parse_args()

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    robot_ip = config["robot_ip"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Handeye sample collection: no motion_enable, no set_mode, no set_state, no motion command.")
    print("Obstacle note: this script will not move in +X or any other direction.")
    print(f"Reading xArm TCP pose from {robot_ip}")
    robot = read_robot_pose(robot_ip)
    print(f"TCP pose code: {robot['tcp_pose_code']}")
    print(f"TCP pose [x,y,z,roll,pitch,yaw] mm/deg: {robot['tcp_pose_mm_deg']}")

    context = rs.context()
    devices = context.query_devices()
    print(f"RealSense device count: {len(devices)}")
    if len(devices) == 0:
        print("No RealSense camera detected.")
        return 2

    device = devices[0]
    device_info = {
        "name": device.get_info(rs.camera_info.name),
        "serial": device.get_info(rs.camera_info.serial_number),
        "firmware": device.get_info(rs.camera_info.firmware_version),
    }
    print(
        f"Using camera: {device_info['name']}, "
        f"serial={device_info['serial']}, firmware={device_info['firmware']}"
    )

    pipeline = rs.pipeline(context)
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    rs_config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)

    profile = pipeline.start(rs_config)
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
        if corners is None:
            color_path = output_dir / "color.png"
            depth_path = output_dir / "depth_raw.png"
            cv2.imwrite(str(color_path), color)
            cv2.imwrite(str(depth_path), depth)
            print("Chessboard not detected; saved raw images for diagnosis.")
            print(f"Saved color image: {color_path}")
            print(f"Saved raw depth image: {depth_path}")
            return 1

        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        camera_matrix = camera_matrix_from_intrinsics(color_intrinsics)
        dist_coeffs = np.array(color_intrinsics.coeffs, dtype=np.float64).reshape(-1, 1)
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

        reprojected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        errors = np.linalg.norm(corners.reshape(-1, 2) - reprojected.reshape(-1, 2), axis=1)
        reprojection_error_px = float(np.mean(errors))

        annotated = color.copy()
        cv2.drawChessboardCorners(annotated, args.pattern, corners, True)
        cv2.drawFrameAxes(
            annotated,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            args.square_size_m * 2.0,
        )

        color_path = output_dir / "color.png"
        depth_path = output_dir / "depth_raw.png"
        annotated_path = output_dir / "annotated.png"
        sample_path = output_dir / "sample.json"
        cv2.imwrite(str(color_path), color)
        cv2.imwrite(str(depth_path), depth)
        cv2.imwrite(str(annotated_path), annotated)

        sample = {
            "timestamp": timestamp,
            "safety": {
                "robot_motion_commanded": False,
                "positive_x_motion_commanded": False,
                "note": "Read-only sample collection; no xArm movement commands are sent.",
            },
            "robot": robot,
            "camera": {
                "device": device_info,
                "stream": {"width": args.width, "height": args.height, "fps": args.fps},
                "intrinsics": {
                    "width": color_intrinsics.width,
                    "height": color_intrinsics.height,
                    "fx": color_intrinsics.fx,
                    "fy": color_intrinsics.fy,
                    "ppx": color_intrinsics.ppx,
                    "ppy": color_intrinsics.ppy,
                    "model": str(color_intrinsics.model),
                    "coeffs": list(color_intrinsics.coeffs),
                },
                "depth_scale_m_per_unit": profile.get_device()
                .first_depth_sensor()
                .get_depth_scale(),
            },
            "target": {
                "type": "chessboard",
                "pattern_inner_corners": list(args.pattern),
                "square_size_m": args.square_size_m,
                "corner_count": int(len(corners)),
                "detection_method": method,
                "camera_T_target_rvec": rvec.reshape(-1).tolist(),
                "camera_T_target_tvec_m": tvec.reshape(-1).tolist(),
                "mean_reprojection_error_px": reprojection_error_px,
            },
            "outputs": {
                "color": str(color_path),
                "depth_raw": str(depth_path),
                "annotated": str(annotated_path),
                "sample": str(sample_path),
            },
        }
        sample_path.write_text(json.dumps(sample, indent=2), encoding="utf-8")

        print(f"Chessboard detected: pattern={args.pattern}, corners={len(corners)}, method={method}")
        print(f"solvePnP target translation in camera frame [m]: {tvec.reshape(-1).tolist()}")
        print(f"mean reprojection error: {reprojection_error_px:.4f} px")
        print(f"Saved annotated image: {annotated_path}")
        print(f"Saved sample: {sample_path}")
        return 0
    finally:
        pipeline.stop()
        print("RealSense pipeline stopped.")


if __name__ == "__main__":
    raise SystemExit(main())
