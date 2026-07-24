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
OUTPUT_ROOT = ROOT / "data" / "handeye_motion_verification"


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
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def read_robot_pose(arm: XArmAPI) -> dict:
    position_code, position = arm.get_position(is_radian=False)
    state = arm.get_state()
    err_warn = arm.get_err_warn_code()
    return {
        "state": state,
        "err_warn": err_warn,
        "tcp_pose_code": position_code,
        "tcp_pose_mm_deg": position,
    }


def call(label: str, func, *args, **kwargs):
    result = func(*args, **kwargs)
    print(f"{label}: {result}")
    return result


def matrix_to_list(transform: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in transform]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Move xArm TCP to a safe point above the detected chessboard center. "
            "Default is dry-run; use --run to command motion."
        )
    )
    parser.add_argument("--run", action="store_true", help="Actually command robot motion.")
    parser.add_argument(
        "--max-x-mm",
        type=float,
        default=None,
        help="Required for --run. Rejects targets with X above this safety limit.",
    )
    parser.add_argument("--min-z-mm", type=float, default=180.0, help="Minimum allowed target Z.")
    parser.add_argument("--above-mm", type=float, default=250.0, help="Height above board center.")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--acc", type=float, default=100.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--pattern", type=parse_pattern, default=(8, 11))
    parser.add_argument("--square-size-m", type=float, default=0.020)
    args = parser.parse_args()

    if args.run and args.max_x_mm is None:
        print("Refusing to move: --run requires an explicit --max-x-mm safety limit.")
        return 2

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        robot_ip = yaml.safe_load(f)["robot_ip"]
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    tcp_t_camera = np.array(calibration["tcp_T_camera"], dtype=np.float64)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Motion verification target: chessboard center above-point.")
    print("Dry run mode." if not args.run else "RUN MODE: robot motion may be commanded.")
    print(f"Using calibration: {args.calibration}")

    arm = XArmAPI(do_not_open=True, is_radian=False)
    pipeline = None
    try:
        call("connect", arm.connect, port=robot_ip)
        robot_before = read_robot_pose(arm)
        current_pose = robot_before["tcp_pose_mm_deg"]
        print(f"Current TCP pose [x,y,z,roll,pitch,yaw] mm/deg: {current_pose}")

        base_t_tcp = tcp_pose_to_base_t_tcp(current_pose)

        context = rs.context()
        if len(context.query_devices()) == 0:
            print("No RealSense camera detected.")
            return 3

        pipeline = rs.pipeline(context)
        config = rs.config()
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        align = rs.align(rs.stream.color)
        profile = pipeline.start(config)
        for _ in range(args.warmup):
            pipeline.wait_for_frames()
        frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            print("Failed to receive synchronized color/depth frames.")
            return 4

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
            return 5

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
            return 6

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
        target_center_base_m = (base_t_target @ target_center_in_target)[:3]
        target_center_base_mm = target_center_base_m * 1000.0
        target_pose = list(current_pose)
        target_pose[0] = float(target_center_base_mm[0])
        target_pose[1] = float(target_center_base_mm[1])
        target_pose[2] = float(max(target_center_base_mm[2] + args.above_mm, args.min_z_mm))

        checks = {
            "max_x_ok": args.max_x_mm is None or target_pose[0] <= args.max_x_mm,
            "min_z_ok": target_pose[2] >= args.min_z_mm,
            "tcp_pose_code_ok": robot_before["tcp_pose_code"] == 0,
        }
        checks["all_ok"] = all(checks.values())

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

        print(f"Chessboard detected: corners={len(corners)}, method={method}")
        print(f"mean reprojection error: {reprojection_error_px:.4f} px")
        print(f"Target center in base frame [mm]: {target_center_base_mm.round(3).tolist()}")
        print(f"Planned above-point TCP pose [mm/deg]: {[round(v, 3) for v in target_pose]}")
        print(f"Safety checks: {checks}")

        moved = False
        robot_after = None
        if args.run:
            if not checks["all_ok"]:
                print("Refusing to move: safety checks failed.")
                return 7
            call("clean_warn", arm.clean_warn)
            call("clean_error", arm.clean_error)
            call("motion_enable", arm.motion_enable, enable=True)
            call("set_mode(0)", arm.set_mode, 0)
            call("set_state(0)", arm.set_state, 0)
            call(
                "move above target",
                arm.set_position,
                x=target_pose[0],
                y=target_pose[1],
                z=target_pose[2],
                roll=target_pose[3],
                pitch=target_pose[4],
                yaw=target_pose[5],
                speed=args.speed,
                mvacc=args.acc,
                wait=True,
            )
            moved = True
            robot_after = read_robot_pose(arm)
            print(f"After TCP pose [x,y,z,roll,pitch,yaw] mm/deg: {robot_after['tcp_pose_mm_deg']}")
        else:
            print("Dry run only. Re-run with --run --max-x-mm <limit> to execute motion.")

        result = {
            "timestamp": timestamp,
            "calibration_path": str(args.calibration),
            "run": bool(args.run),
            "moved": moved,
            "robot_before": robot_before,
            "robot_after": robot_after,
            "target": {
                "pattern_inner_corners": list(args.pattern),
                "square_size_m": args.square_size_m,
                "detection_method": method,
                "mean_reprojection_error_px": reprojection_error_px,
                "base_T_target": matrix_to_list(base_t_target),
                "target_center_base_mm": target_center_base_mm.tolist(),
                "planned_above_tcp_pose_mm_deg": target_pose,
            },
            "safety": {
                "above_mm": args.above_mm,
                "min_z_mm": args.min_z_mm,
                "max_x_mm": args.max_x_mm,
                "checks": checks,
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
        print(f"Saved annotated image: {annotated_path}")
        print(f"Saved result: {result_path}")
        return 0
    finally:
        if pipeline is not None:
            pipeline.stop()
            print("RealSense pipeline stopped.")
        arm.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    raise SystemExit(main())
