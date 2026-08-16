#!/usr/bin/env python3
"""Interactively collect fixed-camera 3D point pairs for eye-to-hand calibration.

Move the xArm only from UFactory. Left-click the physical point corresponding
to the current TCP (or configured marker offset), then press S to save a pair.
This script reads robot state and never enables or commands robot motion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.realsense_capture import camera_info, choose_device  # noqa: E402
from xarm_sdk_control.robot_motion import load_robot_ip  # noqa: E402


CONFIG_PATH = ROOT / "config" / "robot.yaml"
OUTPUT_ROOT = ROOT / "data" / "global_camera_samples"
WINDOW_NAME = "Global camera calibration - read-only robot"


def parse_pixel(text: str) -> tuple[int, int]:
    try:
        u_text, v_text = text.replace("x", ",").split(",", maxsplit=1)
        return int(u_text.strip()), int(v_text.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Pixel must look like 320,240") from exc


def parse_xyz_mm(text: str) -> np.ndarray:
    try:
        values = [float(value.strip()) for value in text.replace(" ", ",").split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Offset must look like 0,0,0") from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Offset must contain exactly three values: x,y,z")
    return np.asarray(values, dtype=np.float64)


def rx(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def ry(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rz(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def marker_point_in_base_m(tcp_pose_mm_deg: list[float], marker_offset_mm: np.ndarray) -> np.ndarray:
    position_m = np.asarray(tcp_pose_mm_deg[:3], dtype=np.float64) / 1000.0
    roll, pitch, yaw = np.radians(tcp_pose_mm_deg[3:6])
    rotation = rz(yaw) @ ry(pitch) @ rx(roll)
    return position_m + rotation @ (marker_offset_mm / 1000.0)


def read_robot_pose(arm: XArmAPI) -> dict[str, Any]:
    state = arm.get_state()
    err_warn = arm.get_err_warn_code()
    position_code, position = arm.get_position(is_radian=False)
    angle_code, angles = arm.get_servo_angle(is_radian=False)
    return {
        "connected": bool(arm.connected),
        "state": state,
        "err_warn": err_warn,
        "tcp_pose_code": position_code,
        "tcp_pose_mm_deg": position,
        "joint_angles_code": angle_code,
        "joint_angles_deg": angles,
    }


def depth_statistics(
    depth_frame: rs.depth_frame,
    u: int,
    v: int,
    radius_px: int,
    layer_tolerance_m: float = 0.008,
) -> tuple[float, float, int]:
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    depths = []
    for y in range(max(0, v - radius_px), min(height, v + radius_px + 1)):
        for x in range(max(0, u - radius_px), min(width, u + radius_px + 1)):
            depth_m = depth_frame.get_distance(x, y)
            if depth_m > 0:
                depths.append(depth_m)
    center_depth_m = float(depth_frame.get_distance(u, v))
    if center_depth_m <= 0 or not depths:
        return 0.0, float("inf"), 0
    values = np.asarray(depths, dtype=np.float64)
    same_surface = values[np.abs(values - center_depth_m) <= layer_tolerance_m]
    if same_surface.size == 0:
        return 0.0, float("inf"), 0
    median_m = float(np.median(same_surface))
    mad_m = float(np.median(np.abs(same_surface - median_m)))
    return median_m, mad_m, int(same_surface.size)


def save_sample(
    output_root: Path,
    sample_index: int,
    color: np.ndarray,
    depth_raw: np.ndarray,
    pixel_uv: tuple[int, int],
    depth_m: float,
    depth_mad_m: float,
    valid_depth_count: int,
    depth_radius_px: int,
    intrinsics: rs.intrinsics,
    robot: dict[str, Any],
    device_info: dict[str, str],
    marker_offset_mm: np.ndarray,
) -> Path:
    u, v = pixel_uv
    camera_point_m = np.asarray(
        rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], depth_m),
        dtype=np.float64,
    )
    tcp_pose = robot["tcp_pose_mm_deg"]
    base_point_m = marker_point_in_base_m(tcp_pose, marker_offset_mm)
    timestamp = datetime.now().astimezone()
    sample_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_dir = output_root.expanduser().resolve() / sample_id
    output_dir.mkdir(parents=True, exist_ok=False)

    annotated = color.copy()
    cv2.drawMarker(annotated, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
    cv2.circle(annotated, (u, v), depth_radius_px, (0, 255, 255), 1)
    cv2.putText(
        annotated,
        f"sample {sample_index:02d} depth={depth_m:.4f}m",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )

    color_path = output_dir / "color.png"
    depth_path = output_dir / "depth_raw.png"
    annotated_path = output_dir / "annotated.png"
    sample_path = output_dir / "sample.json"
    cv2.imwrite(str(color_path), color)
    cv2.imwrite(str(depth_path), depth_raw)
    cv2.imwrite(str(annotated_path), annotated)

    sample = {
        "timestamp": timestamp.isoformat(timespec="milliseconds"),
        "type": "global_camera_point_pair",
        "point_name": f"p{sample_index:02d}",
        "camera": device_info,
        "robot": robot,
        "selection": {
            "pixel_uv": [u, v],
            "depth_radius_px": depth_radius_px,
            "median_depth_m": depth_m,
            "depth_mad_m": depth_mad_m,
            "valid_depth_count": valid_depth_count,
        },
        "point_pair": {
            "camera_point_m": camera_point_m.tolist(),
            "base_point_m": base_point_m.tolist(),
            "base_point_source": "current TCP pose transformed by tcp_marker_offset_mm",
            "tcp_marker_offset_mm": marker_offset_mm.tolist(),
        },
        "outputs": {
            "color": str(color_path),
            "depth_raw": str(depth_path),
            "annotated": str(annotated_path),
            "sample": str(sample_path),
        },
    }
    sample_path.write_text(json.dumps(sample, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(
        f"SAVED p{sample_index:02d}: pixel={[u, v]}, depth={depth_m:.4f} m, "
        f"camera={camera_point_m.round(5).tolist()}, base={base_point_m.round(5).tolist()}"
    )
    print(f"  {sample_path}")
    return sample_path


def draw_live_overlay(
    color: np.ndarray,
    selected_pixel: tuple[int, int] | None,
    depth_frame: rs.depth_frame,
    depth_radius_px: int,
    sample_count: int,
) -> np.ndarray:
    display = color.copy()
    lines = [
        "Left click: select TCP/marker point | S: save | C: clear | Q/Esc: quit",
        f"Saved samples: {sample_count} (recommended: 10-15, spread in X/Y/Z)",
        "Robot motion is read-only here; move only with UFactory",
    ]
    for index, text in enumerate(lines):
        cv2.putText(display, text, (10, 24 + 24 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    if selected_pixel is not None:
        u, v = selected_pixel
        depth_m, mad_m, count = depth_statistics(depth_frame, u, v, depth_radius_px)
        cv2.drawMarker(display, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
        cv2.circle(display, (u, v), depth_radius_px, (0, 255, 255), 1)
        cv2.putText(
            display,
        f"pixel=({u},{v}) surfaceDepth={depth_m:.4f}m MAD={mad_m * 1000.0:.1f}mm n={count}",
            (10, display.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
    return display


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixel", type=parse_pixel, help="Optional one-shot pixel u,v; omit for interactive mode.")
    parser.add_argument("--serial", help="Optional RealSense serial number.")
    parser.add_argument("--depth-radius-px", type=int, default=4)
    parser.add_argument(
        "--depth-layer-tolerance-mm",
        type=float,
        default=8.0,
        help="Only average neighbors within this depth of the clicked center pixel.",
    )
    parser.add_argument("--max-depth-mad-mm", type=float, default=4.0)
    parser.add_argument(
        "--tcp-marker-offset-mm",
        type=parse_xyz_mm,
        default=np.zeros(3, dtype=np.float64),
        help="Selected marker point in TCP coordinates, x,y,z mm. Default 0,0,0 means select the TCP origin.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    robot_ip = load_robot_ip(CONFIG_PATH)

    print("READ-ONLY MODE: this script never enables or commands xArm motion.")
    print("Move the robot only in UFactory, then select the corresponding physical point and press S.")
    print(f"TCP marker offset [mm]: {args.tcp_marker_offset_mm.tolist()}")

    arm = XArmAPI(do_not_open=True, is_radian=False)
    context = rs.context()
    if len(context.query_devices()) == 0:
        print("ERROR: no RealSense camera detected")
        return 2
    try:
        device = choose_device(context, args.serial)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 3
    device_info = camera_info(device)

    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(device_info["serial"])
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)
    selected: dict[str, tuple[int, int] | None] = {"pixel": args.pixel}

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            selected["pixel"] = (x, y)

    started = False
    try:
        connect_code = arm.connect(port=robot_ip)
        if connect_code not in (0, None) or not arm.connected:
            print(f"ERROR: failed to connect to xArm {robot_ip}: {connect_code}")
            return 4
        profile = pipeline.start(config)
        started = True
        for _ in range(args.warmup):
            pipeline.wait_for_frames()
        print(
            f"Connected read-only to xArm {robot_ip}; using {device_info['name']} "
            f"serial={device_info['serial']} USB={device_info['usb_type']}"
        )

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
        sample_count = len(list(args.output_root.glob("*/sample.json")))
        one_shot_pending = args.pixel is not None

        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            display = draw_live_overlay(color, selected["pixel"], depth_frame, args.depth_radius_px, sample_count)
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            should_save = key in (ord("s"), ord("S")) or one_shot_pending
            one_shot_pending = False

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("c"), ord("C")):
                selected["pixel"] = None
                continue
            if not should_save:
                continue
            if selected["pixel"] is None:
                print("NOT SAVED: left-click the TCP/marker point first")
                continue
            u, v = selected["pixel"]
            if not (0 <= u < args.width and 0 <= v < args.height):
                print(f"NOT SAVED: pixel {(u, v)} is outside {args.width}x{args.height}")
                continue
            depth_m, depth_mad_m, valid_count = depth_statistics(
                depth_frame,
                u,
                v,
                args.depth_radius_px,
                args.depth_layer_tolerance_mm / 1000.0,
            )
            if depth_m <= 0 or valid_count < 9:
                print(
                    "NOT SAVED: clicked center has no reliable depth layer. "
                    "The sensor may be seeing a reflective/black tool surface; it will not use background depth."
                )
                continue
            if depth_mad_m * 1000.0 > args.max_depth_mad_mm:
                print(
                    f"NOT SAVED: local depth MAD {depth_mad_m * 1000.0:.1f} mm exceeds "
                    f"{args.max_depth_mad_mm:.1f} mm; select a flatter, clearer point"
                )
                continue
            robot = read_robot_pose(arm)
            if robot["tcp_pose_code"] != 0 or robot["tcp_pose_mm_deg"] is None:
                print(f"NOT SAVED: failed to read TCP pose: {robot}")
                continue
            err_warn = robot.get("err_warn")
            if isinstance(err_warn, tuple) and err_warn[0] != 0:
                print(f"NOT SAVED: robot error/warning query failed: {err_warn}")
                continue
            sample_count += 1
            save_sample(
                output_root=args.output_root,
                sample_index=sample_count,
                color=color,
                depth_raw=depth_raw,
                pixel_uv=(u, v),
                depth_m=depth_m,
                depth_mad_m=depth_mad_m,
                valid_depth_count=valid_count,
                depth_radius_px=args.depth_radius_px,
                intrinsics=intrinsics,
                robot=robot,
                device_info=device_info,
                marker_offset_mm=args.tcp_marker_offset_mm,
            )
            if args.pixel is not None:
                break

        print(f"Collection finished. Total sample files now: {sample_count}")
        return 0
    finally:
        cv2.destroyAllWindows()
        if started:
            pipeline.stop()
        arm.disconnect()
        print("RealSense stopped; xArm disconnected.")


if __name__ == "__main__":
    raise SystemExit(main())
