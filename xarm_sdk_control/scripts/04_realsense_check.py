#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import numpy as np
import pyrealsense2 as rs


def print_devices(context: rs.context) -> int:
    devices = context.query_devices()
    count = len(devices)
    print(f"RealSense device count: {count}")
    for index, device in enumerate(devices):
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        firmware = device.get_info(rs.camera_info.firmware_version)
        print(f"  [{index}] name={name}, serial={serial}, firmware={firmware}")
    return count


def intrinsics_to_text(intrinsics: rs.intrinsics) -> str:
    return (
        f"width={intrinsics.width}, height={intrinsics.height}, "
        f"fx={intrinsics.fx:.3f}, fy={intrinsics.fy:.3f}, "
        f"ppx={intrinsics.ppx:.3f}, ppy={intrinsics.ppy:.3f}, "
        f"model={intrinsics.model}, coeffs={[round(c, 6) for c in intrinsics.coeffs]}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check RealSense RGB-D capture. This script does not connect to or move the xArm."
        )
    )
    parser.add_argument("--width", type=int, default=640, help="Stream width in pixels.")
    parser.add_argument("--height", type=int, default=480, help="Stream height in pixels.")
    parser.add_argument("--fps", type=int, default=30, help="Stream frame rate.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Frames to discard before reading measurements.",
    )
    args = parser.parse_args()

    context = rs.context()
    if print_devices(context) == 0:
        print("\nNo RealSense camera detected. Check USB cable, port, and udev permissions.")
        return 2

    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    print(
        f"\nStarting streams: color/depth {args.width}x{args.height}@{args.fps}. "
        "No robot motion will be commanded."
    )
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        for _ in range(args.warmup):
            pipeline.wait_for_frames()

        frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            print("Failed to receive synchronized color/depth frames.")
            return 3

        color_profile = color_frame.profile.as_video_stream_profile()
        depth_profile = depth_frame.profile.as_video_stream_profile()
        color_intrinsics = color_profile.intrinsics
        depth_intrinsics = depth_profile.intrinsics

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        center_u = color_intrinsics.width // 2
        center_v = color_intrinsics.height // 2
        center_depth_m = depth_frame.get_distance(center_u, center_v)

        print("\n=== Frame Summary ===")
        print(f"color shape: {color.shape}, dtype={color.dtype}")
        print(f"depth shape: {depth.shape}, dtype={depth.dtype}")
        print(f"depth scale: {profile.get_device().first_depth_sensor().get_depth_scale():.8f} m/unit")
        print(f"center pixel: u={center_u}, v={center_v}, depth={center_depth_m:.4f} m")

        print("\n=== Intrinsics ===")
        print(f"color: {intrinsics_to_text(color_intrinsics)}")
        print(f"depth aligned to color: {intrinsics_to_text(depth_intrinsics)}")

        if center_depth_m > 0:
            point = rs.rs2_deproject_pixel_to_point(
                color_intrinsics, [center_u, center_v], center_depth_m
            )
            print(
                "\nCenter pixel in camera coordinates: "
                f"x={point[0]:.4f} m, y={point[1]:.4f} m, z={point[2]:.4f} m"
            )
        else:
            print("\nCenter depth is zero, so the center 3D point is invalid.")

        time.sleep(0.2)
        return 0
    finally:
        pipeline.stop()
        print("\nRealSense pipeline stopped.")


if __name__ == "__main__":
    raise SystemExit(main())
