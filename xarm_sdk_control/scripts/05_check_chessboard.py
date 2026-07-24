#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "data" / "chessboard_checks"


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


def find_chessboard(gray: np.ndarray, patterns: list[tuple[int, int]]):
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    for pattern in patterns:
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags)
        method = "findChessboardCornersSB"
        if not found:
            legacy_flags = (
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_FAST_CHECK
            )
            found, corners = cv2.findChessboardCorners(gray, pattern, legacy_flags)
            method = "findChessboardCorners"
            if found:
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        if found:
            return pattern, corners, method
    return None, None, None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Capture one RealSense color/depth frame and check chessboard corner detection. "
            "This script does not connect to or move the xArm."
        )
    )
    parser.add_argument("--width", type=int, default=640, help="Stream width in pixels.")
    parser.add_argument("--height", type=int, default=480, help="Stream height in pixels.")
    parser.add_argument("--fps", type=int, default=30, help="Stream frame rate.")
    parser.add_argument("--warmup", type=int, default=30, help="Frames to discard first.")
    parser.add_argument(
        "--pattern",
        action="append",
        type=parse_pattern,
        help=(
            "Inner-corner pattern to try, such as 8x11. "
            "Can be repeated. Defaults to 8x11 and 11x8."
        ),
    )
    args = parser.parse_args()

    patterns = args.pattern or [(8, 11), (11, 8)]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

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
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)

    print(
        f"Starting streams {args.width}x{args.height}@{args.fps}; "
        f"trying patterns: {patterns}"
    )
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

        pattern, corners, method = find_chessboard(gray, patterns)
        annotated = color.copy()
        found = pattern is not None
        if found:
            cv2.drawChessboardCorners(annotated, pattern, corners, True)
            print(
                f"Chessboard detected: pattern={pattern}, "
                f"corners={len(corners)}, method={method}"
            )
        else:
            print("Chessboard not detected. Try improving lighting, distance, focus, or board angle.")

        color_path = output_dir / "color.png"
        depth_path = output_dir / "depth_raw.png"
        annotated_path = output_dir / "annotated.png"
        metadata_path = output_dir / "metadata.json"
        cv2.imwrite(str(color_path), color)
        cv2.imwrite(str(depth_path), depth)
        cv2.imwrite(str(annotated_path), annotated)

        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        metadata = {
            "timestamp": timestamp,
            "device": device_info,
            "stream": {"width": args.width, "height": args.height, "fps": args.fps},
            "patterns_tried": patterns,
            "found": found,
            "detected_pattern": pattern,
            "corner_count": int(len(corners)) if found else 0,
            "method": method,
            "color_intrinsics": {
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
            "outputs": {
                "color": str(color_path),
                "depth_raw": str(depth_path),
                "annotated": str(annotated_path),
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(f"Saved color image: {color_path}")
        print(f"Saved raw depth image: {depth_path}")
        print(f"Saved annotated image: {annotated_path}")
        print(f"Saved metadata: {metadata_path}")
        return 0 if found else 1
    finally:
        pipeline.stop()
        print("RealSense pipeline stopped.")


if __name__ == "__main__":
    raise SystemExit(main())
