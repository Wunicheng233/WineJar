#!/usr/bin/env python3
"""Capture one aligned RealSense RGB-D frame without connecting to the robot."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "vision_frames"


def camera_info(device: rs.device) -> dict[str, str]:
    fields = {
        "name": rs.camera_info.name,
        "serial": rs.camera_info.serial_number,
        "firmware": rs.camera_info.firmware_version,
        "usb_type": rs.camera_info.usb_type_descriptor,
    }
    result = {}
    for key, field in fields.items():
        result[key] = device.get_info(field) if device.supports(field) else "unknown"
    return result


def intrinsics_to_dict(intrinsics: rs.intrinsics) -> dict:
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "distortion_model": str(intrinsics.model),
        "distortion_coefficients": list(intrinsics.coeffs),
        "camera_matrix": [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
    }


def make_depth_preview(depth_raw: np.ndarray, depth_scale_m: float) -> np.ndarray:
    depth_m = depth_raw.astype(np.float32) * depth_scale_m
    valid = depth_m > 0
    normalized = np.zeros(depth_raw.shape, dtype=np.uint8)
    if np.any(valid):
        near_m, far_m = np.percentile(depth_m[valid], [2, 98])
        if far_m > near_m:
            clipped = np.clip(depth_m, near_m, far_m)
            normalized[valid] = np.round(
                255.0 * (far_m - clipped[valid]) / (far_m - near_m)
            ).astype(np.uint8)
    preview = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview


def choose_device(context: rs.context, serial: str | None) -> rs.device:
    devices = context.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device detected")
    if serial is None:
        return devices[0]
    for device in devices:
        if device.get_info(rs.camera_info.serial_number) == serial:
            return device
    available = [d.get_info(rs.camera_info.serial_number) for d in devices]
    raise RuntimeError(f"RealSense serial {serial!r} not found; available: {available}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Save one color frame, depth aligned to color, and camera metadata. "
            "This script never connects to or moves the xArm."
        )
    )
    parser.add_argument("--serial", help="Optional RealSense serial number.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    context = rs.context()
    try:
        device = choose_device(context, args.serial)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    info = camera_info(device)
    print(
        f"Using {info['name']}, serial={info['serial']}, "
        f"firmware={info['firmware']}, USB={info['usb_type']}"
    )

    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(info["serial"])
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align_to_color = rs.align(rs.stream.color)
    started = False

    try:
        profile = pipeline.start(config)
        started = True
        for _ in range(args.warmup):
            pipeline.wait_for_frames()

        frames = align_to_color.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            print("ERROR: synchronized aligned color/depth frames were not received")
            return 3

        timestamp = datetime.now().astimezone()
        timestamp_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        output_dir = args.output_root.expanduser().resolve() / timestamp_id
        output_dir.mkdir(parents=True, exist_ok=False)

        color = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_scale_m = profile.get_device().first_depth_sensor().get_depth_scale()
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        valid_depth = depth_raw > 0

        paths = {
            "color_bgr": output_dir / "color.png",
            "depth_raw": output_dir / "depth_raw.png",
            "depth_preview": output_dir / "depth_preview.png",
            "metadata": output_dir / "frame.json",
        }
        if not cv2.imwrite(str(paths["color_bgr"]), color):
            raise RuntimeError("Failed to save color image")
        if not cv2.imwrite(str(paths["depth_raw"]), depth_raw):
            raise RuntimeError("Failed to save raw depth image")
        cv2.imwrite(str(paths["depth_preview"]), make_depth_preview(depth_raw, depth_scale_m))

        metadata = {
            "schema_version": 1,
            "frame_id": timestamp_id,
            "captured_at": timestamp.isoformat(timespec="milliseconds"),
            "source": "realsense",
            "camera": info,
            "stream": {
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "color_format": "BGR8_PNG",
                "depth_format": "Z16_PNG",
                "depth_aligned_to": "color",
                "depth_scale_m_per_unit": depth_scale_m,
                "coordinate_convention": "+x right, +y down, +z forward",
            },
            "intrinsics": intrinsics_to_dict(intrinsics),
            "quality": {
                "valid_depth_fraction": float(np.mean(valid_depth)),
                "minimum_valid_depth_m": (
                    float(np.min(depth_raw[valid_depth]) * depth_scale_m)
                    if np.any(valid_depth)
                    else None
                ),
                "maximum_valid_depth_m": (
                    float(np.max(depth_raw[valid_depth]) * depth_scale_m)
                    if np.any(valid_depth)
                    else None
                ),
            },
            "files": {key: path.name for key, path in paths.items()},
        }
        paths["metadata"].write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )

        print(f"Saved RGB-D frame: {output_dir}")
        print(f"Color: {color.shape}, depth: {depth_raw.shape} ({depth_raw.dtype})")
        print(f"Depth scale: {depth_scale_m:.8f} m/unit")
        print(f"Valid depth: {metadata['quality']['valid_depth_fraction']:.1%}")
        print("Robot connection/motion: none")
        return 0
    finally:
        if started:
            pipeline.stop()
            print("RealSense pipeline stopped.")


if __name__ == "__main__":
    raise SystemExit(main())
