#!/usr/bin/env python3
"""Save one near-simultaneous RGB-D frame from every camera in a fixed rig.

The scene must remain static during capture. The script uses the rig's stable
RealSense serial numbers, stores each camera's colour/depth frame separately,
and never connects to or moves the xArm.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.multiview_rig import FixedCameraModel, load_fixed_camera_rig  # noqa: E402
from xarm_sdk_control.realsense_capture import camera_info, make_depth_preview, choose_device  # noqa: E402
from xarm_sdk_control.run_output import new_timestamped_run_dir  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT / "data" / "multiview_neck" / "captures"


@dataclass
class CameraStream:
    camera: FixedCameraModel
    pipeline: rs.pipeline
    profile: rs.pipeline_profile
    align_to_color: rs.align
    info: dict[str, str]


COLOR_OPTIONS = (
    (rs.option.enable_auto_exposure, "auto_exposure"),
    (rs.option.exposure, "exposure"),
    (rs.option.gain, "gain"),
    (rs.option.enable_auto_white_balance, "auto_white_balance"),
    (rs.option.white_balance, "white_balance"),
)


def start_stream(context: rs.context, camera: FixedCameraModel, fps: int) -> CameraStream:
    choose_device(context, camera.serial)
    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(camera.serial)
    width, height = camera.image_size_px
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    return CameraStream(
        camera=camera,
        pipeline=pipeline,
        profile=profile,
        align_to_color=rs.align(rs.stream.color),
        info=camera_info(choose_device(context, camera.serial)),
    )


def read_frame(stream: CameraStream) -> tuple[np.ndarray, np.ndarray, dict]:
    frames = stream.align_to_color.process(stream.pipeline.wait_for_frames(2000))
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        raise RuntimeError(f"{stream.camera.camera_id}: no aligned RGB-D frame received")
    color = np.asanyarray(color_frame.get_data()).copy()
    depth_raw = np.asanyarray(depth_frame.get_data()).copy()
    return color, depth_raw, {
        "host_captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "host_monotonic_ns": time.monotonic_ns(),
        "color_frame_number": int(color_frame.get_frame_number()),
        "depth_frame_number": int(depth_frame.get_frame_number()),
        "color_device_timestamp_ms": float(color_frame.get_timestamp()),
        "depth_device_timestamp_ms": float(depth_frame.get_timestamp()),
    }


def rgb_sensor(stream: CameraStream) -> rs.sensor:
    """Return the RGB sensor for one D435 stream, rejecting ambiguous devices."""
    sensors = [
        sensor
        for sensor in stream.profile.get_device().query_sensors()
        if sensor.supports(rs.camera_info.name) and "RGB" in sensor.get_info(rs.camera_info.name)
    ]
    if len(sensors) != 1:
        raise RuntimeError(f"{stream.camera.camera_id}: expected exactly one RGB sensor, found {len(sensors)}")
    return sensors[0]


def color_controls(stream: CameraStream) -> dict[str, float | None]:
    sensor = rgb_sensor(stream)
    return {
        label: float(sensor.get_option(option)) if sensor.supports(option) else None
        for option, label in COLOR_OPTIONS
    }


def lock_color_controls(stream: CameraStream) -> dict[str, float | None]:
    """Freeze the RGB image controls at their warmed-up values for repeatable differencing."""
    sensor = rgb_sensor(stream)
    values = color_controls(stream)
    if sensor.supports(rs.option.enable_auto_exposure):
        sensor.set_option(rs.option.enable_auto_exposure, 0.0)
    if values["exposure"] is not None and sensor.supports(rs.option.exposure):
        sensor.set_option(rs.option.exposure, values["exposure"])
    if values["gain"] is not None and sensor.supports(rs.option.gain):
        sensor.set_option(rs.option.gain, values["gain"])
    if sensor.supports(rs.option.enable_auto_white_balance):
        sensor.set_option(rs.option.enable_auto_white_balance, 0.0)
    if values["white_balance"] is not None and sensor.supports(rs.option.white_balance):
        sensor.set_option(rs.option.white_balance, values["white_balance"])
    return color_controls(stream)


def set_color_controls(stream: CameraStream, exposure: float, gain: float, white_balance: float) -> dict[str, float | None]:
    """Set explicit RGB controls before warmup so foreground/background match exactly."""
    sensor = rgb_sensor(stream)
    required = (rs.option.enable_auto_exposure, rs.option.exposure, rs.option.gain, rs.option.enable_auto_white_balance, rs.option.white_balance)
    if not all(sensor.supports(option) for option in required):
        raise RuntimeError(f"{stream.camera.camera_id}: RGB sensor lacks one or more requested manual controls")
    sensor.set_option(rs.option.enable_auto_exposure, 0.0)
    sensor.set_option(rs.option.exposure, exposure)
    sensor.set_option(rs.option.gain, gain)
    sensor.set_option(rs.option.enable_auto_white_balance, 0.0)
    sensor.set_option(rs.option.white_balance, white_balance)
    return color_controls(stream)


def save_camera_frame(
    run_dir: Path,
    stream: CameraStream,
    color: np.ndarray,
    depth_raw: np.ndarray,
    timing: dict,
    controls: dict[str, float | None],
) -> dict:
    output_dir = run_dir / stream.camera.camera_id
    output_dir.mkdir(parents=True, exist_ok=False)
    color_path = output_dir / "color.png"
    depth_path = output_dir / "depth_raw.png"
    preview_path = output_dir / "depth_preview.png"
    if not cv2.imwrite(str(color_path), color) or not cv2.imwrite(str(depth_path), depth_raw):
        raise RuntimeError(f"{stream.camera.camera_id}: failed to write captured RGB-D files")
    depth_scale_m = stream.profile.get_device().first_depth_sensor().get_depth_scale()
    cv2.imwrite(str(preview_path), make_depth_preview(depth_raw, depth_scale_m))
    valid = depth_raw > 0
    metadata = {
        "schema_version": 1,
        "camera_id": stream.camera.camera_id,
        "camera": stream.info,
        "stream": {
            "width": stream.camera.image_size_px[0],
            "height": stream.camera.image_size_px[1],
            "color_format": "BGR8_PNG",
            "depth_format": "Z16_PNG",
            "depth_aligned_to": "color",
            "depth_scale_m_per_unit": depth_scale_m,
            "coordinate_convention": "+x right, +y down, +z forward",
        },
        "timing": timing,
        "color_controls": controls,
        "quality": {
            "valid_depth_fraction": float(np.mean(valid)),
            "minimum_valid_depth_m": float(np.min(depth_raw[valid]) * depth_scale_m) if np.any(valid) else None,
            "maximum_valid_depth_m": float(np.max(depth_raw[valid]) * depth_scale_m) if np.any(valid) else None,
        },
        "files": {
            "color": color_path.name,
            "depth_raw": depth_path.name,
            "depth_preview": preview_path.name,
        },
    }
    (output_dir / "frame.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig", type=Path, required=True, help="Validated fixed-camera rig manifest.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument(
        "--lock-color",
        action="store_true",
        help="After warmup, freeze RGB exposure/gain/white balance for repeatable background differencing.",
    )
    parser.add_argument("--color-exposure", type=float, help="Explicit RGB exposure; requires gain and white balance too.")
    parser.add_argument("--color-gain", type=float, help="Explicit RGB gain; requires exposure and white balance too.")
    parser.add_argument("--color-white-balance", type=float, help="Explicit RGB white balance; requires exposure and gain too.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.fps <= 0 or args.warmup < 0:
        parser.error("--fps must be positive and --warmup must be non-negative")
    explicit_controls = (args.color_exposure, args.color_gain, args.color_white_balance)
    if any(value is not None for value in explicit_controls) and not all(value is not None for value in explicit_controls):
        parser.error("--color-exposure, --color-gain and --color-white-balance must be provided together")
    if args.lock_color and all(value is not None for value in explicit_controls):
        parser.error("Use either --lock-color or explicit --color-* controls, not both")
    try:
        rig = load_fixed_camera_rig(args.rig)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: invalid rig: {exc}")
        return 2
    run_dir = new_timestamped_run_dir(args.output_root.expanduser().resolve())
    context = rs.context()
    streams: list[CameraStream] = []
    try:
        for camera in rig.cameras.values():
            streams.append(start_stream(context, camera, args.fps))
            print(f"Started {camera.camera_id}: serial={camera.serial}, size={camera.image_size_px}")
        controls = {}
        if all(value is not None for value in explicit_controls):
            for stream in streams:
                controls[stream.camera.camera_id] = set_color_controls(
                    stream,
                    args.color_exposure,
                    args.color_gain,
                    args.color_white_balance,
                )
        for _ in range(args.warmup):
            for stream in streams:
                read_frame(stream)
        for stream in streams:
            if stream.camera.camera_id not in controls:
                controls[stream.camera.camera_id] = (
                    lock_color_controls(stream) if args.lock_color else color_controls(stream)
                )
            print(f"{stream.camera.camera_id} RGB controls: {controls[stream.camera.camera_id]}")
        records = {}
        for stream in streams:
            color, depth_raw, timing = read_frame(stream)
            records[stream.camera.camera_id] = save_camera_frame(
                run_dir,
                stream,
                color,
                depth_raw,
                timing,
                controls[stream.camera.camera_id],
            )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2
    finally:
        for stream in streams:
            stream.pipeline.stop()
    host_times = [record["timing"]["host_monotonic_ns"] for record in records.values()] if "records" in locals() else []
    capture = {
        "schema_version": 1,
        "type": "fixed_multiview_rgbd_capture",
        "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "rig_path": str(args.rig.expanduser().resolve()),
        "reference_camera_id": rig.reference_camera_id,
        "camera_ids": list(rig.cameras),
        "host_capture_skew_ms": (max(host_times) - min(host_times)) / 1_000_000.0 if host_times else None,
        "cameras": records,
    }
    (run_dir / "capture.json").write_text(json.dumps(capture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved three-camera RGB-D capture: {run_dir}")
    print(f"Host capture skew: {capture['host_capture_skew_ms']:.1f} ms; scene must remain static.")
    print("Robot connection/motion: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
