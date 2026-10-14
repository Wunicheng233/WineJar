"""Reusable aligned RealSense RGB-D capture session."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


@contextmanager
def _capture_stage(timing: dict[str, float] | None, name: str):
    """Accumulate disjoint stages, including time spent in a failed attempt."""
    started = time.perf_counter() if timing is not None else None
    try:
        yield
    finally:
        if timing is not None:
            timing[name] = timing.get(name, 0.0) + time.perf_counter() - started


def camera_info(device: rs.device) -> dict[str, str]:
    fields = {
        "name": rs.camera_info.name,
        "serial": rs.camera_info.serial_number,
        "firmware": rs.camera_info.firmware_version,
        "usb_type": rs.camera_info.usb_type_descriptor,
    }
    return {
        key: device.get_info(field) if device.supports(field) else "unknown"
        for key, field in fields.items()
    }


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
        depth_m = depth_m[valid]
        near_m, far_m = np.percentile(depth_m, [2, 98])
        if far_m > near_m:
            np.clip(depth_m, near_m, far_m, out=depth_m)
            normalized[valid] = np.round(
                255.0 * (far_m - depth_m) / (far_m - near_m)
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
    available = [device.get_info(rs.camera_info.serial_number) for device in devices]
    raise RuntimeError(f"RealSense serial {serial!r} not found; available: {available}")


class RealSenseCaptureSession:
    """Keep one D435 pipeline warm while saving multiple independent frames."""

    def __init__(
        self,
        serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        warmup_frames: int = 30,
    ) -> None:
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames
        self.context: rs.context | None = None
        self.pipeline: rs.pipeline | None = None
        self.profile: rs.pipeline_profile | None = None
        self.align_to_color: rs.align | None = None
        self.info: dict[str, str] | None = None
        self._frame_condition = threading.Condition()
        self._latest_frame: tuple[np.ndarray, np.ndarray, datetime, dict] | None = None
        self._frame_sequence = 0
        self._recording_thread: threading.Thread | None = None
        self._recording_stop = threading.Event()
        self._recording_error: BaseException | None = None
        self._recording_path: Path | None = None
        self.last_capture_diagnostics: dict | None = None

    def start(self) -> "RealSenseCaptureSession":
        if self.pipeline is not None:
            return self
        self.context = rs.context()
        device = choose_device(self.context, self.serial)
        self.info = camera_info(device)
        print(
            f"Using {self.info['name']}, serial={self.info['serial']}, "
            f"firmware={self.info['firmware']}, USB={self.info['usb_type']}"
        )
        try:
            pipeline, profile = self._start_pipeline(self.context, self.info["serial"])
        except RuntimeError as exc:
            if "VIDIOC_S_FMT" not in str(exc):
                raise
            print("RealSense stream startup failed; resetting the camera and retrying once.")
            device.hardware_reset()
            self.context, device = self._wait_for_device(self.info["serial"])
            self.info = camera_info(device)
            pipeline, profile = self._start_pipeline(self.context, self.info["serial"])
        self.profile = profile
        self.pipeline = pipeline
        self.align_to_color = rs.align(rs.stream.color)
        for _ in range(self.warmup_frames):
            pipeline.wait_for_frames()
        print(f"RealSense session ready after {self.warmup_frames} warmup frames.")
        return self

    def _start_pipeline(self, context: rs.context, serial: str) -> tuple[rs.pipeline, rs.pipeline_profile]:
        pipeline = rs.pipeline(context)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        return pipeline, pipeline.start(config)

    def _wait_for_device(self, serial: str, timeout_seconds: float = 10.0) -> tuple[rs.context, rs.device]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            time.sleep(0.5)
            context = rs.context()
            try:
                return context, choose_device(context, serial)
            except RuntimeError:
                continue
        raise RuntimeError(f"RealSense {serial} did not re-enumerate after hardware reset")

    def capture(self, output_root: Path) -> Path:
        timing: dict[str, float] = {}
        diagnostics = {
            "mode": "direct" if self._recording_thread is None else "recording",
            "status": "running", "read_attempts": 0, "restart_count": 0,
            "timing_seconds": timing,
        }
        self.last_capture_diagnostics = diagnostics
        started = time.perf_counter()
        try:
            if self.pipeline is None or self.profile is None or self.align_to_color is None or self.info is None:
                raise RuntimeError("RealSense session has not been started")
            if self._recording_thread is None:
                try:
                    diagnostics["read_attempts"] += 1
                    frame = self._read_frame_arrays(timing)
                except RuntimeError as exc:
                    diagnostics["first_read_error"] = f"{type(exc).__name__}: {exc}"
                    diagnostics["restart_count"] += 1
                    print(f"RealSense frame capture failed ({exc}); restarting the stream once.")
                    with _capture_stage(timing, "stream_recovery"):
                        self._restart_pipeline()
                    diagnostics["read_attempts"] += 1
                    frame = self._read_frame_arrays(timing)
            else:
                diagnostics["read_attempts"] += 1
                with _capture_stage(timing, "recorded_frame_wait_and_copy"):
                    frame = self._next_recorded_frame()
            output_dir = self._save_frame(output_root, *frame, timing=timing)
            diagnostics["status"] = "success"
            diagnostics["frame_dir"] = str(output_dir)
            return output_dir
        except Exception as exc:
            diagnostics["status"] = "failed"
            diagnostics["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            diagnostics["total_seconds"] = time.perf_counter() - started

    def _restart_pipeline(self) -> None:
        if self._recording_thread is not None:
            raise RuntimeError("Cannot restart RealSense while recording is active")
        serial = self.info["serial"] if self.info is not None else self.serial
        self.stop(raise_recording_error=False)
        deadline = time.monotonic() + 12.0
        last_error: RuntimeError | None = None
        while time.monotonic() < deadline:
            try:
                self.serial = serial
                self.start()
                return
            except RuntimeError as exc:
                last_error = exc
                time.sleep(0.5)
        raise RuntimeError(f"RealSense stream did not recover within 12 seconds: {last_error}")

    def _read_frame_arrays(self, timing: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray, datetime, dict]:
        assert self.pipeline is not None and self.align_to_color is not None
        with _capture_stage(timing, "frame_wait"):
            raw_frames = self.pipeline.wait_for_frames(1000)
        with _capture_stage(timing, "alignment"):
            frames = self.align_to_color.process(raw_frames)
        with _capture_stage(timing, "frame_copy_and_intrinsics"):
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError("Synchronized aligned color/depth frames were not received")
            return (
                np.asanyarray(color_frame.get_data()).copy(),
                np.asanyarray(depth_frame.get_data()).copy(),
                datetime.now().astimezone(),
                intrinsics_to_dict(color_frame.profile.as_video_stream_profile().intrinsics),
            )

    def _next_recorded_frame(self) -> tuple[np.ndarray, np.ndarray, datetime, dict]:
        with self._frame_condition:
            previous_sequence = self._frame_sequence
            ready = self._frame_condition.wait_for(
                lambda: self._frame_sequence > previous_sequence or self._recording_error is not None,
                timeout=2.0,
            )
            if self._recording_error is not None:
                raise RuntimeError("RealSense recording thread failed") from self._recording_error
            if not ready or self._latest_frame is None:
                raise RuntimeError("Timed out waiting for a fresh recorded RGB-D frame")
            color, depth_raw, timestamp, intrinsics = self._latest_frame
            return color.copy(), depth_raw.copy(), timestamp, dict(intrinsics)

    def _save_frame(
        self,
        output_root: Path,
        color: np.ndarray,
        depth_raw: np.ndarray,
        timestamp: datetime,
        intrinsics: dict,
        *,
        timing: dict[str, float] | None = None,
    ) -> Path:
        assert self.profile is not None and self.info is not None

        timestamp_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        output_dir = output_root.expanduser().resolve() / timestamp_id
        output_dir.mkdir(parents=True, exist_ok=False)
        depth_scale_m = self.profile.get_device().first_depth_sensor().get_depth_scale()
        valid_depth = depth_raw > 0
        paths = {
            "color_bgr": output_dir / "color.png",
            "depth_raw": output_dir / "depth_raw.png",
            "depth_preview": output_dir / "depth_preview.png",
            "metadata": output_dir / "frame.json",
        }
        with _capture_stage(timing, "color_png_write"):
            if not cv2.imwrite(str(paths["color_bgr"]), color):
                raise RuntimeError("Failed to save color image")
        with _capture_stage(timing, "depth_png_write"):
            if not cv2.imwrite(str(paths["depth_raw"]), depth_raw):
                raise RuntimeError("Failed to save raw depth image")
        with _capture_stage(timing, "depth_preview_compute"):
            preview = make_depth_preview(depth_raw, depth_scale_m)
        with _capture_stage(timing, "preview_png_write"):
            if not cv2.imwrite(str(paths["depth_preview"]), preview):
                raise RuntimeError("Failed to save depth preview image")
        metadata = {
            "schema_version": 1,
            "frame_id": timestamp_id,
            "captured_at": timestamp.isoformat(timespec="milliseconds"),
            "source": "realsense",
            "camera": self.info,
            "stream": {
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "color_format": "BGR8_PNG",
                "depth_format": "Z16_PNG",
                "depth_aligned_to": "color",
                "depth_scale_m_per_unit": depth_scale_m,
                "coordinate_convention": "+x right, +y down, +z forward",
            },
            "intrinsics": intrinsics,
            "quality": {
                "valid_depth_fraction": float(np.mean(valid_depth)),
                "minimum_valid_depth_m": (
                    float(np.min(depth_raw[valid_depth]) * depth_scale_m) if np.any(valid_depth) else None
                ),
                "maximum_valid_depth_m": (
                    float(np.max(depth_raw[valid_depth]) * depth_scale_m) if np.any(valid_depth) else None
                ),
            },
            "files": {key: path.name for key, path in paths.items()},
        }
        with _capture_stage(timing, "metadata_write"):
            paths["metadata"].write_text(
                json.dumps(metadata, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
        print(f"Saved RGB-D frame: {output_dir}")
        print(f"Color: {color.shape}, depth: {depth_raw.shape} ({depth_raw.dtype})")
        print(f"Depth scale: {depth_scale_m:.8f} m/unit")
        print(f"Valid depth: {metadata['quality']['valid_depth_fraction']:.1%}")
        return output_dir

    def start_recording(self, output_path: Path) -> Path:
        if self.pipeline is None or self.align_to_color is None:
            raise RuntimeError("RealSense session has not been started")
        if self._recording_thread is not None:
            raise RuntimeError("RealSense recording is already active")
        path = output_path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(self.fps),
            (self.width, self.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open global-camera video writer: {path}")
        self._recording_stop.clear()
        self._recording_error = None
        self._recording_path = path

        def record() -> None:
            try:
                while not self._recording_stop.is_set():
                    color, depth_raw, timestamp, intrinsics = self._read_frame_arrays()
                    writer.write(color)
                    with self._frame_condition:
                        self._latest_frame = (color, depth_raw, timestamp, intrinsics)
                        self._frame_sequence += 1
                        self._frame_condition.notify_all()
            except BaseException as exc:
                if not self._recording_stop.is_set():
                    self._recording_error = exc
                with self._frame_condition:
                    self._frame_condition.notify_all()
            finally:
                writer.release()

        self._recording_thread = threading.Thread(
            target=record,
            name="realsense-global-camera-recorder",
            daemon=True,
        )
        self._recording_thread.start()
        print(f"Recording global camera: {path}")
        return path

    def stop_recording(self) -> Path | None:
        thread = self._recording_thread
        if thread is None:
            return self._recording_path
        self._recording_stop.set()
        thread.join(timeout=3.0)
        if thread.is_alive():
            raise RuntimeError("RealSense recording thread did not stop")
        self._recording_thread = None
        path = self._recording_path
        if self._recording_error is not None:
            error = self._recording_error
            self._recording_error = None
            raise RuntimeError("RealSense recording failed") from error
        if path is not None:
            print(f"Saved global-camera video: {path}")
        return path

    def stop(self, raise_recording_error: bool = True) -> None:
        pipeline = self.pipeline
        if pipeline is None:
            return
        recording_error: BaseException | None = None
        try:
            self.stop_recording()
        except BaseException as exc:
            recording_error = exc
        self.pipeline = None
        self.profile = None
        self.align_to_color = None
        try:
            pipeline.stop()
        except RuntimeError as exc:
            print(f"RealSense stream was already stopped ({exc}).")
        print("RealSense session stopped.")
        if recording_error is not None and raise_recording_error:
            raise recording_error

    def __enter__(self) -> "RealSenseCaptureSession":
        return self.start()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.stop()
