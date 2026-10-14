from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
import json

import cv2
import numpy as np
import pytest

from xarm_sdk_control.realsense_capture import RealSenseCaptureSession, make_depth_preview


class AlreadyStoppedPipeline:
    def stop(self) -> None:
        raise RuntimeError("stop() cannot be called before start()")


def test_stop_is_idempotent_when_sdk_pipeline_already_stopped() -> None:
    session = RealSenseCaptureSession(warmup_frames=0)
    session.pipeline = AlreadyStoppedPipeline()
    session.profile = object()
    session.align_to_color = object()

    session.stop()
    session.stop()

    assert session.pipeline is None
    assert session.profile is None
    assert session.align_to_color is None


def test_capture_restarts_once_after_stream_error(monkeypatch, tmp_path) -> None:
    session = RealSenseCaptureSession(warmup_frames=0)
    session.pipeline = object()
    session.profile = object()
    session.align_to_color = object()
    session.info = {"serial": "test"}
    calls = {"read": 0, "restart": 0}
    frame = (
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.ones((2, 2), dtype=np.uint16),
        datetime.now().astimezone(),
        {},
    )

    def read(_timing=None):
        calls["read"] += 1
        if calls["read"] == 1:
            raise RuntimeError("device disconnected")
        return frame

    def restart():
        calls["restart"] += 1

    monkeypatch.setattr(session, "_read_frame_arrays", read)
    monkeypatch.setattr(session, "_restart_pipeline", restart)
    monkeypatch.setattr(session, "_save_frame", lambda *_args, **_kwargs: tmp_path / "frame")

    assert session.capture(tmp_path) == tmp_path / "frame"
    assert calls == {"read": 2, "restart": 1}
    diagnostics = session.last_capture_diagnostics
    assert diagnostics["read_attempts"] == 2
    assert diagnostics["restart_count"] == 1
    assert diagnostics["status"] == "success"
    assert "device disconnected" in diagnostics["first_read_error"]
    assert diagnostics["total_seconds"] >= diagnostics["timing_seconds"]["stream_recovery"] >= 0


def fake_session():
    session = RealSenseCaptureSession(width=4, height=3, warmup_frames=0)
    session.pipeline = object()
    sensor = SimpleNamespace(get_depth_scale=lambda: 0.001)
    device = SimpleNamespace(first_depth_sensor=lambda: sensor)
    session.profile = SimpleNamespace(get_device=lambda: device)
    session.align_to_color = object()
    session.info = {"serial": "test"}
    return session


def fake_frame():
    return (np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
            np.arange(12, dtype=np.uint16).reshape(3, 4) * 100,
            datetime.now().astimezone(), {})


def test_capture_saves_lossless_images_and_disjoint_diagnostics(monkeypatch, tmp_path):
    session = fake_session()
    frame = fake_frame()
    monkeypatch.setattr(session, "_read_frame_arrays", lambda _timing: frame)
    path = session.capture(tmp_path)
    assert np.array_equal(cv2.imread(str(path / "color.png"), -1), frame[0])
    assert np.array_equal(cv2.imread(str(path / "depth_raw.png"), -1), frame[1])
    assert np.array_equal(cv2.imread(str(path / "depth_preview.png"), -1), make_depth_preview(frame[1], .001))
    assert json.loads((path / "frame.json").read_text())["quality"]["valid_depth_fraction"] == 11 / 12
    d = session.last_capture_diagnostics
    assert d["status"] == "success" and d["restart_count"] == 0
    assert d["total_seconds"] >= sum(d["timing_seconds"].values()) >= 0
    assert set(d["timing_seconds"]) == {"color_png_write", "depth_png_write", "depth_preview_compute", "preview_png_write", "metadata_write"}


def test_failed_retry_is_not_retried_again_and_preserves_error(monkeypatch, tmp_path):
    session = fake_session()
    error = RuntimeError("no synchronized frame")
    restarts = []
    def read(_timing):
        raise error
    monkeypatch.setattr(session, "_read_frame_arrays", read)
    monkeypatch.setattr(session, "_restart_pipeline", lambda: restarts.append(True))
    with pytest.raises(RuntimeError) as caught:
        session.capture(tmp_path)
    assert caught.value is error and restarts == [True]
    assert session.last_capture_diagnostics["read_attempts"] == 2
    assert session.last_capture_diagnostics["status"] == "failed"
    assert "no synchronized frame" in session.last_capture_diagnostics["error"]


def test_recording_path_uses_fresh_frame_provider_without_stream_restart(monkeypatch, tmp_path):
    session = fake_session()
    session._recording_thread = object()
    calls = []
    def recorded():
        calls.append(True)
        return fake_frame()
    monkeypatch.setattr(session, "_next_recorded_frame", recorded)
    monkeypatch.setattr(session, "_read_frame_arrays", lambda *_: pytest.fail("direct read while recording"))
    session.capture(tmp_path)
    assert calls == [True]
    d = session.last_capture_diagnostics
    assert d["mode"] == "recording" and d["restart_count"] == 0
    assert "recorded_frame_wait_and_copy" in d["timing_seconds"]


@pytest.mark.parametrize("filename", ["color.png", "depth_raw.png", "depth_preview.png"])
def test_image_write_failure_is_reported_and_does_not_restart_camera(monkeypatch, tmp_path, filename):
    session = fake_session()
    monkeypatch.setattr(session, "_read_frame_arrays", lambda _: fake_frame())
    monkeypatch.setattr(cv2, "imwrite", lambda path, _: not path.endswith(filename))
    with pytest.raises(RuntimeError, match="Failed to save"):
        session.capture(tmp_path)
    assert session.last_capture_diagnostics["status"] == "failed"
    assert session.last_capture_diagnostics["restart_count"] == 0


def test_read_stages_preserve_frame_data_and_intrinsics(monkeypatch):
    from xarm_sdk_control import realsense_capture as module
    session = fake_session()
    color, depth, _, _ = fake_frame()
    stream = SimpleNamespace(as_video_stream_profile=lambda: SimpleNamespace(intrinsics=object()))
    frames = SimpleNamespace(
        get_color_frame=lambda: SimpleNamespace(get_data=lambda: color, profile=stream),
        get_depth_frame=lambda: SimpleNamespace(get_data=lambda: depth),
    )
    waits = []
    def wait(timeout):
        waits.append(timeout)
        return frames
    session.pipeline = SimpleNamespace(wait_for_frames=wait)
    session.align_to_color = SimpleNamespace(process=lambda frames: frames)
    monkeypatch.setattr(module, "intrinsics_to_dict", lambda _: {"fx": 123})
    timing = {}
    result = session._read_frame_arrays(timing)
    assert waits == [1000] and result[3] == {"fx": 123}
    assert np.array_equal(result[0], color) and not np.shares_memory(result[0], color)
    assert np.array_equal(result[1], depth) and not np.shares_memory(result[1], depth)
    assert set(timing) == {"frame_wait", "alignment", "frame_copy_and_intrinsics"}


def test_frame_timeout_and_failed_recovery_both_record_time(monkeypatch, tmp_path):
    session = fake_session()
    def wait(_timeout):
        raise RuntimeError("frame timeout")
    def restart():
        raise RuntimeError("restart failed")
    session.pipeline = SimpleNamespace(wait_for_frames=wait)
    monkeypatch.setattr(session, "_restart_pipeline", restart)
    with pytest.raises(RuntimeError, match="restart failed"):
        session.capture(tmp_path)
    d = session.last_capture_diagnostics
    assert d["read_attempts"] == 1 and d["restart_count"] == 1
    assert "frame timeout" in d["first_read_error"]
    assert "restart failed" in d["error"]
    assert set(d["timing_seconds"]) == {"frame_wait", "stream_recovery"}
    assert d["total_seconds"] >= sum(d["timing_seconds"].values()) > 0


def test_unstarted_capture_replaces_previous_diagnostics(tmp_path):
    session = RealSenseCaptureSession()
    session.last_capture_diagnostics = {"status": "success", "frame_dir": "old"}
    with pytest.raises(RuntimeError, match="not been started"):
        session.capture(tmp_path)
    assert session.last_capture_diagnostics["status"] == "failed"
    assert "frame_dir" not in session.last_capture_diagnostics


def legacy_preview(raw, scale):
    depth = raw.astype(np.float32) * scale
    valid = depth > 0
    normalized = np.zeros(raw.shape, np.uint8)
    if np.any(valid):
        near, far = np.percentile(depth[valid], [2, 98])
        if far > near:
            clipped = np.clip(depth, near, far)
            normalized[valid] = np.round(255.0 * (far - clipped[valid]) / (far - near)).astype(np.uint8)
    preview = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview


@pytest.mark.parametrize("scale", [0.001, 0.0001, 0.0, -0.001])
def test_preview_matches_legacy_pixels_and_preserves_input(scale):
    rng = np.random.default_rng(20260919)
    cases = [np.zeros((48, 64), np.uint16), np.full((48, 64), 500, np.uint16),
             np.array([[0, 1, 2, 65535]], np.uint16)]
    for probability in [0, .2, .9]:
        raw = rng.integers(0, 65536, (48, 64), dtype=np.uint16)
        raw[rng.random(raw.shape) < probability] = 0
        cases.append(raw)
    for raw in cases:
        before = raw.copy()
        assert np.array_equal(make_depth_preview(raw, scale), legacy_preview(raw, scale))
        assert np.array_equal(raw, before)
