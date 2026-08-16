from __future__ import annotations

from datetime import datetime

import numpy as np

from xarm_sdk_control.realsense_capture import RealSenseCaptureSession


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

    def read():
        calls["read"] += 1
        if calls["read"] == 1:
            raise RuntimeError("device disconnected")
        return frame

    def restart():
        calls["restart"] += 1

    monkeypatch.setattr(session, "_read_frame_arrays", read)
    monkeypatch.setattr(session, "_restart_pipeline", restart)
    monkeypatch.setattr(session, "_save_frame", lambda *_args: tmp_path / "frame")

    assert session.capture(tmp_path) == tmp_path / "frame"
    assert calls == {"read": 2, "restart": 1}
