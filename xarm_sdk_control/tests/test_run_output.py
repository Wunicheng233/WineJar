from __future__ import annotations

import json

from xarm_sdk_control.run_output import write_failure_report


def test_failure_report_merges_extra_keys(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = write_failure_report(
        run_dir,
        RuntimeError("boom"),
        extra={"pick_verification_failed": {"leaf": 1, "attempts_used": 3}},
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["error_type"] == "RuntimeError"
    assert report["pick_verification_failed"]["leaf"] == 1


def test_failure_report_without_extra_matches_baseline_shape(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = write_failure_report(run_dir, RuntimeError("boom"), holding=True)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert set(report) == {
        "status",
        "error_type",
        "error_message",
        "traceback",
        "holding_leaf",
        "robot_after_stop",
        "timing",
        "completed_cycles",
    }
