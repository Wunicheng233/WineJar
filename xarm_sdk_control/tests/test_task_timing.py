from __future__ import annotations

from xarm_sdk_control.task_timing import TaskTimeline


def test_timeline_records_delta_and_productive_cycle() -> None:
    values = iter([10.0, 11.0, 13.5, 16.0])
    timeline = TaskTimeline(clock=lambda: next(values))
    timeline.mark("reset_complete")
    timeline.mark("scan_complete")
    timeline.mark("task_complete")

    result = timeline.to_dict(productive_start="reset_complete")
    assert result["total_seconds"] == 6.0
    assert result["productive_cycle_seconds"] == 5.0
    assert result["events"][1]["delta_seconds"] == 2.5
