from __future__ import annotations

import pytest

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


def test_suction_wait_does_not_replace_task_elapsed_time() -> None:
    values = iter([10.0, 20.0, 20.05])
    timeline = TaskTimeline(clock=lambda: next(values))
    timeline.mark("productive_cycle_started")
    event = timeline.mark("pick.suction_confirmed", suction_wait_seconds=0.05)
    assert event["elapsed_seconds"] == pytest.approx(10.05)
    assert event["suction_wait_seconds"] == 0.05
    assert timeline.to_dict("productive_cycle_started")["productive_cycle_seconds"] == pytest.approx(0.05)


@pytest.mark.parametrize("field", ["elapsed_seconds", "delta_seconds"])
def test_reserved_detail_rejected_without_advancing_timeline(field) -> None:
    values = iter([10.0, 12.0])
    timeline = TaskTimeline(clock=lambda: next(values))
    with pytest.raises(ValueError, match="reserved"):
        timeline.mark("invalid", **{field: 0.05})
    event = timeline.mark("valid")
    assert event["delta_seconds"] == 2.0
    assert len(timeline.events) == 1
