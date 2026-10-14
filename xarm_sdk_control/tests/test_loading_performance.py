from __future__ import annotations

from copy import deepcopy

import pytest

from xarm_sdk_control.loading_performance import render_loading_report, summarize_loading_result


def fixture_result():
    return {
        "status": "success", "run_mode": True,
        "cycles": [{"completed": True, "pick_attempts_used": 2}],
        "timing": {
            "total_seconds": 10.0, "productive_cycle_seconds": 8.0,
            "events": [
                {"name": "productive_cycle_started", "delta_seconds": 2.0, "elapsed_seconds": 2.0},
                {"name": "pick.route_planned", "delta_seconds": 3.0, "elapsed_seconds": 5.0,
                 "planning_seconds": 2.9, "planning_diagnostics": {"ik_seconds": 2.0}},
                {"name": "pick.move_01", "delta_seconds": 4.5, "elapsed_seconds": 9.5,
                 "waited_for_completion": True},
                {"name": "pick.suction_confirmed", "delta_seconds": 0.5, "elapsed_seconds": 0.49},
            ],
        },
    }


def test_old_log_reconstruction_is_nonmutating_and_does_not_double_count():
    result = fixture_result()
    before = deepcopy(result)
    report = summarize_loading_result(result)
    assert result == before
    assert report["productive_cycle_seconds"] == 8.0
    assert sum(report["phase_seconds"].values()) == 10.0
    assert report["phase_seconds"]["planning"] == 3.0
    assert report["phase_seconds"]["motion"] == 4.5
    assert report["timestamp_mismatches"] == [
        {"index": 3, "name": "pick.suction_confirmed", "recorded_seconds": 0.49,
         "reconstructed_seconds": 10.0}
    ]
    assert report["intervals"][1]["planning_diagnostics"] == {"ik_seconds": 2.0}
    assert report["pick_attempts_by_cycle"] == [2]


def test_corrected_log_needs_no_timestamp_reconstruction_warning():
    result = fixture_result()
    result["timing"]["events"][-1].update(elapsed_seconds=10.0, suction_wait_seconds=0.49)
    report = summarize_loading_result(result)
    assert report["timestamp_mismatches"] == []
    assert report["warnings"] == []


def test_failure_before_productive_start_is_preserved_and_not_counted_as_success():
    result = fixture_result()
    result["status"] = "failed"
    result["timing"]["events"][0]["name"] = "camera_session_ready"
    result["completed_cycles"] = result.pop("cycles")
    report = summarize_loading_result(result, "failure.json")
    assert report["productive_cycle_seconds"] is None
    assert report["completed_cycles"] == 1
    markdown = render_loading_report([summarize_loading_result(fixture_result()), report])
    assert "failed 1 条" in markdown and "success 1 条" in markdown
    assert "failure.json" in markdown


@pytest.mark.parametrize("delta", [None, -1.0, float("nan"), float("inf"), True])
def test_invalid_or_missing_delta_is_rejected(delta):
    result = fixture_result()
    result["timing"]["events"][0]["delta_seconds"] = delta
    with pytest.raises(ValueError, match="finite nonnegative"):
        summarize_loading_result(result)


def test_unreconciled_total_is_rejected_instead_of_claiming_repair():
    result = fixture_result()
    result["timing"]["total_seconds"] = 11.0
    with pytest.raises(ValueError, match="does not match"):
        summarize_loading_result(result)


def test_duplicate_start_is_rejected():
    result = fixture_result()
    result["timing"]["events"][-1]["name"] = "productive_cycle_started"
    with pytest.raises(ValueError, match="duplicate"):
        summarize_loading_result(result)


def test_failed_capture_diagnostics_are_retained_without_double_counting():
    result = fixture_result()
    result["status"] = "failed"
    event = result["timing"]["events"][-1]
    event.update(name="scan.capture_failed", elapsed_seconds=10.0,
                 capture_diagnostics={"status": "failed", "total_seconds": .49,
                                      "timing_seconds": {"frame_wait": .4}})
    report = summarize_loading_result(result)
    assert report["phase_seconds"]["capture"] == .5
    assert sum(report["phase_seconds"].values()) == 10.0
    assert report["intervals"][-1]["capture_diagnostics"] == event["capture_diagnostics"]
    markdown = render_loading_report([report])
    assert "采图细分：scan.capture_failed（failed）" in markdown
    assert "| frame_wait | 0.400000 |" in markdown
