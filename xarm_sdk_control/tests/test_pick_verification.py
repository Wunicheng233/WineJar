from __future__ import annotations

import pytest

from xarm_sdk_control.pick_verification import policy_for, verify_carried_object


def run(detections):
    return verify_carried_object(
        detections,
        target_kind="ceramic",
        source_point_m=[0.32, -0.14, 0.03],
        expected_object_point_m=[0.32, -0.14, 0.055],
        min_score=0.45,
        minimum_lift_mm=12.0,
        tool_xy_tolerance_mm=35.0,
        height_tolerance_mm=20.0,
        source_xy_tolerance_mm=35.0,
        source_height_tolerance_mm=10.0,
    )


def detection(point, score=0.8):
    return {
        "label": "ceramic disk",
        "score": score,
        "point_base_m": point,
        "touches_image_edge": False,
    }


def test_confirms_lifted_target() -> None:
    result = run([detection([0.321, -0.139, 0.054])])

    assert result.confirmed
    assert result.observed_lift_mm == pytest.approx(24.0)


def test_reports_target_remaining_at_source() -> None:
    result = run([detection([0.321, -0.139, 0.031])])

    assert result.status == "not_confirmed"


def test_missing_detection_is_inconclusive() -> None:
    assert run([]).status == "inconclusive"


def test_ceramic_policy_is_tighter_than_bamboo_policy() -> None:
    assert policy_for("ceramic").tool_xy_tolerance_mm < policy_for("bamboo").tool_xy_tolerance_mm
