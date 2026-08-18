from __future__ import annotations

import pytest

from xarm_sdk_control.pick_test_workflow import build_pick_test_route, profile_for


HOME = [215.0, -2.0, 300.0, 178.0, 0.0, 0.0]


def test_ceramic_route_uses_validated_6_mm_compression_and_40_mm_lift() -> None:
    profile = profile_for("ceramic")
    route = build_pick_test_route(
        home_pose_mm_deg=HOME,
        source_point_m=[0.32, -0.14, 0.03],
        profile=profile,
        compression_mm=profile.default_compression_mm,
    )

    contact = next(step for step in route if step.event == "acquire")
    inspection = next(step for step in route if step.event == "hold")
    assert profile.default_compression_mm == 6.0
    assert contact.pose_mm_deg[:3] == pytest.approx([320.0, -140.0, 24.0])
    assert inspection.pose_mm_deg[:3] == pytest.approx([320.0, -140.0, 70.0])
    assert all(step.pose_mm_deg[:2] == pytest.approx([320.0, -140.0]) for step in route[:-1])


def test_bamboo_test_is_also_vertical_only() -> None:
    profile = profile_for("bamboo")
    route = build_pick_test_route(
        home_pose_mm_deg=HOME,
        source_point_m=[0.47, 0.10, -0.03],
        profile=profile,
        compression_mm=profile.default_compression_mm,
    )

    inspection = next(step for step in route if step.event == "hold")
    assert inspection.pose_mm_deg[:3] == pytest.approx([470.0, 100.0, 10.0])


def test_closed_loop_route_replaces_hold_with_verification() -> None:
    route = build_pick_test_route(
        home_pose_mm_deg=HOME,
        source_point_m=[0.32, -0.14, 0.03],
        profile=profile_for("ceramic"),
        compression_mm=16.0,
        inspection_event="verify",
    )

    assert [step.event for step in route].count("verify") == 1
    assert all(step.event != "hold" for step in route)


def test_unknown_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        profile_for("metal")


def test_compression_may_use_new_30_mm_spring_travel() -> None:
    route = build_pick_test_route(
        home_pose_mm_deg=HOME,
        source_point_m=[0.32, -0.14, 0.03],
        profile=profile_for("ceramic"),
        compression_mm=30.0,
    )

    assert next(step for step in route if step.event == "acquire").pose_mm_deg[2] == pytest.approx(0.0)


def test_lotus_and_paper_profiles_present() -> None:
    lotus = profile_for("lotus-leaf")
    paper = profile_for("paper")
    assert lotus.kind == "lotus"
    assert lotus.text_prompt == "lotus leaf."
    assert lotus.default_compression_mm == 2.0
    assert lotus.test_lift_mm == 40.0
    assert paper.kind == "paper"
    assert paper.text_prompt == "white paper."
    assert paper.default_compression_mm == 1.0


def test_resolve_pick_test_settings_defaults_and_overrides() -> None:
    from xarm_sdk_control.pick_test_workflow import resolve_pick_test_settings

    profile = profile_for("ceramic")
    assert resolve_pick_test_settings(profile) == (0.45, 6.0, 40.0)
    assert resolve_pick_test_settings(profile, min_score=0.30, compression_mm=8.0, test_lift_mm=50.0) == (0.30, 8.0, 50.0)
    assert resolve_pick_test_settings(profile_for("lotus-leaf")) == (0.35, 2.0, 40.0)
    assert resolve_pick_test_settings(profile_for("paper")) == (0.35, 1.0, 40.0)
