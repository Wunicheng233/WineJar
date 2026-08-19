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


def test_execute_pick_test_route_detects_suction_lost_after_lift() -> None:
    import pytest

    from xarm_sdk_control.pick_test_workflow import execute_pick_test_route
    from xarm_sdk_control.suction_sensor import SuctionNotConfirmedError, SuctionSensor, SuctionSensorConfig

    class CounterArm:
        def __init__(self, success_reads: int) -> None:
            self.success_reads = success_reads
            self.reads = 0

        def get_cgpio_digital(self, channel):
            self.reads += 1
            return 0, 0 if self.reads <= self.success_reads else 1

        def set_position(self, *pose, **kwargs):
            return 0

    class Clock:
        def __init__(self):
            self.now = 0.0

        def __call__(self):
            return self.now

        def advance(self, seconds):
            self.now += seconds

    class Vacuum:
        def __init__(self):
            self.actions = []

        def prepare_for_pick(self):
            self.actions.append("prepare")

        def acquire(self, settle_seconds):
            self.actions.append(("acquire", settle_seconds))

        def release(self, settle_seconds):
            self.actions.append(("release", settle_seconds))

    clock = Clock()
    sensor = SuctionSensor(
        CounterArm(8),
        SuctionSensorConfig(input_channel=0, success_value=0, debounce_seconds=0.05, timeout_seconds=2.0),
        clock=clock,
        sleep=lambda s: clock.advance(s),
    )
    full = build_pick_test_route(
        home_pose_mm_deg=[215.0, -2.0, 300.0, 178.0, 0.0, 0.0],
        source_point_m=[0.32, -0.14, 0.03],
        profile=profile_for("ceramic"),
        compression_mm=6.0,
    )
    route = [full[2], full[3]]  # compress (acquire) then lift (loaded)
    args = type("Args", (), {
        "precision_speed": 15.0, "transit_speed": 80.0, "acc": 150.0,
        "vacuum_settle_seconds": 0.9, "release_settle_seconds": 0.5, "inspection_hold_seconds": 2.0,
    })()

    with pytest.raises(SuctionNotConfirmedError):
        execute_pick_test_route(
            CounterArm(8), Vacuum(), route, args, {"value": False}, suction_sensor=sensor
        )
