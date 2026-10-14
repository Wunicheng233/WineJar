from __future__ import annotations

import pytest

from conftest import (
    RecordingArm,
    RecordingTimeline,
    RecordingVacuum,
    TimeoutAtTargetArm,
    TimeoutAwayFromTargetArm,
    load_script_module,
    motion_args,
)

from xarm_sdk_control import cross_place_motion as MOTION


MODULE = load_script_module(
    "visual_cross_place",
    "xarm_sdk_control/scripts/19_visual_cross_place_two_leaves.py",
)


def test_robot_status_snapshot_records_post_stop_diagnostics() -> None:
    snapshot = MOTION.robot_status_snapshot(RecordingArm())
    assert snapshot["state"] == (0, 4)
    assert snapshot["error_warn"] == (0, [0, 0])
    assert snapshot["tcp_pose_mm_deg"][1][:3] == [300.0, 100.0, 200.0]


def test_first_leaf_interpolates_yaw_during_safe_height_transport() -> None:
    route = MOTION.build_cycle_route(
        home_pose=[219, 0, 300, 178, 0, 0],
        leaf_point_m=[0.287, 0.150, -0.044],
        ceramic_point_m=[0.400, -0.119, 0.049],
        place_yaw_offset_deg=90,
        transport_clearance_mm=200,
        approach_clearance_mm=35,
        compression_mm=15,
        release_clearance_mm=5,
        layer_height_mm=0,
    )
    poses = {label: pose for label, pose, _motion, _event in route}
    labels = [item[0] for item in route]
    assert "rotate leaf at safe height" not in labels
    assert poses["move directly above source leaf while interpolating pickup yaw"][2] == -9.0
    assert poses["lift leaf vertically with pickup yaw fixed"][2] == 249.0
    assert poses["lift leaf vertically with pickup yaw fixed"][5] == 0.0
    assert poses["move above ceramic while interpolating placement yaw"][2] == 84.0
    assert poses["move above ceramic while interpolating placement yaw"][5] == 90.0
    assert poses["lower leaf to release height"][5] == 90.0
    assert poses["return home directly after safe release retreat"][5] == 0.0


def test_second_leaf_keeps_original_yaw_and_adds_layer_height() -> None:
    route = MOTION.build_cycle_route(
        home_pose=[219, 0, 300, 178, 0, 0],
        leaf_point_m=[0.287, 0.150, -0.045],
        ceramic_point_m=[0.400, -0.119, 0.049],
        place_yaw_offset_deg=0,
        transport_clearance_mm=200,
        approach_clearance_mm=35,
        compression_mm=15,
        release_clearance_mm=5,
        layer_height_mm=3,
    )
    labels = [item[0] for item in route]
    poses = {label: pose for label, pose, _motion, _event in route}
    assert "rotate leaf at safe height" not in labels
    assert poses["lower leaf to release height"][2] == 57.0
    assert poses["lower leaf to release height"][5] == 0.0


def test_default_profile_compensates_yaw_and_shortens_free_space_motion(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.sys, "argv", ["19_visual_cross_place_two_leaves.py"])
    args = MODULE.parse_args()
    assert args.first_place_yaw_offset_deg == 45.0
    assert args.second_place_yaw_offset_deg == -60.0
    assert args.place_offset_y_mm == -10.0
    assert args.transport_clearance_mm == 100.0
    assert args.transit_speed == 300.0
    assert args.loaded_transit_speed == 220.0
    assert args.vertical_speed == 200.0
    assert args.loaded_vertical_speed == 120.0
    assert args.precision_speed == 60.0
    assert args.contact_speed == 20.0
    assert args.acc == 500.0
    assert args.loaded_acc == 250.0
    assert args.precision_acc == 200.0
    assert args.release_clearance_mm == 0.0
    assert args.vacuum_settle_seconds == 0.9
    assert args.release_settle_seconds == 0.3
    assert args.blend_radius_mm == 10.0
    assert args.joint_velocity_utilization == 0.40
    assert args.joint_acceleration_utilization == 0.40
    assert args.ik_translation_step_mm == 10.0
    assert args.ik_rotation_step_deg == 2.0
    assert args.record_global_camera is False
    assert args.recording_fps == 15
    assert args.second_scan_min_score == 0.10
    assert args.verify_pick is False
    assert args.max_pick_attempts == 3
    assert args.simulate_first_pick_failure is False


def test_place_offset_moves_only_the_task_target() -> None:
    visual_center = [0.350, -0.120, 0.032]
    placement_target = MOTION.offset_point_mm(visual_center, y_mm=-10.0)
    assert visual_center == [0.350, -0.120, 0.032]
    assert placement_target == [0.350, -0.130, 0.032]


def test_pick_and_place_targets_must_not_reuse_the_same_detection() -> None:
    try:
        MOTION.require_distinct_pick_and_place_targets(
            [0.475, 0.098, -0.005],
            [0.475, 0.088, -0.005],
        )
    except RuntimeError as exc:
        assert "same object" in str(exc)
    else:
        raise AssertionError("Nearly identical pick and place targets must be rejected")


def test_two_cycles_share_ceramic_retract_pose_without_intermediate_home() -> None:
    home = [219, 0, 300, 178, 0, 0]
    leaf = [0.287, 0.150, -0.044]
    ceramic = [0.400, -0.119, 0.049]
    first_route = MOTION.build_cycle_route(
        home_pose=home,
        leaf_point_m=leaf,
        ceramic_point_m=ceramic,
        place_yaw_offset_deg=45,
        transport_clearance_mm=200,
        approach_clearance_mm=35,
        compression_mm=20,
        release_clearance_mm=5,
        layer_height_mm=0,
        return_home=False,
    )
    first_labels = [item[0] for item in first_route]
    first_waiting_pose = first_route[-1][1]
    assert "return home after final placement" not in first_labels
    assert first_waiting_pose[:3] == [400.0, -119.0, 249.0]
    assert first_waiting_pose[5] == 45.0

    second_route = MOTION.build_cycle_route(
        home_pose=home,
        leaf_point_m=leaf,
        ceramic_point_m=ceramic,
        place_yaw_offset_deg=-45,
        transport_clearance_mm=200,
        approach_clearance_mm=35,
        compression_mm=20,
        release_clearance_mm=5,
        layer_height_mm=3,
        start_pose=first_waiting_pose,
        return_home=True,
    )
    second_labels = [item[0] for item in second_route]
    second_poses = {label: pose for label, pose, _motion, _event in second_route}
    assert second_labels[0] == "move directly above source leaf while interpolating pickup yaw"
    assert "move to camera-clear waiting column" not in second_labels
    assert second_poses["lift leaf vertically with pickup yaw fixed"][5] == 0.0
    assert second_poses["move above ceramic while interpolating placement yaw"][5] == -45.0
    assert "retract vertically above ceramic" not in second_labels
    assert second_labels[-1] == "return home directly after safe release retreat"


def test_execute_route_blends_only_approach_into_exact_contact() -> None:
    arm = RecordingArm()
    vacuum = RecordingVacuum()
    timeline = RecordingTimeline()
    args = motion_args()
    route = [
        ("descend near source", [400, 100, 20, 180, 0, 0], "precision", None),
        ("compress at source", [400, 100, -20, 180, 0, 0], "contact", "acquire"),
    ]

    MOTION.execute_route(
        arm,
        vacuum,
        route,
        args,
        {"value": False},
        timeline,
        "leaf_1",
    )

    assert arm.commands[0][1]["wait"] is False
    assert arm.commands[0][1]["radius"] == 10.0
    assert arm.commands[1][1]["wait"] is True
    assert arm.commands[1][1]["radius"] == -1
    assert vacuum.actions == ["prepare", ("acquire", 0.9)]


def test_direct_diagonal_approach_can_blend_into_vertical_contact() -> None:
    arm = RecordingArm()
    args = motion_args()
    route = [
        ("direct approach", [480, 80, 10, 178, 0, 0], "empty_transit", None),
        ("contact", [480, 80, -30, 178, 0, 0], "contact", "acquire"),
    ]

    MOTION.execute_route(
        arm,
        RecordingVacuum(),
        route,
        args,
        {"value": False},
        RecordingTimeline(),
        "leaf_1",
    )

    assert arm.commands[0][1]["wait"] is False
    assert arm.commands[0][1]["radius"] == 10.0
    assert arm.commands[1][1]["wait"] is True


def test_loaded_lift_rotation_and_transfer_are_exact_stop_segments() -> None:
    route = [
        ("stabilize suction", [480, 100, 10, 178, 0, 0], "loaded_precision", None),
        ("rotate while lifting", [480, 100, 100, 178, 0, 45], "loaded_vertical", None),
        ("carry to ceramic", [320, -145, 60, 178, 0, 45], "loaded_transit", None),
        ("place", [320, -145, 30, 178, 0, 45], "contact", "release"),
    ]

    assert MOTION.route_blend_radii(route, 10.0) == [-1.0, -1.0, -1.0, -1.0]


def test_execute_route_uses_joint_planner_speed_and_acceleration() -> None:
    arm = RecordingArm()
    route = [("planned move", [300, 100, 50, 178, 0, 0], "empty_transit", None)]
    plan = MOTION.SegmentSpeedPlan(
        label="planned move",
        sample_count=3,
        translation_distance_mm=100.0,
        rotation_distance_deg=0.0,
        requested_speed_mm_s=300.0,
        planned_speed_mm_s=123.0,
        requested_acceleration_mm_s2=500.0,
        planned_acceleration_mm_s2=234.0,
        predicted_peak_joint_speed_deg_s=100.0,
        predicted_peak_joint_acceleration_deg_s2=500.0,
        limiting_velocity_joint=3,
        limiting_acceleration_joint=3,
        end_joint_angles_deg=[0.0] * 6,
    )

    MOTION.execute_route(
        arm,
        RecordingVacuum(),
        route,
        motion_args(),
        {"value": False},
        RecordingTimeline(),
        "leaf_1",
        [plan],
    )

    assert arm.commands[0][1]["speed"] == 123.0
    assert arm.commands[0][1]["mvacc"] == 234.0


def test_motion_profile_rejects_values_above_validated_task_limits() -> None:
    args = motion_args(transit_speed=401.0)
    try:
        MOTION.validate_motion_profile(args)
    except ValueError as exc:
        assert "empty transit speed" in str(exc)
    else:
        raise AssertionError("Unsafe task speed override must be rejected")


def test_negative_settle_time_is_rejected_by_main_validation(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        ["19_visual_cross_place_two_leaves.py", "--vacuum-settle-seconds", "-0.1"],
    )
    try:
        MODULE.main()
    except ValueError as exc:
        assert "settle times" in str(exc)
    else:
        raise AssertionError("Negative settle time must be rejected")


def _reject_argv(monkeypatch, argv: list[str], expected: str) -> None:
    monkeypatch.setattr(MODULE.sys, "argv", ["19_visual_cross_place_two_leaves.py", *argv])
    try:
        MODULE.main()
    except ValueError as exc:
        assert expected in str(exc)
    else:
        raise AssertionError(f"{expected!r} must be rejected")


def test_verify_pick_requires_run(monkeypatch) -> None:
    _reject_argv(monkeypatch, ["--verify-pick"], "--verify-pick requires --run")


def test_confirm_no_pressure_sensor_flag_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        ["19_visual_cross_place_two_leaves.py", "--confirm-no-pressure-sensor"],
    )
    try:
        MODULE.parse_args()
    except SystemExit:
        pass
    else:
        raise AssertionError("--confirm-no-pressure-sensor must be rejected")


def test_max_pick_attempts_must_stay_within_three(monkeypatch) -> None:
    _reject_argv(monkeypatch, ["--max-pick-attempts", "4"], "--max-pick-attempts must be within 1~3")


def test_simulated_failure_needs_at_least_two_attempts(monkeypatch) -> None:
    _reject_argv(
        monkeypatch,
        [
            "--run",
            "--simulate-first-pick-failure",
            "--max-pick-attempts",
            "1",
            "--min-x-mm", "-1000", "--max-x-mm", "1000",
            "--min-y-mm", "-1000", "--max-y-mm", "1000",
            "--min-z-mm", "-1000", "--max-z-mm", "1000",
        ],
        "requires --max-pick-attempts 2 or greater",
    )


def test_loaded_yaw_starts_only_after_vertical_lift_during_horizontal_carry() -> None:
    route = MOTION.build_cycle_route(
        home_pose=[219, 0, 300, 178, 0, 0],
        leaf_point_m=[0.480, 0.090, -0.030],
        ceramic_point_m=[0.350, -0.150, 0.032],
        place_yaw_offset_deg=45,
        transport_clearance_mm=140,
        approach_clearance_mm=25,
        compression_mm=20,
        release_clearance_mm=0,
        layer_height_mm=0,
        return_home=False,
    )
    poses = {label: pose for label, pose, _motion, _event in route}
    motions = {label: motion for label, _pose, motion, _event in route}

    assert poses["lift leaf slowly until suction is stable"][5] == 0.0
    assert poses["lift leaf vertically with pickup yaw fixed"][5] == 0.0
    assert poses["move above ceramic while interpolating placement yaw"][5] == 45.0
    assert motions["lift leaf vertically with pickup yaw fixed"] == "loaded_vertical"
    assert motions["move above ceramic while interpolating placement yaw"] == "loaded_transit"
    assert poses["lower leaf to release height"][2] == 32.0


def test_feedback_timeout_is_accepted_only_when_target_is_verified() -> None:
    MOTION.require_motion_completed(
        TimeoutAtTargetArm(),
        "contact",
        3,
        [300.0, 150.0, -60.0, 180.0, 0.0, 90.0],
    )


def test_feedback_timeout_away_from_target_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(MOTION.time, "sleep", lambda _seconds: None)
    try:
        MOTION.require_motion_completed(
            TimeoutAwayFromTargetArm(),
            "contact",
            3,
            [300.0, 150.0, -60.0, 180.0, 0.0, 90.0],
        )
    except RuntimeError as exc:
        assert "could not be verified" in str(exc)
    else:
        raise AssertionError("A timed-out motion away from target must be rejected")


def test_verify_step_inserts_stationary_hold_at_transport_height() -> None:
    route = MOTION.build_cycle_route(
        home_pose=[215.194, -2.409, 300.0, 178.467, 0.581, -0.014],
        leaf_point_m=[0.50, 0.06, -0.015],
        ceramic_point_m=[0.30, -0.16, 0.018],
        place_yaw_offset_deg=45.0,
        transport_clearance_mm=100,
        approach_clearance_mm=25,
        compression_mm=13,
        release_clearance_mm=0.0,
        layer_height_mm=0.0,
        inspection_event="verify",
    )

    verify_indices = [index for index, item in enumerate(route) if item[3] == "verify"]
    assert len(verify_indices) == 1
    index = verify_indices[0]
    assert route[index][0] == "hold leaf above source for visual verification"
    assert route[index][2] == "loaded_vertical"
    assert route[index - 1][0] == "lift leaf vertically with pickup yaw fixed"
    assert route[index][1] == route[index - 1][1]
    assert route[index][1][2] == 118.0


def test_invalid_inspection_event_is_rejected() -> None:
    try:
        MOTION.build_cycle_route(
            home_pose=[215.194, -2.409, 300.0, 178.467, 0.581, -0.014],
            leaf_point_m=[0.50, 0.06, -0.015],
            ceramic_point_m=[0.30, -0.16, 0.018],
            place_yaw_offset_deg=45.0,
            transport_clearance_mm=100,
            approach_clearance_mm=25,
            compression_mm=13,
            release_clearance_mm=0.0,
            layer_height_mm=0.0,
            inspection_event="hold",
        )
    except ValueError as exc:
        assert "inspection event" in str(exc)
    else:
        raise AssertionError("Unsupported inspection events must be rejected")


def test_recovery_route_vents_at_source_and_retreats() -> None:
    route = MOTION.build_pick_recovery_route(
        source_point_m=[0.50, 0.06, -0.015],
        transport_z_mm=118.0,
        release_clearance_mm=0.0,
        home_pose=[215.194, -2.409, 300.0, 178.467, 0.581, -0.014],
        start_pose=[500.0, 60.0, 118.0, 178.467, 0.581, -0.014],
    )

    assert len(route) == 3
    assert route[0][0] == "descend loaded to source release height"
    assert route[0][1][2] == -15.0
    assert route[1][3] == "release"
    assert route[1][1] == route[0][1]
    assert route[2][0] == "retreat vertically above the source stack"
    assert route[2][1][2] == 118.0
    assert MOTION.route_blend_radii(route, 10.0) == [-1.0, -1.0, -1.0]


def test_recovery_route_rejects_transport_above_camera_corridor() -> None:
    try:
        MOTION.build_pick_recovery_route(
            source_point_m=[0.50, 0.06, -0.015],
            transport_z_mm=300.0,
            release_clearance_mm=0.0,
            home_pose=[215.194, -2.409, 300.0, 178.467, 0.581, -0.014],
            start_pose=[500.0, 60.0, 118.0, 178.467, 0.581, -0.014],
        )
    except RuntimeError as exc:
        assert "camera-safe corridor" in str(exc)
    else:
        raise AssertionError("Recovery transport above the camera corridor must be rejected")


def test_transport_height_uses_max_of_leaf_and_ceramic() -> None:
    assert MOTION.transport_height_mm([0.50, 0.06, -0.015], [0.30, -0.16, 0.018], 100.0) == 118.0
    assert MOTION.transport_height_mm([0.50, 0.06, 0.05], [0.30, -0.16, 0.018], 100.0) == 150.0


def test_verify_event_invokes_callback_and_commands_no_hold_motion() -> None:
    arm = RecordingArm()
    vacuum = RecordingVacuum()
    timeline = RecordingTimeline()
    route = [
        ("lift", [500.0, 60.0, 118.0, 178.0, 0.0, 0.0], "loaded_vertical", None),
        ("verify hold", [500.0, 60.0, 118.0, 178.0, 0.0, 0.0], "loaded_vertical", "verify"),
    ]
    seen: list[str] = []

    MOTION.execute_route(
        arm,
        vacuum,
        route,
        motion_args(),
        {"value": False},
        timeline,
        "leaf_1",
        event_callback=seen.append,
    )

    assert len(arm.commands) == 1
    assert seen == ["verify hold"]


def test_verify_event_without_callback_is_rejected() -> None:
    arm = RecordingArm()
    vacuum = RecordingVacuum()
    timeline = RecordingTimeline()
    route = [
        ("lift", [500.0, 60.0, 118.0, 178.0, 0.0, 0.0], "loaded_vertical", None),
        ("verify hold", [500.0, 60.0, 118.0, 178.0, 0.0, 0.0], "loaded_vertical", "verify"),
    ]
    try:
        MOTION.execute_route(arm, vacuum, route, motion_args(), {"value": False}, timeline, "leaf_1")
    except RuntimeError as exc:
        assert "no event callback" in str(exc)
    else:
        raise AssertionError("A verify event without a callback must be rejected")


def test_failure_injected_acquire_skips_pump_and_leaves_holding_unset() -> None:
    arm = RecordingArm()
    vacuum = RecordingVacuum()
    timeline = RecordingTimeline()
    holding = {"value": False}
    route = [
        ("contact", [500.0, 60.0, -28.0, 178.0, 0.0, 0.0], "contact", "acquire"),
    ]

    MOTION.execute_route(
        arm,
        vacuum,
        route,
        motion_args(),
        holding,
        timeline,
        "leaf_1",
        failure_injected=True,
    )

    assert vacuum.actions == ["prepare"]
    assert holding["value"] is False
    names = [name for name, _details in timeline.events]
    assert "leaf_1.vacuum_acquire_skipped" in names


def test_prepare_vacuum_false_skips_preparation() -> None:
    arm = RecordingArm()
    vacuum = RecordingVacuum()
    timeline = RecordingTimeline()
    route = [
        ("lift", [500.0, 60.0, 118.0, 178.0, 0.0, 0.0], "loaded_vertical", None),
    ]

    MOTION.execute_route(
        arm,
        vacuum,
        route,
        motion_args(),
        {"value": False},
        timeline,
        "leaf_1",
        prepare_vacuum=False,
    )

    assert vacuum.actions == []


def test_stationary_hold_step_is_planned_without_motion_preflight(monkeypatch) -> None:
    class FakeArm:
        joint_speed_limit = (None, 100.0)
        joint_acc_limit = (None, 200.0)

    route = [
        ("lift", [500.0, 60.0, 118.0, 178.0, 0.0, 0.0], "loaded_vertical", None),
        ("verify hold", [500.0, 60.0, 118.0, 178.0, 0.0, 0.0], "loaded_vertical", "verify"),
    ]
    captured: dict = {}

    def fake_plan(arm, start_pose, start_angles, requests, **kwargs):
        captured["plan_labels"] = [request.label for request in requests]
        return [
            MOTION.SegmentSpeedPlan(
                label=request.label,
                sample_count=2,
                translation_distance_mm=100.0,
                rotation_distance_deg=0.0,
                requested_speed_mm_s=request.requested_speed_mm_s,
                planned_speed_mm_s=request.requested_speed_mm_s,
                requested_acceleration_mm_s2=request.requested_acceleration_mm_s2,
                planned_acceleration_mm_s2=request.requested_acceleration_mm_s2,
                predicted_peak_joint_speed_deg_s=10.0,
                predicted_peak_joint_acceleration_deg_s2=20.0,
                limiting_velocity_joint=1,
                limiting_acceleration_joint=1,
                end_joint_angles_deg=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )
            for request in requests
        ]

    def fake_preflight(arm, requests, plans, radii):
        captured["preflight_labels"] = [request.label for request in requests]
        captured["preflight_radii"] = radii

    monkeypatch.setattr(MOTION, "plan_cartesian_route_speeds", fake_plan)
    monkeypatch.setattr(MOTION, "preflight_cartesian_route", fake_preflight)

    plans = MOTION.plan_route_motion_profile(
        FakeArm(),
        [400.0, 60.0, 118.0, 178.0, 0.0, 0.0],
        [0.0] * 6,
        route,
        motion_args(),
    )

    assert captured["plan_labels"] == ["lift"]
    assert captured["preflight_labels"] == ["lift"]
    assert len(plans) == 2
    assert plans[1].translation_distance_mm == 0.0
    assert plans[1].label == "verify hold"
    assert plans[1].end_joint_angles_deg == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_accumulated_layer_heights_running_sums() -> None:
    assert MOTION.accumulated_layer_heights_mm([4.0, 0.5, 5.0, 3.0, 3.0]) == [0.0, 4.0, 4.5, 9.5, 12.5]
    assert MOTION.accumulated_layer_heights_mm([]) == []
    try:
        MOTION.accumulated_layer_heights_mm([4.0, -1.0])
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Negative layer thicknesses must be rejected")


class SuctionFakeArm(RecordingArm):
    def __init__(self, values) -> None:
        super().__init__()
        self.sensor_values = list(values)
        self._last_value = 0

    def get_cgpio_digital(self, channel):
        if self.sensor_values:
            self._last_value = self.sensor_values.pop(0)
        return 0, self._last_value


def make_sensor_for_test(arm, clock=None):
    from xarm_sdk_control.suction_sensor import SuctionSensor, SuctionSensorConfig

    if clock is None:
        class AdvancingClock:
            def __init__(self):
                self.now = 0.0

            def __call__(self):
                return self.now

            def advance(self, seconds):
                self.now += seconds

        clock = AdvancingClock()
    config = SuctionSensorConfig(input_channel=0, success_value=0, debounce_seconds=0.05, timeout_seconds=2.0)
    return SuctionSensor(arm, config, clock=clock, sleep=lambda s: clock.advance(s))


@pytest.mark.parametrize("outcome", ["confirmed", "failed", "lost", "loss_advisory"])
def test_suction_events_preserve_real_timeline_fields(outcome) -> None:
    from itertools import count
    from xarm_sdk_control.suction_sensor import SuctionNotConfirmedError
    from xarm_sdk_control.task_timing import TaskTimeline

    arm = SuctionFakeArm([0] if outcome == "confirmed" else [1])
    if outcome in {"lost", "loss_advisory"}:
        arm = SuctionCounterArm(8)
    ticks = count(100.0)
    timeline = TaskTimeline(clock=lambda: next(ticks))
    timeline.mark("productive_cycle_started")
    route = [("contact", [500.0, 60.0, -28.0, 178.0, 0.0, 0.0], "contact", "acquire")]
    if outcome in {"lost", "loss_advisory"}:
        route.append(("lift", [500.0, 60.0, 20.0, 178.0, 0.0, 0.0], "loaded_precision", None))

    def execute():
        MOTION.execute_route(
            arm, RecordingVacuum(), route, motion_args(), {"value": False}, timeline, "pick",
            suction_sensor=make_sensor_for_test(arm),
            advisory_suction_loss_motions=frozenset({"loaded_precision"})
            if outcome == "loss_advisory" else frozenset(),
        )

    if outcome in {"failed", "lost"}:
        with pytest.raises(SuctionNotConfirmedError):
            execute()
    else:
        execute()
    event = timeline.events[-1]
    assert event["name"] == f"pick.suction_{outcome}"
    assert event["suction_wait_seconds"] > 0
    assert [e["elapsed_seconds"] for e in timeline.events] == list(range(1, len(timeline.events) + 1))
    assert timeline.to_dict()["total_seconds"] == len(timeline.events)


def test_execute_route_confirms_suction_on_acquire() -> None:
    arm = SuctionFakeArm([0, 0, 0, 0, 0, 0])
    vacuum = RecordingVacuum()
    timeline = RecordingTimeline()
    holding = {"value": False}
    sensor = make_sensor_for_test(arm)
    route = [
        ("contact", [500.0, 60.0, -28.0, 178.0, 0.0, 0.0], "contact", "acquire"),
    ]

    MOTION.execute_route(
        arm, vacuum, route, motion_args(), holding, timeline, "leaf_1", suction_sensor=sensor
    )

    assert holding["value"] is True
    names = [name for name, _details in timeline.events]
    assert "leaf_1.vacuum_acquired" in names
    assert "leaf_1.suction_confirmed" in names


def test_execute_route_raises_when_suction_never_confirms() -> None:
    from xarm_sdk_control.suction_sensor import SuctionNotConfirmedError

    arm = SuctionFakeArm([1, 1, 1, 1])
    vacuum = RecordingVacuum()
    timeline = RecordingTimeline()
    holding = {"value": False}
    sensor = make_sensor_for_test(arm)
    route = [
        ("contact", [500.0, 60.0, -28.0, 178.0, 0.0, 0.0], "contact", "acquire"),
    ]

    try:
        MOTION.execute_route(
            arm, vacuum, route, motion_args(), holding, timeline, "leaf_1", suction_sensor=sensor
        )
    except SuctionNotConfirmedError:
        pass
    else:
        raise AssertionError("A never-confirming sensor must abort the acquire")

    assert holding["value"] is True  # stays set so recovery can vent safely
    names = [name for name, _details in timeline.events]
    assert "leaf_1.suction_failed" in names


def test_execute_route_failure_injected_with_sensor_raises() -> None:
    from xarm_sdk_control.suction_sensor import SuctionNotConfirmedError

    arm = SuctionFakeArm([1, 1, 1, 1])
    vacuum = RecordingVacuum()
    timeline = RecordingTimeline()
    holding = {"value": False}
    sensor = make_sensor_for_test(arm)
    route = [
        ("contact", [500.0, 60.0, -28.0, 178.0, 0.0, 0.0], "contact", "acquire"),
    ]

    try:
        MOTION.execute_route(
            arm, vacuum, route, motion_args(), holding, timeline, "leaf_1",
            failure_injected=True, suction_sensor=sensor,
        )
    except SuctionNotConfirmedError:
        pass
    else:
        raise AssertionError("Controlled pump-skip with a sensor must fail suction")

    assert holding["value"] is False


class SuctionCounterArm(RecordingArm):
    """Returns 0 for the first `success_reads` calls, then 1 forever."""

    def __init__(self, success_reads: int) -> None:
        super().__init__()
        self.success_reads = success_reads
        self.reads = 0

    def get_cgpio_digital(self, channel):
        self.reads += 1
        return 0, 0 if self.reads <= self.success_reads else 1


def test_execute_route_holds_through_loaded_segments() -> None:
    arm = SuctionCounterArm(1000)  # always confirms
    vacuum = RecordingVacuum()
    timeline = RecordingTimeline()
    holding = {"value": False}
    sensor = make_sensor_for_test(arm)
    route = [
        ("contact", [500.0, 60.0, -28.0, 178.0, 0.0, 0.0], "contact", "acquire"),
        ("lift", [500.0, 60.0, 20.0, 178.0, 0.0, 0.0], "loaded_precision", None),
    ]

    MOTION.execute_route(
        arm, vacuum, route, motion_args(), holding, timeline, "leaf_1", suction_sensor=sensor
    )

    assert holding["value"] is True


def test_execute_route_detects_suction_lost_after_lift() -> None:
    from xarm_sdk_control.suction_sensor import SuctionNotConfirmedError

    arm = SuctionCounterArm(8)  # confirms at acquire, then loses the hold
    vacuum = RecordingVacuum()
    timeline = RecordingTimeline()
    holding = {"value": False}
    sensor = make_sensor_for_test(arm)
    route = [
        ("contact", [500.0, 60.0, -28.0, 178.0, 0.0, 0.0], "contact", "acquire"),
        ("lift", [500.0, 60.0, 20.0, 178.0, 0.0, 0.0], "loaded_precision", None),
    ]

    try:
        MOTION.execute_route(
            arm, vacuum, route, motion_args(), holding, timeline, "leaf_1", suction_sensor=sensor
        )
    except SuctionNotConfirmedError:
        pass
    else:
        raise AssertionError("A mid-lift suction loss must abort the carry")

    assert holding["value"] is True
    names = [name for name, _details in timeline.events]
    assert "leaf_1.suction_lost" in names


def test_execute_route_allows_scoped_porous_load_sustain_warning() -> None:
    arm = SuctionCounterArm(8)  # confirms acquire, then loses the pressure signal
    vacuum = RecordingVacuum()
    timeline = RecordingTimeline()
    holding = {"value": False}
    sensor = make_sensor_for_test(arm)
    route = [
        ("contact", [500.0, 60.0, -28.0, 178.0, 0.0, 0.0], "contact", "acquire"),
        ("initial lift", [500.0, 60.0, 20.0, 178.0, 0.0, 0.0], "loaded_precision", None),
        ("carry", [300.0, -200.0, 100.0, 178.0, 0.0, 0.0], "loaded_transit", None),
    ]

    MOTION.execute_route(
        arm,
        vacuum,
        route,
        motion_args(),
        holding,
        timeline,
        "paper",
        suction_sensor=sensor,
        advisory_suction_loss_motions=frozenset(
            {"loaded_precision", "loaded_vertical", "loaded_transit"}
        ),
    )

    assert holding["value"] is True
    names = [name for name, _details in timeline.events]
    assert names.count("paper.suction_loss_advisory") == 1
    assert "paper.suction_lost" not in names


def test_route_planning_falls_back_to_endpoint_interpolation(monkeypatch) -> None:
    class FakeArm:
        joint_speed_limit = (None, 100.0)
        joint_acc_limit = (None, 200.0)

    route = [
        ("lift", [500.0, 60.0, 118.0, 178.0, 0.0, 0.0], "loaded_vertical", None),
    ]
    calls = {"endpoint": 0}

    def always_fail(arm, start_pose, start_angles, requests, **kwargs):
        raise MOTION.TrajectoryPlanningError("IK discontinuity on joint 4")

    def fake_endpoint(arm, start_pose, start_angles, requests, **kwargs):
        calls["endpoint"] += 1
        return [
            MOTION.SegmentSpeedPlan(
                label=r.label,
                sample_count=2,
                translation_distance_mm=100.0,
                rotation_distance_deg=0.0,
                requested_speed_mm_s=r.requested_speed_mm_s,
                planned_speed_mm_s=r.requested_speed_mm_s,
                requested_acceleration_mm_s2=r.requested_acceleration_mm_s2,
                planned_acceleration_mm_s2=r.requested_acceleration_mm_s2,
                predicted_peak_joint_speed_deg_s=10.0,
                predicted_peak_joint_acceleration_deg_s2=20.0,
                limiting_velocity_joint=1,
                limiting_acceleration_joint=1,
                end_joint_angles_deg=[0.0] * 6,
            )
            for r in requests
        ]

    monkeypatch.setattr(MOTION, "plan_cartesian_route_speeds", always_fail)
    monkeypatch.setattr(MOTION, "plan_cartesian_route_speeds_endpoint", fake_endpoint)
    monkeypatch.setattr(MOTION, "preflight_cartesian_route", lambda *a, **k: None)

    plans = MOTION.plan_route_motion_profile(
        FakeArm(),
        [400.0, 60.0, 118.0, 178.0, 0.0, 0.0],
        [0.0] * 6,
        route,
        motion_args(),
    )

    assert calls["endpoint"] == 1
    assert len(plans) == 1
