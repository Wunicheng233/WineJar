from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from xarm_sdk_control import cross_place_motion as MOTION


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "xarm_sdk_control" / "scripts" / "19_visual_cross_place_two_leaves.py"
SPEC = importlib.util.spec_from_file_location("visual_cross_place", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TimeoutAtTargetArm:
    def get_position(self, is_radian=False):
        return 0, [300.0, 150.0, -60.0, 180.0, 0.0, 90.0]

    def get_state(self):
        return 0, 2

    def get_err_warn_code(self):
        return 0, [0, 0]


class TimeoutAwayFromTargetArm(TimeoutAtTargetArm):
    def get_position(self, is_radian=False):
        return 0, [300.0, 150.0, -50.0, 180.0, 0.0, 90.0]


class RecordingArm:
    def __init__(self) -> None:
        self.commands = []

    def set_position(self, *pose, **kwargs):
        self.commands.append((pose, kwargs))
        return 0

    def get_state(self):
        return 0, 4

    def get_err_warn_code(self):
        return 0, [0, 0]

    def get_position(self, is_radian=False):
        return 0, [300.0, 100.0, 200.0, 178.0, 0.0, 0.0]

    def get_servo_angle(self, is_radian=False):
        return 0, [0.0] * 7


class RecordingVacuum:
    def __init__(self) -> None:
        self.actions = []

    def prepare_for_pick(self) -> None:
        self.actions.append("prepare")

    def acquire(self, settle_seconds: float) -> None:
        self.actions.append(("acquire", settle_seconds))

    def release(self, settle_seconds: float) -> None:
        self.actions.append(("release", settle_seconds))


class RecordingTimeline:
    def __init__(self) -> None:
        self.events = []

    def mark(self, name: str, **details) -> None:
        self.events.append((name, details))


def motion_args(**overrides):
    values = {
        "transit_speed": 300.0,
        "loaded_transit_speed": 220.0,
        "vertical_speed": 200.0,
        "loaded_vertical_speed": 120.0,
        "precision_speed": 60.0,
        "contact_speed": 20.0,
        "acc": 500.0,
        "loaded_acc": 250.0,
        "precision_acc": 200.0,
        "blend_radius_mm": 10.0,
        "vacuum_settle_seconds": 0.9,
        "release_settle_seconds": 0.3,
        "joint_velocity_utilization": 0.40,
        "joint_acceleration_utilization": 0.40,
        "ik_translation_step_mm": 10.0,
        "ik_rotation_step_deg": 2.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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
    assert args.second_scan_min_score == 0.20
    assert args.skip_second_controller_preflight is False


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
