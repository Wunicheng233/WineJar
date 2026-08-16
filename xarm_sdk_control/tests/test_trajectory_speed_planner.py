from __future__ import annotations

import pytest

from xarm_sdk_control.trajectory_speed_planner import (
    CartesianSegmentRequest,
    ControllerPreflightError,
    JointConstraintConfig,
    TrajectoryPlanningError,
    interpolate_pose_mm_deg,
    plan_cartesian_route_speeds,
    preflight_cartesian_route,
)


class LinearIKArm:
    def __init__(self, gain_deg_per_mm: float = 1.0, jump: bool = False) -> None:
        self.gain = gain_deg_per_mm
        self.jump = jump
        self.motion_commands = []

    def get_inverse_kinematics(self, pose, **_kwargs):
        x = float(pose[0])
        joint_1 = self.gain * x
        if self.jump and x >= 15.0:
            joint_1 += 90.0
        return 0, [joint_1, 0.0, 0.0, 0.0, 0.0, float(pose[5])]


class ControllerCheckArm:
    def __init__(self, maximum_speed: float, failure_result: int = 24) -> None:
        self.maximum_speed = maximum_speed
        self.failure_result = failure_result
        self.only_check_result = 0
        self.check_modes = []
        self.checked_speeds = []

    def set_only_check_type(self, mode):
        self.check_modes.append(mode)

    def set_position(self, *_pose, **kwargs):
        speed = float(kwargs["speed"])
        self.checked_speeds.append(speed)
        if speed > self.maximum_speed:
            self.only_check_result = self.failure_result
            return 1
        self.only_check_result = 0
        return 0


def request(speed=100.0, acceleration=100.0):
    return CartesianSegmentRequest(
        label="test segment",
        target_pose_mm_deg=[20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        requested_speed_mm_s=speed,
        requested_acceleration_mm_s2=acceleration,
    )


def test_pose_interpolation_uses_shortest_angle() -> None:
    midpoint = interpolate_pose_mm_deg(
        [0, 0, 0, 0, 0, 170],
        [10, 0, 0, 0, 0, -170],
        0.5,
    )
    assert midpoint[:3] == [5.0, 0.0, 0.0]
    assert midpoint[5] == 180.0


def test_planner_scales_speed_to_joint_margin_without_motion() -> None:
    arm = LinearIKArm(gain_deg_per_mm=2.0)
    plans = plan_cartesian_route_speeds(
        arm,
        start_pose_mm_deg=[0, 0, 0, 0, 0, 0],
        start_joint_angles_deg=[0] * 6,
        requests=[request(speed=100.0, acceleration=100.0)],
        joint_velocity_limit_deg_s=180.0,
        joint_acceleration_limit_deg_s2=1000.0,
        config=JointConstraintConfig(max_translation_step_mm=5.0),
    )
    assert len(plans) == 1
    assert plans[0].planned_speed_mm_s == pytest.approx(58.5)
    assert plans[0].predicted_peak_joint_speed_deg_s == pytest.approx(117.0)
    assert plans[0].limiting_velocity_joint == 1
    assert arm.motion_commands == []


def test_planner_keeps_safe_requested_speed() -> None:
    plans = plan_cartesian_route_speeds(
        LinearIKArm(gain_deg_per_mm=0.2),
        start_pose_mm_deg=[0, 0, 0, 0, 0, 0],
        start_joint_angles_deg=[0] * 6,
        requests=[request(speed=100.0, acceleration=100.0)],
        joint_velocity_limit_deg_s=180.0,
        joint_acceleration_limit_deg_s2=1000.0,
    )
    assert plans[0].planned_speed_mm_s == 100.0
    assert plans[0].planned_acceleration_mm_s2 == 100.0


def test_planner_rejects_discontinuous_ik_branch() -> None:
    with pytest.raises(TrajectoryPlanningError, match="IK discontinuity"):
        plan_cartesian_route_speeds(
            LinearIKArm(jump=True),
            start_pose_mm_deg=[0, 0, 0, 0, 0, 0],
            start_joint_angles_deg=[0] * 6,
            requests=[request()],
            joint_velocity_limit_deg_s=180.0,
            joint_acceleration_limit_deg_s2=1000.0,
            config=JointConstraintConfig(max_translation_step_mm=5.0),
        )


def test_planner_rejects_pure_rotation_segment() -> None:
    with pytest.raises(TrajectoryPlanningError, match="Pure orientation"):
        plan_cartesian_route_speeds(
            LinearIKArm(),
            start_pose_mm_deg=[0, 0, 0, 0, 0, 0],
            start_joint_angles_deg=[0] * 6,
            requests=[
                CartesianSegmentRequest(
                    "rotation",
                    [0, 0, 0, 0, 0, 30],
                    50,
                    100,
                )
            ],
            joint_velocity_limit_deg_s=180.0,
            joint_acceleration_limit_deg_s2=1000.0,
        )


def test_invalid_joint_limit_vector_is_rejected() -> None:
    with pytest.raises(ValueError, match="contain 6"):
        plan_cartesian_route_speeds(
            LinearIKArm(),
            start_pose_mm_deg=[0, 0, 0, 0, 0, 0],
            start_joint_angles_deg=[0] * 6,
            requests=[request()],
            joint_velocity_limit_deg_s=[180.0] * 5,
            joint_acceleration_limit_deg_s2=1000.0,
        )


def test_controller_preflight_rejects_unsafe_route_without_rescaling() -> None:
    request_item = request(speed=200.0, acceleration=400.0)
    estimated = plan_cartesian_route_speeds(
        LinearIKArm(gain_deg_per_mm=0.1),
        start_pose_mm_deg=[0, 0, 0, 0, 0, 0],
        start_joint_angles_deg=[0] * 6,
        requests=[request_item],
        joint_velocity_limit_deg_s=180.0,
        joint_acceleration_limit_deg_s2=1000.0,
    )
    arm = ControllerCheckArm(maximum_speed=120.0)

    with pytest.raises(ControllerPreflightError, match="C24"):
        preflight_cartesian_route(arm, [request_item], estimated, [-1.0])

    assert arm.checked_speeds == [pytest.approx(200.0)]
    assert arm.check_modes[-1] == 0


def test_controller_preflight_does_not_hide_non_c24_failure() -> None:
    request_item = request(speed=200.0, acceleration=400.0)
    estimated = plan_cartesian_route_speeds(
        LinearIKArm(gain_deg_per_mm=0.1),
        start_pose_mm_deg=[0, 0, 0, 0, 0, 0],
        start_joint_angles_deg=[0] * 6,
        requests=[request_item],
        joint_velocity_limit_deg_s=180.0,
        joint_acceleration_limit_deg_s2=1000.0,
    )

    with pytest.raises(ControllerPreflightError, match="C23"):
        preflight_cartesian_route(
            ControllerCheckArm(maximum_speed=120.0, failure_result=23),
            [request_item],
            estimated,
            [-1.0],
        )

