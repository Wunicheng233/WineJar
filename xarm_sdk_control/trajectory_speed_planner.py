"""Joint-aware speed scaling for precomputed Cartesian xArm routes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from xarm_sdk_control.geometry import wrapped_delta_deg


class TrajectoryPlanningError(RuntimeError):
    """Raised when a route cannot be certified before robot motion."""


@dataclass(frozen=True)
class JointConstraintConfig:
    joint_count: int = 6
    velocity_utilization: float = 0.65
    acceleration_utilization: float = 0.50
    max_translation_step_mm: float = 10.0
    max_rotation_step_deg: float = 2.0
    max_joint_step_deg: float = 15.0
    minimum_speed_mm_s: float = 5.0
    minimum_acceleration_mm_s2: float = 10.0

    def __post_init__(self) -> None:
        if self.joint_count <= 0:
            raise ValueError("joint_count must be positive")
        for name, value in (
            ("velocity_utilization", self.velocity_utilization),
            ("acceleration_utilization", self.acceleration_utilization),
        ):
            if not 0 < value < 1:
                raise ValueError(f"{name} must be within (0, 1)")
        for name, value in (
            ("max_translation_step_mm", self.max_translation_step_mm),
            ("max_rotation_step_deg", self.max_rotation_step_deg),
            ("max_joint_step_deg", self.max_joint_step_deg),
            ("minimum_speed_mm_s", self.minimum_speed_mm_s),
            ("minimum_acceleration_mm_s2", self.minimum_acceleration_mm_s2),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class CartesianSegmentRequest:
    label: str
    target_pose_mm_deg: list[float]
    requested_speed_mm_s: float
    requested_acceleration_mm_s2: float


@dataclass(frozen=True)
class SegmentSpeedPlan:
    label: str
    sample_count: int
    translation_distance_mm: float
    rotation_distance_deg: float
    requested_speed_mm_s: float
    planned_speed_mm_s: float
    requested_acceleration_mm_s2: float
    planned_acceleration_mm_s2: float
    predicted_peak_joint_speed_deg_s: float
    predicted_peak_joint_acceleration_deg_s2: float
    limiting_velocity_joint: int
    limiting_acceleration_joint: int
    end_joint_angles_deg: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ControllerPreflightError(TrajectoryPlanningError):
    def __init__(self, segment_index: int, label: str, api_code: int, controller_result: int) -> None:
        self.segment_index = segment_index
        self.label = label
        self.api_code = api_code
        self.controller_result = controller_result
        super().__init__(
            f"Controller preflight rejected segment {segment_index + 1} {label!r}: "
            f"API code={api_code}, controller result=C{controller_result}"
        )


def interpolate_pose_mm_deg(start: Sequence[float], target: Sequence[float], alpha: float) -> list[float]:
    start_array = np.asarray(start, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if start_array.shape != (6,) or target_array.shape != (6,):
        raise ValueError("Cartesian poses must contain [x, y, z, roll, pitch, yaw]")
    pose = start_array.copy()
    pose[:3] += float(alpha) * (target_array[:3] - start_array[:3])
    pose[3:] += float(alpha) * wrapped_delta_deg(target_array[3:], start_array[3:])
    return pose.tolist()


def _joint_limit_array(value: float | Sequence[float], joint_count: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(joint_count, float(array), dtype=float)
    if array.shape != (joint_count,):
        raise ValueError(f"{label} must be a scalar or contain {joint_count} values")
    if np.any(array <= 0):
        raise ValueError(f"{label} must be positive")
    return array


def _solve_ik(arm: Any, pose: list[float], reference_deg: np.ndarray, joint_count: int) -> np.ndarray:
    result = arm.get_inverse_kinematics(
        pose,
        input_is_radian=False,
        return_is_radian=False,
        limited=True,
        ref_angles=reference_deg.tolist(),
    )
    code = result[0] if isinstance(result, (tuple, list)) else None
    if code != 0 or not isinstance(result, (tuple, list)) or len(result) < 2:
        raise TrajectoryPlanningError(f"Inverse kinematics failed for pose {pose}: {result}")
    angles = np.asarray(result[1], dtype=float)[:joint_count]
    if angles.shape != (joint_count,) or not np.all(np.isfinite(angles)):
        raise TrajectoryPlanningError(f"Inverse kinematics returned invalid joint angles: {result}")
    return reference_deg + wrapped_delta_deg(angles, reference_deg)


def _sample_segment(
    arm: Any,
    start_pose: np.ndarray,
    target_pose: np.ndarray,
    start_angles_deg: np.ndarray,
    config: JointConstraintConfig,
) -> tuple[np.ndarray, float, float, int]:
    translation_mm = float(np.linalg.norm(target_pose[:3] - start_pose[:3]))
    rotation_deg = float(np.max(np.abs(wrapped_delta_deg(target_pose[3:], start_pose[3:]))))
    if translation_mm < 1e-6:
        raise TrajectoryPlanningError(
            "Pure orientation segments are not supported because xArm Cartesian speed is expressed in mm/s"
        )
    sample_count = max(
        2,
        int(np.ceil(translation_mm / config.max_translation_step_mm)),
        int(np.ceil(rotation_deg / config.max_rotation_step_deg)),
    )
    samples = [np.asarray(start_angles_deg, dtype=float)]
    reference = samples[0]
    for alpha in np.linspace(0.0, 1.0, sample_count + 1, dtype=float)[1:]:
        pose = interpolate_pose_mm_deg(start_pose, target_pose, float(alpha))
        angles = _solve_ik(arm, pose, reference, config.joint_count)
        joint_step = np.abs(angles - reference)
        if float(np.max(joint_step)) > config.max_joint_step_deg:
            joint = int(np.argmax(joint_step)) + 1
            raise TrajectoryPlanningError(
                f"IK discontinuity on joint {joint}: {joint_step[joint - 1]:.1f} deg between adjacent samples"
            )
        samples.append(angles)
        reference = angles
    return np.vstack(samples), translation_mm, rotation_deg, sample_count


def _plan_segment(
    joint_samples_deg: np.ndarray,
    request: CartesianSegmentRequest,
    translation_mm: float,
    rotation_deg: float,
    sample_count: int,
    velocity_limits_deg_s: np.ndarray,
    acceleration_limits_deg_s2: np.ndarray,
    config: JointConstraintConfig,
) -> SegmentSpeedPlan:
    ds_mm = translation_mm / sample_count
    joint_prime = np.diff(joint_samples_deg, axis=0) / ds_mm
    max_abs_prime = np.max(np.abs(joint_prime), axis=0)
    if joint_prime.shape[0] >= 2:
        joint_second = np.diff(joint_prime, axis=0) / ds_mm
        max_abs_second = np.max(np.abs(joint_second), axis=0)
    else:
        max_abs_second = np.zeros(config.joint_count, dtype=float)

    allowed_velocity = velocity_limits_deg_s * config.velocity_utilization
    allowed_acceleration = acceleration_limits_deg_s2 * config.acceleration_utilization
    velocity_caps = np.full(config.joint_count, np.inf, dtype=float)
    moving = max_abs_prime > 1e-9
    velocity_caps[moving] = allowed_velocity[moving] / max_abs_prime[moving]

    curvature_caps = np.full(config.joint_count, np.inf, dtype=float)
    curved = max_abs_second > 1e-12
    curvature_caps[curved] = np.sqrt(allowed_acceleration[curved] / max_abs_second[curved])
    planned_speed = float(min(request.requested_speed_mm_s, np.min(velocity_caps), np.min(curvature_caps)))
    if planned_speed < config.minimum_speed_mm_s:
        raise TrajectoryPlanningError(
            f"Safe speed for {request.label!r} is only {planned_speed:.2f} mm/s, below the configured minimum"
        )

    curvature_acceleration = max_abs_second * planned_speed**2
    remaining_acceleration = np.maximum(0.0, allowed_acceleration - curvature_acceleration)
    acceleration_caps = np.full(config.joint_count, np.inf, dtype=float)
    acceleration_caps[moving] = remaining_acceleration[moving] / max_abs_prime[moving]
    planned_acceleration = float(min(request.requested_acceleration_mm_s2, np.min(acceleration_caps)))
    if planned_acceleration < config.minimum_acceleration_mm_s2:
        raise TrajectoryPlanningError(
            f"Safe acceleration for {request.label!r} is only {planned_acceleration:.2f} mm/s^2"
        )

    predicted_velocity = max_abs_prime * planned_speed
    predicted_acceleration = curvature_acceleration + max_abs_prime * planned_acceleration
    velocity_joint = int(np.argmax(predicted_velocity))
    acceleration_joint = int(np.argmax(predicted_acceleration))
    return SegmentSpeedPlan(
        label=request.label,
        sample_count=sample_count,
        translation_distance_mm=translation_mm,
        rotation_distance_deg=rotation_deg,
        requested_speed_mm_s=float(request.requested_speed_mm_s),
        planned_speed_mm_s=planned_speed,
        requested_acceleration_mm_s2=float(request.requested_acceleration_mm_s2),
        planned_acceleration_mm_s2=planned_acceleration,
        predicted_peak_joint_speed_deg_s=float(predicted_velocity[velocity_joint]),
        predicted_peak_joint_acceleration_deg_s2=float(predicted_acceleration[acceleration_joint]),
        limiting_velocity_joint=velocity_joint + 1,
        limiting_acceleration_joint=acceleration_joint + 1,
        end_joint_angles_deg=joint_samples_deg[-1].tolist(),
    )


def plan_cartesian_route_speeds(
    arm: Any,
    start_pose_mm_deg: Sequence[float],
    start_joint_angles_deg: Sequence[float],
    requests: Sequence[CartesianSegmentRequest],
    joint_velocity_limit_deg_s: float | Sequence[float],
    joint_acceleration_limit_deg_s2: float | Sequence[float],
    config: JointConstraintConfig | None = None,
) -> list[SegmentSpeedPlan]:
    """Estimate joint-aware per-segment speeds without commanding robot motion."""
    config = config or JointConstraintConfig()
    velocity_limits = _joint_limit_array(
        joint_velocity_limit_deg_s,
        config.joint_count,
        "joint_velocity_limit_deg_s",
    )
    acceleration_limits = _joint_limit_array(
        joint_acceleration_limit_deg_s2,
        config.joint_count,
        "joint_acceleration_limit_deg_s2",
    )
    pose = np.asarray(start_pose_mm_deg, dtype=float)
    angles = np.asarray(start_joint_angles_deg, dtype=float)[: config.joint_count]
    if pose.shape != (6,) or angles.shape != (config.joint_count,):
        raise ValueError("Invalid start pose or joint-angle vector")

    plans: list[SegmentSpeedPlan] = []
    for request in requests:
        target = np.asarray(request.target_pose_mm_deg, dtype=float)
        if target.shape != (6,):
            raise ValueError(f"Invalid target pose for {request.label!r}")
        samples, distance, rotation, sample_count = _sample_segment(
            arm,
            pose,
            target,
            angles,
            config,
        )
        plan = _plan_segment(
            samples,
            request,
            distance,
            rotation,
            sample_count,
            velocity_limits,
            acceleration_limits,
            config,
        )
        plans.append(plan)
        pose = target
        angles = samples[-1]
    return plans


def plan_cartesian_route_speeds_endpoint(
    arm: Any,
    start_pose_mm_deg: Sequence[float],
    start_joint_angles_deg: Sequence[float],
    requests: Sequence[CartesianSegmentRequest],
    joint_velocity_limit_deg_s: float | Sequence[float],
    joint_acceleration_limit_deg_s2: float | Sequence[float],
    config: JointConstraintConfig | None = None,
) -> list[SegmentSpeedPlan]:
    """Reduced-fidelity fallback: linear joint interpolation between endpoint IK
    solutions.

    The dense per-sample IK sampling is fragile near the wrist singularity of
    the home pose (roll ~178.5 deg): the controller's solver intermittently
    flips its J4/J6 branch for nearly identical inputs. This fallback solves
    IK only at the segment endpoints (each anchored to the previous endpoint),
    so it cannot drift onto a flipped branch mid-segment. The cost is losing
    the curvature caps from interior samples; the controller no-motion
    preflight and runtime joint-limit enforcement still apply.
    """
    config = config or JointConstraintConfig()
    velocity_limits = _joint_limit_array(
        joint_velocity_limit_deg_s,
        config.joint_count,
        "joint_velocity_limit_deg_s",
    )
    acceleration_limits = _joint_limit_array(
        joint_acceleration_limit_deg_s2,
        config.joint_count,
        "joint_acceleration_limit_deg_s2",
    )
    pose = np.asarray(start_pose_mm_deg, dtype=float)
    angles = np.asarray(start_joint_angles_deg, dtype=float)[: config.joint_count]
    if pose.shape != (6,) or angles.shape != (config.joint_count,):
        raise ValueError("Invalid start pose or joint-angle vector")

    plans: list[SegmentSpeedPlan] = []
    for request in requests:
        target = np.asarray(request.target_pose_mm_deg, dtype=float)
        if target.shape != (6,):
            raise ValueError(f"Invalid target pose for {request.label!r}")
        # Anchor the endpoint solve to the previous endpoint's angles. Unlike the
        # dense sampler there is no per-step continuity check: a whole segment
        # can legitimately move a joint far more than max_joint_step_deg, and a
        # flipped branch merely yields a conservative speed estimate that the
        # controller preflight and runtime limits still guard.
        end_angles = _solve_ik(arm, target.tolist(), angles, config.joint_count)
        samples = np.vstack([angles, end_angles])
        distance = float(np.linalg.norm(target[:3] - pose[:3]))
        rotation = float(np.max(np.abs(wrapped_delta_deg(target[3:], pose[3:]))))
        plan = _plan_segment(
            samples,
            request,
            distance,
            rotation,
            2,
            velocity_limits,
            acceleration_limits,
            config,
        )
        plans.append(plan)
        pose = target
        angles = end_angles
    return plans


def _api_code(result: Any) -> int:
    code = result[0] if isinstance(result, (tuple, list)) else result
    return 0 if code is None else int(code)


def preflight_cartesian_route(
    arm: Any,
    requests: Sequence[CartesianSegmentRequest],
    plans: Sequence[SegmentSpeedPlan],
    blend_radii_mm: Sequence[float],
) -> None:
    """Ask the xArm controller to validate an exact route without moving the robot."""
    if not (len(requests) == len(plans) == len(blend_radii_mm)):
        raise ValueError("Preflight request, plan, and radius counts must match")
    if not requests:
        return

    try:
        arm.set_only_check_type(1)
        for index, (request, plan, radius) in enumerate(zip(requests, plans, blend_radii_mm)):
            if index == 1:
                arm.set_only_check_type(3)
            result = arm.set_position(
                *request.target_pose_mm_deg,
                speed=plan.planned_speed_mm_s,
                mvacc=plan.planned_acceleration_mm_s2,
                wait=False,
                radius=float(radius),
                is_radian=False,
            )
            code = _api_code(result)
            controller_result = int(getattr(arm, "only_check_result", 0) or 0)
            if code != 0 or controller_result != 0:
                raise ControllerPreflightError(
                    index,
                    request.label,
                    code,
                    controller_result,
                )
    finally:
        arm.set_only_check_type(0)

