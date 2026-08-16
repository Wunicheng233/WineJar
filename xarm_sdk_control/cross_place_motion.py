"""Motion planning and execution primitives for the two-shell placement task."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from xarm_sdk_control.cross_place_route import (  # noqa: F401  (re-exported for task scripts)
    BLENDABLE_TRANSITIONS,
    build_cycle_route,
    build_pick_recovery_route,
    choose_second_source_leaf,
    offset_detection,
    offset_point_mm,
    require_distinct_pick_and_place_targets,
    route_blend_radii,
    transport_height_mm,
)
from xarm_sdk_control.geometry import wrapped_angle_deg
from xarm_sdk_control.robot_motion import (  # noqa: F401  (re-exported for task scripts)
    require_code,
    require_emergency_stop_released,
    robot_status_snapshot,
)
from xarm_sdk_control.task_timing import TaskTimeline
from xarm_sdk_control.trajectory_speed_planner import (
    CartesianSegmentRequest,
    JointConstraintConfig,
    SegmentSpeedPlan,  # noqa: F401  (re-exported for tests)
    plan_cartesian_route_speeds,
    preflight_cartesian_route,
)
from xarm_sdk_control.vacuum_control import VacuumController


MAX_TASK_EMPTY_TRANSIT_SPEED_MM_S = 400.0
MAX_TASK_LOADED_TRANSIT_SPEED_MM_S = 400.0
MAX_TASK_VERTICAL_SPEED_MM_S = 250.0
MAX_TASK_LOADED_VERTICAL_SPEED_MM_S = 250.0
MAX_TASK_PRECISION_SPEED_MM_S = 60.0
MAX_TASK_CONTACT_SPEED_MM_S = 20.0
MAX_TASK_ACCELERATION_MM_S2 = 500.0
MAX_TASK_LOADED_ACCELERATION_MM_S2 = 350.0
MAX_TASK_PRECISION_ACCELERATION_MM_S2 = 200.0


@dataclass(frozen=True)
class MotionProfile:
    """Validated stability-aware speed profile for the two-shell cross-place task."""

    transit_speed: float
    loaded_transit_speed: float
    vertical_speed: float
    loaded_vertical_speed: float
    precision_speed: float
    contact_speed: float
    acc: float
    loaded_acc: float
    precision_acc: float
    blend_radius_mm: float
    vacuum_settle_seconds: float
    release_settle_seconds: float
    joint_velocity_utilization: float
    joint_acceleration_utilization: float
    ik_translation_step_mm: float
    ik_rotation_step_deg: float


DEFAULT_MOTION_PROFILE = MotionProfile(
    transit_speed=300.0,
    loaded_transit_speed=220.0,
    vertical_speed=200.0,
    loaded_vertical_speed=120.0,
    precision_speed=60.0,
    contact_speed=20.0,
    acc=500.0,
    loaded_acc=250.0,
    precision_acc=200.0,
    blend_radius_mm=10.0,
    vacuum_settle_seconds=0.9,
    release_settle_seconds=0.3,
    joint_velocity_utilization=0.40,
    joint_acceleration_utilization=0.40,
    ik_translation_step_mm=10.0,
    ik_rotation_step_deg=2.0,
)


def require_motion_completed(
    arm: Any,
    label: str,
    result: Any,
    target_pose: list[float],
    position_tolerance_mm: float = 1.0,
    angle_tolerance_deg: float = 2.0,
) -> None:
    code = result[0] if isinstance(result, tuple) else result
    print(f"{label}: {result}")
    if code in (0, None):
        return
    if code != 3:
        raise RuntimeError(f"{label} failed with code/result: {result}")

    # Code 3 is a feedback timeout. The controller may still have reached the target.
    for _ in range(5):
        pose_code, actual_pose = arm.get_position(is_radian=False)
        state_code, state = arm.get_state()
        err_code, err_warn = arm.get_err_warn_code()
        position_error = float(np.max(np.abs(np.asarray(actual_pose[:3]) - np.asarray(target_pose[:3]))))
        angle_error = float(
            np.max(
                np.abs(
                    [wrapped_angle_deg(actual_pose[index] - target_pose[index]) for index in range(3, 6)]
                )
            )
        )
        if (
            pose_code == 0
            and state_code == 0
            and state == 2
            and err_code == 0
            and list(err_warn) == [0, 0]
            and position_error <= position_tolerance_mm
            and angle_error <= angle_tolerance_deg
        ):
            print(
                f"WARNING: {label} feedback timed out, but target was reached "
                f"(position error={position_error:.3f} mm, angle error={angle_error:.3f} deg)."
            )
            return
        time.sleep(0.2)
    raise RuntimeError(
        f"{label} timed out and target completion could not be verified; "
        f"last position error={position_error:.3f} mm, angle error={angle_error:.3f} deg, "
        f"state={state}, err/warn={err_warn}"
    )


def validate_motion_profile(args: Any) -> None:
    limits = {
        "empty transit speed": (args.transit_speed, MAX_TASK_EMPTY_TRANSIT_SPEED_MM_S),
        "loaded transit speed": (args.loaded_transit_speed, MAX_TASK_LOADED_TRANSIT_SPEED_MM_S),
        "vertical speed": (args.vertical_speed, MAX_TASK_VERTICAL_SPEED_MM_S),
        "loaded vertical speed": (args.loaded_vertical_speed, MAX_TASK_LOADED_VERTICAL_SPEED_MM_S),
        "precision speed": (args.precision_speed, MAX_TASK_PRECISION_SPEED_MM_S),
        "contact speed": (args.contact_speed, MAX_TASK_CONTACT_SPEED_MM_S),
        "TCP acceleration": (args.acc, MAX_TASK_ACCELERATION_MM_S2),
        "loaded acceleration": (args.loaded_acc, MAX_TASK_LOADED_ACCELERATION_MM_S2),
        "precision acceleration": (args.precision_acc, MAX_TASK_PRECISION_ACCELERATION_MM_S2),
    }
    for label, (value, maximum) in limits.items():
        if not 0 < value <= maximum:
            raise ValueError(f"{label} must be within (0, {maximum}], got {value}")
    for label, value in (
        ("joint velocity utilization", args.joint_velocity_utilization),
        ("joint acceleration utilization", args.joint_acceleration_utilization),
    ):
        if not 0 < value < 1:
            raise ValueError(f"{label} must be within (0, 1), got {value}")
    for label, value in (
        ("IK translation step", args.ik_translation_step_mm),
        ("IK rotation step", args.ik_rotation_step_deg),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {value}")


def requested_motion_profiles(args: Any) -> tuple[dict[str, float], dict[str, float]]:
    speeds = {
        "empty_transit": args.transit_speed,
        "loaded_transit": args.loaded_transit_speed,
        "vertical": args.vertical_speed,
        "loaded_vertical": args.loaded_vertical_speed,
        "precision": args.precision_speed,
        "loaded_precision": args.precision_speed,
        "contact": args.contact_speed,
    }
    accelerations = {
        "empty_transit": args.acc,
        "loaded_transit": args.loaded_acc,
        "vertical": args.acc,
        "loaded_vertical": args.loaded_acc,
        "precision": args.precision_acc,
        "loaded_precision": args.precision_acc,
        "contact": args.precision_acc,
    }
    return speeds, accelerations


def _stationary_step_plan(
    label: str,
    target_pose_mm_deg: list[float],
    end_joint_angles_deg: list[float],
    requested_speed_mm_s: float,
    requested_acceleration_mm_s2: float,
) -> SegmentSpeedPlan:
    """Synthetic plan for a hold step that commands no motion."""
    return SegmentSpeedPlan(
        label=label,
        sample_count=1,
        translation_distance_mm=0.0,
        rotation_distance_deg=0.0,
        requested_speed_mm_s=float(requested_speed_mm_s),
        planned_speed_mm_s=float(requested_speed_mm_s),
        requested_acceleration_mm_s2=float(requested_acceleration_mm_s2),
        planned_acceleration_mm_s2=float(requested_acceleration_mm_s2),
        predicted_peak_joint_speed_deg_s=0.0,
        predicted_peak_joint_acceleration_deg_s2=0.0,
        limiting_velocity_joint=0,
        limiting_acceleration_joint=0,
        end_joint_angles_deg=list(map(float, end_joint_angles_deg)),
    )


def plan_route_motion_profile(
    arm: Any,
    start_pose: list[float],
    start_angles: list[float],
    route: list[tuple[str, list[float], str, str | None]],
    args: Any,
    run_controller_preflight: bool = True,
) -> list[SegmentSpeedPlan]:
    speeds, accelerations = requested_motion_profiles(args)
    requests = [
        CartesianSegmentRequest(
            label=label,
            target_pose_mm_deg=target_pose,
            requested_speed_mm_s=speeds[motion],
            requested_acceleration_mm_s2=accelerations[motion],
        )
        for label, target_pose, motion, _event in route
    ]
    config = JointConstraintConfig(
        joint_count=6,
        velocity_utilization=args.joint_velocity_utilization,
        acceleration_utilization=args.joint_acceleration_utilization,
        max_translation_step_mm=args.ik_translation_step_mm,
        max_rotation_step_deg=args.ik_rotation_step_deg,
    )

    # Stationary hold steps (e.g. the verification hold) share the previous
    # step's pose. The segment planner rejects zero-distance segments, so plan
    # only the motion steps and synthesize zero-distance plans for the holds.
    motion_indices: list[int] = []
    previous_pose: list[float] | None = None
    for index, request in enumerate(requests):
        stationary = previous_pose is not None and np.allclose(
            np.asarray(request.target_pose_mm_deg, dtype=float),
            np.asarray(previous_pose, dtype=float),
            atol=1e-6,
        )
        if not stationary:
            motion_indices.append(index)
        previous_pose = request.target_pose_mm_deg

    motion_requests = [requests[index] for index in motion_indices]
    motion_plans = plan_cartesian_route_speeds(
        arm,
        start_pose,
        start_angles,
        motion_requests,
        joint_velocity_limit_deg_s=float(arm.joint_speed_limit[1]),
        joint_acceleration_limit_deg_s2=float(arm.joint_acc_limit[1]),
        config=config,
    )
    blend_radii = route_blend_radii(route, args.blend_radius_mm)
    if run_controller_preflight:
        preflight_cartesian_route(
            arm,
            motion_requests,
            motion_plans,
            [blend_radii[index] for index in motion_indices],
        )
    else:
        print("WARNING: controller no-motion preflight skipped for this route.")

    running_angles = list(map(float, start_angles))
    motion_plan_by_index = dict(zip(motion_indices, motion_plans))
    estimated_plans: list[SegmentSpeedPlan] = []
    for index, request in enumerate(requests):
        plan = motion_plan_by_index.get(index)
        if plan is not None:
            running_angles = list(map(float, plan.end_joint_angles_deg))
            estimated_plans.append(plan)
        else:
            estimated_plans.append(
                _stationary_step_plan(
                    label=request.label,
                    target_pose_mm_deg=request.target_pose_mm_deg,
                    end_joint_angles_deg=running_angles,
                    requested_speed_mm_s=request.requested_speed_mm_s,
                    requested_acceleration_mm_s2=request.requested_acceleration_mm_s2,
                )
            )
    print(
        "Joint-aware route plan "
        f"(speed target={arm.joint_speed_limit[1] * config.velocity_utilization:.1f} deg/s, "
        f"acc target={arm.joint_acc_limit[1] * config.acceleration_utilization:.1f} deg/s^2, "
        f"controller checks={int(run_controller_preflight)}):"
    )
    for index, plan in enumerate(estimated_plans, start=1):
        stationary_note = " (stationary hold)" if plan.translation_distance_mm == 0.0 else ""
        print(
            f"  {index:02d} {plan.label}: speed {plan.requested_speed_mm_s:.1f} -> "
            f"{plan.planned_speed_mm_s:.1f} mm/s, acc {plan.requested_acceleration_mm_s2:.1f} -> "
            f"{plan.planned_acceleration_mm_s2:.1f} mm/s^2, predicted joint peak "
            f"J{plan.limiting_velocity_joint}={plan.predicted_peak_joint_speed_deg_s:.1f} deg/s"
            f"{stationary_note}"
        )
    return estimated_plans


def execute_route(
    arm: Any,
    vacuum: VacuumController,
    route: list[tuple[str, list[float], str, str | None]],
    args: Any,
    holding_state: dict[str, bool],
    timeline: TaskTimeline,
    cycle_name: str,
    segment_plans: list[SegmentSpeedPlan] | None = None,
    *,
    event_callback: Callable[[str], None] | None = None,
    failure_injected: bool = False,
    prepare_vacuum: bool = True,
) -> None:
    """Drive one planned route with exact-stop discipline and vacuum events.

    Keyword options: ``event_callback`` receives the label of any "verify" step
    and may raise to abort the remaining segments; ``failure_injected`` skips
    pump activation on "acquire" for controlled tests (holding state stays
    unset); ``prepare_vacuum=False`` skips the initial prepare call and is used
    by recovery routes. Steps whose pose equals the previous step's pose are
    stationary holds: no motion is commanded and only the event fires.
    """
    if prepare_vacuum:
        vacuum.prepare_for_pick()
        timeline.mark(f"{cycle_name}.vacuum_path_prepared")
    speeds, accelerations = requested_motion_profiles(args)
    if segment_plans is not None and len(segment_plans) != len(route):
        raise ValueError("Segment plan count does not match route segment count")
    radii = route_blend_radii(route, args.blend_radius_mm)
    previous_target: list[float] | None = None
    for index, (label, target_pose, motion, event) in enumerate(route, start=1):
        segment_plan = segment_plans[index - 1] if segment_plans is not None else None
        if segment_plan is not None and segment_plan.label != label:
            raise ValueError(f"Segment plan label mismatch at route step {index}")
        stationary = previous_target is not None and np.allclose(
            np.asarray(target_pose, dtype=float),
            np.asarray(previous_target, dtype=float),
            atol=1e-6,
        )
        if stationary:
            timeline.mark(
                f"{cycle_name}.move_{index:02d}",
                label=label,
                motion=motion,
                stationary_hold=True,
                waited_for_completion=True,
            )
        else:
            command_speed = segment_plan.planned_speed_mm_s if segment_plan is not None else speeds[motion]
            command_acceleration = (
                segment_plan.planned_acceleration_mm_s2 if segment_plan is not None else accelerations[motion]
            )
            radius = radii[index - 1]
            wait = radius < 0
            result = arm.set_position(
                *target_pose,
                speed=command_speed,
                mvacc=command_acceleration,
                wait=wait,
                radius=radius,
                is_radian=False,
            )
            if wait:
                require_motion_completed(
                    arm,
                    f"move {index:02d}: {label}",
                    result,
                    target_pose,
                )
            else:
                require_code(f"queue blended move {index:02d}: {label}", result)
            timeline.mark(
                f"{cycle_name}.move_{index:02d}",
                label=label,
                motion=motion,
                speed_mm_s=command_speed,
                acceleration_mm_s2=command_acceleration,
                blend_radius_mm=radius,
                waited_for_completion=wait,
            )
        previous_target = list(map(float, target_pose))
        if event == "acquire":
            if failure_injected:
                print("CONTROLLED TEST: pump activation skipped; holding state remains unset.")
                timeline.mark(f"{cycle_name}.vacuum_acquire_skipped", controlled_test=True)
            else:
                vacuum.acquire(args.vacuum_settle_seconds)
                holding_state["value"] = True
                timeline.mark(
                    f"{cycle_name}.vacuum_acquired",
                    settle_seconds=args.vacuum_settle_seconds,
                )
        elif event == "verify":
            if event_callback is None:
                raise RuntimeError(
                    f"Route step {index} requests verification but no event callback was provided"
                )
            event_callback(label)
        elif event == "release":
            vacuum.release(args.release_settle_seconds)
            holding_state["value"] = False
            timeline.mark(
                f"{cycle_name}.vacuum_released",
                settle_seconds=args.release_settle_seconds,
            )
