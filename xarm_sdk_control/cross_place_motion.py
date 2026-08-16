"""Motion planning and execution primitives for the two-shell placement task."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from xarm_sdk_control.robot_motion import (
    CAMERA_TRANSIT_Z_MM,
    require_overhead_camera_safe_route,
)
from xarm_sdk_control.task_timing import TaskTimeline
from xarm_sdk_control.trajectory_speed_planner import (
    CartesianSegmentRequest,
    JointConstraintConfig,
    SegmentSpeedPlan,
    plan_cartesian_route_speeds,
    preflight_cartesian_route,
)
from xarm_sdk_control.vacuum_control import VacuumController
from xarm_sdk_control.visual_targeting import select_leaf_waypoints


MAX_TASK_EMPTY_TRANSIT_SPEED_MM_S = 400.0
MAX_TASK_LOADED_TRANSIT_SPEED_MM_S = 400.0
MAX_TASK_VERTICAL_SPEED_MM_S = 250.0
MAX_TASK_LOADED_VERTICAL_SPEED_MM_S = 250.0
MAX_TASK_PRECISION_SPEED_MM_S = 60.0
MAX_TASK_CONTACT_SPEED_MM_S = 20.0
MAX_TASK_ACCELERATION_MM_S2 = 500.0
MAX_TASK_LOADED_ACCELERATION_MM_S2 = 350.0
MAX_TASK_PRECISION_ACCELERATION_MM_S2 = 200.0
MIN_PICK_PLACE_SEPARATION_MM = 100.0
BLENDABLE_TRANSITIONS = {
    ("empty_transit", "precision"),
    ("precision", "vertical"),
}


def require_code(label: str, result: Any) -> Any:
    code = result[0] if isinstance(result, tuple) else result
    print(f"{label}: {result}")
    if code not in (0, None):
        raise RuntimeError(f"{label} failed with code/result: {result}")
    return result


def require_emergency_stop_released(arm: Any) -> None:
    result = arm.get_err_warn_code()
    require_code("get controller error/warn before motion", result)
    error_code, warn_code = result[1]
    if error_code == 1:
        raise RuntimeError(
            "Controller error C1: the control-box emergency-stop button is pressed. "
            "Twist/release the physical emergency stop, then enable the robot in UFactory before retrying."
        )
    if error_code != 0:
        raise RuntimeError(f"Controller has unresolved error C{error_code} (warning={warn_code})")


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


def offset_detection(item: dict[str, Any], bias_m: np.ndarray) -> dict[str, Any]:
    result = dict(item)
    result["point_base_m"] = (np.asarray(item["point_base_m"], dtype=float) + bias_m).tolist()
    return result


def offset_point_mm(
    point_m: list[float],
    x_mm: float = 0.0,
    y_mm: float = 0.0,
    z_mm: float = 0.0,
) -> list[float]:
    offset_m = np.asarray([x_mm, y_mm, z_mm], dtype=float) / 1000.0
    return (np.asarray(point_m, dtype=float) + offset_m).tolist()


def require_distinct_pick_and_place_targets(
    source_point_m: list[float],
    placement_point_m: list[float],
    minimum_xy_separation_mm: float = MIN_PICK_PLACE_SEPARATION_MM,
) -> None:
    separation_mm = float(
        np.linalg.norm(
            np.asarray(source_point_m, dtype=float)[:2]
            - np.asarray(placement_point_m, dtype=float)[:2]
        )
        * 1000.0
    )
    if separation_mm < minimum_xy_separation_mm:
        raise RuntimeError(
            f"Visual pick and placement targets are only {separation_mm:.1f} mm apart; "
            "the bamboo shell and ceramic disk detections may refer to the same object"
        )


def wrapped_angle_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def pose(point_m: list[float], z_mm: float, orientation: list[float]) -> list[float]:
    point_mm = np.asarray(point_m, dtype=float) * 1000.0
    return [float(point_mm[0]), float(point_mm[1]), float(z_mm), *map(float, orientation)]


def build_cycle_route(
    home_pose: list[float],
    leaf_point_m: list[float],
    ceramic_point_m: list[float],
    place_yaw_offset_deg: float,
    transport_clearance_mm: float,
    approach_clearance_mm: float,
    compression_mm: float,
    release_clearance_mm: float,
    layer_height_mm: float,
    start_pose: list[float] | None = None,
    return_home: bool = True,
    precontact_clearance_mm: float = 5.0,
    precision_lift_mm: float = 20.0,
) -> list[tuple[str, list[float], str, str | None]]:
    cycle_start = list(map(float, start_pose if start_pose is not None else home_pose))
    pickup_orientation = list(map(float, home_pose[3:6]))
    place_orientation = pickup_orientation.copy()
    place_orientation[2] = wrapped_angle_deg(place_orientation[2] + place_yaw_offset_deg)
    leaf_mm = np.asarray(leaf_point_m, dtype=float) * 1000.0
    ceramic_mm = np.asarray(ceramic_point_m, dtype=float) * 1000.0
    transport_z = float(max(leaf_mm[2], ceramic_mm[2]) + transport_clearance_mm)
    if transport_z > CAMERA_TRANSIT_Z_MM:
        raise RuntimeError(
            f"Transport Z {transport_z:.1f} mm exceeds the {CAMERA_TRANSIT_Z_MM:.1f} mm camera-safe corridor"
        )

    route: list[tuple[str, list[float], str, str | None]] = [
        (
            "move directly above source leaf while interpolating pickup yaw",
            pose(leaf_point_m, float(leaf_mm[2] + approach_clearance_mm), pickup_orientation),
            "empty_transit",
            None,
        ),
        (
            "descend slowly to source precontact height",
            pose(leaf_point_m, float(leaf_mm[2] + precontact_clearance_mm), pickup_orientation),
            "precision",
            None,
        ),
        (
            "compress suction cup on source leaf",
            pose(leaf_point_m, float(leaf_mm[2] - compression_mm), pickup_orientation),
            "contact",
            "acquire",
        ),
        (
            "lift leaf slowly until suction is stable",
            pose(leaf_point_m, float(leaf_mm[2] + precision_lift_mm), pickup_orientation),
            "loaded_precision",
            None,
        ),
        (
            "lift leaf vertically with pickup yaw fixed",
            pose(leaf_point_m, transport_z, pickup_orientation),
            "loaded_vertical",
            None,
        ),
        (
            "move above ceramic while interpolating placement yaw",
            pose(
                ceramic_point_m,
                float(ceramic_mm[2] + approach_clearance_mm + layer_height_mm),
                place_orientation,
            ),
            "loaded_transit",
            None,
        ),
        (
            "lower leaf to release height",
            pose(
                ceramic_point_m,
                float(ceramic_mm[2] + release_clearance_mm + layer_height_mm),
                place_orientation,
            ),
            "contact",
            "release",
        ),
        (
            "retreat slowly after releasing leaf",
            pose(
                ceramic_point_m,
                float(ceramic_mm[2] + approach_clearance_mm + layer_height_mm),
                place_orientation,
            ),
            "precision",
            None,
        ),
    ]
    if return_home:
        route.append(
            (
                "return home directly after safe release retreat",
                list(map(float, home_pose)),
                "empty_transit",
                None,
            )
        )
    else:
        route.append(
            (
                "retract vertically above ceramic",
                pose(ceramic_point_m, transport_z, place_orientation),
                "vertical",
                None,
            )
        )
    require_overhead_camera_safe_route([cycle_start, *[item[1] for item in route]])
    return route


def choose_second_source_leaf(
    detections: list[dict[str, Any]],
    first_source_m: list[float],
    allow_clipping: bool,
    max_shift_m: float,
) -> dict[str, Any]:
    candidates, _ = select_leaf_waypoints(
        detections,
        distinct_distance_m=0.05,
        allow_any_clipping=allow_clipping,
    )
    reference_xy = np.asarray(first_source_m, dtype=float)[:2]
    selected = min(
        candidates,
        key=lambda item: float(np.linalg.norm(np.asarray(item["point_base_m"], dtype=float)[:2] - reference_xy)),
    )
    shift_m = float(np.linalg.norm(np.asarray(selected["point_base_m"], dtype=float)[:2] - reference_xy))
    if shift_m > max_shift_m:
        raise RuntimeError(
            f"Second-scan leaf is {shift_m * 1000.0:.1f} mm from the original source stack; "
            "refusing to risk picking the leaf already placed on the ceramic disk"
        )
    return selected


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


def route_blend_radii(
    route: list[tuple[str, list[float], str, str | None]],
    blend_radius_mm: float,
) -> list[float]:
    radii: list[float] = []
    for index, (_label, _target_pose, motion, event) in enumerate(route):
        next_event = route[index + 1][3] if index + 1 < len(route) else None
        next_motion = route[index + 1][2] if index + 1 < len(route) else None
        blendable = (
            blend_radius_mm > 0
            and event is None
            and (
                (next_event == "acquire" and motion in ("empty_transit", "precision"))
                or (motion, next_motion) in BLENDABLE_TRANSITIONS
            )
        )
        radii.append(float(blend_radius_mm if blendable else -1.0))
    return radii


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
    estimated_plans = plan_cartesian_route_speeds(
        arm,
        start_pose,
        start_angles,
        requests,
        joint_velocity_limit_deg_s=float(arm.joint_speed_limit[1]),
        joint_acceleration_limit_deg_s2=float(arm.joint_acc_limit[1]),
        config=config,
    )
    blend_radii = route_blend_radii(route, args.blend_radius_mm)
    if run_controller_preflight:
        preflight_cartesian_route(arm, requests, estimated_plans, blend_radii)
    else:
        print("WARNING: controller no-motion preflight skipped for this route.")
    print(
        "Joint-aware route plan "
        f"(speed target={arm.joint_speed_limit[1] * config.velocity_utilization:.1f} deg/s, "
        f"acc target={arm.joint_acc_limit[1] * config.acceleration_utilization:.1f} deg/s^2, "
        f"controller checks={int(run_controller_preflight)}):"
    )
    for index, plan in enumerate(estimated_plans, start=1):
        print(
            f"  {index:02d} {plan.label}: speed {plan.requested_speed_mm_s:.1f} -> "
            f"{plan.planned_speed_mm_s:.1f} mm/s, acc {plan.requested_acceleration_mm_s2:.1f} -> "
            f"{plan.planned_acceleration_mm_s2:.1f} mm/s^2, predicted joint peak "
            f"J{plan.limiting_velocity_joint}={plan.predicted_peak_joint_speed_deg_s:.1f} deg/s"
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
) -> None:
    vacuum.prepare_for_pick()
    timeline.mark(f"{cycle_name}.vacuum_path_prepared")
    speeds, accelerations = requested_motion_profiles(args)
    if segment_plans is not None and len(segment_plans) != len(route):
        raise ValueError("Segment plan count does not match route segment count")
    radii = route_blend_radii(route, args.blend_radius_mm)
    for index, (label, target_pose, motion, event) in enumerate(route, start=1):
        segment_plan = segment_plans[index - 1] if segment_plans is not None else None
        if segment_plan is not None and segment_plan.label != label:
            raise ValueError(f"Segment plan label mismatch at route step {index}")
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
        if event == "acquire":
            vacuum.acquire(args.vacuum_settle_seconds)
            holding_state["value"] = True
            timeline.mark(
                f"{cycle_name}.vacuum_acquired",
                settle_seconds=args.vacuum_settle_seconds,
            )
        elif event == "release":
            vacuum.release(args.release_settle_seconds)
            holding_state["value"] = False
            timeline.mark(
                f"{cycle_name}.vacuum_released",
                settle_seconds=args.release_settle_seconds,
            )


def robot_status_snapshot(arm: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name, getter in (
        ("state", arm.get_state),
        ("error_warn", arm.get_err_warn_code),
        ("tcp_pose_mm_deg", lambda: arm.get_position(is_radian=False)),
        ("joint_angles_deg", lambda: arm.get_servo_angle(is_radian=False)),
    ):
        try:
            snapshot[name] = getter()
        except Exception as exc:
            snapshot[name] = {"read_error": f"{type(exc).__name__}: {exc}"}
    return snapshot
