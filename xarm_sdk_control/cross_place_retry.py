"""Closed-loop pick verification, recovery, and retry for the two-shell cross-place task."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np

from xarm_sdk_control.cli_utils import route_safety_report
from xarm_sdk_control.cross_place_motion import execute_route, plan_route_motion_profile
from xarm_sdk_control.cross_place_route import (
    build_cycle_route,
    build_pick_recovery_route,
    choose_second_source_leaf,
    offset_detection,
    require_distinct_pick_and_place_targets,
    transport_height_mm,
)
from xarm_sdk_control.pick_verification import (
    PickVerification,
    policy_for,
    verify_carried_object,
)
from xarm_sdk_control.robot_motion import move_robot_home, require_code
from xarm_sdk_control.suction_sensor import SuctionNotConfirmedError, SuctionSensor
from xarm_sdk_control.task_timing import TaskTimeline
from xarm_sdk_control.vacuum_control import VacuumController
from xarm_sdk_control.visual_targeting import run_scan


RetryAction = Literal["proceed", "recover_retry", "abort"]

RetrySelect = Callable[[list[dict[str, Any]], list[float]], dict[str, Any]]

DEFAULT_RETRY_OFFSETS_MM: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, -5.0),
    (5.0, 0.0),
)


def select_nearest_of_kind(
    kind: str,
    detections: list[dict[str, Any]],
    reference_m: list[float],
    max_shift_m: float,
    *,
    display_name: str | None = None,
) -> dict[str, Any]:
    """Best-confidence candidate of one kind within max_shift_m of the reference point.

    Used as a ``retry_select`` for single-kind retry targeting: refuse anything
    that moved far from its original position instead of picking it anyway.
    """
    from xarm_sdk_control.visual_targeting import detection_object_kind

    name = display_name if display_name is not None else kind
    candidates = [
        item
        for item in detections
        if detection_object_kind(item) == kind
        and not bool(item.get("touches_image_edge", False))
    ]
    if not candidates:
        raise RuntimeError(f"No {name} detection near the original position; re-stage or abort")
    reference_xy = np.asarray(reference_m, dtype=float)[:2]
    selected = min(
        candidates,
        key=lambda item: float(
            np.linalg.norm(np.asarray(item["point_base_m"], dtype=float)[:2] - reference_xy)
        ),
    )
    shift_m = float(
        np.linalg.norm(np.asarray(selected["point_base_m"], dtype=float)[:2] - reference_xy)
    )
    if shift_m > max_shift_m:
        raise RuntimeError(
            f"{name.title()} is {shift_m * 1000.0:.1f} mm from the original position; "
            "refusing to pick it after it moved"
        )
    return selected


@dataclass(frozen=True)
class RetryDecision:
    action: RetryAction
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PickNotConfirmedError(RuntimeError):
    """Raised by a verify callback when the retry policy requires recovery or abort."""

    def __init__(self, verification: PickVerification, decision: RetryDecision) -> None:
        self.verification = verification
        self.decision = decision
        super().__init__(
            f"pick verification not confirmed: {verification.status} ({verification.reason}); "
            f"retry policy action: {decision.action} ({decision.reason})"
        )


class AllPickAttemptsFailedError(RuntimeError):
    """Raised by the task script when every attempt for a leaf fails verification."""

    def __init__(self, leaf_index: int, outcome: "LeafPickOutcome") -> None:
        self.leaf_index = leaf_index
        self.outcome = outcome
        super().__init__(f"Leaf {leaf_index} was not confirmed after {len(outcome.attempts)} pick attempt(s)")

    @property
    def attempts_used(self) -> int:
        return len(self.outcome.attempts)


def decide_pick_retry(
    verification: PickVerification | None,
    attempt_number: int,
    max_attempts: int,
) -> RetryDecision:
    """Map a pick verification to the current retry policy.

    Confirmed and inconclusive both proceed for now (the user decision records
    inconclusive results so the policy can be tightened once the visual check
    is reliable). not_confirmed recovers and retries while attempts remain.
    """
    if verification is None:
        return RetryDecision("proceed", "verification unavailable; proceeding per task policy")
    if verification.status == "confirmed":
        return RetryDecision("proceed", "target detected at the lifted tool position")
    if verification.status == "inconclusive":
        return RetryDecision("proceed", "inconclusive verification proceeds for now and is recorded")
    if attempt_number < max_attempts:
        return RetryDecision(
            "recover_retry",
            f"attempt {attempt_number}/{max_attempts} was not confirmed; recovery and retry",
        )
    return RetryDecision(
        "abort",
        f"attempt {attempt_number}/{max_attempts} was not confirmed and no attempts remain",
    )


def decide_pick_retry_on_suction_failure(attempt_number: int, max_attempts: int) -> RetryDecision:
    """Retry policy mirroring the vision not_confirmed branch for pressure-switch failures."""
    if attempt_number < max_attempts:
        return RetryDecision(
            "recover_retry",
            f"attempt {attempt_number}/{max_attempts} had no suction confirmation; recovery and retry",
        )
    return RetryDecision(
        "abort",
        f"attempt {attempt_number}/{max_attempts} had no suction confirmation and no attempts remain",
    )


@dataclass(frozen=True)
class LeafPickOutcome:
    success: bool
    leaf_index: int
    source_point_m: list[float]
    attempts: list[dict[str, Any]]
    final_verification: PickVerification | None

    def cycle_record(self) -> dict[str, Any]:
        """One result.json ``cycles`` entry; existing keys describe the final attempt."""
        final = self.attempts[-1]
        return {
            "leaf": self.leaf_index,
            "completed": self.success,
            "frame_dir": final.get("targeting_frame"),
            "source_point_m": list(map(float, self.source_point_m)),
            "route": final.get("route"),
            "joint_speed_planning_seconds": final.get("joint_speed_planning_seconds"),
            "joint_speed_plan": final.get("joint_speed_plan"),
            "controller_preflight": final.get("controller_preflight"),
            "pick_attempts_used": len(self.attempts),
            "final_pick_verification_status": (
                self.final_verification.status if self.final_verification is not None else None
            ),
            "pick_attempts": self.attempts,
        }


def _capture_verification(
    *,
    capture_root: Path,
    base_t_camera: np.ndarray,
    vision_env: str,
    min_score: float,
    camera: Any,
    vision_socket: Path,
    require_service: bool,
    position_bias_m: np.ndarray,
    source_point_m: list[float],
    expected_object_point_m: list[float],
    allow_clipped_candidate: bool,
    timeline: TaskTimeline,
    timeline_prefix: str,
) -> tuple[Path | None, PickVerification]:
    policy = policy_for("bamboo")
    try:
        frame_dir, detections = run_scan(
            capture_root,
            base_t_camera,
            vision_env,
            min_score,
            camera=camera,
            text_prompt="bamboo leaf.",
            service_socket=vision_socket,
            require_service=require_service,
            timeline=timeline,
            timeline_prefix=timeline_prefix,
        )
        detections = [offset_detection(item, position_bias_m) for item in detections]
        verification = verify_carried_object(
            detections,
            target_kind="bamboo",
            source_point_m=source_point_m,
            expected_object_point_m=expected_object_point_m,
            min_score=min_score,
            minimum_lift_mm=policy.minimum_lift_mm,
            tool_xy_tolerance_mm=policy.tool_xy_tolerance_mm,
            height_tolerance_mm=policy.height_tolerance_mm,
            source_xy_tolerance_mm=policy.source_xy_tolerance_mm,
            source_height_tolerance_mm=policy.source_height_tolerance_mm,
            allow_clipped_candidate=allow_clipped_candidate,
        )
        return frame_dir, verification
    except Exception as exc:
        return None, PickVerification(
            "inconclusive",
            f"verification capture or inference failed: {exc}",
            "bamboo",
            None,
            None,
            None,
        )


def _execute_recovery(
    *,
    arm: Any,
    vacuum: VacuumController,
    source_point_m: list[float],
    transport_z_mm: float,
    home_pose: list[float],
    args: Any,
    holding_state: dict[str, bool],
    timeline: TaskTimeline,
    cycle_name: str,
) -> None:
    pose_code, start_pose = arm.get_position(is_radian=False)
    require_code("get TCP pose before recovery", (pose_code, start_pose))
    angles_code, start_angles = arm.get_servo_angle(is_radian=False)
    require_code("get joint angles before recovery", (angles_code, start_angles))
    route = build_pick_recovery_route(
        source_point_m,
        transport_z_mm,
        args.release_clearance_mm,
        home_pose,
        start_pose,
    )
    plans = plan_route_motion_profile(
        arm,
        start_pose,
        start_angles,
        route,
        args,
        run_controller_preflight=True,
    )
    timeline.mark(f"{cycle_name}.recovery.route_planned", segment_count=len(plans))
    execute_route(
        arm,
        vacuum,
        route,
        args,
        holding_state,
        timeline,
        f"{cycle_name}.recovery",
        plans,
        prepare_vacuum=False,
    )
    final_pose = move_robot_home(arm, home_pose, args.transit_speed, args.acc)
    timeline.mark(f"{cycle_name}.recovery.home_reached", pose_mm_deg=final_pose)


def run_leaf_pick_with_retry(
    *,
    leaf_index: int,
    arm: Any,
    vacuum: VacuumController,
    home_pose: list[float],
    placement_target_m: list[float],
    initial_target_m: list[float],
    source_stack_reference_m: list[float],
    place_yaw_offset_deg: float,
    layer_height_mm: float,
    return_home: bool,
    args: Any,
    holding_state: dict[str, bool],
    timeline: TaskTimeline,
    run_dir: Path,
    base_t_camera: np.ndarray,
    position_bias_m: np.ndarray,
    camera: Any,
    initial_targeting_frame: Path | None = None,
    suction_sensor: SuctionSensor | None = None,
    inspection_event: str | None = "verify",
    retry_scan_prompt: str = "bamboo leaf.",
    retry_select: RetrySelect | None = None,
    compression_mm: float | None = None,
    cycle_prefix: str | None = None,
    retry_offsets_mm: Sequence[tuple[float, float]] = DEFAULT_RETRY_OFFSETS_MM,
    advisory_suction_loss_motions: frozenset[str] = frozenset(),
) -> LeafPickOutcome:
    """Pick one object with verification, recovery, and bounded retries.

    ``inspection_event=None`` runs pressure-only confirmation (no visual verify
    step); ``retry_scan_prompt``/``retry_select`` customize the retry targeting
    scan (default: bamboo with the near-source gate); ``compression_mm``
    overrides ``args.compression_mm`` for callers whose args lack that field.
    """
    verification_score = args.second_scan_min_score if leaf_index == 2 else args.min_score
    attempts: list[dict[str, Any]] = []
    final_verification: PickVerification | None = None
    prefix = cycle_prefix if cycle_prefix is not None else f"leaf_{leaf_index}"
    for attempt_number in range(1, args.max_pick_attempts + 1):
        cycle_name = f"{prefix}.attempt_{attempt_number:02d}"
        attempt_dir = run_dir / f"{prefix}_attempt_{attempt_number:02d}"
        if attempt_number == 1:
            target_m = list(map(float, initial_target_m))
            targeting_frame = initial_targeting_frame
        else:
            retry_frame, retry_detections = run_scan(
                attempt_dir / "targeting" / "captures",
                base_t_camera,
                args.vision_env,
                args.second_scan_min_score,
                camera=camera,
                text_prompt=retry_scan_prompt,
                service_socket=args.vision_socket,
                require_service=args.require_vision_service,
                timeline=timeline,
                timeline_prefix=f"{cycle_name}.targeting",
            )
            targeting_frame = retry_frame
            retry_detections = [offset_detection(item, position_bias_m) for item in retry_detections]
            if retry_select is not None:
                retry_target = retry_select(retry_detections, source_stack_reference_m)
            else:
                retry_target = choose_second_source_leaf(
                    retry_detections,
                    source_stack_reference_m,
                    args.allow_validated_clipped_leaf,
                    args.max_second_source_shift_mm / 1000.0,
                )
            target_m = list(map(float, retry_target["point_base_m"]))
        # Vary the suction point per attempt: re-pressing the exact same spot
        # reproduces the same failed seal. Small offsets probe nearby spots.
        offset_x_mm, offset_y_mm = retry_offsets_mm[(attempt_number - 1) % len(retry_offsets_mm)]
        target_m = [
            target_m[0] + offset_x_mm / 1000.0,
            target_m[1] + offset_y_mm / 1000.0,
            target_m[2],
        ]
        require_distinct_pick_and_place_targets(target_m, placement_target_m)

        pose_code, start_pose = arm.get_position(is_radian=False)
        require_code("get current TCP pose for planning", (pose_code, start_pose))
        transport_z_mm = transport_height_mm(target_m, placement_target_m, args.transport_clearance_mm)
        route = build_cycle_route(
            home_pose,
            target_m,
            placement_target_m,
            place_yaw_offset_deg,
            args.transport_clearance_mm,
            args.approach_clearance_mm,
            args.compression_mm if compression_mm is None else compression_mm,
            args.release_clearance_mm,
            layer_height_mm,
            start_pose=list(map(float, start_pose)),
            return_home=return_home,
            precontact_clearance_mm=args.precontact_clearance_mm,
            precision_lift_mm=args.precision_lift_mm,
            inspection_event=inspection_event,
        )
        safety = route_safety_report([item[1] for item in route], args)
        if not safety["all_ok"]:
            raise RuntimeError("Pick route is outside explicit workspace limits")
        angles_code, start_angles = arm.get_servo_angle(is_radian=False)
        require_code("get joint angles for route planning", (angles_code, start_angles))
        planning_started = time.perf_counter()
        preflight_result: dict[str, Any] = {}
        plans = plan_route_motion_profile(
            arm,
            start_pose,
            start_angles,
            route,
            args,
            run_controller_preflight=True,
            preflight_result=preflight_result,
        )
        planning_seconds = time.perf_counter() - planning_started
        timeline.mark(
            f"{cycle_name}.route_planned",
            planning_seconds=planning_seconds,
            segment_count=len(plans),
        )
        failure_injected = bool(
            getattr(args, "simulate_first_pick_failure", False)
            and leaf_index == 1
            and attempt_number == 1
        )
        attempt_record: dict[str, Any] = {
            "attempt": attempt_number,
            "targeting_frame": str(targeting_frame) if targeting_frame is not None else None,
            "source_point_base_m": target_m,
            "pick_offset_mm": [offset_x_mm, offset_y_mm],
            "route": route,
            "safety": safety,
            "joint_speed_planning_seconds": planning_seconds,
                "joint_speed_plan": [plan.to_dict() for plan in plans],
                "controller_preflight": {
                    "checks_run": 1,
                    "status": preflight_result.get("status"),
                    "error": preflight_result.get("error"),
                    "skipped_by_operator": False,
                },
            "failure_injected": failure_injected,
            "suction_checked": suction_sensor is not None,
            "advisory_suction_loss_motions": sorted(advisory_suction_loss_motions),
        }
        attempts.append(attempt_record)
        if failure_injected:
            print("CONTROLLED TEST: leaf-1 attempt 1 will contact normally but intentionally keep the pump OFF.")

        verification_state: dict[str, Any] = {}

        def verify_step(_label: str) -> None:
            expected_object_point_m = [target_m[0], target_m[1], transport_z_mm / 1000.0]
            frame_dir, verification = _capture_verification(
                capture_root=attempt_dir / "verification" / "captures",
                base_t_camera=base_t_camera,
                vision_env=args.vision_env,
                min_score=verification_score,
                camera=camera,
                vision_socket=args.vision_socket,
                require_service=args.require_vision_service,
                position_bias_m=position_bias_m,
                source_point_m=target_m,
                expected_object_point_m=expected_object_point_m,
                allow_clipped_candidate=args.allow_validated_clipped_leaf,
                timeline=timeline,
                timeline_prefix=f"{cycle_name}.verification",
            )
            verification_state["verification"] = verification
            verification_state["verification_frame"] = str(frame_dir) if frame_dir is not None else None
            attempt_record["verification_frame"] = verification_state["verification_frame"]
            attempt_record["verification"] = verification.to_dict()
            timeline.mark(
                f"{cycle_name}.verification",
                status=verification.status,
                reason=verification.reason,
                observed_lift_mm=verification.observed_lift_mm,
                tool_xy_error_mm=verification.tool_xy_error_mm,
            )
            decision = decide_pick_retry(verification, attempt_number, args.max_pick_attempts)
            attempt_record["retry_decision"] = decision.to_dict()
            print(
                "Pick verification: "
                f"{verification.status} | {verification.reason} | "
                f"observed_lift_mm={verification.observed_lift_mm} | "
                f"tool_xy_error_mm={verification.tool_xy_error_mm} | "
                f"policy={decision.action}"
            )
            if decision.action != "proceed":
                raise PickNotConfirmedError(verification, decision)

        try:
            execute_route(
                arm,
                vacuum,
                route,
                args,
                holding_state,
                timeline,
                cycle_name,
                plans,
                event_callback=verify_step if inspection_event is not None else None,
                failure_injected=failure_injected,
                suction_sensor=suction_sensor,
                advisory_suction_loss_motions=advisory_suction_loss_motions,
            )
        except (PickNotConfirmedError, SuctionNotConfirmedError) as exc:
            if isinstance(exc, PickNotConfirmedError):
                verification = verification_state.get("verification")
                timeline.mark(
                    f"{cycle_name}.verification_failed",
                    status=verification.status if verification is not None else "unknown",
                    reason=verification.reason if verification is not None else "unknown",
                )
                attempt_record["failure_type"] = "verification_not_confirmed"
            else:
                verification = None
                attempt_record["failure_type"] = "suction_not_confirmed"
                attempt_record["suction_failure"] = {
                    "elapsed_seconds": exc.elapsed_seconds,
                    "timeout_seconds": exc.timeout_seconds,
                    "last_value": exc.last_value,
                }
            _execute_recovery(
                arm=arm,
                vacuum=vacuum,
                source_point_m=target_m,
                transport_z_mm=transport_z_mm,
                home_pose=home_pose,
                args=args,
                holding_state=holding_state,
                timeline=timeline,
                cycle_name=cycle_name,
            )
            attempt_record["recovery_completed"] = True
            if attempt_number >= args.max_pick_attempts:
                return LeafPickOutcome(
                    success=False,
                    leaf_index=leaf_index,
                    source_point_m=target_m,
                    attempts=attempts,
                    final_verification=verification,
                )
            continue
        timeline.mark(f"{cycle_name}.cycle_complete")
        final_verification = verification_state.get("verification")
        if suction_sensor is not None and suction_sensor.last_wait is not None:
            attempt_record["suction"] = {
                "confirmed": True,
                "elapsed_seconds": suction_sensor.last_wait.elapsed_seconds,
                "timeout_seconds": suction_sensor.last_wait.timeout_seconds,
            }
        return LeafPickOutcome(
            success=True,
            leaf_index=leaf_index,
            source_point_m=target_m,
            attempts=attempts,
            final_verification=final_verification,
        )
    return LeafPickOutcome(
        success=False,
        leaf_index=leaf_index,
        source_point_m=list(map(float, initial_target_m)),
        attempts=attempts,
        final_verification=final_verification,
    )
