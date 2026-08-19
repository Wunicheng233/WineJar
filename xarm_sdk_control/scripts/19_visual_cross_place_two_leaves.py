#!/usr/bin/env python3
"""Visually pick two stacked bamboo leaves and place them crossed on a ceramic disk."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.positioning_bias import load_bias_mm  # noqa: E402
from xarm_sdk_control.end_effector_config import load_end_effector_config  # noqa: E402
from xarm_sdk_control.realsense_capture import RealSenseCaptureSession  # noqa: E402
from xarm_sdk_control.cli_utils import (  # noqa: E402
    add_motion_profile_args,
    add_vision_common_args,
    add_workspace_limit_args,
    route_safety_report,
)
from xarm_sdk_control.run_output import (  # noqa: E402
    new_timestamped_run_dir,
    write_failure_report,
    write_json_report,
)
from xarm_sdk_control.robot_motion import (  # noqa: E402
    connect_robot,
    load_robot_ip,
    prepare_robot_for_motion,
    reset_robot_at_ip,
)
from xarm_sdk_control.task_timing import TaskTimeline  # noqa: E402
from xarm_sdk_control.cross_place_motion import (  # noqa: E402
    DEFAULT_MOTION_PROFILE,
    build_cycle_route,
    choose_second_source_leaf,
    execute_route,
    offset_detection,
    offset_point_mm,
    plan_route_motion_profile,
    require_code,
    require_distinct_pick_and_place_targets,
    robot_status_snapshot,
    validate_motion_profile,
)
from xarm_sdk_control.cross_place_retry import (  # noqa: E402
    AllPickAttemptsFailedError,
    run_leaf_pick_with_retry,
)
from xarm_sdk_control.suction_sensor import SuctionSensor  # noqa: E402
from xarm_sdk_control.vacuum_control import VacuumController  # noqa: E402
from xarm_sdk_control.visual_targeting import (  # noqa: E402
    DEFAULT_VISION_SOCKET,
    detection_object_kind,
    latest_global_calibration,
    load_base_t_camera,
    mask_principal_axis_deg,
    run_scan,
    select_ceramic,
    select_leaf_waypoints,
    undirected_axis_delta_deg,
)


ROBOT_CONFIG = ROOT / "config" / "robot.yaml"
POSITION_BIAS_CONFIG = ROOT / "config" / "visual_position_bias.json"
OUTPUT_ROOT = ROOT / "data" / "visual_cross_place"
END_EFFECTOR = load_end_effector_config()


def _build_suction_sensor(arm: Any) -> SuctionSensor:
    if END_EFFECTOR.suction_sensor is None:
        raise RuntimeError(
            "Suction confirmation is mandatory: add a suction_sensor section to "
            "config/end_effector.yaml (input_channel, success_value, debounce_seconds, timeout_seconds)."
        )
    return SuctionSensor(arm, END_EFFECTOR.suction_sensor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--skip-second-controller-preflight",
        action="store_true",
        help="Diagnostic only: retain software limits but skip xArm only_check for leaf 2",
    )
    parser.add_argument("--allow-validated-clipped-leaf", action="store_true")
    add_vision_common_args(parser, latest_global_calibration())
    parser.add_argument("--vision-socket", type=Path, default=DEFAULT_VISION_SOCKET)
    parser.add_argument("--require-vision-service", action="store_true")
    parser.add_argument("--record-global-camera", action="store_true")
    parser.add_argument(
        "--recording-fps",
        type=int,
        choices=(15, 30),
        default=15,
        help="RealSense stream and video FPS while --record-global-camera is active",
    )
    parser.add_argument(
        "--second-scan-min-score",
        type=float,
        default=0.10,
        help="Lower confidence accepted only near the previously observed source stack; the remaining shell can score below 0.20",
    )
    parser.add_argument("--tcp-offset-z-mm", type=float, default=END_EFFECTOR.tcp_offset_mm_deg[2])
    parser.add_argument("--first-place-yaw-offset-deg", type=float, default=45.0)
    parser.add_argument("--second-place-yaw-offset-deg", type=float, default=-60.0)
    parser.add_argument("--place-offset-y-mm", type=float, default=-10.0)
    parser.add_argument("--second-layer-height-mm", type=float, default=3.0)
    parser.add_argument("--max-second-source-shift-mm", type=float, default=80.0)
    parser.add_argument("--transport-clearance-mm", type=float, default=100.0)
    parser.add_argument("--approach-clearance-mm", type=float, default=25.0)
    parser.add_argument("--precontact-clearance-mm", type=float, default=5.0)
    parser.add_argument("--precision-lift-mm", type=float, default=20.0)
    parser.add_argument("--compression-mm", type=float, default=END_EFFECTOR.default_compression_mm)
    parser.add_argument("--release-clearance-mm", type=float, default=0.0)
    parser.add_argument(
        "--verify-pick",
        action="store_true",
        help="Classify every lift from a fresh RGB-D frame; on failure vent at the "
             "source stack, replan from a new targeting frame, and retry.",
    )
    parser.add_argument(
        "--max-pick-attempts",
        type=int,
        default=3,
        help="Pick attempts per leaf including retries (1-3).",
    )
    parser.add_argument(
        "--simulate-first-pick-failure",
        action="store_true",
        help="Controlled test: the pump stays OFF on leaf-1 attempt 1, so the CI0 pressure "
             "switch reports no suction - a real deterministic failure that exercises "
             "recovery + replan + retry.",
    )
    add_motion_profile_args(parser, DEFAULT_MOTION_PROFILE)
    add_workspace_limit_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timeline = TaskTimeline()
    timeline.mark("program_started")
    limits = (args.min_x_mm, args.max_x_mm, args.min_y_mm, args.max_y_mm, args.min_z_mm, args.max_z_mm)
    if args.run and any(value is None for value in limits):
        print("ERROR: --run requires explicit min/max X, Y, and Z limits")
        return 3
    if args.calibration is None or not args.calibration.expanduser().exists():
        raise RuntimeError("Global camera calibration is missing")
    if not 0 <= args.min_score <= 1 or not 0 <= args.second_scan_min_score <= 1:
        raise ValueError("Vision confidence thresholds must remain within 0~1")
    if not 0 <= args.compression_mm <= 20:
        raise ValueError("Compression must remain within the stated 0~20 mm cup travel")
    if args.vacuum_settle_seconds < 0 or args.release_settle_seconds < 0:
        raise ValueError("Vacuum and release settle times must be non-negative")
    if not 0 <= args.blend_radius_mm <= 50:
        raise ValueError("Blend radius must remain within 0~50 mm for this task")
    if not 0 <= args.precontact_clearance_mm < args.approach_clearance_mm:
        raise ValueError("Precontact clearance must be non-negative and below approach clearance")
    if args.precision_lift_mm <= args.precontact_clearance_mm:
        raise ValueError("Precision lift height must exceed precontact clearance")
    validate_motion_profile(args)
    if not 1 <= args.max_pick_attempts <= 3:
        raise ValueError("--max-pick-attempts must be within 1~3")
    if args.verify_pick and not args.run:
        raise ValueError("--verify-pick requires --run")
    if args.simulate_first_pick_failure and args.max_pick_attempts < 2:
        raise ValueError("Simulated first-pick failure requires --max-pick-attempts 2 or greater")
    if args.verify_pick and not args.require_vision_service:
        print(
            "WARNING: --verify-pick without --require-vision-service falls back to one-shot "
            "inference for every targeting and verification capture; the persistent vision "
            "service is strongly recommended."
        )

    calibration_path, base_t_camera = load_base_t_camera(args.calibration)
    position_bias_m = load_bias_mm(POSITION_BIAS_CONFIG) / 1000.0
    robot_ip = load_robot_ip(ROBOT_CONFIG)
    camera_fps = args.recording_fps if args.record_global_camera else 30
    camera = RealSenseCaptureSession(serial=args.serial, fps=camera_fps)
    camera.start()
    timeline.mark("camera_session_ready")
    try:
        if args.run:
            print("Task step 1: moving to the configured camera-clear reset pose.")
            timeline.mark("reset_started")
            reset_robot_at_ip(
                robot_ip,
                [0, 0, args.tcp_offset_z_mm, 0, 0, 0],
                END_EFFECTOR.home_pose_mm_deg,
                args.transit_speed,
                args.acc,
            )
            timeline.mark("reset_complete")
        else:
            timeline.mark("reset_skipped")
        timeline.mark("productive_cycle_started")
        run_dir = new_timestamped_run_dir(OUTPUT_ROOT)
        global_camera_video = run_dir / "global_camera.mp4" if args.record_global_camera else None
        if global_camera_video is not None:
            camera.start_recording(global_camera_video)
            timeline.mark("global_camera_recording_started", path=str(global_camera_video))

        first_frame, first_detections = run_scan(
            run_dir / "scan_1" / "captures",
            base_t_camera,
            args.vision_env,
            args.min_score,
            camera=camera,
            service_socket=args.vision_socket,
            require_service=args.require_vision_service,
            timeline=timeline,
            timeline_prefix="scan_1",
        )
        ceramic = offset_detection(select_ceramic(first_detections), position_bias_m)
        placement_target_m = offset_point_mm(
            ceramic["point_base_m"],
            y_mm=args.place_offset_y_mm,
        )
        first_leaf = offset_detection(
            select_leaf_waypoints(
                first_detections,
                distinct_distance_m=0.05,
                allow_any_clipping=args.allow_validated_clipped_leaf,
            )[0][0],
            position_bias_m,
        )
        require_distinct_pick_and_place_targets(first_leaf["point_base_m"], placement_target_m)
        first_source_axis_deg, first_source_elongation = mask_principal_axis_deg(first_frame, first_leaf)
    except BaseException:
        camera.stop(raise_recording_error=False)
        raise

    arm: Any | None = None
    vacuum: VacuumController | None = None
    controller_ready = False
    holding_state = {"value": False}
    records: list[dict[str, Any]] = []
    try:
        arm, home_pose = connect_robot(robot_ip, [0, 0, args.tcp_offset_z_mm, 0, 0, 0])

        print(f"System positioning bias [mm]: {(position_bias_m * 1000).round(2).tolist()}")
        print(f"First source [mm]: {(np.asarray(first_leaf['point_base_m']) * 1000).round(2).tolist()}")
        print(f"Detected ceramic center [mm]: {(np.asarray(ceramic['point_base_m']) * 1000).round(2).tolist()}")
        print(
            "Placement target with task offset [mm]: "
            f"{(np.asarray(placement_target_m) * 1000).round(2).tolist()} "
            f"(Y offset={args.place_offset_y_mm:.1f} mm)"
        )

        if not args.verify_pick:
            if not args.run:
                first_route = build_cycle_route(
                    home_pose,
                    first_leaf["point_base_m"],
                    placement_target_m,
                    args.first_place_yaw_offset_deg,
                    args.transport_clearance_mm,
                    args.approach_clearance_mm,
                    args.compression_mm,
                    args.release_clearance_mm,
                    0.0,
                    return_home=False,
                    precontact_clearance_mm=args.precontact_clearance_mm,
                    precision_lift_mm=args.precision_lift_mm,
                )
                first_safety = route_safety_report([item[1] for item in first_route], args)
                if not first_safety["all_ok"]:
                    raise RuntimeError("First-leaf route is outside explicit workspace limits")
                print("Preparing controller for no-motion route preflight.")
                prepare_robot_for_motion(arm)
                controller_ready = True

                angles_code, first_start_angles = arm.get_servo_angle(is_radian=False)
                require_code("get joint angles for first-route planning", (angles_code, first_start_angles))
                planning_started = time.perf_counter()
                first_plans = plan_route_motion_profile(
                    arm,
                    home_pose,
                    first_start_angles,
                    first_route,
                    args,
                )
                first_planning_seconds = time.perf_counter() - planning_started
                timeline.mark(
                    "leaf_1.route_planned",
                    planning_seconds=first_planning_seconds,
                    segment_count=len(first_plans),
                )
                require_code("stop robot after no-motion preflight", arm.set_state(4))
                controller_ready = False
                print(
                    "DRY RUN: first scan, workspace limits, IK continuity, and joint limits validated; "
                    "second scan requires the first physical placement."
                )
                return 0

            prepare_robot_for_motion(arm, clean_error=True)
            controller_ready = True

            vacuum = VacuumController(arm, END_EFFECTOR.vacuum_io)
            vacuum.configure()
            sensor = _build_suction_sensor(arm)

            leaf_1_outcome = run_leaf_pick_with_retry(
                leaf_index=1,
                arm=arm,
                vacuum=vacuum,
                home_pose=home_pose,
                placement_target_m=placement_target_m,
                initial_target_m=first_leaf["point_base_m"],
                source_stack_reference_m=first_leaf["point_base_m"],
                place_yaw_offset_deg=args.first_place_yaw_offset_deg,
                layer_height_mm=0.0,
                return_home=False,
                args=args,
                holding_state=holding_state,
                timeline=timeline,
                run_dir=run_dir,
                base_t_camera=base_t_camera,
                position_bias_m=position_bias_m,
                camera=camera,
                initial_targeting_frame=first_frame,
                suction_sensor=sensor,
                inspection_event=None,
            )
            records.append(leaf_1_outcome.cycle_record())
            if not leaf_1_outcome.success:
                raise AllPickAttemptsFailedError(1, leaf_1_outcome)
            timeline.mark(
                "leaf_1.pick_complete",
                final_status=(
                    leaf_1_outcome.final_verification.status
                    if leaf_1_outcome.final_verification is not None
                    else None
                ),
                attempts_used=len(leaf_1_outcome.attempts),
            )
        else:
            prepare_robot_for_motion(arm, clean_error=True)
            controller_ready = True

            vacuum = VacuumController(arm, END_EFFECTOR.vacuum_io)
            vacuum.configure()
            sensor = _build_suction_sensor(arm)

            leaf_1_outcome = run_leaf_pick_with_retry(
                leaf_index=1,
                arm=arm,
                vacuum=vacuum,
                home_pose=home_pose,
                placement_target_m=placement_target_m,
                initial_target_m=first_leaf["point_base_m"],
                source_stack_reference_m=first_leaf["point_base_m"],
                place_yaw_offset_deg=args.first_place_yaw_offset_deg,
                layer_height_mm=0.0,
                return_home=False,
                args=args,
                holding_state=holding_state,
                timeline=timeline,
                run_dir=run_dir,
                base_t_camera=base_t_camera,
                position_bias_m=position_bias_m,
                camera=camera,
                initial_targeting_frame=first_frame,
                suction_sensor=sensor,
            )
            records.append(leaf_1_outcome.cycle_record())
            if not leaf_1_outcome.success:
                raise AllPickAttemptsFailedError(1, leaf_1_outcome)
            timeline.mark(
                "leaf_1.pick_complete",
                final_status=(
                    leaf_1_outcome.final_verification.status
                    if leaf_1_outcome.final_verification is not None
                    else None
                ),
                attempts_used=len(leaf_1_outcome.attempts),
            )

        second_frame, second_detections = run_scan(
            run_dir / "scan_2" / "captures",
            base_t_camera,
            args.vision_env,
            args.second_scan_min_score,
            camera=camera,
            text_prompt="bamboo leaf.",
            service_socket=args.vision_socket,
            require_service=args.require_vision_service,
            timeline=timeline,
            timeline_prefix="scan_2",
        )
        second_detections = [offset_detection(item, position_bias_m) for item in second_detections]
        second_leaf = choose_second_source_leaf(
            second_detections,
            first_leaf["point_base_m"],
            args.allow_validated_clipped_leaf,
            args.max_second_source_shift_mm / 1000.0,
        )
        require_distinct_pick_and_place_targets(second_leaf["point_base_m"], placement_target_m)
        print(f"Second source [mm]: {(np.asarray(second_leaf['point_base_m']) * 1000).round(2).tolist()}")
        orientation_diagnostic: dict[str, Any] = {
            "source_axis_image_deg": first_source_axis_deg,
            "source_elongation_ratio": first_source_elongation,
        }
        try:
            placed_candidates = [
                item
                for item in second_detections
                if detection_object_kind(item) == "bamboo"
                and item.get("mask_path") != second_leaf.get("mask_path")
            ]
            placed_first = min(
                placed_candidates,
                key=lambda item: float(
                    np.linalg.norm(
                        np.asarray(item["point_base_m"], dtype=float)[:2]
                        - np.asarray(placement_target_m, dtype=float)[:2]
                    )
                ),
            )
            placed_axis_deg, placed_elongation = mask_principal_axis_deg(second_frame, placed_first)
            actual_rotation_deg = undirected_axis_delta_deg(first_source_axis_deg, placed_axis_deg)
            orientation_diagnostic.update(
                {
                    "placed_axis_image_deg": placed_axis_deg,
                    "placed_elongation_ratio": placed_elongation,
                    "observed_first_leaf_rotation_deg": actual_rotation_deg,
                    "commanded_first_tcp_yaw_delta_deg": args.first_place_yaw_offset_deg,
                }
            )
            print(
                "First-leaf orientation diagnostic: "
                f"source axis={first_source_axis_deg:.1f} deg, "
                f"placed axis={placed_axis_deg:.1f} deg, "
                f"observed rotation={actual_rotation_deg:.1f} deg, "
                f"commanded TCP yaw={args.first_place_yaw_offset_deg:.1f} deg"
            )
        except Exception as exc:
            orientation_diagnostic["error"] = f"{type(exc).__name__}: {exc}"
            print(f"WARNING: first-leaf orientation diagnostic unavailable: {exc}")
        if args.verify_pick:
            leaf_2_outcome = run_leaf_pick_with_retry(
                leaf_index=2,
                arm=arm,
                vacuum=vacuum,
                home_pose=home_pose,
                placement_target_m=placement_target_m,
                initial_target_m=second_leaf["point_base_m"],
                source_stack_reference_m=first_leaf["point_base_m"],
                place_yaw_offset_deg=args.second_place_yaw_offset_deg,
                layer_height_mm=args.second_layer_height_mm,
                return_home=True,
                args=args,
                holding_state=holding_state,
                timeline=timeline,
                run_dir=run_dir,
                base_t_camera=base_t_camera,
                position_bias_m=position_bias_m,
                camera=camera,
                initial_targeting_frame=second_frame,
                suction_sensor=sensor,
            )
        else:
            leaf_2_outcome = run_leaf_pick_with_retry(
                leaf_index=2,
                arm=arm,
                vacuum=vacuum,
                home_pose=home_pose,
                placement_target_m=placement_target_m,
                initial_target_m=second_leaf["point_base_m"],
                source_stack_reference_m=first_leaf["point_base_m"],
                place_yaw_offset_deg=args.second_place_yaw_offset_deg,
                layer_height_mm=args.second_layer_height_mm,
                return_home=True,
                args=args,
                holding_state=holding_state,
                timeline=timeline,
                run_dir=run_dir,
                base_t_camera=base_t_camera,
                position_bias_m=position_bias_m,
                camera=camera,
                initial_targeting_frame=second_frame,
                suction_sensor=sensor,
                inspection_event=None,
            )
        records.append(leaf_2_outcome.cycle_record())
        if not leaf_2_outcome.success:
            raise AllPickAttemptsFailedError(2, leaf_2_outcome)
        timeline.mark(
            "leaf_2.pick_complete",
            final_status=(
                leaf_2_outcome.final_verification.status
                if leaf_2_outcome.final_verification is not None
                else None
            ),
            attempts_used=len(leaf_2_outcome.attempts),
        )
        vacuum.safe_shutdown()
        if global_camera_video is not None:
            camera.stop_recording()
            timeline.mark("global_camera_recording_complete", path=str(global_camera_video))
        timeline.mark("task_complete")
        timing = timeline.to_dict(productive_start="productive_cycle_started")

        result_path = write_json_report(
            run_dir,
            {
                    "status": "success",
                    "calibration": str(calibration_path),
                    "pick_verification": {
                        "enabled": bool(args.verify_pick),
                        "max_pick_attempts": args.max_pick_attempts,
                        "simulate_first_pick_failure": bool(args.simulate_first_pick_failure),
                        "require_vision_service": bool(args.require_vision_service),
                    },
                    "position_bias_base_mm": (position_bias_m * 1000).tolist(),
                    "ceramic_target": ceramic,
                    "placement_offset_base_mm": [0.0, args.place_offset_y_mm, 0.0],
                    "placement_target_base_m": placement_target_m,
                    "global_camera_video": str(global_camera_video) if global_camera_video else None,
                    "performance_parameters": {
                        "transport_clearance_mm": args.transport_clearance_mm,
                        "approach_clearance_mm": args.approach_clearance_mm,
                        "precontact_clearance_mm": args.precontact_clearance_mm,
                        "precision_lift_mm": args.precision_lift_mm,
                        "transit_speed_mm_s": args.transit_speed,
                        "loaded_transit_speed_mm_s": args.loaded_transit_speed,
                        "vertical_speed_mm_s": args.vertical_speed,
                        "loaded_vertical_speed_mm_s": args.loaded_vertical_speed,
                        "precision_speed_mm_s": args.precision_speed,
                        "contact_speed_mm_s": args.contact_speed,
                        "acceleration_mm_s2": args.acc,
                        "loaded_acceleration_mm_s2": args.loaded_acc,
                        "precision_acceleration_mm_s2": args.precision_acc,
                        "blend_radius_mm": args.blend_radius_mm,
                        "joint_velocity_utilization": args.joint_velocity_utilization,
                        "joint_acceleration_utilization": args.joint_acceleration_utilization,
                        "ik_translation_step_mm": args.ik_translation_step_mm,
                        "ik_rotation_step_deg": args.ik_rotation_step_deg,
                        "vacuum_settle_seconds": args.vacuum_settle_seconds,
                        "release_settle_seconds": args.release_settle_seconds,
                        "first_scan_min_score": args.min_score,
                        "second_scan_min_score": args.second_scan_min_score,
                    },
                    "timing": timing,
                    "orientation_diagnostic": orientation_diagnostic,
                    "cycles": records,
            },
        )
        print(
            f"Productive cycle: {timing['productive_cycle_seconds']:.3f}s; "
            f"total including reset: {timing['total_seconds']:.3f}s"
        )
        print(f"Saved result: {result_path}")
        return 0
    except BaseException as exc:
        timeline.mark("task_failed", error=f"{type(exc).__name__}: {exc}")
        if args.run or controller_ready:
            try:
                require_code("stop robot after task interruption", arm.set_state(4))
            except Exception as stop_exc:
                print(f"WARNING: failed to confirm robot stop: {stop_exc}", file=sys.stderr)
        if holding_state["value"]:
            print(
                "EMERGENCY NOTE: vacuum may still be holding a leaf. Stop the robot and make the area safe "
                "before manually releasing it.",
                file=sys.stderr,
            )
        elif vacuum is not None:
            vacuum.safe_shutdown()
        extra: dict[str, Any] | None = None
        if isinstance(exc, AllPickAttemptsFailedError):
            final_status = (
                exc.outcome.final_verification.status
                if exc.outcome.final_verification is not None
                else None
            )
            final_reason = (
                exc.outcome.final_verification.reason
                if exc.outcome.final_verification is not None
                else None
            )
            extra = {
                "pick_verification_failed": {
                    "leaf": exc.leaf_index,
                    "attempts_used": exc.attempts_used,
                    "final_status": final_status,
                    "final_reason": final_reason,
                }
            }
            result_path = write_json_report(
                run_dir,
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "calibration": str(calibration_path),
                    "pick_verification": {
                        "enabled": bool(args.verify_pick),
                        "max_pick_attempts": args.max_pick_attempts,
                        "simulate_first_pick_failure": bool(args.simulate_first_pick_failure),
                        "require_vision_service": bool(args.require_vision_service),
                    },
                    "timing": timeline.to_dict(productive_start="productive_cycle_started"),
                    "cycles": records,
                },
                name="result.json",
            )
            print(f"Saved result: {result_path}", file=sys.stderr)
        failure_path = write_failure_report(
            run_dir,
            exc,
            holding=holding_state["value"],
            timeline=timeline,
            productive_start="productive_cycle_started",
            completed_records=records,
            robot_snapshot=robot_status_snapshot(arm) if arm is not None else None,
            extra=extra,
        )
        print(f"Saved failure report: {failure_path}", file=sys.stderr)
        if isinstance(exc, AllPickAttemptsFailedError):
            return 2
        raise
    finally:
        if arm is not None:
            arm.disconnect()
        camera.stop(raise_recording_error=False)


if __name__ == "__main__":
    raise SystemExit(main())
