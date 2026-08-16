#!/usr/bin/env python3
"""Visually pick two stacked bamboo leaves and place them crossed on a ceramic disk."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.positioning_bias import load_bias_mm  # noqa: E402
from xarm_sdk_control.end_effector_config import load_end_effector_config  # noqa: E402
from xarm_sdk_control.realsense_capture import RealSenseCaptureSession  # noqa: E402
from xarm_sdk_control.robot_motion import prepare_robot_for_motion, reset_robot_at_ip  # noqa: E402
from xarm_sdk_control.task_timing import TaskTimeline  # noqa: E402
from xarm_sdk_control.cross_place_motion import (  # noqa: E402
    build_cycle_route,
    choose_second_source_leaf,
    execute_route,
    offset_detection,
    offset_point_mm,
    plan_route_motion_profile,
    require_code,
    require_distinct_pick_and_place_targets,
    require_emergency_stop_released,
    robot_status_snapshot,
    validate_motion_profile,
)
from xarm_sdk_control.vacuum_control import VacuumController  # noqa: E402
from xarm_sdk_control.visual_targeting import (  # noqa: E402
    DEFAULT_VISION_SOCKET,
    detection_object_kind,
    detections_in_base,
    latest_global_calibration,
    mask_principal_axis_deg,
    run_vision,
    safety_report,
    select_ceramic,
    select_leaf_waypoints,
    undirected_axis_delta_deg,
)


ROBOT_CONFIG = ROOT / "config" / "robot.yaml"
POSITION_BIAS_CONFIG = ROOT / "config" / "visual_position_bias.json"
OUTPUT_ROOT = ROOT / "data" / "visual_cross_place"
END_EFFECTOR = load_end_effector_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--skip-second-controller-preflight",
        action="store_true",
        help="Diagnostic only: retain software limits but skip xArm only_check for leaf 2",
    )
    parser.add_argument("--confirm-no-pressure-sensor", action="store_true")
    parser.add_argument("--allow-validated-clipped-leaf", action="store_true")
    parser.add_argument("--calibration", type=Path, default=latest_global_calibration())
    parser.add_argument("--vision-env", default="winejar-vision")
    parser.add_argument("--vision-socket", type=Path, default=DEFAULT_VISION_SOCKET)
    parser.add_argument("--require-vision-service", action="store_true")
    parser.add_argument("--serial")
    parser.add_argument("--record-global-camera", action="store_true")
    parser.add_argument(
        "--recording-fps",
        type=int,
        choices=(15, 30),
        default=15,
        help="RealSense stream and video FPS while --record-global-camera is active",
    )
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument(
        "--second-scan-min-score",
        type=float,
        default=0.20,
        help="Lower confidence accepted only near the previously observed source stack",
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
    parser.add_argument("--vacuum-settle-seconds", type=float, default=0.9)
    parser.add_argument("--release-settle-seconds", type=float, default=0.3)
    parser.add_argument("--transit-speed", type=float, default=300.0)
    parser.add_argument("--loaded-transit-speed", type=float, default=220.0)
    parser.add_argument("--vertical-speed", type=float, default=200.0)
    parser.add_argument("--loaded-vertical-speed", type=float, default=120.0)
    parser.add_argument("--precision-speed", type=float, default=60.0)
    parser.add_argument("--contact-speed", type=float, default=20.0)
    parser.add_argument("--acc", type=float, default=500.0)
    parser.add_argument("--loaded-acc", type=float, default=250.0)
    parser.add_argument("--precision-acc", type=float, default=200.0)
    parser.add_argument("--blend-radius-mm", type=float, default=10.0)
    parser.add_argument("--joint-velocity-utilization", type=float, default=0.40)
    parser.add_argument("--joint-acceleration-utilization", type=float, default=0.40)
    parser.add_argument("--ik-translation-step-mm", type=float, default=10.0)
    parser.add_argument("--ik-rotation-step-deg", type=float, default=2.0)
    parser.add_argument("--min-x-mm", type=float)
    parser.add_argument("--max-x-mm", type=float)
    parser.add_argument("--min-y-mm", type=float)
    parser.add_argument("--max-y-mm", type=float)
    parser.add_argument("--min-z-mm", type=float)
    parser.add_argument("--max-z-mm", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timeline = TaskTimeline()
    timeline.mark("program_started")
    if args.run and not args.confirm_no_pressure_sensor:
        print("ERROR: --run requires --confirm-no-pressure-sensor")
        return 2
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

    calibration_path = args.calibration.expanduser().resolve()
    base_t_camera = np.asarray(json.loads(calibration_path.read_text())["base_T_camera"], dtype=float)
    position_bias_m = load_bias_mm(POSITION_BIAS_CONFIG) / 1000.0
    with ROBOT_CONFIG.open("r", encoding="utf-8") as file:
        robot_ip = str(yaml.safe_load(file)["robot_ip"])
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
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        run_dir = OUTPUT_ROOT / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)
        global_camera_video = run_dir / "global_camera.mp4" if args.record_global_camera else None
        if global_camera_video is not None:
            camera.start_recording(global_camera_video)
            timeline.mark("global_camera_recording_started", path=str(global_camera_video))

        timeline.mark("scan_1.capture_started")
        first_frame = camera.capture(run_dir / "scan_1" / "captures")
        timeline.mark("scan_1.capture_complete", frame_dir=str(first_frame))
        first_vision = run_vision(
            first_frame,
            args.vision_env,
            args.min_score,
            service_socket=args.vision_socket,
            require_service=args.require_vision_service,
        )
        timeline.mark(
            "scan_1.vision_complete",
            inference_timing_seconds=first_vision.get("timing_seconds", {}),
        )
        first_detections = detections_in_base(first_vision.get("detections", []), base_t_camera, args.min_score)
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

    from xarm.wrapper import XArmAPI

    arm = XArmAPI(do_not_open=True, is_radian=False)
    vacuum: VacuumController | None = None
    controller_ready = False
    holding_state = {"value": False}
    records: list[dict[str, Any]] = []
    try:
        require_code("connect", arm.connect(port=robot_ip))
        require_code(
            "set suction-cup TCP offset",
            arm.set_tcp_offset([0, 0, args.tcp_offset_z_mm, 0, 0, 0], is_radian=False, wait=True),
        )
        pose_code, home_pose = arm.get_position(is_radian=False)
        require_code("get camera-clear home pose", (pose_code, home_pose))

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
        print(f"System positioning bias [mm]: {(position_bias_m * 1000).round(2).tolist()}")
        print(f"First source [mm]: {(np.asarray(first_leaf['point_base_m']) * 1000).round(2).tolist()}")
        print(f"Detected ceramic center [mm]: {(np.asarray(ceramic['point_base_m']) * 1000).round(2).tolist()}")
        print(
            "Placement target with task offset [mm]: "
            f"{(np.asarray(placement_target_m) * 1000).round(2).tolist()} "
            f"(Y offset={args.place_offset_y_mm:.1f} mm)"
        )

        bound_values = {
            "min_x_mm": args.min_x_mm if args.min_x_mm is not None else -1000.0,
            "max_x_mm": args.max_x_mm if args.max_x_mm is not None else 1000.0,
            "min_y_mm": args.min_y_mm if args.min_y_mm is not None else -1000.0,
            "max_y_mm": args.max_y_mm if args.max_y_mm is not None else 1000.0,
            "min_z_mm": args.min_z_mm if args.min_z_mm is not None else -1000.0,
            "max_z_mm": args.max_z_mm if args.max_z_mm is not None else 1000.0,
        }
        first_safety = safety_report([item[1] for item in first_route], **bound_values)
        if not first_safety["all_ok"]:
            raise RuntimeError("First-leaf route is outside explicit workspace limits")

        if args.run:
            require_emergency_stop_released(arm)
            require_code("clean error", arm.clean_error())
            require_code("clean warn", arm.clean_warn())
            require_code("motion enable", arm.motion_enable(enable=True))
            require_code("set mode 0", arm.set_mode(0))
            require_code("set state 0", arm.set_state(0))
        else:
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

        if not args.run:
            require_code("stop robot after no-motion preflight", arm.set_state(4))
            controller_ready = False
            print(
                "DRY RUN: first scan, workspace limits, IK continuity, and joint limits validated; "
                "second scan requires the first physical placement."
            )
            return 0

        vacuum = VacuumController(arm, END_EFFECTOR.vacuum_io)
        vacuum.configure()

        execute_route(
            arm,
            vacuum,
            first_route,
            args,
            holding_state,
            timeline,
            "leaf_1",
            first_plans,
        )
        timeline.mark("leaf_1.cycle_complete")
        records.append(
            {
                "leaf": 1,
                "frame_dir": str(first_frame),
                "route": first_route,
                "joint_speed_planning_seconds": first_planning_seconds,
                "joint_speed_plan": [plan.to_dict() for plan in first_plans],
                "controller_preflight": {"checks_run": 1, "passed": True},
            }
        )

        pose_code, second_start_pose = arm.get_position(is_radian=False)
        require_code("get continuous-cycle waiting pose", (pose_code, second_start_pose))
        angles_code, second_start_angles = arm.get_servo_angle(is_radian=False)
        require_code("get joint angles for second-route planning", (angles_code, second_start_angles))

        timeline.mark("scan_2.capture_started")
        second_frame = camera.capture(run_dir / "scan_2" / "captures")
        timeline.mark("scan_2.capture_complete", frame_dir=str(second_frame))
        second_vision = run_vision(
            second_frame,
            args.vision_env,
            args.second_scan_min_score,
            text_prompt="bamboo leaf.",
            service_socket=args.vision_socket,
            require_service=args.require_vision_service,
        )
        timeline.mark(
            "scan_2.vision_complete",
            inference_timing_seconds=second_vision.get("timing_seconds", {}),
        )
        second_detections = detections_in_base(
            second_vision.get("detections", []),
            base_t_camera,
            args.second_scan_min_score,
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
        second_route = build_cycle_route(
            home_pose,
            second_leaf["point_base_m"],
            placement_target_m,
            args.second_place_yaw_offset_deg,
            args.transport_clearance_mm,
            args.approach_clearance_mm,
            args.compression_mm,
            args.release_clearance_mm,
            args.second_layer_height_mm,
            start_pose=second_start_pose,
            return_home=True,
            precontact_clearance_mm=args.precontact_clearance_mm,
            precision_lift_mm=args.precision_lift_mm,
        )
        second_safety = safety_report([item[1] for item in second_route], **bound_values)
        if not second_safety["all_ok"]:
            raise RuntimeError("Second-leaf route is outside explicit workspace limits")
        planning_started = time.perf_counter()
        second_plans = plan_route_motion_profile(
            arm,
            second_start_pose,
            second_start_angles,
            second_route,
            args,
            run_controller_preflight=not args.skip_second_controller_preflight,
        )
        second_planning_seconds = time.perf_counter() - planning_started
        timeline.mark(
            "leaf_2.route_planned",
            planning_seconds=second_planning_seconds,
            segment_count=len(second_plans),
        )
        execute_route(
            arm,
            vacuum,
            second_route,
            args,
            holding_state,
            timeline,
            "leaf_2",
            second_plans,
        )
        timeline.mark("leaf_2.cycle_complete")
        records.append(
            {
                "leaf": 2,
                "frame_dir": str(second_frame),
                "route": second_route,
                "joint_speed_planning_seconds": second_planning_seconds,
                "joint_speed_plan": [plan.to_dict() for plan in second_plans],
                "controller_preflight": {
                    "checks_run": int(not args.skip_second_controller_preflight),
                    "passed": None if args.skip_second_controller_preflight else True,
                    "skipped_by_operator": args.skip_second_controller_preflight,
                },
            }
        )
        vacuum.safe_shutdown()
        if global_camera_video is not None:
            camera.stop_recording()
            timeline.mark("global_camera_recording_complete", path=str(global_camera_video))
        timeline.mark("task_complete")
        timing = timeline.to_dict(productive_start="productive_cycle_started")

        result_path = run_dir / "result.json"
        result_path.write_text(
            json.dumps(
                {
                    "calibration": str(calibration_path),
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
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
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
        failure_path = run_dir / "failure.json"
        failure_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "holding_leaf": holding_state["value"],
                    "robot_after_stop": robot_status_snapshot(arm),
                    "timing": timeline.to_dict(productive_start="productive_cycle_started"),
                    "completed_cycles": records,
                },
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved failure report: {failure_path}", file=sys.stderr)
        raise
    finally:
        arm.disconnect()
        camera.stop(raise_recording_error=False)


if __name__ == "__main__":
    raise SystemExit(main())
