#!/usr/bin/env python3
"""Load a wine jar mouth with lotus leaf, white paper, ceramic disk, and two crossed bamboo shells."""

from __future__ import annotations

import argparse
import sys
import time
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
    require_code,
    robot_status_snapshot,
)
from xarm_sdk_control.task_timing import TaskTimeline  # noqa: E402
from xarm_sdk_control.cross_place_motion import (  # noqa: E402
    DEFAULT_MOTION_PROFILE,
    accumulated_layer_heights_mm,
    build_cycle_route,
    choose_second_source_leaf,
    execute_route,
    offset_detection,
    plan_route_motion_profile,
    require_distinct_pick_and_place_targets,
    validate_motion_profile,
)
from xarm_sdk_control.vacuum_control import VacuumController  # noqa: E402
from xarm_sdk_control.visual_targeting import (  # noqa: E402
    DEFAULT_VISION_SOCKET,
    latest_global_calibration,
    load_base_t_camera,
    run_scan,
    select_leaf_waypoints,
    select_single_of_kind,
)


CONFIG_PATH = ROOT / "config" / "robot.yaml"
POSITION_BIAS_CONFIG = ROOT / "config" / "visual_position_bias.json"
JAR_POSITION_CONFIG = ROOT / "config" / "jar_position.yaml"
OUTPUT_ROOT = ROOT / "data" / "complete_loading"
END_EFFECTOR = load_end_effector_config()

STACK_A_PRESENCE_SHIFT_MM = 30.0


def load_jar_point_mm(args: argparse.Namespace) -> list[float]:
    """The fixed jar-mouth teach point, from CLI overrides or config/jar_position.yaml."""
    provided = (args.jar_x_mm is not None, args.jar_y_mm is not None, args.jar_z_mm is not None)
    if any(provided) and not all(provided):
        raise ValueError("--jar-x-mm/--jar-y-mm/--jar-z-mm must be provided together")
    if all(provided):
        return [float(args.jar_x_mm), float(args.jar_y_mm), float(args.jar_z_mm)]
    if not JAR_POSITION_CONFIG.exists():
        raise FileNotFoundError(
            f"Jar mouth teach point is missing: create {JAR_POSITION_CONFIG} with "
            "x_mm/y_mm/z_mm or pass --jar-x-mm/--jar-y-mm/--jar-z-mm"
        )
    data = yaml.safe_load(JAR_POSITION_CONFIG.read_text(encoding="utf-8"))
    return [float(data["x_mm"]), float(data["y_mm"]), float(data["z_mm"])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--confirm-no-pressure-sensor", action="store_true")
    add_vision_common_args(parser, latest_global_calibration(), min_score_default=0.35)
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
    parser.add_argument("--position-bias-config", type=Path, default=POSITION_BIAS_CONFIG)
    parser.add_argument("--jar-x-mm", type=float)
    parser.add_argument("--jar-y-mm", type=float)
    parser.add_argument("--jar-z-mm", type=float)
    parser.add_argument("--lotus-layer-mm", type=float, default=4.0)
    parser.add_argument("--paper-layer-mm", type=float, default=0.5)
    parser.add_argument("--ceramic-layer-mm", type=float, default=5.0)
    parser.add_argument("--shell-layer-mm", type=float, default=3.0)
    parser.add_argument("--second-layer-height-mm", type=float, default=3.0)
    parser.add_argument("--lotus-compression-mm", type=float, default=2.0)
    parser.add_argument("--paper-compression-mm", type=float, default=1.0)
    parser.add_argument("--ceramic-compression-mm", type=float, default=6.0)
    parser.add_argument("--shell-compression-mm", type=float, default=END_EFFECTOR.default_compression_mm)
    parser.add_argument("--first-place-yaw-offset-deg", type=float, default=45.0)
    parser.add_argument("--second-place-yaw-offset-deg", type=float, default=-60.0)
    parser.add_argument("--transport-clearance-mm", type=float, default=100.0)
    parser.add_argument("--approach-clearance-mm", type=float, default=25.0)
    parser.add_argument("--precontact-clearance-mm", type=float, default=5.0)
    parser.add_argument("--precision-lift-mm", type=float, default=20.0)
    parser.add_argument("--release-clearance-mm", type=float, default=0.0)
    parser.add_argument("--second-scan-min-score", type=float, default=0.20)
    parser.add_argument("--max-second-source-shift-mm", type=float, default=80.0)
    parser.add_argument("--allow-validated-clipped-leaf", action="store_true")
    parser.add_argument("--tcp-offset-z-mm", type=float, default=END_EFFECTOR.tcp_offset_mm_deg[2])
    parser.add_argument(
        "--max-legs",
        type=int,
        default=5,
        help="Bring-up aid: stop after this many legs (1-5) and return home.",
    )
    parser.add_argument(
        "--intermediate-home",
        action="store_true",
        help="Return home after every leg instead of parking above the jar mouth.",
    )
    add_motion_profile_args(parser, DEFAULT_MOTION_PROFILE)
    add_workspace_limit_args(parser)
    return parser.parse_args()


def require_stack_a_present(
    detections: list[dict[str, Any]],
    shell1_point_m: list[float],
    bias_m: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Reject runs where the bamboo stack has moved away from its original position."""
    dets = [offset_detection(item, bias_m) for item in detections]
    try:
        choose_second_source_leaf(
            dets,
            shell1_point_m,
            args.allow_validated_clipped_leaf,
            STACK_A_PRESENCE_SHIFT_MM / 1000.0,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Stack A bamboo shell no longer detected near its original position; "
            "re-stage the stacks or abort"
        ) from exc


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
    if not 1 <= args.max_legs <= 5:
        raise ValueError("--max-legs must be within 1~5")
    for label, value in (
        ("lotus compression", args.lotus_compression_mm),
        ("paper compression", args.paper_compression_mm),
        ("ceramic compression", args.ceramic_compression_mm),
        ("shell compression", args.shell_compression_mm),
    ):
        if not 0 <= value <= 30:
            raise ValueError(f"{label} must remain within 0~30 mm")
    for label, value in (
        ("lotus layer", args.lotus_layer_mm),
        ("paper layer", args.paper_layer_mm),
        ("ceramic layer", args.ceramic_layer_mm),
        ("shell layer", args.shell_layer_mm),
        ("second layer", args.second_layer_height_mm),
    ):
        if value < 0:
            raise ValueError(f"{label} must be non-negative")
    if not 0 <= args.precontact_clearance_mm < args.approach_clearance_mm:
        raise ValueError("Precontact clearance must be non-negative and below approach clearance")
    if args.precision_lift_mm <= args.precontact_clearance_mm:
        raise ValueError("Precision lift height must exceed precontact clearance")
    validate_motion_profile(args)

    jar_point_mm = load_jar_point_mm(args)
    jar_mouth_point_m = [value / 1000.0 for value in jar_point_mm]
    print(f"Jar mouth teach point [mm]: {jar_point_mm}")

    calibration_path, base_t_camera = load_base_t_camera(args.calibration)
    position_bias_m = load_bias_mm(args.position_bias_config.expanduser().resolve()) / 1000.0
    robot_ip = load_robot_ip(CONFIG_PATH)
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
            text_prompt="lotus leaf. bamboo leaf.",
            service_socket=args.vision_socket,
            require_service=args.require_vision_service,
            timeline=timeline,
            timeline_prefix="scan_1",
        )
        lotus = offset_detection(
            select_single_of_kind("lotus", first_detections, display_name="lotus leaf"),
            position_bias_m,
        )
        shell1 = offset_detection(
            select_leaf_waypoints(
                first_detections,
                distinct_distance_m=0.05,
                allow_any_clipping=args.allow_validated_clipped_leaf,
            )[0][0],
            position_bias_m,
        )
        for label, source_m in (("lotus stack", lotus["point_base_m"]), ("shell stack", shell1["point_base_m"])):
            require_distinct_pick_and_place_targets(source_m, jar_mouth_point_m)
        require_distinct_pick_and_place_targets(lotus["point_base_m"], shell1["point_base_m"])
        print(f"Lotus leaf [mm]: {(np.asarray(lotus['point_base_m']) * 1000).round(2).tolist()}")
        print(f"Shell stack [mm]: {(np.asarray(shell1['point_base_m']) * 1000).round(2).tolist()}")

        legs = [
            ("lotus", 0.0, args.lotus_compression_mm, args.lotus_layer_mm),
            ("paper", 0.0, args.paper_compression_mm, args.paper_layer_mm),
            ("ceramic", 0.0, args.ceramic_compression_mm, args.ceramic_layer_mm),
            ("shell_1", args.first_place_yaw_offset_deg, args.shell_compression_mm, args.shell_layer_mm),
            ("shell_2", args.second_place_yaw_offset_deg, args.shell_compression_mm, args.second_layer_height_mm),
        ]
        heights = accumulated_layer_heights_mm([thickness for _name, _yaw, _comp, thickness in legs])
        print(f"Accumulated layer heights [mm]: {heights}")
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
        if args.run:
            prepare_robot_for_motion(arm, clean_error=True)
        else:
            print("Preparing controller for no-motion route preflight.")
            prepare_robot_for_motion(arm)
        controller_ready = True

        vacuum = VacuumController(arm, END_EFFECTOR.vacuum_io)
        vacuum.configure()

        for leg_index, (material, yaw_deg, compression_mm, _thickness_mm) in enumerate(
            legs[: args.max_legs], start=1
        ):
            cycle_name = f"material_{leg_index:02d}_{material}"
            timeline.mark(f"{cycle_name}.pick_started")
            if material == "lotus":
                source = lotus
            elif material == "paper":
                paper_frame, paper_dets = run_scan(
                    run_dir / "scan_2" / "captures",
                    base_t_camera,
                    args.vision_env,
                    args.min_score,
                    camera=camera,
                    text_prompt="white paper. bamboo leaf.",
                    service_socket=args.vision_socket,
                    require_service=args.require_vision_service,
                    timeline=timeline,
                    timeline_prefix="scan_2",
                )
                require_stack_a_present(paper_dets, shell1["point_base_m"], position_bias_m, args)
                source = offset_detection(
                    select_single_of_kind("paper", paper_dets, display_name="white paper"),
                    position_bias_m,
                )
            elif material == "ceramic":
                ceramic_frame, ceramic_dets = run_scan(
                    run_dir / "scan_3" / "captures",
                    base_t_camera,
                    args.vision_env,
                    args.min_score,
                    camera=camera,
                    text_prompt="ceramic disk. bamboo leaf.",
                    service_socket=args.vision_socket,
                    require_service=args.require_vision_service,
                    timeline=timeline,
                    timeline_prefix="scan_3",
                )
                require_stack_a_present(ceramic_dets, shell1["point_base_m"], position_bias_m, args)
                source = offset_detection(
                    select_single_of_kind("ceramic", ceramic_dets, display_name="ceramic disk"),
                    position_bias_m,
                )
            elif material == "shell_1":
                source = shell1
            else:  # shell_2
                shell2_frame, shell2_dets = run_scan(
                    run_dir / "scan_4" / "captures",
                    base_t_camera,
                    args.vision_env,
                    args.second_scan_min_score,
                    camera=camera,
                    text_prompt="bamboo leaf.",
                    service_socket=args.vision_socket,
                    require_service=args.require_vision_service,
                    timeline=timeline,
                    timeline_prefix="scan_4",
                )
                shell2_dets = [offset_detection(item, position_bias_m) for item in shell2_dets]
                source = choose_second_source_leaf(
                    shell2_dets,
                    shell1["point_base_m"],
                    args.allow_validated_clipped_leaf,
                    args.max_second_source_shift_mm / 1000.0,
                )
            require_distinct_pick_and_place_targets(source["point_base_m"], jar_mouth_point_m)

            if leg_index == 1 or args.intermediate_home:
                start_pose = None
            else:
                pose_code, start_pose = arm.get_position(is_radian=False)
                require_code(f"get waiting pose before {material} leg", (pose_code, start_pose))
            route = build_cycle_route(
                home_pose,
                source["point_base_m"],
                jar_mouth_point_m,
                yaw_deg,
                args.transport_clearance_mm,
                args.approach_clearance_mm,
                compression_mm,
                args.release_clearance_mm,
                heights[leg_index - 1],
                start_pose=start_pose,
                return_home=(leg_index == len(legs[: args.max_legs]) or args.intermediate_home),
                precontact_clearance_mm=args.precontact_clearance_mm,
                precision_lift_mm=args.precision_lift_mm,
            )
            safety = route_safety_report([item[1] for item in route], args)
            if not safety["all_ok"]:
                raise RuntimeError(f"{material} route is outside explicit workspace limits")

            pose_code, leg_start_pose = arm.get_position(is_radian=False)
            require_code(f"get pose for {material} route planning", (pose_code, leg_start_pose))
            angles_code, leg_start_angles = arm.get_servo_angle(is_radian=False)
            require_code(f"get joint angles for {material} route planning", (angles_code, leg_start_angles))
            planning_started = time.perf_counter()
            plans = plan_route_motion_profile(arm, leg_start_pose, leg_start_angles, route, args)
            planning_seconds = time.perf_counter() - planning_started
            timeline.mark(
                f"{cycle_name}.route_planned",
                planning_seconds=planning_seconds,
                segment_count=len(plans),
            )
            records.append(
                {
                    "material": material,
                    "leg": leg_index,
                    "layer_height_mm": heights[leg_index - 1],
                    "source_point_m": source["point_base_m"],
                    "route": route,
                    "safety": safety,
                    "joint_speed_planning_seconds": planning_seconds,
                    "joint_speed_plan": [plan.to_dict() for plan in plans],
                    "controller_preflight": {"checks_run": 1, "passed": True},
                }
            )

            if not args.run:
                require_code("stop robot after no-motion preflight", arm.set_state(4))
                controller_ready = False
                print(
                    "DRY RUN: first scan, jar teach point, staging gates, IK continuity, and joint "
                    "limits validated; subsequent legs require physical placement."
                )
                return 0

            execute_route(
                arm,
                vacuum,
                route,
                args,
                holding_state,
                timeline,
                cycle_name,
                plans,
            )
            timeline.mark(f"{cycle_name}.cycle_complete")
            print(f"Leg {leg_index}/{len(legs[: args.max_legs])} ({material}) complete.")

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
                "run_mode": bool(args.run),
                "calibration": str(calibration_path),
                "position_bias_base_mm": (position_bias_m * 1000).tolist(),
                "jar": {
                    "teach_point_mm": jar_point_mm,
                    "target_base_m": jar_mouth_point_m,
                },
                "layers": {
                    "lotus_layer_mm": args.lotus_layer_mm,
                    "paper_layer_mm": args.paper_layer_mm,
                    "ceramic_layer_mm": args.ceramic_layer_mm,
                    "shell_layer_mm": args.shell_layer_mm,
                    "second_layer_height_mm": args.second_layer_height_mm,
                    "accumulated_layer_heights_mm": heights,
                },
                "performance_parameters": {
                    "transport_clearance_mm": args.transport_clearance_mm,
                    "approach_clearance_mm": args.approach_clearance_mm,
                    "precontact_clearance_mm": args.precontact_clearance_mm,
                    "precision_lift_mm": args.precision_lift_mm,
                    "release_clearance_mm": args.release_clearance_mm,
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
                    "vacuum_settle_seconds": args.vacuum_settle_seconds,
                    "release_settle_seconds": args.release_settle_seconds,
                    "min_score": args.min_score,
                    "second_scan_min_score": args.second_scan_min_score,
                },
                "global_camera_video": str(global_camera_video) if global_camera_video else None,
                "timing": timing,
                "cycles": records,
            },
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
                "EMERGENCY NOTE: vacuum may still be holding a material. Stop the robot and make "
                "the area safe before manually releasing it.",
                file=sys.stderr,
            )
        elif vacuum is not None:
            vacuum.safe_shutdown()
        failure_path = write_failure_report(
            run_dir,
            exc,
            holding=holding_state["value"],
            timeline=timeline,
            productive_start="productive_cycle_started",
            completed_records=records,
            robot_snapshot=robot_status_snapshot(arm) if arm is not None else None,
        )
        print(f"Saved failure report: {failure_path}", file=sys.stderr)
        raise
    finally:
        if arm is not None:
            arm.disconnect()
        camera.stop(raise_recording_error=False)


if __name__ == "__main__":
    raise SystemExit(main())
