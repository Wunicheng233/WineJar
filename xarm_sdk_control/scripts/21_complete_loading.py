#!/usr/bin/env python3
"""Load a jar mouth with an optional lotus leaf, paper, ceramic disk, and two crossed shells."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
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
    offset_detection,
    plan_route_motion_profile,
    require_distinct_pick_and_place_targets,
    validate_motion_profile,
)
from xarm_sdk_control.cross_place_retry import (  # noqa: E402
    AllPickAttemptsFailedError,
    run_leaf_pick_with_retry,
    select_nearest_of_kind,
)
from xarm_sdk_control.suction_sensor import SuctionSensor  # noqa: E402
from xarm_sdk_control.vacuum_control import VacuumController  # noqa: E402
from xarm_sdk_control.visual_targeting import (  # noqa: E402
    DEFAULT_VISION_SOCKET,
    detection_object_kind,
    latest_global_calibration,
    load_base_t_camera,
    run_scan,
    select_single_of_kind,
)


CONFIG_PATH = ROOT / "config" / "robot.yaml"
POSITION_BIAS_CONFIG = ROOT / "config" / "visual_position_bias.json"
JAR_POSITION_CONFIG = ROOT / "config" / "jar_position.yaml"
OUTPUT_ROOT = ROOT / "data" / "complete_loading"
END_EFFECTOR = load_end_effector_config()

# Conservative defaults for first end-to-end bring-up. Individual values stay
# exposed as CLI arguments so they can be raised after the workflow is stable.
STABILITY_FIRST_MOTION_PROFILE = replace(
    DEFAULT_MOTION_PROFILE,
    transit_speed=250.0,
    loaded_transit_speed=180.0,
    vertical_speed=170.0,
    loaded_vertical_speed=100.0,
    precision_speed=50.0,
    contact_speed=15.0,
    acc=400.0,
    loaded_acc=200.0,
    precision_acc=160.0,
    joint_velocity_utilization=0.35,
    joint_acceleration_utilization=0.35,
)

# Single-kind materials retain an 80 mm near-source retry gate. Shell entries
# are replaced at runtime by the stricter material-table source-region gate.
RETRY_TARGETING: dict[str, tuple[str, Any | None]] = {
    "lotus": ("lotus leaf.", lambda dets, ref: select_nearest_of_kind(
        "lotus", dets, ref, 80.0 / 1000.0, display_name="lotus leaf")),
    "paper": ("white paper.", lambda dets, ref: select_nearest_of_kind(
        "paper", dets, ref, 80.0 / 1000.0, display_name="white paper")),
    "ceramic": ("ceramic disk.", lambda dets, ref: select_nearest_of_kind(
        "ceramic", dets, ref, 80.0 / 1000.0, display_name="ceramic disk")),
    "shell_1": ("bamboo leaf.", None),
    "shell_2": ("bamboo leaf.", None),
}


def select_execution_legs(
    all_legs: list[tuple[str, float, float, float]],
    accumulated_heights_mm: list[float],
    *,
    skip_lotus: bool,
    max_legs: int,
    start_material: str | None = None,
) -> list[tuple[str, float, float, float, float]]:
    """Select executable legs while preserving heights of preplaced layers."""
    if len(all_legs) != len(accumulated_heights_mm):
        raise ValueError("Each loading leg must have one accumulated layer height")
    materials = [leg[0] for leg in all_legs]
    if start_material is not None:
        if start_material not in materials:
            raise ValueError(f"Unknown start material: {start_material}")
        first_index = materials.index(start_material)
    else:
        first_index = 1 if skip_lotus else 0
    available = [
        (*leg, accumulated_heights_mm[index])
        for index, leg in enumerate(all_legs[first_index:], start=first_index)
    ]
    return available[:max_legs]


def material_position_bias_m(
    base_bias_m: np.ndarray,
    material: str,
    ceramic_pick_offset_x_mm: float,
) -> np.ndarray:
    """Return the calibrated bias plus a narrowly scoped material correction."""
    bias_m = np.asarray(base_bias_m, dtype=float).copy()
    if material == "ceramic":
        bias_m[0] += ceramic_pick_offset_x_mm / 1000.0
    return bias_m


def select_shell_source_candidate(
    detections: list[dict[str, Any]],
    *,
    min_x_mm: float,
    max_x_mm: float,
    min_y_mm: float,
    max_y_mm: float,
    min_box_elongation: float,
    allow_axial_clipping: bool = False,
) -> dict[str, Any]:
    """Select a bamboo shell only from its fixed material-table source region."""
    candidates: list[dict[str, Any]] = []
    for item in detections:
        if detection_object_kind(item) != "bamboo":
            continue
        point_m = item.get("point_base_m")
        box = item.get("box_xyxy")
        if not isinstance(point_m, list) or len(point_m) < 2:
            continue
        if not isinstance(box, list) or len(box) != 4:
            continue
        x_mm, y_mm = float(point_m[0]) * 1000.0, float(point_m[1]) * 1000.0
        if not (min_x_mm <= x_mm <= max_x_mm and min_y_mm <= y_mm <= max_y_mm):
            continue
        width = max(float(box[2]) - float(box[0]), 1.0)
        height = max(float(box[3]) - float(box[1]), 1.0)
        elongation = max(width / height, height / width)
        if elongation < min_box_elongation:
            continue
        if bool(item.get("touches_image_edge", False)):
            edge_sides = set(item.get("edge_sides", []))
            axial_sides = {"left", "right"} if width >= height else {"top", "bottom"}
            if not allow_axial_clipping or not edge_sides or not edge_sides <= axial_sides:
                continue
        candidate = dict(item)
        candidate["source_region_box_elongation"] = elongation
        candidates.append(candidate)
    if not candidates:
        raise RuntimeError(
            "No complete elongated bamboo-shell detection was found inside the configured "
            f"source region X=[{min_x_mm:.0f}, {max_x_mm:.0f}] mm, "
            f"Y=[{min_y_mm:.0f}, {max_y_mm:.0f}] mm"
        )
    return max(candidates, key=lambda item: float(item.get("score", 0.0)))


def select_shell_source_for_args(
    detections: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return select_shell_source_candidate(
        detections,
        min_x_mm=args.shell_source_min_x_mm,
        max_x_mm=args.shell_source_max_x_mm,
        min_y_mm=args.shell_source_min_y_mm,
        max_y_mm=args.shell_source_max_y_mm,
        min_box_elongation=args.shell_min_box_elongation,
        allow_axial_clipping=args.allow_validated_clipped_leaf,
    )


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
    add_vision_common_args(parser, latest_global_calibration(), min_score_default=0.35)
    parser.add_argument("--vision-socket", type=Path, default=DEFAULT_VISION_SOCKET)
    parser.add_argument("--require-vision-service", action="store_true")
    parser.add_argument("--record-global-camera", action="store_true")
    parser.add_argument(
        "--skip-lotus",
        action="store_true",
        help="Treat the lotus leaf as already placed and start the loading cycle with white paper.",
    )
    parser.add_argument(
        "--start-material",
        choices=("lotus", "paper", "ceramic", "shell_1", "shell_2"),
        help="Resume with this material, treating every earlier layer as already placed.",
    )
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
    parser.add_argument("--paper-compression-mm", type=float, default=17.0)
    parser.add_argument("--ceramic-compression-mm", type=float, default=7.0)
    parser.add_argument("--shell-compression-mm", type=float, default=13.0)
    parser.add_argument(
        "--ceramic-pick-offset-x-mm",
        type=float,
        default=8.0,
        help="Ceramic-only X correction added after the shared camera calibration bias.",
    )
    parser.add_argument("--first-place-yaw-offset-deg", type=float, default=45.0)
    parser.add_argument("--second-place-yaw-offset-deg", type=float, default=-60.0)
    parser.add_argument("--transport-clearance-mm", type=float, default=100.0)
    parser.add_argument("--approach-clearance-mm", type=float, default=25.0)
    parser.add_argument("--precontact-clearance-mm", type=float, default=5.0)
    parser.add_argument("--precision-lift-mm", type=float, default=20.0)
    parser.add_argument("--release-clearance-mm", type=float, default=0.0)
    parser.add_argument(
        "--minimum-planned-speed-mm-s",
        type=float,
        default=3.0,
        help="Lowest speed accepted from the conservative joint-aware route estimator.",
    )
    parser.add_argument("--second-scan-min-score", type=float, default=0.10)
    parser.add_argument("--shell-scan-min-score", type=float, default=0.20)
    parser.add_argument("--shell-source-min-x-mm", type=float, default=400.0)
    parser.add_argument("--shell-source-max-x-mm", type=float, default=540.0)
    parser.add_argument("--shell-source-min-y-mm", type=float, default=50.0)
    parser.add_argument("--shell-source-max-y-mm", type=float, default=180.0)
    parser.add_argument("--shell-min-box-elongation", type=float, default=2.2)
    parser.add_argument("--max-second-source-shift-mm", type=float, default=80.0)
    parser.add_argument("--allow-validated-clipped-leaf", action="store_true")
    parser.add_argument("--tcp-offset-z-mm", type=float, default=END_EFFECTOR.tcp_offset_mm_deg[2])
    parser.add_argument(
        "--max-pick-attempts",
        type=int,
        default=3,
        help="Pick attempts per leg including retries (1-3).",
    )
    parser.add_argument(
        "--max-legs",
        type=int,
        default=5,
        help="Bring-up aid: stop after this many available legs (1-5) and return home.",
    )
    parser.add_argument(
        "--intermediate-home",
        action="store_true",
        help="Return home after every leg instead of parking above the jar mouth.",
    )
    add_motion_profile_args(parser, STABILITY_FIRST_MOTION_PROFILE)
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
    if not all(
        0 <= value <= 1
        for value in (args.min_score, args.second_scan_min_score, args.shell_scan_min_score)
    ):
        raise ValueError("Vision confidence thresholds must remain within 0~1")
    if not 1 <= args.max_legs <= 5:
        raise ValueError("--max-legs must be within 1~5")
    if not 1 <= args.max_pick_attempts <= 3:
        raise ValueError("--max-pick-attempts must be within 1~3")
    if abs(args.ceramic_pick_offset_x_mm) > 50:
        raise ValueError("Ceramic X correction must remain within +/-50 mm")
    if not 0 < args.minimum_planned_speed_mm_s <= 5:
        raise ValueError("Minimum planned speed must remain within (0, 5] mm/s")
    if not args.shell_source_min_x_mm < args.shell_source_max_x_mm:
        raise ValueError("Shell source X bounds must be ordered")
    if not args.shell_source_min_y_mm < args.shell_source_max_y_mm:
        raise ValueError("Shell source Y bounds must be ordered")
    if args.shell_min_box_elongation < 1:
        raise ValueError("Shell box elongation must be at least 1")
    for label, value in (
        ("lotus compression", args.lotus_compression_mm),
        ("paper compression", args.paper_compression_mm),
        ("ceramic compression", args.ceramic_compression_mm),
        ("shell compression", args.shell_compression_mm),
    ):
        if not 0 <= value <= END_EFFECTOR.max_compression_mm:
            raise ValueError(
                f"{label} must remain within 0~{END_EFFECTOR.max_compression_mm:g} mm"
            )
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

        lotus = None
        first_frame = None
        if not args.skip_lotus:
            first_frame, first_detections = run_scan(
                run_dir / "scan_1" / "captures",
                base_t_camera,
                args.vision_env,
                args.min_score,
                camera=camera,
                text_prompt="lotus leaf.",
                service_socket=args.vision_socket,
                require_service=args.require_vision_service,
                timeline=timeline,
                timeline_prefix="scan_1",
            )
            lotus = offset_detection(
                select_single_of_kind("lotus", first_detections, display_name="lotus leaf"),
                position_bias_m,
            )
        shell_frame = None
        shell1 = None
        if not args.skip_lotus:
            shell_frame, shell_detections = run_scan(
                run_dir / "scan_2" / "captures",
                base_t_camera,
                args.vision_env,
                args.shell_scan_min_score,
                camera=camera,
                text_prompt="bamboo leaf.",
                service_socket=args.vision_socket,
                require_service=args.require_vision_service,
                timeline=timeline,
                timeline_prefix="scan_2",
            )
            shell_detections = [
                offset_detection(item, position_bias_m) for item in shell_detections
            ]
            shell1 = select_shell_source_for_args(
                shell_detections,
                args,
            )
            require_distinct_pick_and_place_targets(shell1["point_base_m"], jar_mouth_point_m)
        if lotus is not None:
            require_distinct_pick_and_place_targets(lotus["point_base_m"], jar_mouth_point_m)
            assert shell1 is not None
            require_distinct_pick_and_place_targets(lotus["point_base_m"], shell1["point_base_m"])
            print(f"Lotus leaf [mm]: {(np.asarray(lotus['point_base_m']) * 1000).round(2).tolist()}")
        else:
            print("Lotus leaf: preplaced on the jar mouth; pickup leg skipped.")
            print("Shell stack scan: deferred until paper and ceramic have been removed.")
        if shell1 is not None:
            print(f"Shell stack [mm]: {(np.asarray(shell1['point_base_m']) * 1000).round(2).tolist()}")

        all_legs = [
            ("lotus", 0.0, args.lotus_compression_mm, args.lotus_layer_mm),
            ("paper", 0.0, args.paper_compression_mm, args.paper_layer_mm),
            ("ceramic", 0.0, args.ceramic_compression_mm, args.ceramic_layer_mm),
            ("shell_1", args.first_place_yaw_offset_deg, args.shell_compression_mm, args.shell_layer_mm),
            ("shell_2", args.second_place_yaw_offset_deg, args.shell_compression_mm, args.second_layer_height_mm),
        ]
        heights = accumulated_layer_heights_mm([thickness for _name, _yaw, _comp, thickness in all_legs])
        legs = select_execution_legs(
            all_legs,
            heights,
            skip_lotus=args.skip_lotus,
            max_legs=args.max_legs,
            start_material=args.start_material,
        )
        print(f"Accumulated layer heights [mm]: {heights}")
        print(f"Execution legs: {[material for material, *_rest in legs]}")
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

        sensor: SuctionSensor | None = None
        if args.run:
            vacuum = VacuumController(arm, END_EFFECTOR.vacuum_io)
            vacuum.configure()
            if END_EFFECTOR.suction_sensor is None:
                raise RuntimeError(
                    "Suction confirmation is mandatory: add a suction_sensor section to "
                    "config/end_effector.yaml (input_channel, success_value, debounce_seconds, timeout_seconds)."
                )
            sensor = SuctionSensor(arm, END_EFFECTOR.suction_sensor)

        for leg_index, (material, yaw_deg, compression_mm, _thickness_mm, layer_height_mm) in enumerate(
            legs, start=1
        ):
            cycle_name = f"material_{leg_index:02d}_{material}"
            timeline.mark(f"{cycle_name}.pick_started")
            leg_position_bias_m = material_position_bias_m(
                position_bias_m,
                material,
                args.ceramic_pick_offset_x_mm,
            )
            if material == "lotus":
                assert lotus is not None and first_frame is not None
                source = lotus
                leg_frame = first_frame
            elif material == "paper":
                paper_frame, paper_dets = run_scan(
                    run_dir / "scan_3" / "captures",
                    base_t_camera,
                    args.vision_env,
                    args.min_score,
                    camera=camera,
                    text_prompt="white paper.",
                    service_socket=args.vision_socket,
                    require_service=args.require_vision_service,
                    timeline=timeline,
                    timeline_prefix="scan_3",
                )
                source = offset_detection(
                    select_single_of_kind("paper", paper_dets, display_name="white paper"),
                    leg_position_bias_m,
                )
                leg_frame = paper_frame
            elif material == "ceramic":
                ceramic_frame, ceramic_dets = run_scan(
                    run_dir / "scan_4" / "captures",
                    base_t_camera,
                    args.vision_env,
                    args.min_score,
                    camera=camera,
                    text_prompt="ceramic disk.",
                    service_socket=args.vision_socket,
                    require_service=args.require_vision_service,
                    timeline=timeline,
                    timeline_prefix="scan_4",
                )
                source = offset_detection(
                    select_single_of_kind("ceramic", ceramic_dets, display_name="ceramic disk"),
                    leg_position_bias_m,
                )
                print(
                    "Ceramic-only pick correction [mm]: "
                    f"X={args.ceramic_pick_offset_x_mm:+.1f}"
                )
                leg_frame = ceramic_frame
            elif material == "shell_1":
                if shell1 is None:
                    shell_frame, shell_detections = run_scan(
                        run_dir / "scan_shell_after_ceramic" / "captures",
                        base_t_camera,
                        args.vision_env,
                        args.shell_scan_min_score,
                        camera=camera,
                        text_prompt="bamboo leaf.",
                        service_socket=args.vision_socket,
                        require_service=args.require_vision_service,
                        timeline=timeline,
                        timeline_prefix="scan_shell_after_ceramic",
                    )
                    shell_detections = [
                        offset_detection(item, leg_position_bias_m) for item in shell_detections
                    ]
                    shell1 = select_shell_source_for_args(
                        shell_detections,
                        args,
                    )
                    require_distinct_pick_and_place_targets(shell1["point_base_m"], jar_mouth_point_m)
                    print(
                        "Uncovered shell stack [mm]: "
                        f"{(np.asarray(shell1['point_base_m']) * 1000).round(2).tolist()}"
                    )
                assert shell_frame is not None
                source = shell1
                leg_frame = shell_frame
            else:  # shell_2
                assert shell1 is not None
                shell2_frame, shell2_dets = run_scan(
                    run_dir / "scan_5" / "captures",
                    base_t_camera,
                    args.vision_env,
                    args.shell_scan_min_score,
                    camera=camera,
                    text_prompt="bamboo leaf.",
                    service_socket=args.vision_socket,
                    require_service=args.require_vision_service,
                    timeline=timeline,
                    timeline_prefix="scan_5",
                )
                shell2_dets = [offset_detection(item, leg_position_bias_m) for item in shell2_dets]
                source = select_shell_source_for_args(
                    shell2_dets,
                    args,
                )
                leg_frame = shell2_frame
            require_distinct_pick_and_place_targets(source["point_base_m"], jar_mouth_point_m)

            retry_scan_prompt, retry_select = RETRY_TARGETING[material]
            if material.startswith("shell_"):
                retry_scan_prompt = "bamboo leaf."
                retry_select = lambda dets, _reference: select_shell_source_for_args(dets, args)

            if not args.run:
                pose_code, leg_start_pose = arm.get_position(is_radian=False)
                require_code(f"get pose for {material} route planning", (pose_code, leg_start_pose))
                route = build_cycle_route(
                    home_pose,
                    source["point_base_m"],
                    jar_mouth_point_m,
                    yaw_deg,
                    args.transport_clearance_mm,
                    args.approach_clearance_mm,
                    compression_mm,
                    args.release_clearance_mm,
                    layer_height_mm,
                    start_pose=list(map(float, leg_start_pose)),
                    return_home=(leg_index == len(legs) or args.intermediate_home),
                    precontact_clearance_mm=args.precontact_clearance_mm,
                    precision_lift_mm=args.precision_lift_mm,
                )
                safety = route_safety_report([item[1] for item in route], args)
                if not safety["all_ok"]:
                    raise RuntimeError(f"{material} route is outside explicit workspace limits")
                angles_code, leg_start_angles = arm.get_servo_angle(is_radian=False)
                require_code(f"get joint angles for {material} route planning", (angles_code, leg_start_angles))
                planning_started = time.perf_counter()
                preflight_result = {}
                plans = plan_route_motion_profile(
                    arm, leg_start_pose, leg_start_angles, route, args,
                    preflight_result=preflight_result,
                )
                planning_seconds = time.perf_counter() - planning_started
                timeline.mark(
                    f"{cycle_name}.route_planned",
                    planning_seconds=planning_seconds,
                    planning_diagnostics=preflight_result.get("planning_diagnostics"),
                    segment_count=len(plans),
                )
                records.append(
                    {
                        "material": material,
                        "leg": leg_index,
                        "layer_height_mm": layer_height_mm,
                        "source_point_m": source["point_base_m"],
                        "route": route,
                        "safety": safety,
                        "joint_speed_planning_seconds": planning_seconds,
                        "planning_diagnostics": preflight_result.get("planning_diagnostics"),
                        "joint_speed_plan": [plan.to_dict() for plan in plans],
                        "controller_preflight": {"checks_run": 1, "passed": True},
                    }
                )
                require_code("stop robot after no-motion preflight", arm.set_state(4))
                controller_ready = False
                print(
                    "DRY RUN: first scan, jar teach point, staging gates, IK continuity, and joint "
                    "limits validated; subsequent legs require physical placement."
                )
                return 0

            assert vacuum is not None and sensor is not None
            outcome = run_leaf_pick_with_retry(
                leaf_index=leg_index,
                arm=arm,
                vacuum=vacuum,
                home_pose=home_pose,
                placement_target_m=jar_mouth_point_m,
                initial_target_m=source["point_base_m"],
                source_stack_reference_m=(
                    shell1["point_base_m"] if material == "shell_2" else source["point_base_m"]
                ),
                place_yaw_offset_deg=yaw_deg,
                layer_height_mm=layer_height_mm,
                return_home=(leg_index == len(legs) or args.intermediate_home),
                args=args,
                holding_state=holding_state,
                timeline=timeline,
                run_dir=run_dir,
                base_t_camera=base_t_camera,
                position_bias_m=leg_position_bias_m,
                camera=camera,
                initial_targeting_frame=leg_frame,
                suction_sensor=sensor,
                inspection_event=None,
                retry_scan_prompt=retry_scan_prompt,
                retry_select=retry_select,
                compression_mm=compression_mm,
                cycle_prefix=cycle_name,
                advisory_suction_loss_motions=(
                    frozenset({"loaded_precision", "loaded_vertical", "loaded_transit"})
                    if material == "paper"
                    else frozenset()
                ),
            )
            record = outcome.cycle_record()
            record.update(
                {"material": material, "leg": leg_index, "layer_height_mm": layer_height_mm}
            )
            records.append(record)
            if not outcome.success:
                raise AllPickAttemptsFailedError(leg_index, outcome)
            timeline.mark(f"{cycle_name}.cycle_complete", attempts_used=len(outcome.attempts))
            print(f"Leg {leg_index}/{len(legs)} ({material}) complete.")

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
                "lotus_preplaced": bool(args.skip_lotus),
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
        extra: dict[str, Any] | None = None
        if isinstance(exc, AllPickAttemptsFailedError):
            final_status = (
                exc.outcome.final_verification.status
                if exc.outcome.final_verification is not None
                else None
            )
            extra = {
                "pick_verification_failed": {
                    "leg": exc.leaf_index,
                    "attempts_used": exc.attempts_used,
                    "final_status": final_status,
                }
            }
            result_path = write_json_report(
                run_dir,
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "jar": {
                        "teach_point_mm": jar_point_mm,
                        "target_base_m": jar_mouth_point_m,
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
