#!/usr/bin/env python3
"""Visually pick one bamboo leaf and place it on the detected ceramic disk."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.end_effector_config import load_end_effector_config  # noqa: E402
from xarm_sdk_control.vacuum_control import VacuumController  # noqa: E402
from xarm_sdk_control.positioning_bias import load_bias_mm  # noqa: E402
from xarm_sdk_control.robot_motion import (  # noqa: E402
    CAMERA_TRANSIT_Z_MM,
    require_overhead_camera_safe_route,
    reset_robot_at_ip,
)
from xarm_sdk_control.visual_targeting import (  # noqa: E402
    capture_rgbd,
    detections_in_base,
    latest_global_calibration,
    run_vision,
    safety_report,
    select_ceramic,
    select_leaf_waypoints,
)


CONFIG_PATH = ROOT / "config" / "robot.yaml"
OUTPUT_ROOT = ROOT / "data" / "visual_vacuum_pick"
DEFAULT_GRASP_CONFIG = ROOT / "config" / "leaf_grasp_target.json"
DEFAULT_POSITION_BIAS_CONFIG = ROOT / "config" / "visual_position_bias.json"
END_EFFECTOR = load_end_effector_config()


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


def pose_at(point_mm: np.ndarray, z_mm: float, orientation: list[float]) -> list[float]:
    return [float(point_mm[0]), float(point_mm[1]), float(z_mm), *map(float, orientation)]


def build_route(
    initial_pose: list[float],
    leaf_point_m: list[float],
    ceramic_point_m: list[float],
    stage: str,
    transport_clearance_mm: float,
    approach_clearance_mm: float,
    compression_mm: float,
    release_clearance_mm: float,
) -> list[tuple[str, list[float], str]]:
    orientation = list(map(float, initial_pose[3:6]))
    leaf_mm = np.asarray(leaf_point_m, dtype=float) * 1000.0
    ceramic_mm = np.asarray(ceramic_point_m, dtype=float) * 1000.0
    transport_z = float(max(leaf_mm[2], ceramic_mm[2]) + transport_clearance_mm)
    if transport_z > CAMERA_TRANSIT_Z_MM:
        raise RuntimeError(
            f"Transport Z {transport_z:.1f} mm exceeds the {CAMERA_TRANSIT_Z_MM:.1f} mm camera-safe corridor"
        )
    leaf_transit = pose_at(leaf_mm, transport_z, orientation)
    leaf_approach = pose_at(leaf_mm, float(leaf_mm[2] + approach_clearance_mm), orientation)

    def add_safe_departure(route: list[tuple[str, list[float], str]]) -> None:
        if abs(float(initial_pose[2]) - transport_z) > 1.0:
            route.append(
                (
                    "move vertically to camera-safe transit height",
                    [float(initial_pose[0]), float(initial_pose[1]), transport_z, *orientation],
                    "descent",
                )
            )

    def add_safe_return(route: list[tuple[str, list[float], str]]) -> None:
        route.append(
            (
                "move to camera-safe initial column",
                [float(initial_pose[0]), float(initial_pose[1]), transport_z, *orientation],
                "transit",
            )
        )
        route.append(("return vertically to initial pose", list(map(float, initial_pose)), "descent"))

    def checked(route: list[tuple[str, list[float], str]]) -> list[tuple[str, list[float], str]]:
        require_overhead_camera_safe_route([list(map(float, initial_pose)), *[item[1] for item in route]])
        return route

    if stage == "approach":
        route: list[tuple[str, list[float], str]] = []
        add_safe_departure(route)
        route.extend(
            [
                ("move directly above leaf", leaf_transit, "transit"),
                ("descend to leaf approach pose", leaf_approach, "descent"),
                ("retract vertically above leaf", leaf_transit, "descent"),
            ]
        )
        add_safe_return(route)
        return checked(route)

    leaf_contact = pose_at(leaf_mm, float(leaf_mm[2] - compression_mm), orientation)
    if stage == "pick-test":
        leaf_release = pose_at(leaf_mm, float(leaf_mm[2] + release_clearance_mm), orientation)
        route = []
        add_safe_departure(route)
        route.extend(
            [
                ("move directly above leaf", leaf_transit, "transit"),
                ("descend to leaf approach pose", leaf_approach, "descent"),
                ("contact source leaf", leaf_contact, "descent"),
                ("lift leaf vertically for inspection", leaf_transit, "descent"),
                ("descend above original source", leaf_approach, "descent"),
                ("lower leaf back to source", leaf_release, "descent"),
                ("retract vertically above source", leaf_transit, "descent"),
            ]
        )
        add_safe_return(route)
        return checked(route)

    ceramic_transit = pose_at(ceramic_mm, transport_z, orientation)
    ceramic_approach = pose_at(ceramic_mm, float(ceramic_mm[2] + approach_clearance_mm), orientation)
    ceramic_release = pose_at(ceramic_mm, float(ceramic_mm[2] + release_clearance_mm), orientation)
    route = []
    add_safe_departure(route)
    route.extend([
        ("move directly above leaf", leaf_transit, "transit"),
        ("descend to leaf approach pose", leaf_approach, "descent"),
        ("contact source leaf", leaf_contact, "descent"),
        ("lift leaf vertically", leaf_transit, "descent"),
        ("carry leaf above ceramic disk", ceramic_transit, "transit"),
        ("descend to ceramic approach pose", ceramic_approach, "descent"),
        ("lower leaf to release clearance", ceramic_release, "descent"),
        ("retract vertically above ceramic", ceramic_transit, "descent"),
    ])
    add_safe_return(route)
    return checked(route)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Enable real robot motion; default is a dry run.")
    parser.add_argument("--stage", choices=("approach", "pick-test", "full"), default="approach")
    parser.add_argument("--confirm-no-pressure-sensor", action="store_true")
    parser.add_argument("--calibration", type=Path, default=latest_global_calibration())
    parser.add_argument("--vision-env", default="winejar-vision")
    parser.add_argument("--serial")
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--leaf-grasp-config", type=Path, default=DEFAULT_GRASP_CONFIG)
    parser.add_argument("--position-bias-config", type=Path, default=DEFAULT_POSITION_BIAS_CONFIG)
    parser.add_argument(
        "--allow-clipped-leaf-for-approach",
        action="store_true",
        help="Allow a clipped leaf only for non-contact approach inspection; never affects full pick mode.",
    )
    parser.add_argument(
        "--allow-validated-clipped-leaf-for-full",
        action="store_true",
        help="Allow a clipped leaf during contact motion only after its compensated center was physically hover-validated.",
    )
    parser.add_argument("--max-calibration-residual-mm", type=float, default=15.0)
    parser.add_argument("--tcp-offset-z-mm", type=float, default=END_EFFECTOR.tcp_offset_mm_deg[2])
    parser.add_argument(
        "--apply-tcp-offset",
        action="store_true",
        help="Apply the configured TCP offset during a dry run; this changes coordinates but commands no motion.",
    )
    parser.add_argument("--transport-clearance-mm", type=float, default=200.0)
    parser.add_argument("--approach-clearance-mm", type=float, default=35.0)
    parser.add_argument("--compression-mm", type=float, default=END_EFFECTOR.default_compression_mm)
    parser.add_argument("--release-clearance-mm", type=float, default=5.0)
    parser.add_argument("--vacuum-settle-seconds", type=float, default=1.5)
    parser.add_argument("--release-settle-seconds", type=float, default=1.0)
    parser.add_argument("--approach-hold-seconds", type=float, default=3.0)
    parser.add_argument("--pick-hold-seconds", type=float, default=3.0)
    parser.add_argument("--transit-speed", type=float, default=35.0)
    parser.add_argument("--descent-speed", type=float, default=10.0)
    parser.add_argument("--acc", type=float, default=120.0)
    parser.add_argument("--min-x-mm", type=float)
    parser.add_argument("--max-x-mm", type=float)
    parser.add_argument("--min-y-mm", type=float)
    parser.add_argument("--max-y-mm", type=float)
    parser.add_argument("--min-z-mm", type=float)
    parser.add_argument("--max-z-mm", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.calibration is None or not args.calibration.expanduser().exists():
        print("ERROR: fixed global-camera calibration is missing.")
        return 2
    if args.run and any(
        value is None
        for value in (args.min_x_mm, args.max_x_mm, args.min_y_mm, args.max_y_mm, args.min_z_mm, args.max_z_mm)
    ):
        print("ERROR: --run requires explicit min/max X, Y, and Z workspace limits.")
        return 3
    if args.run and args.stage != "approach" and not args.confirm_no_pressure_sensor:
        print("ERROR: contact motion requires --confirm-no-pressure-sensor.")
        return 4
    if not 0.0 <= args.compression_mm <= 20.0:
        raise ValueError("--compression-mm must stay within the stated 0~20 mm cup travel")
    if min(args.transport_clearance_mm, args.approach_clearance_mm, args.release_clearance_mm) < 0.0:
        raise ValueError("Clearance values cannot be negative")

    calibration_path = args.calibration.expanduser().resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    base_t_camera = np.asarray(calibration["base_T_camera"], dtype=float)
    residual_max_mm = calibration.get("quality", {}).get("residual_max_mm")
    if base_t_camera.shape != (4, 4):
        raise RuntimeError(f"Invalid base_T_camera shape: {base_t_camera.shape}")
    if residual_max_mm is None or float(residual_max_mm) > args.max_calibration_residual_mm:
        print(f"ERROR: calibration residual {residual_max_mm} mm exceeds the configured limit.")
        return 5

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        robot_ip = str(yaml.safe_load(file)["robot_ip"])
    if args.run:
        print("Task step 1: moving to the configured camera-clear reset pose.")
        reset_robot_at_ip(
            robot_ip,
            [0, 0, args.tcp_offset_z_mm, 0, 0, 0],
            END_EFFECTOR.home_pose_mm_deg,
            args.transit_speed,
            args.acc,
        )

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_dir = OUTPUT_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    frame_dir = capture_rgbd(run_dir, args.serial)
    vision_result = run_vision(frame_dir, args.vision_env, args.min_score)
    detections = detections_in_base(vision_result.get("detections", []), base_t_camera, args.min_score)
    ceramic = select_ceramic(detections) if args.stage == "full" else None
    leaves, _reused = select_leaf_waypoints(
        detections,
        distinct_distance_m=0.05,
        allow_any_clipping=bool(
            (args.stage == "approach" and args.allow_clipped_leaf_for_approach)
            or (args.stage in {"pick-test", "full"} and args.allow_validated_clipped_leaf_for_full)
        ),
    )
    leaf = leaves[0]
    raw_leaf_point_m = np.asarray(leaf["point_base_m"], dtype=float)
    grasp_config_path = args.leaf_grasp_config.expanduser().resolve()
    grasp_offset_m = np.zeros(3, dtype=float)
    if grasp_config_path.exists():
        grasp_config = json.loads(grasp_config_path.read_text(encoding="utf-8"))
        grasp_offset_m = np.asarray(grasp_config["offset_base_mm"], dtype=float) / 1000.0
        if grasp_offset_m.shape != (3,):
            raise RuntimeError(f"Invalid leaf grasp offset shape in {grasp_config_path}")
    position_bias_path = args.position_bias_config.expanduser().resolve()
    position_bias_m = load_bias_mm(position_bias_path) / 1000.0
    leaf = dict(leaf)
    leaf["point_base_m"] = (raw_leaf_point_m + grasp_offset_m + position_bias_m).tolist()
    if ceramic is not None:
        ceramic = dict(ceramic)
        ceramic["point_base_m"] = (np.asarray(ceramic["point_base_m"], dtype=float) + position_bias_m).tolist()

    from xarm.wrapper import XArmAPI

    arm = XArmAPI(do_not_open=True, is_radian=False)
    vacuum: VacuumController | None = None
    holding_leaf = False
    route: list[tuple[str, list[float], str]] = []
    safety: dict[str, Any] = {}
    try:
        require_code("connect", arm.connect(port=robot_ip))
        if args.run or args.apply_tcp_offset:
            require_code(
                "set suction-cup TCP offset",
                arm.set_tcp_offset([0, 0, args.tcp_offset_z_mm, 0, 0, 0], is_radian=False, wait=True),
            )
            time.sleep(0.5)
        pose_code, initial_pose = arm.get_position(is_radian=False)
        require_code("get initial TCP pose", (pose_code, initial_pose))
        reported_offset = list(map(float, arm.tcp_offset))
        if abs(reported_offset[2] - args.tcp_offset_z_mm) > 1.0:
            raise RuntimeError(
                f"Controller TCP Z offset is {reported_offset[2]:.2f} mm, expected {args.tcp_offset_z_mm:.2f} mm"
            )

        route = build_route(
            initial_pose,
            leaf["point_base_m"],
            ceramic["point_base_m"] if ceramic is not None else leaf["point_base_m"],
            args.stage,
            args.transport_clearance_mm,
            args.approach_clearance_mm,
            args.compression_mm,
            args.release_clearance_mm,
        )
        limits = {
            "min_x_mm": args.min_x_mm if args.min_x_mm is not None else -1000.0,
            "max_x_mm": args.max_x_mm if args.max_x_mm is not None else 1000.0,
            "min_y_mm": args.min_y_mm if args.min_y_mm is not None else -1000.0,
            "max_y_mm": args.max_y_mm if args.max_y_mm is not None else 1000.0,
            "min_z_mm": args.min_z_mm if args.min_z_mm is not None else -1000.0,
            "max_z_mm": args.max_z_mm if args.max_z_mm is not None else 1000.0,
        }
        safety = safety_report([pose for _label, pose, _kind in route], **limits)

        print(f"Calibration: {calibration_path}")
        print(f"TCP offset [mm/deg]: {[round(value, 2) for value in reported_offset]}")
        print(f"Raw leaf mask centroid [mm]: {(raw_leaf_point_m * 1000.0).round(2).tolist()}")
        print(f"Leaf grasp offset [mm]: {(grasp_offset_m * 1000.0).round(2).tolist()}")
        print(f"System positioning bias [mm]: {(position_bias_m * 1000.0).round(2).tolist()}")
        print(f"Compensated leaf grasp point [mm]: {(np.asarray(leaf['point_base_m']) * 1000.0).round(2).tolist()}")
        if ceramic is not None:
            print(f"Ceramic surface point [mm]: {(np.asarray(ceramic['point_base_m']) * 1000.0).round(2).tolist()}")
        else:
            print("Ceramic detection: not required for approach-only inspection")
        print(f"Stage: {args.stage}; compression={args.compression_mm:.1f} mm")
        for index, (label, pose, kind) in enumerate(route, start=1):
            print(f"Step {index:02d} [{kind}] {label}: {[round(value, 2) for value in pose]}")
        print(f"Safety checks: {safety}")

        if args.run:
            if not safety["all_ok"]:
                raise RuntimeError("Refusing robot motion because a waypoint is outside explicit safety limits")
            require_emergency_stop_released(arm)
            if args.stage != "approach":
                vacuum = VacuumController(
                    arm,
                    END_EFFECTOR.vacuum_io,
                )
                vacuum.configure()
                vacuum.prepare_for_pick()
            require_code("clean error", arm.clean_error())
            require_code("clean warn", arm.clean_warn())
            require_code("motion enable", arm.motion_enable(enable=True))
            require_code("set mode 0", arm.set_mode(0))
            require_code("set state 0", arm.set_state(0))

            for index, (label, pose, kind) in enumerate(route, start=1):
                speed = args.descent_speed if kind == "descent" else args.transit_speed
                require_code(
                    f"move step {index:02d}: {label}",
                    arm.set_position(
                        *pose,
                        speed=speed,
                        mvacc=args.acc,
                        wait=True,
                        radius=-1,
                        is_radian=False,
                    ),
                )
                if args.stage == "approach" and label == "descend to leaf approach pose":
                    print(f"Holding {args.approach_hold_seconds:.1f} seconds for visual inspection.")
                    time.sleep(args.approach_hold_seconds)
                if args.stage != "approach" and label == "contact source leaf":
                    assert vacuum is not None
                    vacuum.acquire(args.vacuum_settle_seconds)
                    holding_leaf = True
                if args.stage == "pick-test" and label == "lift leaf vertically for inspection":
                    print(f"Holding lifted leaf for {args.pick_hold_seconds:.1f} seconds.")
                    time.sleep(args.pick_hold_seconds)
                if (
                    (args.stage == "full" and label == "lower leaf to release clearance")
                    or (args.stage == "pick-test" and label == "lower leaf back to source")
                ):
                    assert vacuum is not None
                    vacuum.release(args.release_settle_seconds)
                    holding_leaf = False
            if vacuum is not None:
                vacuum.safe_shutdown()
        else:
            print("DRY RUN: vision and route planning completed; no robot motion or GPIO output was commanded.")

        result = {
            "run_mode": bool(args.run),
            "stage": args.stage,
            "calibration": str(calibration_path),
            "frame_dir": str(frame_dir),
            "tcp_offset_mm_deg": reported_offset,
            "leaf_detection": leaf,
            "raw_leaf_mask_centroid_base_m": raw_leaf_point_m.tolist(),
            "leaf_grasp_config": str(grasp_config_path),
            "position_bias_config": str(position_bias_path),
            "position_bias_base_mm": (position_bias_m * 1000.0).tolist(),
            "ceramic_detection": ceramic,
            "compression_mm": args.compression_mm,
            "route": [{"label": label, "pose_mm_deg": pose, "motion": kind} for label, pose, kind in route],
            "safety": safety,
        }
        result_path = run_dir / "result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(f"Saved result: {result_path}")
        return 0
    except Exception:
        if holding_leaf:
            print(
                "EMERGENCY NOTE: vacuum remains ON to avoid dropping a carried leaf. "
                "Stop the robot, make the area safe, then run 15_vacuum_io_check.py --test off.",
                file=sys.stderr,
            )
        elif vacuum is not None:
            vacuum.safe_shutdown()
        raise
    finally:
        arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
