#!/usr/bin/env python3
"""Descend above the detected bamboo leaf and hold for non-contact approach inspection."""

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

from xarm_sdk_control.end_effector_config import load_end_effector_config  # noqa: E402
from xarm_sdk_control.positioning_bias import load_bias_mm  # noqa: E402
from xarm_sdk_control.geometry import compose_pose_mm  # noqa: E402
from xarm_sdk_control.cli_utils import (  # noqa: E402
    add_vision_common_args,
    add_workspace_limit_args,
    explicit_workspace_limits,
    route_safety_report,
)
from xarm_sdk_control.run_output import new_timestamped_run_dir, write_json_report  # noqa: E402
from xarm_sdk_control.robot_motion import (  # noqa: E402
    CAMERA_TRANSIT_Z_MM,
    connect_robot,
    load_robot_ip,
    prepare_robot_for_motion,
    require_code,
    require_overhead_camera_safe_route,
    reset_robot_at_ip,
)
from xarm_sdk_control.visual_targeting import (  # noqa: E402
    DEFAULT_MAX_CALIBRATION_RESIDUAL_MM,
    CalibrationResidualError,
    latest_global_calibration,
    load_base_t_camera,
    run_scan,
    select_leaf_waypoints,
)


CONFIG_PATH = ROOT / "config" / "robot.yaml"
DEFAULT_POSITION_BIAS_CONFIG = ROOT / "config" / "visual_position_bias.json"
OUTPUT_ROOT = ROOT / "data" / "visual_vacuum_pick"
END_EFFECTOR = load_end_effector_config()


def build_route(
    initial_pose: list[float],
    leaf_point_m: list[float],
    transport_clearance_mm: float,
    approach_clearance_mm: float,
) -> list[tuple[str, list[float], str]]:
    orientation = list(map(float, initial_pose[3:6]))
    leaf_mm = np.asarray(leaf_point_m, dtype=float) * 1000.0
    transport_z = float(leaf_mm[2] + transport_clearance_mm)
    if transport_z > CAMERA_TRANSIT_Z_MM:
        raise RuntimeError(
            f"Transport Z {transport_z:.1f} mm exceeds the {CAMERA_TRANSIT_Z_MM:.1f} mm camera-safe corridor"
        )
    leaf_transit = compose_pose_mm(leaf_mm, transport_z, orientation)
    leaf_approach = compose_pose_mm(leaf_mm, float(leaf_mm[2] + approach_clearance_mm), orientation)

    route: list[tuple[str, list[float], str]] = []
    if abs(float(initial_pose[2]) - transport_z) > 1.0:
        route.append(
            (
                "move vertically to camera-safe transit height",
                [float(initial_pose[0]), float(initial_pose[1]), transport_z, *orientation],
                "descent",
            )
        )
    route.extend(
        [
            ("move directly above leaf", leaf_transit, "transit"),
            ("descend to leaf approach pose", leaf_approach, "descent"),
            ("retract vertically above leaf", leaf_transit, "descent"),
        ]
    )
    route.append(
        (
            "move to camera-safe initial column",
            [float(initial_pose[0]), float(initial_pose[1]), transport_z, *orientation],
            "transit",
        )
    )
    route.append(("return vertically to initial pose", list(map(float, initial_pose)), "descent"))
    require_overhead_camera_safe_route([list(map(float, initial_pose)), *[item[1] for item in route]])
    return route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Enable real robot motion; default is a dry run.")
    add_vision_common_args(parser, latest_global_calibration())
    parser.add_argument("--position-bias-config", type=Path, default=DEFAULT_POSITION_BIAS_CONFIG)
    parser.add_argument(
        "--allow-clipped-leaf-for-approach",
        action="store_true",
        help="Allow a clipped leaf for non-contact approach inspection.",
    )
    parser.add_argument(
        "--max-calibration-residual-mm",
        type=float,
        default=DEFAULT_MAX_CALIBRATION_RESIDUAL_MM,
    )
    parser.add_argument("--tcp-offset-z-mm", type=float, default=END_EFFECTOR.tcp_offset_mm_deg[2])
    parser.add_argument(
        "--apply-tcp-offset",
        action="store_true",
        help="Apply the configured TCP offset during a dry run; this changes coordinates but commands no motion.",
    )
    parser.add_argument("--transport-clearance-mm", type=float, default=200.0)
    parser.add_argument("--approach-clearance-mm", type=float, default=35.0)
    parser.add_argument("--approach-hold-seconds", type=float, default=3.0)
    parser.add_argument("--transit-speed", type=float, default=35.0)
    parser.add_argument("--descent-speed", type=float, default=10.0)
    parser.add_argument("--acc", type=float, default=120.0)
    add_workspace_limit_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.calibration is None:
        print("ERROR: fixed global-camera calibration is missing.")
        return 2
    try:
        calibration_path, base_t_camera = load_base_t_camera(
            args.calibration,
            args.max_calibration_residual_mm,
        )
    except FileNotFoundError:
        print("ERROR: fixed global-camera calibration is missing.")
        return 2
    except CalibrationResidualError as exc:
        print(f"ERROR: {exc}")
        return 5
    if args.run and not explicit_workspace_limits(args):
        print("ERROR: --run requires explicit min/max X, Y, and Z workspace limits.")
        return 3
    if min(args.transport_clearance_mm, args.approach_clearance_mm) < 0.0:
        raise ValueError("Clearance values cannot be negative")

    robot_ip = load_robot_ip(CONFIG_PATH)
    if args.run:
        print("Task step 1: moving to the configured camera-clear reset pose.")
        reset_robot_at_ip(
            robot_ip,
            [0, 0, args.tcp_offset_z_mm, 0, 0, 0],
            END_EFFECTOR.home_pose_mm_deg,
            args.transit_speed,
            args.acc,
        )

    run_dir = new_timestamped_run_dir(OUTPUT_ROOT)
    frame_dir, detections = run_scan(
        run_dir,
        base_t_camera,
        args.vision_env,
        args.min_score,
        serial=args.serial,
    )
    leaves, _reused = select_leaf_waypoints(
        detections,
        allow_any_clipping=args.allow_clipped_leaf_for_approach,
    )
    leaf = leaves[0]
    raw_leaf_point_m = np.asarray(leaf["point_base_m"], dtype=float)
    position_bias_path = args.position_bias_config.expanduser().resolve()
    position_bias_m = load_bias_mm(position_bias_path) / 1000.0
    leaf = dict(leaf)
    leaf["point_base_m"] = (raw_leaf_point_m + position_bias_m).tolist()

    arm: Any | None = None
    route: list[tuple[str, list[float], str]] = []
    safety: dict[str, Any] = {}
    try:
        if args.run or args.apply_tcp_offset:
            arm, initial_pose = connect_robot(
                robot_ip,
                [0, 0, args.tcp_offset_z_mm, 0, 0, 0],
                settle_seconds=0.5,
            )
        else:
            arm, initial_pose = connect_robot(robot_ip)
        reported_offset = list(map(float, arm.tcp_offset))
        if abs(reported_offset[2] - args.tcp_offset_z_mm) > 1.0:
            raise RuntimeError(
                f"Controller TCP Z offset is {reported_offset[2]:.2f} mm, expected {args.tcp_offset_z_mm:.2f} mm"
            )

        route = build_route(
            initial_pose,
            leaf["point_base_m"],
            args.transport_clearance_mm,
            args.approach_clearance_mm,
        )
        safety = route_safety_report([pose for _label, pose, _kind in route], args)

        print(f"Calibration: {calibration_path}")
        print(f"TCP offset [mm/deg]: {[round(value, 2) for value in reported_offset]}")
        print(f"Raw leaf mask centroid [mm]: {(raw_leaf_point_m * 1000.0).round(2).tolist()}")
        print(f"System positioning bias [mm]: {(position_bias_m * 1000.0).round(2).tolist()}")
        print(
            f"Compensated leaf approach point [mm]: "
            f"{(np.asarray(leaf['point_base_m']) * 1000.0).round(2).tolist()}"
        )
        for index, (label, pose, kind) in enumerate(route, start=1):
            print(f"Step {index:02d} [{kind}] {label}: {[round(value, 2) for value in pose]}")
        print(f"Safety checks: {safety}")

        if args.run:
            if not safety["all_ok"]:
                raise RuntimeError("Refusing robot motion because a waypoint is outside explicit safety limits")
            prepare_robot_for_motion(arm, clean_error=True)

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
                if label == "descend to leaf approach pose":
                    print(f"Holding {args.approach_hold_seconds:.1f} seconds for visual inspection.")
                    time.sleep(args.approach_hold_seconds)
        else:
            print("DRY RUN: vision and route planning completed; no robot motion or GPIO output was commanded.")

        result = {
            "run_mode": bool(args.run),
            "calibration": str(calibration_path),
            "frame_dir": str(frame_dir),
            "tcp_offset_mm_deg": reported_offset,
            "leaf_detection": leaf,
            "raw_leaf_mask_centroid_base_m": raw_leaf_point_m.tolist(),
            "position_bias_config": str(position_bias_path),
            "position_bias_base_mm": (position_bias_m * 1000.0).tolist(),
            "route": [{"label": label, "pose_mm_deg": pose, "motion": kind} for label, pose, kind in route],
            "safety": safety,
        }
        result_path = write_json_report(run_dir, result)
        print(f"Saved result: {result_path}")
        return 0
    finally:
        if arm is not None:
            arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
