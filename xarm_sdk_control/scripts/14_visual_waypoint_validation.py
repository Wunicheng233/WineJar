#!/usr/bin/env python3
"""Validate the real fixed-camera vision-to-xArm waypoint chain without grasping.

The default mode captures RGB-D, runs GroundingDINO + SAM2, and prints the
planned route. Robot motion requires --run and explicit workspace limits.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "robot.yaml"
OUTPUT_ROOT = ROOT / "data" / "visual_waypoint_validation"
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from xarm_sdk_control.visual_targeting import (  # noqa: E402
    capture_rgbd,
    detections_in_base,
    latest_global_calibration,
    make_waypoint,
    run_vision,
    safety_report,
    select_ceramic,
    select_leaf_waypoints,
    transform_point,
)


def require_code(label: str, result: Any) -> Any:
    code = result[0] if isinstance(result, tuple) else result
    print(f"{label}: {result}")
    if code not in (0, None):
        raise RuntimeError(f"{label} failed with code/result: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Command the real robot; default is vision/planning only.")
    parser.add_argument("--calibration", type=Path, default=latest_global_calibration())
    parser.add_argument("--vision-env", default="winejar-vision")
    parser.add_argument("--serial", help="Optional RealSense serial number.")
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--max-calibration-residual-mm", type=float, default=15.0)
    parser.add_argument("--above-mm", type=float, default=150.0)
    parser.add_argument("--distinct-leaf-distance-mm", type=float, default=50.0)
    parser.add_argument("--speed", type=float, default=35.0)
    parser.add_argument("--acc", type=float, default=180.0)
    parser.add_argument("--min-x-mm", type=float)
    parser.add_argument("--max-x-mm", type=float)
    parser.add_argument("--min-y-mm", type=float)
    parser.add_argument("--max-y-mm", type=float)
    parser.add_argument("--min-z-mm", type=float)
    parser.add_argument("--max-z-mm", type=float)
    args = parser.parse_args()

    if args.calibration is None or not args.calibration.expanduser().exists():
        print("ERROR: fixed global-camera calibration is missing.")
        print("Collect samples with 11_collect_global_camera_sample.py, then solve with 12_solve_global_camera_calibration.py.")
        return 2
    if args.run and any(
        value is None
        for value in (args.min_x_mm, args.max_x_mm, args.min_y_mm, args.max_y_mm, args.min_z_mm, args.max_z_mm)
    ):
        print("ERROR: --run requires explicit min/max X, Y, and Z workspace limits.")
        return 3

    calibration_path = args.calibration.expanduser().resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    base_t_camera = np.asarray(calibration["base_T_camera"], dtype=float)
    if base_t_camera.shape != (4, 4):
        raise RuntimeError(f"Invalid base_T_camera shape: {base_t_camera.shape}")
    residual_max_mm = calibration.get("quality", {}).get("residual_max_mm")
    if residual_max_mm is None or float(residual_max_mm) > args.max_calibration_residual_mm:
        print(
            "ERROR: calibration quality is missing or exceeds the allowed maximum residual "
            f"({residual_max_mm} mm > {args.max_calibration_residual_mm} mm)."
        )
        return 4

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_dir = OUTPUT_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    frame_dir = capture_rgbd(run_dir, args.serial)
    vision_result = run_vision(frame_dir, args.vision_env, args.min_score)
    detections = detections_in_base(vision_result.get("detections", []), base_t_camera, args.min_score)
    ceramic = select_ceramic(detections)
    leaves, stacked_or_occluded = select_leaf_waypoints(
        detections,
        args.distinct_leaf_distance_mm / 1000.0,
    )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        robot_ip = str(yaml.safe_load(file)["robot_ip"])
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(do_not_open=True, is_radian=False)
    initial_pose: list[float] | None = None
    try:
        require_code("connect", arm.connect(port=robot_ip))
        pose_code, initial_pose = arm.get_position(is_radian=False)
        require_code("get initial TCP pose", (pose_code, initial_pose))
        orientation = [float(value) for value in initial_pose[3:6]]

        leaf_1_pose = make_waypoint(leaves[0]["point_base_m"], args.above_mm, orientation)
        ceramic_pose = make_waypoint(ceramic["point_base_m"], args.above_mm, orientation)
        leaf_2_pose = make_waypoint(leaves[1]["point_base_m"], args.above_mm, orientation)
        route = [leaf_1_pose, ceramic_pose, leaf_2_pose, ceramic_pose, list(initial_pose)]

        limits = {
            "min_x_mm": args.min_x_mm if args.min_x_mm is not None else -1000.0,
            "max_x_mm": args.max_x_mm if args.max_x_mm is not None else 1000.0,
            "min_y_mm": args.min_y_mm if args.min_y_mm is not None else -1000.0,
            "max_y_mm": args.max_y_mm if args.max_y_mm is not None else 1000.0,
            "min_z_mm": args.min_z_mm if args.min_z_mm is not None else -1000.0,
            "max_z_mm": args.max_z_mm if args.max_z_mm is not None else 1000.0,
        }
        safety = safety_report(route, **limits)

        print(f"Calibration: {calibration_path}")
        print(f"Leaf 1 base point [mm]: {(np.asarray(leaves[0]['point_base_m']) * 1000.0).round(2).tolist()}")
        print(f"Ceramic base point [mm]: {(np.asarray(ceramic['point_base_m']) * 1000.0).round(2).tolist()}")
        print(f"Leaf 2 base point [mm]: {(np.asarray(leaves[1]['point_base_m']) * 1000.0).round(2).tolist()}")
        if stacked_or_occluded:
            print("Leaf 2 is not independently visible; reusing the detected leaf-stack center.")
        for index, pose in enumerate(route, start=1):
            print(f"Waypoint {index}: {[round(value, 2) for value in pose]}")
        print(f"Safety checks: {safety}")

        if args.run:
            if not safety["all_ok"]:
                raise RuntimeError("Refusing robot motion because a waypoint is outside explicit safety limits")
            require_code("clean error", arm.clean_error())
            require_code("clean warn", arm.clean_warn())
            require_code("motion enable", arm.motion_enable(enable=True))
            require_code("set mode 0", arm.set_mode(0))
            require_code("set state 0", arm.set_state(0))
            for index, pose in enumerate(route, start=1):
                require_code(
                    f"move waypoint {index}",
                    arm.set_position(
                        *pose,
                        speed=args.speed,
                        mvacc=args.acc,
                        wait=True,
                        radius=-1,
                        is_radian=False,
                    ),
                )
        else:
            print("DRY RUN: robot was queried for its initial pose but no motion was commanded.")

        result = {
            "run_mode": bool(args.run),
            "calibration": str(calibration_path),
            "frame_dir": str(frame_dir),
            "minimum_confidence": args.min_score,
            "detections_base": detections,
            "leaf_2_reuses_stack_center": stacked_or_occluded,
            "route_mm_deg": route,
            "safety": safety,
        }
        result_path = run_dir / "result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(f"Saved result: {result_path}")
        return 0
    finally:
        arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
