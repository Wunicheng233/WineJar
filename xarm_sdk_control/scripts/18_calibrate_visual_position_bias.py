#!/usr/bin/env python3
"""Measure fixed-camera/TCP XY bias by manually jogging above a visual target."""

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

from xarm_sdk_control.positioning_bias import append_bias_sample, load_bias_mm  # noqa: E402
from xarm_sdk_control.end_effector_config import load_end_effector_config  # noqa: E402
from xarm_sdk_control.robot_motion import (  # noqa: E402
    CAMERA_TRANSIT_Z_MM,
    connect_robot,
    load_robot_ip,
    pose_in_overhead_camera_keepout,
    prepare_robot_for_motion,
    require_code,
    require_overhead_camera_safe_route,
)
from xarm_sdk_control.visual_targeting import (  # noqa: E402
    capture_rgbd,
    detections_in_base,
    latest_global_calibration,
    run_vision,
    select_ceramic,
    select_leaf_waypoints,
)


CONFIG_PATH = ROOT / "config" / "robot.yaml"
DEFAULT_OUTPUT = ROOT / "config" / "visual_position_bias.json"
OUTPUT_ROOT = ROOT / "data" / "visual_position_bias"
END_EFFECTOR = load_end_effector_config()




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Enable robot motion; otherwise only detect and print.")
    parser.add_argument("--target", choices=("leaf", "ceramic"), default="leaf")
    parser.add_argument("--calibration", type=Path, default=latest_global_calibration())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--use-existing-bias",
        action="store_true",
        help="Start at the currently configured bias; saved correction remains relative to the raw visual target.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Hold at the predicted/compensated pose without interactive jogging or saving.",
    )
    parser.add_argument(
        "--leave-at-target",
        action="store_true",
        help="On successful verification, disconnect without returning so the physical offset can be measured.",
    )
    parser.add_argument("--hold-seconds", type=float, default=8.0)
    parser.add_argument("--vision-env", default="winejar-vision")
    parser.add_argument("--serial")
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--tcp-offset-z-mm", type=float, default=END_EFFECTOR.tcp_offset_mm_deg[2])
    parser.add_argument("--approach-clearance-mm", type=float, default=35.0)
    parser.add_argument("--transport-clearance-mm", type=float, default=200.0)
    parser.add_argument("--step-mm", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--acc", type=float, default=100.0)
    parser.add_argument("--min-x-mm", type=float)
    parser.add_argument("--max-x-mm", type=float)
    parser.add_argument("--min-y-mm", type=float)
    parser.add_argument("--max-y-mm", type=float)
    parser.add_argument("--min-z-mm", type=float)
    parser.add_argument("--max-z-mm", type=float)
    return parser.parse_args()


def inside_limits(pose: list[float], args: argparse.Namespace) -> bool:
    return (
        args.min_x_mm <= pose[0] <= args.max_x_mm
        and args.min_y_mm <= pose[1] <= args.max_y_mm
        and args.min_z_mm <= pose[2] <= args.max_z_mm
    )


def jog_until_aligned(arm: Any, predicted_pose: list[float], args: argparse.Namespace) -> list[float] | None:
    pose = predicted_pose.copy()
    step = float(args.step_mm)
    print("\nAlign the physical suction-cup center with the visual target.")
    print("Commands: x+ x- y+ y- | step N | save | cancel")
    while True:
        command = input(f"bias-jog (step={step:g} mm)> ").strip().lower()
        if command in {"cancel", "q", "quit"}:
            return None
        if command == "save":
            return pose
        if command.startswith("step "):
            step = float(command.split(maxsplit=1)[1])
            if not 0.1 <= step <= 10.0:
                print("Step must be between 0.1 and 10 mm.")
                step = float(args.step_mm)
            continue
        delta = {
            "x+": (step, 0.0),
            "x-": (-step, 0.0),
            "y+": (0.0, step),
            "y-": (0.0, -step),
        }.get(command)
        if delta is None:
            print("Unknown command. Use x+/x-/y+/y-, step N, save, or cancel.")
            continue
        candidate = pose.copy()
        candidate[0] += delta[0]
        candidate[1] += delta[1]
        if not inside_limits(candidate, args):
            print("Refusing jog: candidate pose is outside the explicit workspace limits.")
            continue
        if pose_in_overhead_camera_keepout(candidate):
            print("Refusing jog: candidate pose enters the overhead-camera keepout.")
            continue
        require_code(
            "jog",
            arm.set_position(
                *candidate,
                speed=args.speed,
                mvacc=args.acc,
                wait=True,
                radius=-1,
                is_radian=False,
            ),
        )
        pose = candidate
        correction = np.asarray(pose[:2]) - np.asarray(predicted_pose[:2])
        print(f"Accumulated XY correction [mm]: {correction.round(2).tolist()}")


def main() -> int:
    args = parse_args()
    required_limits = (args.min_x_mm, args.max_x_mm, args.min_y_mm, args.max_y_mm, args.min_z_mm, args.max_z_mm)
    if args.run and any(value is None for value in required_limits):
        print("ERROR: --run requires explicit min/max X, Y, and Z workspace limits.")
        return 2
    if args.calibration is None or not args.calibration.expanduser().exists():
        raise RuntimeError("Global camera calibration is missing")

    calibration_path = args.calibration.expanduser().resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    base_t_camera = np.asarray(calibration["base_T_camera"], dtype=float)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_dir = OUTPUT_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    frame_dir = capture_rgbd(run_dir, args.serial)
    vision_result = run_vision(frame_dir, args.vision_env, args.min_score)
    detections = detections_in_base(vision_result.get("detections", []), base_t_camera, args.min_score)
    if args.target == "leaf":
        target = select_leaf_waypoints(detections, distinct_distance_m=0.05)[0][0]
    else:
        target = select_ceramic(detections)
    target_mm = np.asarray(target["point_base_m"], dtype=float) * 1000.0
    print(f"Detected {args.target} surface point [mm]: {target_mm.round(2).tolist()}")
    if not args.run:
        print("DRY RUN: no robot motion was commanded.")
        return 0

    robot_ip = load_robot_ip(CONFIG_PATH)
    arm: Any | None = None
    initial_pose: list[float] | None = None
    transit_pose: list[float] | None = None
    initial_column_pose: list[float] | None = None
    completed_at_target = False
    try:
        arm, initial_pose = connect_robot(robot_ip, [0, 0, args.tcp_offset_z_mm, 0, 0, 0])
        orientation = list(map(float, initial_pose[3:6]))
        transit_z = float(target_mm[2] + args.transport_clearance_mm)
        if transit_z > CAMERA_TRANSIT_Z_MM:
            raise RuntimeError(
                f"Target transit Z {transit_z:.1f} mm exceeds the "
                f"{CAMERA_TRANSIT_Z_MM:.1f} mm camera-safe corridor"
            )
        initial_column_pose = [
            float(initial_pose[0]),
            float(initial_pose[1]),
            transit_z,
            *orientation,
        ]
        transit_pose = [float(target_mm[0]), float(target_mm[1]), transit_z, *orientation]
        raw_predicted_pose = [
            float(target_mm[0]),
            float(target_mm[1]),
            float(target_mm[2] + args.approach_clearance_mm),
            *orientation,
        ]
        starting_bias_mm = (
            load_bias_mm(args.output.expanduser().resolve()) if args.use_existing_bias else np.zeros(3, dtype=float)
        )
        predicted_pose = raw_predicted_pose.copy()
        predicted_pose[0] += float(starting_bias_mm[0])
        predicted_pose[1] += float(starting_bias_mm[1])
        transit_pose[0] += float(starting_bias_mm[0])
        transit_pose[1] += float(starting_bias_mm[1])
        if args.use_existing_bias:
            print(f"Starting from existing XY bias [mm]: {starting_bias_mm[:2].round(2).tolist()}")
        if not inside_limits(transit_pose, args) or not inside_limits(predicted_pose, args):
            raise RuntimeError("Predicted calibration route is outside the explicit workspace limits")
        require_overhead_camera_safe_route(
            [list(map(float, initial_pose)), initial_column_pose, transit_pose, predicted_pose]
        )

        prepare_robot_for_motion(arm, clean_error=True)
        require_code(
            "move vertically to camera-safe transit height",
            arm.set_position(
                *initial_column_pose,
                speed=args.speed,
                mvacc=args.acc,
                wait=True,
                radius=-1,
                is_radian=False,
            ),
        )
        require_code(
            "move directly above target",
            arm.set_position(*transit_pose, speed=args.speed, mvacc=args.acc, wait=True, radius=-1, is_radian=False),
        )
        require_code(
            "descend to inspection height",
            arm.set_position(*predicted_pose, speed=args.speed / 2, mvacc=args.acc, wait=True, radius=-1, is_radian=False),
        )
        if args.verify_only:
            print(f"VERIFY ONLY: holding {args.hold_seconds:.1f} seconds; no sample will be saved.")
            time.sleep(args.hold_seconds)
            completed_at_target = True
            return 0
        aligned_pose = jog_until_aligned(arm, predicted_pose, args)
        if aligned_pose is None:
            print("Cancelled; no bias sample was saved.")
            return 1

        correction = np.array(
            [aligned_pose[0] - raw_predicted_pose[0], aligned_pose[1] - raw_predicted_pose[1], 0.0],
            dtype=float,
        )
        sample = {
            "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "target": args.target,
            "frame_dir": str(frame_dir),
            "calibration": str(calibration_path),
            "raw_predicted_pose_mm_deg": raw_predicted_pose,
            "starting_bias_base_mm": starting_bias_mm.tolist(),
            "aligned_pose_mm_deg": aligned_pose,
            "correction_base_mm": correction.tolist(),
        }
        result = append_bias_sample(args.output.expanduser().resolve(), sample)
        print(f"Saved sample correction [mm]: {correction.round(2).tolist()}")
        print(f"Median system bias [mm]: {np.asarray(result['bias_base_mm']).round(2).tolist()}")
        print(f"Sample range [mm]: {np.asarray(result['sample_range_base_mm']).round(2).tolist()}")
        print(f"Saved: {args.output.expanduser().resolve()}")
        completed_at_target = True
        return 0
    finally:
        should_return = not (args.leave_at_target and completed_at_target)
        if arm is not None and arm.connected and initial_pose is not None and should_return:
            try:
                if transit_pose is not None:
                    arm.set_position(*transit_pose, speed=args.speed, mvacc=args.acc, wait=True, radius=-1, is_radian=False)
                if initial_column_pose is not None:
                    arm.set_position(
                        *initial_column_pose,
                        speed=args.speed,
                        mvacc=args.acc,
                        wait=True,
                        radius=-1,
                        is_radian=False,
                    )
                arm.set_position(*initial_pose, speed=args.speed, mvacc=args.acc, wait=True, radius=-1, is_radian=False)
            except Exception as exc:
                print(f"WARNING: automatic safe return failed: {exc}", file=sys.stderr)
        elif arm is not None and arm.connected and not should_return:
            print("Leaving the robot at the verified target by request; no automatic return was commanded.")
        if arm is not None:
            arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
