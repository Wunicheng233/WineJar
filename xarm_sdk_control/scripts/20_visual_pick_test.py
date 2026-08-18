#!/usr/bin/env python3
"""Visually pick up and return every ceramic disk detected in one RGB-D frame."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.end_effector_config import load_end_effector_config  # noqa: E402
from xarm_sdk_control.pick_test_workflow import (  # noqa: E402
    MAX_COMPRESSION_MM,
    build_pick_test_route,
    execute_pick_test_route,
    profile_for,
)
from xarm_sdk_control.positioning_bias import load_bias_mm  # noqa: E402
from xarm_sdk_control.realsense_capture import RealSenseCaptureSession  # noqa: E402
from xarm_sdk_control.cli_utils import (  # noqa: E402
    add_vision_common_args,
    add_workspace_limit_args,
    explicit_workspace_limits,
    route_safety_report,
)
from xarm_sdk_control.run_output import new_timestamped_run_dir, write_json_report  # noqa: E402
from xarm_sdk_control.robot_motion import (  # noqa: E402
    connect_robot,
    load_robot_ip,
    prepare_robot_for_motion,
    reset_robot_at_ip,
)
from xarm_sdk_control.task_timing import TaskTimeline  # noqa: E402
from xarm_sdk_control.vacuum_control import VacuumController  # noqa: E402
from xarm_sdk_control.visual_targeting import (  # noqa: E402
    DEFAULT_VISION_SOCKET,
    latest_global_calibration,
    load_base_t_camera,
    run_scan,
    select_all_ceramic,
)


CONFIG_PATH = ROOT / "config" / "robot.yaml"
POSITION_BIAS_PATH = ROOT / "config" / "visual_position_bias.json"
OUTPUT_ROOT = ROOT / "data" / "visual_pick_test"
END_EFFECTOR = load_end_effector_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Enable robot motion; default is camera/route dry-run.")
    parser.add_argument("--confirm-no-pressure-sensor", action="store_true")
    add_vision_common_args(parser, latest_global_calibration())
    parser.add_argument("--vision-socket", type=Path, default=DEFAULT_VISION_SOCKET)
    parser.add_argument("--require-vision-service", action="store_true")
    parser.add_argument("--position-bias-config", type=Path, default=POSITION_BIAS_PATH)
    parser.add_argument(
        "--compression-mm",
        type=float,
        default=6.0,
        help="Ceramic-disk spring compression; keep shallow on a rigid table to avoid joint overload.",
    )
    parser.add_argument("--approach-clearance-mm", type=float, default=35.0)
    parser.add_argument(
        "--release-clearance-mm",
        type=float,
        default=10.0,
        help="Height above the detected surface at which the held disk is released; "
             "raise it if the disk presses into the table at release.",
    )
    parser.add_argument(
        "--pick-offset-x-mm",
        type=float,
        default=5.0,
        help="X offset of the suction point relative to the detected disk center "
             "(applied after the shared positioning bias). Tune until the disk hangs level.",
    )
    parser.add_argument(
        "--test-lift-mm",
        type=float,
        default=40.0,
        help="Vertical lift height for operator inspection; raise it if the disk "
             "does not fully clear the table surface.",
    )
    parser.add_argument("--vacuum-settle-seconds", type=float, default=0.9)
    parser.add_argument("--release-settle-seconds", type=float, default=0.5)
    parser.add_argument(
        "--inspection-hold-seconds",
        type=float,
        default=2.0,
        help="Hold every lifted disk at inspection height for operator observation.",
    )
    parser.add_argument("--transit-speed", type=float, default=80.0)
    parser.add_argument("--precision-speed", type=float, default=15.0)
    parser.add_argument("--acc", type=float, default=150.0)
    add_workspace_limit_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = replace(profile_for("ceramic"), test_lift_mm=args.test_lift_mm)
    if not 0.0 <= args.compression_mm <= MAX_COMPRESSION_MM:
        raise ValueError(
            f"--compression-mm must remain within the configured 0~{MAX_COMPRESSION_MM:g} mm spring travel"
        )
    if min(args.vacuum_settle_seconds, args.release_settle_seconds, args.inspection_hold_seconds) < 0.0:
        raise ValueError("Hold and settle times must be non-negative")
    if min(args.transit_speed, args.precision_speed, args.acc) <= 0.0:
        raise ValueError("Motion speeds and acceleration must be positive")
    if args.approach_clearance_mm < 0.0:
        raise ValueError("--approach-clearance-mm must be non-negative")
    if args.test_lift_mm <= 0.0:
        raise ValueError("--test-lift-mm must be positive")
    if args.run and not explicit_workspace_limits(args):
        print("ERROR: --run requires explicit min/max X, Y, and Z workspace limits.")
        return 3
    if args.run and not args.confirm_no_pressure_sensor:
        print("ERROR: --run requires --confirm-no-pressure-sensor.")
        return 2
    if args.calibration is None:
        raise FileNotFoundError("Fixed global-camera calibration is missing")
    calibration_path, base_t_camera = load_base_t_camera(args.calibration)
    bias_mm = load_bias_mm(args.position_bias_config.expanduser().resolve())
    robot_ip = load_robot_ip(CONFIG_PATH)

    timeline = TaskTimeline()
    timeline.mark("program_started")
    run_dir = new_timestamped_run_dir(OUTPUT_ROOT)
    camera = RealSenseCaptureSession(serial=args.serial)
    arm: Any | None = None
    vacuum: VacuumController | None = None
    holding_state = {"value": False}
    disk_records: list[dict[str, Any]] = []
    try:
        camera.start()
        timeline.mark("camera_session_ready")
        if args.run:
            print("Moving to the configured camera-clear reset pose before the pick test.")
            timeline.mark("reset_started")
            reset_robot_at_ip(
                robot_ip,
                END_EFFECTOR.tcp_offset_mm_deg,
                END_EFFECTOR.home_pose_mm_deg,
                args.transit_speed,
                args.acc,
            )
            timeline.mark("reset_complete")
            arm, home_pose = connect_robot(robot_ip, END_EFFECTOR.tcp_offset_mm_deg)
            prepare_robot_for_motion(arm)
            vacuum = VacuumController(arm, END_EFFECTOR.vacuum_io)
            vacuum.configure()
        else:
            timeline.mark("reset_skipped")
            home_pose = list(map(float, END_EFFECTOR.home_pose_mm_deg))

        timeline.mark("productive_cycle_started")
        frame_dir, detections = run_scan(
            run_dir / "scan" / "captures",
            base_t_camera,
            args.vision_env,
            args.min_score,
            camera=camera,
            text_prompt="ceramic disk.",
            service_socket=args.vision_socket,
            require_service=args.require_vision_service,
            timeline=timeline,
            timeline_prefix="scan",
        )
        disks = [dict(item) for item in select_all_ceramic(detections, args.min_score)]
        for item in disks:
            point = np.asarray(item["point_base_m"], dtype=float)
            point += bias_mm / 1000.0
            point[0] += args.pick_offset_x_mm / 1000.0
            item["point_base_m"] = point.tolist()

        print(f"Calibration: {calibration_path}")
        print(f"Position bias [mm]: {bias_mm.round(2).tolist()}")
        print(f"Detected {len(disks)} ceramic disk(s); processing best confidence first.")
        for index, disk in enumerate(disks, start=1):
            source_mm = (np.asarray(disk["point_base_m"], dtype=float) * 1000.0).round(2).tolist()
            print(f"  disk {index}: source [mm] = {source_mm}, score = {float(disk.get('score', 0.0)):.3f}")

        for index, disk in enumerate(disks, start=1):
            cycle_name = f"disk_{index:02d}"
            timeline.mark(f"{cycle_name}.pick_started")
            source_point_m = list(map(float, disk["point_base_m"]))
            source_mm = (np.asarray(source_point_m, dtype=float) * 1000.0).round(2).tolist()
            route = build_pick_test_route(
                home_pose_mm_deg=list(map(float, home_pose)),
                source_point_m=source_point_m,
                profile=profile,
                compression_mm=args.compression_mm,
                approach_clearance_mm=args.approach_clearance_mm,
                release_clearance_mm=args.release_clearance_mm,
                inspection_event="hold",
            )
            safety = route_safety_report([step.pose_mm_deg for step in route], args)
            if not safety["all_ok"]:
                raise RuntimeError(f"Disk {index} route is outside explicit workspace limits")
            disk_records.append(
                {
                    "disk": index,
                    "targeting_frame": str(frame_dir),
                    "source_point_mm": source_mm,
                    "route": [
                        {
                            "label": step.label,
                            "pose_mm_deg": step.pose_mm_deg,
                            "motion": step.motion,
                            "event": step.event,
                        }
                        for step in route
                    ],
                    "safety": safety,
                }
            )
            if not args.run:
                print(
                    f"DRY RUN disk {index}: route validated at {source_mm}; "
                    "no robot motion or GPIO output was commanded."
                )
                timeline.mark(f"{cycle_name}.pick_complete", dry_run=True)
                continue

            assert arm is not None and vacuum is not None
            print(f"\n=== Ceramic disk {index}/{len(disks)} at {source_mm} ===")
            execute_pick_test_route(
                arm,
                vacuum,
                route,
                args,
                holding_state,
                label_prefix=f"disk {index} ",
            )
            timeline.mark(f"{cycle_name}.pick_complete")

        if vacuum is not None:
            vacuum.safe_shutdown()
        timeline.mark("task_complete")
        timing = timeline.to_dict(productive_start="productive_cycle_started")
        result = {
            "status": "success",
            "run_mode": bool(args.run),
            "calibration": str(calibration_path),
            "position_bias_mm": bias_mm.tolist(),
            "pick_offset_x_mm": args.pick_offset_x_mm,
            "compression_mm": args.compression_mm,
            "release_clearance_mm": args.release_clearance_mm,
            "test_lift_mm": profile.test_lift_mm,
            "disks": disk_records,
            "picked_count": len(disks) if args.run else 0,
            "timing": timing,
        }
        result_path = write_json_report(run_dir, result)
        print(f"Saved result: {result_path}")
        return 0
    except Exception:
        if holding_state["value"]:
            print(
                "EMERGENCY NOTE: vacuum remains ON because a disk may be held. "
                "Make the area safe before manually releasing it.",
                file=sys.stderr,
            )
        elif vacuum is not None:
            vacuum.safe_shutdown()
        raise
    finally:
        if vacuum is not None and not holding_state["value"]:
            vacuum.safe_shutdown()
        if arm is not None:
            arm.disconnect()
        camera.stop(raise_recording_error=False)


if __name__ == "__main__":
    raise SystemExit(main())
