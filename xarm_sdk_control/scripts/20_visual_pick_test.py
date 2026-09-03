#!/usr/bin/env python3
"""Visually test-pick a material, optionally transferring it onto a ceramic disk."""

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
    build_pick_place_test_route,
    build_pick_test_route,
    execute_pick_test_route,
    profile_for,
    recover_failed_pick_test,
    resolve_pick_test_settings,
)
from xarm_sdk_control.cross_place_retry import DEFAULT_RETRY_OFFSETS_MM  # noqa: E402
from xarm_sdk_control.cross_place_route import require_distinct_pick_and_place_targets  # noqa: E402
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
from xarm_sdk_control.suction_sensor import SuctionNotConfirmedError, SuctionSensor  # noqa: E402
from xarm_sdk_control.task_timing import TaskTimeline  # noqa: E402
from xarm_sdk_control.vacuum_control import VacuumController  # noqa: E402
from xarm_sdk_control.visual_targeting import (  # noqa: E402
    DEFAULT_VISION_SOCKET,
    latest_global_calibration,
    load_base_t_camera,
    run_scan,
    select_all_of_kind,
)


CONFIG_PATH = ROOT / "config" / "robot.yaml"
POSITION_BIAS_PATH = ROOT / "config" / "visual_position_bias.json"
OUTPUT_ROOT = ROOT / "data" / "visual_pick_test"
END_EFFECTOR = load_end_effector_config()
CERAMIC_DESTINATION_PROMPT = "round brown ceramic disk."
CERAMIC_DESTINATION_MIN_SCORE = 0.45
MIN_PICK_PLACE_SEPARATION_M = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Enable robot motion; default is camera/route dry-run.")
    parser.add_argument(
        "--material",
        choices=("ceramic", "lotus-leaf", "paper", "bamboo"),
        default="ceramic",
        help="Which material to pick and return; drives the vision prompt and per-material defaults.",
    )
    add_vision_common_args(parser, latest_global_calibration(), min_score_default=None)
    parser.add_argument("--vision-socket", type=Path, default=DEFAULT_VISION_SOCKET)
    parser.add_argument("--require-vision-service", action="store_true")
    parser.add_argument("--position-bias-config", type=Path, default=POSITION_BIAS_PATH)
    parser.add_argument(
        "--compression-mm",
        type=float,
        default=None,
        help="Spring compression; defaults: ceramic/lotus-leaf/paper 13 mm, bamboo 11 mm.",
    )
    parser.add_argument("--approach-clearance-mm", type=float, default=35.0)
    parser.add_argument(
        "--place-on-ceramic",
        action="store_true",
        help="Visually locate a ceramic disk and transfer each selected material onto it.",
    )
    parser.add_argument("--transport-clearance-mm", type=float, default=150.0)
    parser.add_argument("--placement-clearance-mm", type=float, default=0.0)
    parser.add_argument(
        "--release-clearance-mm",
        type=float,
        default=10.0,
        help="Height above the detected surface at which the held material is released; "
             "raise it if the material presses into the table at release.",
    )
    parser.add_argument(
        "--pick-offset-x-mm",
        type=float,
        default=0.0,
        help="X offset of the suction point relative to the detected center "
             "(applied after the shared positioning bias).",
    )
    parser.add_argument(
        "--pick-offset-y-mm",
        type=float,
        default=0.0,
        help="Y offset of the suction point relative to the detected center "
             "(applied after the shared positioning bias).",
    )
    parser.add_argument(
        "--test-lift-mm",
        type=float,
        default=None,
        help="Vertical lift height for operator inspection; per-material default 100 mm.",
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
    profile = profile_for(args.material)
    min_score, compression_mm, test_lift_mm = resolve_pick_test_settings(
        profile,
        min_score=args.min_score,
        compression_mm=args.compression_mm,
        test_lift_mm=args.test_lift_mm,
    )
    profile = replace(profile, test_lift_mm=test_lift_mm)
    if not 0.0 <= compression_mm <= MAX_COMPRESSION_MM:
        raise ValueError(
            f"--compression-mm must remain within the configured 0~{MAX_COMPRESSION_MM:g} mm spring travel"
        )
    if min(args.vacuum_settle_seconds, args.release_settle_seconds, args.inspection_hold_seconds) < 0.0:
        raise ValueError("Hold and settle times must be non-negative")
    if min(args.transit_speed, args.precision_speed, args.acc) <= 0.0:
        raise ValueError("Motion speeds and acceleration must be positive")
    if min(
        args.approach_clearance_mm,
        args.transport_clearance_mm,
        args.placement_clearance_mm,
    ) < 0.0:
        raise ValueError("Route clearances must be non-negative")
    if test_lift_mm <= 0.0:
        raise ValueError("--test-lift-mm must be positive")
    if args.run and not explicit_workspace_limits(args):
        print("ERROR: --run requires explicit min/max X, Y, and Z workspace limits.")
        return 3
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
            if END_EFFECTOR.suction_sensor is None:
                raise RuntimeError(
                    "Suction confirmation is mandatory: add a suction_sensor section to "
                    "config/end_effector.yaml (input_channel, success_value, debounce_seconds, timeout_seconds)."
                )
            sensor = SuctionSensor(arm, END_EFFECTOR.suction_sensor)
        else:
            timeline.mark("reset_skipped")
            home_pose = list(map(float, END_EFFECTOR.home_pose_mm_deg))

        timeline.mark("productive_cycle_started")
        frame_dir, detections = run_scan(
            run_dir / "scan" / "captures",
            base_t_camera,
            args.vision_env,
            min_score,
            camera=camera,
            text_prompt=profile.text_prompt,
            service_socket=args.vision_socket,
            require_service=args.require_vision_service,
            timeline=timeline,
            timeline_prefix="scan",
        )
        items = [dict(item) for item in select_all_of_kind(profile.kind, detections, min_score)]
        for item in items:
            point = np.asarray(item["point_base_m"], dtype=float)
            point += bias_mm / 1000.0
            point[0] += args.pick_offset_x_mm / 1000.0
            point[1] += args.pick_offset_y_mm / 1000.0
            item["point_base_m"] = point.tolist()

        destination: dict[str, Any] | None = None
        destination_frame: Path | None = None
        if args.place_on_ceramic:
            destination_frame, destination_detections = run_scan(
                run_dir / "destination_scan" / "captures",
                base_t_camera,
                args.vision_env,
                CERAMIC_DESTINATION_MIN_SCORE,
                camera=camera,
                text_prompt=CERAMIC_DESTINATION_PROMPT,
                service_socket=args.vision_socket,
                require_service=args.require_vision_service,
                timeline=timeline,
                timeline_prefix="destination_scan",
            )
            destination_candidates = [
                dict(item)
                for item in select_all_of_kind(
                    "ceramic",
                    destination_detections,
                    CERAMIC_DESTINATION_MIN_SCORE,
                    display_name="ceramic disk",
                )
            ]
            source_points = [np.asarray(item["point_base_m"], dtype=float) for item in items]
            separated_candidates: list[dict[str, Any]] = []
            for candidate in destination_candidates:
                candidate_point = np.asarray(candidate["point_base_m"], dtype=float)
                if all(
                    np.linalg.norm(candidate_point[:2] - source_point[:2])
                    >= MIN_PICK_PLACE_SEPARATION_M
                    for source_point in source_points
                ):
                    separated_candidates.append(candidate)
            if not separated_candidates:
                raise RuntimeError(
                    "No ceramic disk candidate is spatially distinct from the source material; "
                    "check the annotated destination image before enabling robot motion"
                )
            destination = max(
                separated_candidates,
                key=lambda item: float(item.get("score", 0.0)),
            )
            destination_point = np.asarray(destination["point_base_m"], dtype=float)
            destination_point += bias_mm / 1000.0
            destination["point_base_m"] = destination_point.tolist()
            print(
                "Ceramic destination [mm]: "
                f"{(destination_point * 1000.0).round(2).tolist()}"
            )

        print(f"Calibration: {calibration_path}")
        print(f"Material: {args.material} (kind={profile.kind!r}, prompt={profile.text_prompt!r})")
        print(f"Position bias [mm]: {bias_mm.round(2).tolist()}")
        print(f"Detected {len(items)} item(s); processing best confidence first.")
        for index, item in enumerate(items, start=1):
            source_mm = (np.asarray(item["point_base_m"], dtype=float) * 1000.0).round(2).tolist()
            print(f"  item {index}: source [mm] = {source_mm}, score = {float(item.get('score', 0.0)):.3f}")

        for index, item in enumerate(items, start=1):
            cycle_name = f"item_{index:02d}"
            timeline.mark(f"{cycle_name}.pick_started")
            source_point_m = list(map(float, item["point_base_m"]))
            source_mm = (np.asarray(source_point_m, dtype=float) * 1000.0).round(2).tolist()
            if destination is None:
                route = build_pick_test_route(
                    home_pose_mm_deg=list(map(float, home_pose)),
                    source_point_m=source_point_m,
                    profile=profile,
                    compression_mm=compression_mm,
                    approach_clearance_mm=args.approach_clearance_mm,
                    release_clearance_mm=args.release_clearance_mm,
                    inspection_event="hold",
                )
            else:
                require_distinct_pick_and_place_targets(
                    source_point_m,
                    destination["point_base_m"],
                )
                route = build_pick_place_test_route(
                    home_pose_mm_deg=list(map(float, home_pose)),
                    source_point_m=source_point_m,
                    destination_point_m=destination["point_base_m"],
                    compression_mm=compression_mm,
                    transport_clearance_mm=args.transport_clearance_mm,
                    approach_clearance_mm=args.approach_clearance_mm,
                    placement_clearance_mm=args.placement_clearance_mm,
                )
            safety = route_safety_report([step.pose_mm_deg for step in route], args)
            if not safety["all_ok"]:
                raise RuntimeError(f"Item {index} route is outside explicit workspace limits")
            disk_records.append(
                {
                    "item": index,
                    "targeting_frame": str(frame_dir),
                    "destination_frame": str(destination_frame) if destination_frame else None,
                    "source_point_mm": source_mm,
                    "destination_point_mm": (
                        (np.asarray(destination["point_base_m"]) * 1000.0).round(2).tolist()
                        if destination is not None
                        else None
                    ),
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
                    f"DRY RUN item {index}: route validated at {source_mm}; "
                    "no robot motion or GPIO output was commanded."
                )
                timeline.mark(f"{cycle_name}.pick_complete", dry_run=True)
                continue

            assert arm is not None and vacuum is not None
            print(f"\n=== {args.material} {index}/{len(items)} at {source_mm} ===")
            attempts = []
            last_error: SuctionNotConfirmedError | None = None
            for attempt in range(1, 4):
                offset_x_mm, offset_y_mm = DEFAULT_RETRY_OFFSETS_MM[(attempt - 1) % len(DEFAULT_RETRY_OFFSETS_MM)]
                attempt_point_m = [
                    source_point_m[0] + offset_x_mm / 1000.0,
                    source_point_m[1] + offset_y_mm / 1000.0,
                    source_point_m[2],
                ]
                attempt_mm = (np.asarray(attempt_point_m, dtype=float) * 1000.0).round(2).tolist()
                if attempt > 1:
                    print(f"RETRY attempt {attempt}: offset [{offset_x_mm:+.1f}, {offset_y_mm:+.1f}] mm -> {attempt_mm}")
                    if destination is None:
                        route = build_pick_test_route(
                            home_pose_mm_deg=list(map(float, home_pose)),
                            source_point_m=attempt_point_m,
                            profile=profile,
                            compression_mm=compression_mm,
                            approach_clearance_mm=args.approach_clearance_mm,
                            release_clearance_mm=args.release_clearance_mm,
                            inspection_event="hold",
                        )
                    else:
                        route = build_pick_place_test_route(
                            home_pose_mm_deg=list(map(float, home_pose)),
                            source_point_m=attempt_point_m,
                            destination_point_m=destination["point_base_m"],
                            compression_mm=compression_mm,
                            transport_clearance_mm=args.transport_clearance_mm,
                            approach_clearance_mm=args.approach_clearance_mm,
                            placement_clearance_mm=args.placement_clearance_mm,
                        )
                    safety = route_safety_report([step.pose_mm_deg for step in route], args)
                    if not safety["all_ok"]:
                        raise RuntimeError(f"Item {index} retry route is outside explicit workspace limits")
                try:
                    execute_pick_test_route(
                        arm,
                        vacuum,
                        route,
                        args,
                        holding_state,
                        label_prefix=f"disk {index} ",
                        suction_sensor=sensor,
                    )
                except SuctionNotConfirmedError as exc:
                    last_error = exc
                    print(f"\nSUCTION NOT CONFIRMED on attempt {attempt}: {exc}")
                    timeline.mark(
                        f"{cycle_name}.suction_failed",
                        attempt=attempt,
                        offset_mm=[offset_x_mm, offset_y_mm],
                        elapsed_seconds=exc.elapsed_seconds,
                        timeout_seconds=exc.timeout_seconds,
                        last_value=exc.last_value,
                    )
                    attempts.append(
                        {
                            "attempt": attempt,
                            "offset_mm": [offset_x_mm, offset_y_mm],
                            "confirmed": False,
                            "elapsed_seconds": exc.elapsed_seconds,
                            "timeout_seconds": exc.timeout_seconds,
                            "last_value": exc.last_value,
                        }
                    )
                    holding_state["value"] = False  # the sensor proves nothing is held
                    if vacuum is not None:
                        vacuum.safe_shutdown()
                    recover_failed_pick_test(
                        arm,
                        route,
                        args,
                        return_home=(attempt == 3),
                        label_prefix=f"item {index} attempt {attempt} ",
                    )
                    continue
                confirmed_at = sensor.last_wait
                attempts.append(
                    {
                        "attempt": attempt,
                        "offset_mm": [offset_x_mm, offset_y_mm],
                        "confirmed": True,
                        "elapsed_seconds": confirmed_at.elapsed_seconds if confirmed_at else None,
                    }
                )
                disk_records[-1]["suction"] = {
                    "confirmed": True,
                    "attempt": attempt,
                    "elapsed_seconds": confirmed_at.elapsed_seconds if confirmed_at else None,
                }
                break
            else:
                assert last_error is not None
                disk_records[-1]["suction"] = {"confirmed": False, "attempts": attempts}
                result = {
                    "status": "failed",
                    "run_mode": bool(args.run),
                    "material": args.material,
                    "error": str(last_error),
                    "items": disk_records,
                    "picked_count": index - 1,
                    "timing": timeline.to_dict(productive_start="productive_cycle_started"),
                }
                write_json_report(run_dir, result)
                return 2
            disk_records[-1]["suction_attempts"] = attempts
            timeline.mark(f"{cycle_name}.pick_complete")

        if vacuum is not None:
            vacuum.safe_shutdown()
        timeline.mark("task_complete")
        timing = timeline.to_dict(productive_start="productive_cycle_started")
        result = {
            "status": "success",
            "run_mode": bool(args.run),
            "material": args.material,
            "min_score": min_score,
            "calibration": str(calibration_path),
            "position_bias_mm": bias_mm.tolist(),
            "pick_offset_x_mm": args.pick_offset_x_mm,
            "pick_offset_y_mm": args.pick_offset_y_mm,
            "compression_mm": compression_mm,
            "release_clearance_mm": args.release_clearance_mm,
            "test_lift_mm": profile.test_lift_mm,
            "place_on_ceramic": bool(args.place_on_ceramic),
            "transport_clearance_mm": args.transport_clearance_mm,
            "placement_clearance_mm": args.placement_clearance_mm,
            "items": disk_records,
            "picked_count": len(items) if args.run else 0,
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
