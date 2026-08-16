#!/usr/bin/env python3
"""Visually target, lift, and return one bamboo leaf or ceramic disk for operator inspection."""

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
from xarm_sdk_control.pick_test_workflow import (  # noqa: E402
    MAX_COMPRESSION_MM,
    PickTestProfile,
    PickTestStep,
    build_pick_test_route,
    profile_for,
)
from xarm_sdk_control.pick_verification import (  # noqa: E402
    PickVerification,
    policy_for,
    verify_carried_object,
)
from xarm_sdk_control.positioning_bias import load_bias_mm  # noqa: E402
from xarm_sdk_control.realsense_capture import RealSenseCaptureSession  # noqa: E402
from xarm_sdk_control.cli_utils import (  # noqa: E402
    add_vision_common_args,
    add_workspace_limit_args,
    route_safety_report,
)
from xarm_sdk_control.run_output import new_timestamped_run_dir, write_json_report  # noqa: E402
from xarm_sdk_control.robot_motion import (  # noqa: E402
    connect_robot,
    load_robot_ip,
    prepare_robot_for_motion,
    require_code,
    reset_robot_at_ip,
)
from xarm_sdk_control.vacuum_control import VacuumController  # noqa: E402
from xarm_sdk_control.visual_targeting import (  # noqa: E402
    DEFAULT_MAX_CALIBRATION_RESIDUAL_MM,
    DEFAULT_VISION_SOCKET,
    latest_global_calibration,
    load_base_t_camera,
    run_scan,
    select_ceramic,
    select_leaf_waypoints,
)


CONFIG_PATH = ROOT / "config" / "robot.yaml"
POSITION_BIAS_PATH = ROOT / "config" / "visual_position_bias.json"
OUTPUT_ROOT = ROOT / "data" / "visual_pick_test"
END_EFFECTOR = load_end_effector_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("bamboo", "ceramic"), required=True)
    parser.add_argument("--run", action="store_true", help="Enable robot motion; default is camera/route dry-run.")
    parser.add_argument("--confirm-no-pressure-sensor", action="store_true")
    add_vision_common_args(parser, latest_global_calibration(), min_score_default=None)
    parser.add_argument("--position-bias-config", type=Path, default=POSITION_BIAS_PATH)
    parser.add_argument("--vision-socket", type=Path, default=DEFAULT_VISION_SOCKET)
    parser.add_argument("--require-vision-service", action="store_true")
    parser.add_argument("--compression-mm", type=float)
    parser.add_argument("--allow-validated-clipped-target", action="store_true")
    parser.add_argument("--vacuum-settle-seconds", type=float, default=0.9)
    parser.add_argument("--release-settle-seconds", type=float, default=0.5)
    parser.add_argument("--inspection-hold-seconds", type=float, default=2.0)
    parser.add_argument(
        "--verify-pick",
        action="store_true",
        help="Use a post-lift RGB-D frame to verify the pick and optionally replan.",
    )
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument(
        "--simulate-first-pick-failure",
        action="store_true",
        help="Skip pump activation on attempt 1 to test visual failure detection safely.",
    )
    parser.add_argument("--transit-speed", type=float, default=80.0)
    parser.add_argument("--precision-speed", type=float, default=15.0)
    parser.add_argument("--acc", type=float, default=150.0)
    add_workspace_limit_args(parser)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, profile: PickTestProfile) -> tuple[float, float]:
    min_score = profile.default_min_score if args.min_score is None else float(args.min_score)
    compression_mm = profile.default_compression_mm if args.compression_mm is None else float(args.compression_mm)
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("--min-score must be within 0~1")
    if not 0.0 <= compression_mm <= MAX_COMPRESSION_MM:
        raise ValueError(
            f"--compression-mm must remain within the configured 0~{MAX_COMPRESSION_MM:g} mm spring travel"
        )
    if min(args.vacuum_settle_seconds, args.release_settle_seconds, args.inspection_hold_seconds) < 0.0:
        raise ValueError("Hold and settle times must be non-negative")
    if min(args.transit_speed, args.precision_speed, args.acc) <= 0.0:
        raise ValueError("Motion speeds and acceleration must be positive")
    limits = (args.min_x_mm, args.max_x_mm, args.min_y_mm, args.max_y_mm, args.min_z_mm, args.max_z_mm)
    if args.run and any(value is None for value in limits):
        raise ValueError("--run requires explicit min/max X, Y, and Z workspace limits")
    if args.run and not args.confirm_no_pressure_sensor:
        raise ValueError("--run requires --confirm-no-pressure-sensor")
    if not 1 <= args.max_attempts <= 3:
        raise ValueError("--max-attempts must be within 1~3")
    if args.verify_pick and not args.run:
        raise ValueError("--verify-pick requires --run")
    if args.max_attempts > 1 and not args.verify_pick:
        raise ValueError("Multiple attempts require --verify-pick")
    if args.simulate_first_pick_failure and not args.verify_pick:
        raise ValueError("--simulate-first-pick-failure requires --verify-pick")
    if args.simulate_first_pick_failure and args.max_attempts < 2:
        raise ValueError("Simulated first-pick failure requires --max-attempts 2 or greater")
    return min_score, compression_mm


def load_calibration(path: Path | None) -> tuple[Path, np.ndarray]:
    if path is None or not path.expanduser().exists():
        raise FileNotFoundError("Fixed global-camera calibration is missing")
    return load_base_t_camera(path, DEFAULT_MAX_CALIBRATION_RESIDUAL_MM)


def select_target(
    detections: list[dict[str, Any]],
    profile: PickTestProfile,
    allow_clipped: bool,
) -> dict[str, Any]:
    if profile.kind == "ceramic":
        return select_ceramic(detections)
    leaves, _stacked = select_leaf_waypoints(
        detections,
        distinct_distance_m=0.05,
        allow_any_clipping=allow_clipped,
    )
    return leaves[0]


def speed_for(step: PickTestStep, args: argparse.Namespace) -> float:
    return float(args.precision_speed if step.motion != "empty_transit" else args.transit_speed)


def route_safety(route: list[PickTestStep], args: argparse.Namespace) -> dict[str, Any]:
    return route_safety_report([step.pose_mm_deg for step in route], args)


def apply_position_bias(detections: list[dict[str, Any]], bias_mm: np.ndarray) -> None:
    for detection in detections:
        point = np.asarray(detection["point_base_m"], dtype=float)
        detection["point_base_m"] = (point + bias_mm / 1000.0).tolist()


def capture_detections(
    *,
    camera: RealSenseCaptureSession,
    capture_root: Path,
    args: argparse.Namespace,
    profile: PickTestProfile,
    min_score: float,
    base_t_camera: np.ndarray,
    bias_mm: np.ndarray,
) -> tuple[Path, list[dict[str, Any]]]:
    frame_dir, detections = run_scan(
        capture_root,
        base_t_camera,
        args.vision_env,
        min_score,
        camera=camera,
        text_prompt=profile.text_prompt,
        service_socket=args.vision_socket,
        require_service=args.require_vision_service,
    )
    apply_position_bias(detections, bias_mm)
    return frame_dir, detections


def result_exit_code(
    *,
    run_mode: bool,
    verify_pick: bool,
    final_verification: PickVerification | None,
) -> int:
    if run_mode and verify_pick and (final_verification is None or not final_verification.confirmed):
        return 2
    return 0


def main() -> int:
    args = parse_args()
    profile = profile_for(args.target)
    min_score, compression_mm = validate_args(args, profile)
    calibration_path, base_t_camera = load_calibration(args.calibration)
    bias_mm = load_bias_mm(args.position_bias_config.expanduser().resolve())
    robot_ip = load_robot_ip(CONFIG_PATH)

    run_dir = new_timestamped_run_dir(OUTPUT_ROOT)
    camera = RealSenseCaptureSession(serial=args.serial)
    arm: Any | None = None
    vacuum: VacuumController | None = None
    holding_target = False
    attempts: list[dict[str, Any]] = []
    try:
        camera.start()
        if args.run:
            print("Moving to the configured camera-clear reset pose before the pick test.")
            reset_robot_at_ip(
                robot_ip,
                END_EFFECTOR.tcp_offset_mm_deg,
                END_EFFECTOR.home_pose_mm_deg,
                args.transit_speed,
                args.acc,
            )
            arm, home_pose = connect_robot(robot_ip, END_EFFECTOR.tcp_offset_mm_deg)
            prepare_robot_for_motion(arm)
            vacuum = VacuumController(arm, END_EFFECTOR.vacuum_io)
            vacuum.configure()
        else:
            home_pose = list(map(float, END_EFFECTOR.home_pose_mm_deg))

        final_verification: PickVerification | None = None
        requested_attempts = args.max_attempts if args.verify_pick else 1
        for attempt_number in range(1, requested_attempts + 1):
            print(f"\n=== Pick attempt {attempt_number}/{requested_attempts} ===")
            attempt_dir = run_dir / f"attempt_{attempt_number:02d}"
            frame_dir, detections = capture_detections(
                camera=camera,
                capture_root=attempt_dir / "targeting" / "captures",
                args=args,
                profile=profile,
                min_score=min_score,
                base_t_camera=base_t_camera,
                bias_mm=bias_mm,
            )
            target = select_target(detections, profile, args.allow_validated_clipped_target)
            source_point_m = list(map(float, target["point_base_m"]))
            route = build_pick_test_route(
                home_pose_mm_deg=list(map(float, home_pose)),
                source_point_m=source_point_m,
                profile=profile,
                compression_mm=compression_mm,
                inspection_event="verify" if args.verify_pick else "hold",
            )
            safety = route_safety(route, args)
            failure_injected = bool(args.simulate_first_pick_failure and attempt_number == 1)
            attempt_result: dict[str, Any] = {
                "attempt": attempt_number,
                "targeting_frame": str(frame_dir),
                "target_detection": target,
                "source_point_m": source_point_m,
                "failure_injected": failure_injected,
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
            attempts.append(attempt_result)

            print(f"Target: {profile.kind}")
            print(f"Calibration: {calibration_path}")
            print(f"Position bias [mm]: {bias_mm.round(2).tolist()}")
            print(f"Detected source [mm]: {(np.asarray(source_point_m) * 1000.0).round(2).tolist()}")
            print(f"Compression: {compression_mm:.1f} mm; test lift: {profile.test_lift_mm:.1f} mm")
            if failure_injected:
                print("CONTROLLED TEST: attempt 1 will contact normally but intentionally keep the pump OFF.")
            for index, step in enumerate(route, start=1):
                print(f"  {index:02d} [{step.motion}] {step.label}: {[round(v, 2) for v in step.pose_mm_deg]}")
            print(f"Safety: {safety}")

            if not args.run:
                print("DRY RUN: detection and route planning completed; no robot motion or GPIO was commanded.")
                break
            if not safety["all_ok"]:
                raise RuntimeError("Refusing motion because a test waypoint is outside explicit workspace limits")

            assert arm is not None and vacuum is not None
            vacuum.prepare_for_pick()
            verification: PickVerification | None = None
            for index, step in enumerate(route, start=1):
                require_code(
                    f"attempt {attempt_number} move {index:02d}: {step.label}",
                    arm.set_position(
                        *step.pose_mm_deg,
                        speed=speed_for(step, args),
                        mvacc=args.acc,
                        wait=True,
                        radius=-1,
                        is_radian=False,
                    ),
                )
                if step.event == "acquire":
                    if failure_injected:
                        print("CONTROLLED TEST: pump activation skipped; valve remains in the suction path state.")
                    else:
                        vacuum.acquire(args.vacuum_settle_seconds)
                        holding_target = True
                elif step.event == "hold":
                    print(f"Holding for operator inspection: {args.inspection_hold_seconds:.1f} s")
                    time.sleep(args.inspection_hold_seconds)
                elif step.event == "verify":
                    expected_object_point_m = list(source_point_m)
                    expected_object_point_m[2] += profile.test_lift_mm / 1000.0
                    try:
                        verification_frame, verification_detections = capture_detections(
                            camera=camera,
                            capture_root=attempt_dir / "verification" / "captures",
                            args=args,
                            profile=profile,
                            min_score=min_score,
                            base_t_camera=base_t_camera,
                            bias_mm=bias_mm,
                        )
                        policy = policy_for(profile.kind)
                        verification = verify_carried_object(
                            verification_detections,
                            target_kind=profile.kind,
                            source_point_m=source_point_m,
                            expected_object_point_m=expected_object_point_m,
                            min_score=min_score,
                            minimum_lift_mm=policy.minimum_lift_mm,
                            tool_xy_tolerance_mm=policy.tool_xy_tolerance_mm,
                            height_tolerance_mm=policy.height_tolerance_mm,
                            source_xy_tolerance_mm=policy.source_xy_tolerance_mm,
                            source_height_tolerance_mm=policy.source_height_tolerance_mm,
                            allow_clipped_candidate=args.allow_validated_clipped_target,
                        )
                        attempt_result["verification_frame"] = str(verification_frame)
                    except Exception as exc:
                        verification = PickVerification(
                            "inconclusive",
                            f"verification capture or inference failed: {exc}",
                            profile.kind,
                            None,
                            None,
                            None,
                        )
                    attempt_result["verification"] = verification.to_dict()
                    print(
                        "Visual pick verification: "
                        f"{verification.status} | {verification.reason} | "
                        f"observed_lift_mm={verification.observed_lift_mm} | "
                        f"tool_xy_error_mm={verification.tool_xy_error_mm}"
                    )
                elif step.event == "release":
                    vacuum.release(args.release_settle_seconds)
                    holding_target = False

            final_verification = verification
            if not args.verify_pick or (verification is not None and verification.confirmed):
                break
            if attempt_number < requested_attempts:
                print(
                    f"Attempt {attempt_number} was {verification.status if verification else 'inconclusive'}; "
                    "the next attempt will use a new RGB-D frame and a newly planned source point."
                )

        result = {
            "run_mode": bool(args.run),
            "target": profile.kind,
            "calibration": str(calibration_path),
            "position_bias_mm": bias_mm.tolist(),
            "compression_mm": compression_mm,
            "test_lift_mm": profile.test_lift_mm,
            "verify_pick": bool(args.verify_pick),
            "max_attempts": requested_attempts,
            "attempts": attempts,
            "final_verification_status": final_verification.status if final_verification else None,
        }
        result_path = write_json_report(run_dir, result)
        print(f"Saved result: {result_path}")
        exit_code = result_exit_code(
            run_mode=bool(args.run),
            verify_pick=bool(args.verify_pick),
            final_verification=final_verification,
        )
        if exit_code:
            print("Closed-loop pick test ended without a visually confirmed pickup.", file=sys.stderr)
        return exit_code
    except Exception:
        if holding_target:
            print(
                "EMERGENCY NOTE: vacuum remains ON because an object may be held. "
                "Make the area safe before manually releasing it.",
                file=sys.stderr,
            )
        elif vacuum is not None:
            vacuum.safe_shutdown()
        raise
    finally:
        if vacuum is not None and not holding_target:
            vacuum.safe_shutdown()
        if arm is not None:
            arm.disconnect()
        camera.stop(raise_recording_error=False)


if __name__ == "__main__":
    raise SystemExit(main())
