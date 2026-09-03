#!/usr/bin/env python3
"""Pick one material using a synchronized wrist RGB-D surface candidate."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.cli_utils import add_workspace_limit_args, explicit_workspace_limits, route_safety_report  # noqa: E402
from xarm_sdk_control.end_effector_config import load_end_effector_config  # noqa: E402
from xarm_sdk_control.hand_eye_calibration import flange_transform_from_tcp_pose  # noqa: E402
from xarm_sdk_control.pick_test_workflow import (  # noqa: E402
    build_pick_place_test_route,
    build_pick_test_route,
    execute_pick_test_route,
    profile_for,
    recover_failed_pick_test,
)
from xarm_sdk_control.realsense_capture import RealSenseCaptureSession  # noqa: E402
from xarm_sdk_control.robot_motion import (  # noqa: E402
    connect_robot,
    load_robot_ip,
    prepare_robot_for_motion,
    require_code,
    reset_robot_at_ip,
)
from xarm_sdk_control.suction_sensor import SuctionNotConfirmedError, SuctionSensor  # noqa: E402
from xarm_sdk_control.suction_surface import (  # noqa: E402
    SuctionCandidate,
    analyze_suction_surface,
    candidate_to_dict,
    draw_suction_candidates,
    load_surface_profiles,
    masked_center_camera_point,
)
from xarm_sdk_control.vacuum_control import VacuumController  # noqa: E402
from xarm_sdk_control.visual_targeting import DEFAULT_VISION_SOCKET, detection_object_kind, run_vision  # noqa: E402


END_EFFECTOR = load_end_effector_config()
OUTPUT_ROOT = ROOT / "data" / "wrist_guided_pick_test"
HAND_EYE_PATH = ROOT / "calibration" / "hand_eye" / "wrist_hand_eye_20260903_141526_672.json"
TARGETS = {
    "lotus": ("lotus", "lotus leaf.", "lotus-leaf"),
    "bamboo": ("bamboo", "bamboo leaf.", "bamboo"),
    "ceramic": ("ceramic", "ceramic disk.", "ceramic"),
    "paper": ("paper", "white paper.", "paper"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--target", choices=tuple(TARGETS), default="lotus")
    parser.add_argument("--serial", default="011422070172")
    parser.add_argument("--hand-eye", type=Path, default=HAND_EYE_PATH)
    parser.add_argument("--compression-mm", type=float, default=13.0)
    parser.add_argument("--lift-z-mm", type=float, default=280.0)
    parser.add_argument("--place-on-ceramic", action="store_true")
    parser.add_argument("--placement-clearance-mm", type=float, default=3.0)
    parser.add_argument("--inspection-hold-seconds", type=float, default=5.0)
    parser.add_argument("--vacuum-settle-seconds", type=float, default=0.9)
    parser.add_argument("--release-settle-seconds", type=float, default=0.5)
    parser.add_argument("--transit-speed", type=float, default=60.0)
    parser.add_argument("--precision-speed", type=float, default=15.0)
    parser.add_argument("--acc", type=float, default=120.0)
    parser.add_argument("--vision-socket", type=Path, default=DEFAULT_VISION_SOCKET)
    parser.add_argument("--require-vision-service", action="store_true")
    parser.add_argument("--vision-env", default="winejar-vision")
    parser.add_argument("--min-score", type=float, default=0.30)
    parser.add_argument("--max-attempts", type=int, default=3)
    add_workspace_limit_args(parser)
    return parser.parse_args()


def load_rgbd(frame_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata = json.loads((frame_dir / "frame.json").read_text(encoding="utf-8"))
    color = cv2.imread(str(frame_dir / metadata["files"]["color_bgr"]), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(frame_dir / metadata["files"]["depth_raw"]), cv2.IMREAD_UNCHANGED)
    if color is None or depth_raw is None:
        raise RuntimeError("Synchronized wrist RGB-D frame is unreadable")
    depth_m = depth_raw.astype(np.float32) * float(metadata["stream"]["depth_scale_m_per_unit"])
    camera_matrix = np.asarray(metadata["intrinsics"]["camera_matrix"], dtype=np.float64)
    return color, depth_m, camera_matrix


def load_flange_from_camera(path: Path, serial: str) -> np.ndarray:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("type") != "eye_in_hand_chessboard_calibration":
        raise ValueError("The selected file is not an eye-in-hand calibration")
    if payload.get("camera", {}).get("serial") != serial:
        raise ValueError("Hand-eye calibration belongs to a different camera serial")
    transform = np.asarray(payload["transform_flange_from_camera"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Hand-eye calibration contains an invalid transform")
    return transform


def spatially_distinct_candidates(
    candidates: list[SuctionCandidate],
    minimum_center_distance_px: float,
    count: int,
) -> list[SuctionCandidate]:
    selected: list[SuctionCandidate] = []
    for candidate in candidates:
        center = np.asarray(candidate.center_px)
        if all(np.linalg.norm(center - np.asarray(prior.center_px)) >= minimum_center_distance_px for prior in selected):
            selected.append(candidate)
            if len(selected) >= count:
                break
    return selected


def base_point_from_candidate(
    candidate: SuctionCandidate,
    base_from_flange: np.ndarray,
    flange_from_camera: np.ndarray,
) -> list[float]:
    point_camera = np.asarray([*candidate.center_camera_m, 1.0], dtype=np.float64)
    return (base_from_flange @ flange_from_camera @ point_camera)[:3].tolist()


def main() -> int:
    args = parse_args()
    if args.run and not explicit_workspace_limits(args):
        raise ValueError("--run requires explicit min/max X, Y, and Z workspace limits")
    if not 0 < args.compression_mm <= END_EFFECTOR.max_compression_mm:
        raise ValueError("Compression must remain within the configured spring travel")
    if args.lift_z_mm > 300.0:
        raise ValueError("--lift-z-mm must remain at or below the overhead-camera limit of 300 mm")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    if args.placement_clearance_mm < 0:
        raise ValueError("--placement-clearance-mm must be non-negative")
    if args.place_on_ceramic and args.target == "ceramic":
        raise ValueError("A ceramic disk cannot also be its own placement destination")

    from xarm.wrapper import XArmAPI

    robot_ip = load_robot_ip()
    if args.run:
        print("Resetting robot before wrist-camera targeting.")
        reset_robot_at_ip(
            robot_ip,
            END_EFFECTOR.tcp_offset_mm_deg,
            END_EFFECTOR.home_pose_mm_deg,
            args.transit_speed,
            args.acc,
        )

    read_arm = XArmAPI(do_not_open=True, is_radian=False, report_type="rich")
    camera = RealSenseCaptureSession(serial=args.serial)
    try:
        require_code("connect read-only robot client", read_arm.connect(port=robot_ip))
        if read_arm.get_is_moving():
            raise RuntimeError("Robot must be stationary during wrist targeting")
        camera.start()
        frame_dir = camera.capture(OUTPUT_ROOT)
        pose_code, tcp_pose = read_arm.get_position(is_radian=False)
        joint_code, joint_angles = read_arm.get_servo_angle(is_radian=False)
        require_code("read synchronized TCP pose", (pose_code, tcp_pose))
        require_code("read synchronized joint angles", (joint_code, joint_angles))
        active_tcp_offset = list(map(float, read_arm.tcp_offset))
    finally:
        camera.stop()
        read_arm.disconnect()

    color, depth_m, camera_matrix = load_rgbd(frame_dir)
    kind, prompt, pick_profile_name = TARGETS[args.target]
    vision_prompt = f"{prompt} ceramic disk." if args.place_on_ceramic else prompt
    vision_result = run_vision(
        frame_dir,
        args.vision_env,
        args.min_score,
        text_prompt=vision_prompt,
        service_socket=args.vision_socket,
        require_service=args.require_vision_service,
    )
    detections = [
        detection
        for detection in vision_result.get("detections", [])
        if detection_object_kind(detection) == kind and detection.get("mask_path")
    ]
    if not detections:
        raise RuntimeError(f"No segmented {args.target} was found in the wrist frame")
    detection = max(detections, key=lambda item: float(item.get("score", 0.0)))
    mask = cv2.imread(str(frame_dir / detection["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError("Target segmentation mask is unreadable")
    analysis = analyze_suction_surface(
        color,
        depth_m,
        mask,
        camera_matrix,
        load_surface_profiles()[args.target],
        grid_step_px=5,
        angle_step_deg=15.0,
        preferred_angle_deg=0.0,
        max_candidates=100,
    )
    retry_candidates = spatially_distinct_candidates(
        analysis.candidates,
        minimum_center_distance_px=max(analysis.cup_radius_px * 2.5, 8.0),
        count=args.max_attempts,
    )
    if not retry_candidates:
        raise RuntimeError("No dual-cup pose passed the surface checks")

    destination_camera_m: tuple[float, float, float] | None = None
    destination_detection: dict[str, Any] | None = None
    if args.place_on_ceramic:
        ceramic_detections = [
            detection
            for detection in vision_result.get("detections", [])
            if detection_object_kind(detection) == "ceramic" and detection.get("mask_path")
        ]
        if not ceramic_detections:
            raise RuntimeError("No segmented ceramic disk was found in the wrist frame")
        destination_detection = max(
            ceramic_detections,
            key=lambda item: float(item.get("score", 0.0)),
        )
        destination_mask = cv2.imread(
            str(frame_dir / destination_detection["mask_path"]),
            cv2.IMREAD_GRAYSCALE,
        )
        if destination_mask is None:
            raise RuntimeError("Ceramic destination mask is unreadable")
        destination_camera_m = masked_center_camera_point(depth_m, destination_mask, camera_matrix)

    flange_from_camera = load_flange_from_camera(args.hand_eye, args.serial)
    base_from_flange = flange_transform_from_tcp_pose(tcp_pose, active_tcp_offset)
    base_points = [
        base_point_from_candidate(candidate, base_from_flange, flange_from_camera)
        for candidate in retry_candidates
    ]
    destination_point: list[float] | None = None
    if destination_camera_m is not None:
        destination_point = (
            base_from_flange
            @ flange_from_camera
            @ np.asarray([*destination_camera_m, 1.0], dtype=np.float64)
        )[:3].tolist()
        print(
            "Ceramic destination: base="
            f"{[round(value * 1000.0, 2) for value in destination_point]} mm"
        )
    target_overlay = draw_suction_candidates(color, replace(analysis, candidates=retry_candidates), count=len(retry_candidates))
    if destination_detection is not None:
        x1, y1, x2, y2 = map(int, destination_detection["box_xyxy"])
        center = (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)))
        cv2.drawMarker(target_overlay, center, (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(
            target_overlay,
            "ceramic destination",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
    overlay_path = frame_dir / f"wrist_pick_candidates_{args.target}.png"
    cv2.imwrite(str(overlay_path), target_overlay)

    profile = profile_for(pick_profile_name)
    routes = []
    for candidate, point in zip(retry_candidates, base_points):
        point_z_mm = point[2] * 1000.0
        if args.lift_z_mm <= point_z_mm:
            raise RuntimeError("Requested inspection height is not above the selected material surface")
        if destination_point is None:
            route_profile = replace(profile, test_lift_mm=args.lift_z_mm - point_z_mm)
            route = build_pick_test_route(
                home_pose_mm_deg=list(END_EFFECTOR.home_pose_mm_deg),
                source_point_m=point,
                profile=route_profile,
                compression_mm=args.compression_mm,
                approach_clearance_mm=35.0,
                release_clearance_mm=10.0,
            )
        else:
            destination_z_mm = destination_point[2] * 1000.0
            transport_clearance_mm = args.lift_z_mm - max(point_z_mm, destination_z_mm)
            if transport_clearance_mm <= 0:
                raise RuntimeError("Requested transport height is not above both surfaces")
            route = build_pick_place_test_route(
                home_pose_mm_deg=list(END_EFFECTOR.home_pose_mm_deg),
                source_point_m=point,
                destination_point_m=destination_point,
                compression_mm=args.compression_mm,
                transport_clearance_mm=transport_clearance_mm,
                approach_clearance_mm=35.0,
                placement_clearance_mm=args.placement_clearance_mm,
            )
        safety = route_safety_report([step.pose_mm_deg for step in route], args)
        if args.run and not safety["all_ok"]:
            raise RuntimeError(f"Candidate route is outside explicit workspace limits: {safety}")
        routes.append(route)
        print(
            f"Candidate {len(routes)}: base={[round(value * 1000.0, 2) for value in point]} mm, "
            f"image angle={candidate.angle_deg:.1f} deg, score={candidate.score:.3f}"
        )
    report = {
        "frame_dir": str(frame_dir),
        "tcp_pose_mm_deg": list(map(float, tcp_pose)),
        "active_tcp_offset_mm_deg": active_tcp_offset,
        "joint_angles_deg": list(map(float, joint_angles)),
        "hand_eye": str(args.hand_eye.expanduser().resolve()),
        "candidates": [
            {**candidate_to_dict(candidate), "point_base_m": point}
            for candidate, point in zip(retry_candidates, base_points)
        ],
        "overlay": overlay_path.name,
        "destination_point_base_m": destination_point,
    }
    (frame_dir / "wrist_pick_targeting.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    if not args.run:
        print("DRY RUN: targeting and routes validated; no robot motion or GPIO command was sent.")
        return 0

    arm: Any | None = None
    vacuum: VacuumController | None = None
    holding = {"value": False}
    try:
        arm, _home = connect_robot(robot_ip, END_EFFECTOR.tcp_offset_mm_deg)
        prepare_robot_for_motion(arm)
        vacuum = VacuumController(arm, END_EFFECTOR.vacuum_io)
        vacuum.configure()
        if END_EFFECTOR.suction_sensor is None:
            raise RuntimeError("CI0 suction sensor configuration is required")
        sensor = SuctionSensor(arm, END_EFFECTOR.suction_sensor)
        for attempt, route in enumerate(routes, start=1):
            try:
                print(f"Executing wrist-selected suction attempt {attempt}/{len(routes)}")
                execute_pick_test_route(
                    arm,
                    vacuum,
                    route,
                    args,
                    holding,
                    label_prefix=f"attempt {attempt} ",
                    suction_sensor=sensor,
                )
                if destination_point is None:
                    print("Wrist-guided pick test completed and material was returned to its source.")
                else:
                    print("Wrist-guided pick-and-place completed on the ceramic destination.")
                return 0
            except SuctionNotConfirmedError as exc:
                print(f"Attempt {attempt} suction not confirmed: {exc}")
                holding["value"] = False
                vacuum.release(args.release_settle_seconds)
                recover_failed_pick_test(
                    arm,
                    route,
                    args,
                    return_home=attempt == len(routes),
                    label_prefix=f"attempt {attempt} ",
                )
        raise RuntimeError("All spatially distinct wrist-selected suction candidates failed")
    finally:
        if arm is not None:
            if vacuum is not None and not holding["value"]:
                vacuum.safe_shutdown()
            elif holding["value"]:
                print("EMERGENCY NOTE: vacuum remains on because a material may still be held.")
            arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
