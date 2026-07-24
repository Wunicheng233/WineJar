#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from xarm.wrapper import XArmAPI


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "robot.yaml"
DEFAULT_CALIBRATION = ROOT / "calibration" / "handeye_20260703_200745.json"
OUTPUT_ROOT = ROOT / "data" / "cube_pick_preview"


def set_pneumatic_gripper(
    arm: XArmAPI,
    ionum: int,
    value: int,
    label: str,
    settle_sec: float,
):
    require_ok(f"set {label} DO{ionum} function to general output", arm.set_cgpio_digital_output_function, ionum, 0)
    require_ok(f"{label} pneumatic gripper DO{ionum}={value}", arm.set_cgpio_digital, ionum, value)
    if settle_sec > 0:
        import time

        time.sleep(settle_sec)


def rx(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def ry(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rz(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def tcp_pose_to_base_t_tcp(pose_mm_deg: list[float]) -> np.ndarray:
    x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg = pose_mm_deg
    rotation = (
        rz(math.radians(yaw_deg))
        @ ry(math.radians(pitch_deg))
        @ rx(math.radians(roll_deg))
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0
    return transform


def call(label: str, func, *args, **kwargs):
    result = func(*args, **kwargs)
    print(f"{label}: {result}")
    return result


def require_ok(label: str, func, *args, **kwargs):
    result = call(label, func, *args, **kwargs)
    code = result[0] if isinstance(result, tuple) else result
    if code not in (0, None):
        raise RuntimeError(f"{label} failed with code/result: {result}")
    return result


def read_robot_pose(arm: XArmAPI) -> dict:
    position_code, position = arm.get_position(is_radian=False)
    state = arm.get_state()
    err_warn = arm.get_err_warn_code()
    return {
        "state": state,
        "err_warn": err_warn,
        "tcp_pose_code": position_code,
        "tcp_pose_mm_deg": position,
    }


def detect_colored_cube(
    color: np.ndarray,
    min_area_px: int,
    saturation_min: int,
    value_min: int,
    max_bbox_px: int,
):
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((saturation >= saturation_min) & (value >= value_min)).astype(np.uint8) * 255

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_edge = x <= 2 or y <= 2 or x + w >= color.shape[1] - 2 or y + h >= color.shape[0] - 2
        aspect = w / max(h, 1)
        if touches_edge:
            continue
        if not (0.55 <= aspect <= 1.8):
            continue
        if max(w, h) > max_bbox_px:
            continue
        if min(w, h) < 20:
            continue
        fill_ratio = area / max(w * h, 1)
        score = area * (1.0 - min(abs(math.log(aspect)), 1.0)) * min(fill_ratio / 0.7, 1.0)
        candidates.append(
            {
                "label": label,
                "area_px": area,
                "bbox_xywh": [x, y, w, h],
                "centroid_uv": [float(centroids[label][0]), float(centroids[label][1])],
                "aspect": float(aspect),
                "fill_ratio": float(fill_ratio),
                "score": float(score),
            }
        )

    if not candidates:
        return None, mask, []
    best = max(candidates, key=lambda item: item["score"])
    best_mask = (labels == best["label"]).astype(np.uint8) * 255
    return best, best_mask, candidates


def median_depth_for_mask(depth_frame: rs.depth_frame, mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    step = max(1, len(xs) // 500)
    depths = []
    for u, v in zip(xs[::step], ys[::step]):
        d = depth_frame.get_distance(int(u), int(v))
        if d > 0:
            depths.append(d)
    if not depths:
        return 0.0
    return float(np.median(depths))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect a colorful Rubik's cube with RealSense and move above it or pick it. "
            "Default is dry-run; use --run to command robot motion, and --pick to grasp/lift."
        )
    )
    parser.add_argument("--run", action="store_true", help="Actually command robot motion.")
    parser.add_argument("--pick", action="store_true", help="After moving above, descend, close gripper, and lift.")
    parser.add_argument(
        "--max-x-mm",
        type=float,
        default=None,
        help="Required for --run. Rejects targets with X above this safety limit.",
    )
    parser.add_argument("--min-z-mm", type=float, default=180.0)
    parser.add_argument("--above-mm", type=float, default=140.0)
    parser.add_argument("--offset-x-mm", type=float, default=0.0, help="Manual base-frame X offset for the detected cube point.")
    parser.add_argument("--offset-y-mm", type=float, default=0.0, help="Manual base-frame Y offset for the detected cube point.")
    parser.add_argument("--offset-z-mm", type=float, default=0.0, help="Manual base-frame Z offset for the detected cube point.")
    parser.add_argument(
        "--grasp-above-surface-mm",
        type=float,
        default=35.0,
        help="TCP Z offset above detected cube surface for closing the gripper, used only when --grasp-z-mm is not set.",
    )
    parser.add_argument(
        "--grasp-drop-mm",
        type=float,
        default=60.0,
        help="Fallback grasp plan: descend this much from the above pose when visual surface Z is below min-z.",
    )
    parser.add_argument(
        "--grasp-z-mm",
        type=float,
        default=None,
        help="Explicit TCP Z for closing the gripper. Recommended for the first real pick attempt.",
    )
    parser.add_argument("--lift-mm", type=float, default=100.0, help="Lift distance after closing gripper.")
    parser.add_argument(
        "--pneumatic-do",
        type=int,
        default=9,
        help="Controller CGPIO output ionum for the pneumatic valve. DO1 is usually ionum=9.",
    )
    parser.add_argument(
        "--open-value",
        type=int,
        default=0,
        choices=[0, 1],
        help="CGPIO value that opens the pneumatic gripper.",
    )
    parser.add_argument(
        "--close-value",
        type=int,
        default=1,
        choices=[0, 1],
        help="CGPIO value that closes the pneumatic gripper.",
    )
    parser.add_argument(
        "--gripper-settle-sec",
        type=float,
        default=0.5,
        help="Wait time after switching the pneumatic valve.",
    )
    parser.add_argument(
        "--target-yaw-deg",
        type=float,
        default=None,
        help="Override TCP yaw for alignment, e.g. 45 for cube grasp approach.",
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--acc", type=float, default=100.0)
    parser.add_argument("--move-timeout-sec", type=float, default=30.0, help="Timeout for each blocking Cartesian move.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--min-area-px", type=int, default=700)
    parser.add_argument("--saturation-min", type=int, default=140)
    parser.add_argument("--value-min", type=int, default=120)
    parser.add_argument("--max-bbox-px", type=int, default=360)
    args = parser.parse_args()

    if args.run and args.max_x_mm is None:
        print("Refusing to move: --run requires an explicit --max-x-mm safety limit.")
        return 2
    if args.pick and not args.run:
        print("Pick requested in dry-run mode: planning full pick sequence, but no motion/gripper commands.")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        robot_ip = yaml.safe_load(f)["robot_ip"]
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    tcp_t_camera = np.array(calibration["tcp_T_camera"], dtype=np.float64)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Rubik's cube visual target: high-saturation color segmentation.")
    print("Dry run mode." if not args.run else "RUN MODE: robot motion may be commanded.")
    print(f"Using calibration: {args.calibration}")

    arm = XArmAPI(do_not_open=True, is_radian=False)
    pipeline = None
    try:
        call("connect", arm.connect, port=robot_ip)
        robot_before = read_robot_pose(arm)
        current_pose = robot_before["tcp_pose_mm_deg"]
        print(f"Current TCP pose [x,y,z,roll,pitch,yaw] mm/deg: {current_pose}")
        base_t_tcp = tcp_pose_to_base_t_tcp(current_pose)
        base_t_camera = base_t_tcp @ tcp_t_camera

        context = rs.context()
        if len(context.query_devices()) == 0:
            print("No RealSense camera detected.")
            return 3

        pipeline = rs.pipeline(context)
        config = rs.config()
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        align = rs.align(rs.stream.color)
        pipeline.start(config)
        for _ in range(args.warmup):
            pipeline.wait_for_frames()
        frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            print("Failed to receive synchronized color/depth frames.")
            return 4

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        detection, mask, candidates = detect_colored_cube(
            color,
            args.min_area_px,
            args.saturation_min,
            args.value_min,
            args.max_bbox_px,
        )

        color_path = output_dir / "color.png"
        depth_path = output_dir / "depth_raw.png"
        mask_path = output_dir / "mask.png"
        annotated_path = output_dir / "annotated.png"
        result_path = output_dir / "result.json"
        cv2.imwrite(str(color_path), color)
        cv2.imwrite(str(depth_path), depth)
        cv2.imwrite(str(mask_path), mask)

        if detection is None:
            print("No colored cube candidate detected.")
            print(f"Saved color image: {color_path}")
            print(f"Saved mask image: {mask_path}")
            return 5

        depth_m = median_depth_for_mask(depth_frame, mask)
        if depth_m <= 0:
            print("Detected cube mask, but valid depth was not available.")
            return 6

        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        centroid_u, centroid_v = detection["centroid_uv"]
        camera_point_m = np.array(
            rs.rs2_deproject_pixel_to_point(
                intrinsics, [float(centroid_u), float(centroid_v)], depth_m
            )
            + [1.0],
            dtype=np.float64,
        )
        base_point_m = (base_t_camera @ camera_point_m)[:3]
        base_point_mm = base_point_m * 1000.0
        offset_mm = np.array([args.offset_x_mm, args.offset_y_mm, args.offset_z_mm], dtype=np.float64)
        compensated_base_point_mm = base_point_mm + offset_mm

        target_pose = list(current_pose)
        target_pose[0] = float(compensated_base_point_mm[0])
        target_pose[1] = float(compensated_base_point_mm[1])
        target_pose[2] = float(max(compensated_base_point_mm[2] + args.above_mm, args.min_z_mm))
        if args.target_yaw_deg is not None:
            target_pose[5] = float(args.target_yaw_deg)
        grasp_pose = list(target_pose)
        if args.grasp_z_mm is not None:
            grasp_pose[2] = float(args.grasp_z_mm)
            grasp_z_source = "explicit --grasp-z-mm"
        else:
            surface_based_z = float(compensated_base_point_mm[2] + args.grasp_above_surface_mm)
            if surface_based_z >= args.min_z_mm:
                grasp_pose[2] = surface_based_z
                grasp_z_source = "surface depth + --grasp-above-surface-mm"
            else:
                grasp_pose[2] = float(max(target_pose[2] - args.grasp_drop_mm, args.min_z_mm))
                grasp_z_source = "above pose - --grasp-drop-mm clamped by --min-z-mm"
        lift_pose = list(grasp_pose)
        lift_pose[2] = float(max(grasp_pose[2] + args.lift_mm, args.min_z_mm))

        checks = {
            "max_x_ok": args.max_x_mm is None or target_pose[0] <= args.max_x_mm,
            "min_z_ok": target_pose[2] >= args.min_z_mm,
            "grasp_min_z_ok": grasp_pose[2] >= args.min_z_mm,
            "lift_min_z_ok": lift_pose[2] >= args.min_z_mm,
            "tcp_pose_code_ok": robot_before["tcp_pose_code"] == 0,
        }
        checks["all_ok"] = all(checks.values())

        annotated = color.copy()
        x, y, w, h = detection["bbox_xywh"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.circle(annotated, (int(round(centroid_u)), int(round(centroid_v))), 6, (0, 0, 255), -1)
        cv2.imwrite(str(annotated_path), annotated)

        print(f"Cube candidate bbox [x,y,w,h]: {detection['bbox_xywh']}, area={detection['area_px']} px")
        print(f"Cube centroid pixel [u,v]: {[round(centroid_u, 2), round(centroid_v, 2)]}")
        print(f"Median depth in mask: {depth_m:.4f} m")
        print(f"Cube surface point in base frame [mm]: {base_point_mm.round(3).tolist()}")
        print(f"Manual base-frame offset [mm]: {offset_mm.round(3).tolist()}")
        print(f"Compensated cube point in base frame [mm]: {compensated_base_point_mm.round(3).tolist()}")
        print(f"Planned above-cube TCP pose [mm/deg]: {[round(v, 3) for v in target_pose]}")
        if args.pick:
            print(f"Planned grasp TCP pose [mm/deg]: {[round(v, 3) for v in grasp_pose]}")
            print(f"Planned lift TCP pose [mm/deg]: {[round(v, 3) for v in lift_pose]}")
            print(f"Grasp Z source: {grasp_z_source}")
            print(
                "Pneumatic gripper plan: "
                f"DO ionum={args.pneumatic_do}, open_value={args.open_value}, "
                f"close_value={args.close_value}, settle_sec={args.gripper_settle_sec}"
            )
        print(f"Safety checks: {checks}")

        moved = False
        picked = False
        robot_after = None
        if args.run:
            if not checks["all_ok"]:
                print("Refusing to move: safety checks failed.")
                return 7
            require_ok("clean_error", arm.clean_error)
            call("clean_warn", arm.clean_warn)
            require_ok("motion_enable", arm.motion_enable, enable=True)
            require_ok("set_mode(0)", arm.set_mode, 0)
            require_ok("set_state(0)", arm.set_state, 0)
            err_warn = call("get_err_warn_code after prepare", arm.get_err_warn_code)
            if isinstance(err_warn, tuple) and err_warn[0] == 0 and err_warn[1][0] != 0:
                raise RuntimeError(f"xArm still has controller error after prepare: {err_warn}")
            if args.pick:
                set_pneumatic_gripper(
                    arm,
                    args.pneumatic_do,
                    args.open_value,
                    "open gripper",
                    args.gripper_settle_sec,
                )
            require_ok(
                "move above cube",
                arm.set_position,
                x=target_pose[0],
                y=target_pose[1],
                z=target_pose[2],
                roll=target_pose[3],
                pitch=target_pose[4],
                yaw=target_pose[5],
                speed=args.speed,
                mvacc=args.acc,
                wait=True,
                timeout=args.move_timeout_sec,
            )
            moved = True
            if args.pick:
                require_ok(
                    "descend to grasp pose",
                    arm.set_position,
                    x=grasp_pose[0],
                    y=grasp_pose[1],
                    z=grasp_pose[2],
                    roll=grasp_pose[3],
                    pitch=grasp_pose[4],
                    yaw=grasp_pose[5],
                    speed=min(args.speed, 10.0),
                    mvacc=args.acc,
                    wait=True,
                    timeout=args.move_timeout_sec,
                )
                set_pneumatic_gripper(
                    arm,
                    args.pneumatic_do,
                    args.close_value,
                    "close gripper",
                    args.gripper_settle_sec,
                )
                require_ok("set_state(0) before lift", arm.set_state, 0)
                require_ok(
                    "lift cube",
                    arm.set_position,
                    x=lift_pose[0],
                    y=lift_pose[1],
                    z=lift_pose[2],
                    roll=lift_pose[3],
                    pitch=lift_pose[4],
                    yaw=lift_pose[5],
                    speed=min(args.speed, 15.0),
                    mvacc=args.acc,
                    wait=True,
                    timeout=args.move_timeout_sec,
                )
                picked = True
            robot_after = read_robot_pose(arm)
            print(f"After TCP pose [x,y,z,roll,pitch,yaw] mm/deg: {robot_after['tcp_pose_mm_deg']}")
        else:
            if args.pick:
                print("Dry run only. Re-run with --run --pick --max-x-mm <limit> to execute pick.")
            else:
                print("Dry run only. Re-run with --run --max-x-mm <limit> to execute motion.")

        result = {
            "timestamp": timestamp,
            "calibration_path": str(args.calibration),
            "run": bool(args.run),
            "pick": bool(args.pick),
            "moved": moved,
            "picked": picked,
            "robot_before": robot_before,
            "robot_after": robot_after,
            "detection": {
                "method": "HSV saturation threshold + largest connected component",
                "saturation_min": args.saturation_min,
                "value_min": args.value_min,
                "max_bbox_px": args.max_bbox_px,
                "min_area_px": args.min_area_px,
                "selected": detection,
                "candidates": candidates,
                "median_depth_m": depth_m,
                "camera_point_m": camera_point_m[:3].tolist(),
                "base_point_mm": base_point_mm.tolist(),
                "offset_mm": offset_mm.tolist(),
                "compensated_base_point_mm": compensated_base_point_mm.tolist(),
                "planned_above_tcp_pose_mm_deg": target_pose,
                "planned_grasp_tcp_pose_mm_deg": grasp_pose,
                "planned_lift_tcp_pose_mm_deg": lift_pose,
                "grasp_z_source": grasp_z_source,
                "target_yaw_override_deg": args.target_yaw_deg,
            },
            "safety": {
                "above_mm": args.above_mm,
                "grasp_above_surface_mm": args.grasp_above_surface_mm,
                "grasp_drop_mm": args.grasp_drop_mm,
                "grasp_z_mm": args.grasp_z_mm,
                "lift_mm": args.lift_mm,
                "min_z_mm": args.min_z_mm,
                "max_x_mm": args.max_x_mm,
                "move_timeout_sec": args.move_timeout_sec,
                "checks": checks,
            },
            "gripper": {
                "type": "pneumatic_cgpio",
                "ionum": args.pneumatic_do,
                "open_value": args.open_value,
                "close_value": args.close_value,
                "settle_sec": args.gripper_settle_sec,
            },
            "outputs": {
                "color": str(color_path),
                "depth_raw": str(depth_path),
                "mask": str(mask_path),
                "annotated": str(annotated_path),
                "result": str(result_path),
            },
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved annotated image: {annotated_path}")
        print(f"Saved result: {result_path}")
        return 0
    finally:
        if pipeline is not None:
            pipeline.stop()
            print("RealSense pipeline stopped.")
        arm.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    raise SystemExit(main())
