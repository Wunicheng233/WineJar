#!/usr/bin/env python3
"""Visually pick two stacked bamboo leaves and place them crossed on a ceramic disk."""

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

from xarm_sdk_control.positioning_bias import load_bias_mm  # noqa: E402
from xarm_sdk_control.vacuum_control import VacuumController, VacuumIOConfig  # noqa: E402
from xarm_sdk_control.visual_targeting import (  # noqa: E402
    capture_rgbd,
    detections_in_base,
    latest_global_calibration,
    run_vision,
    safety_report,
    select_ceramic,
    select_leaf_waypoints,
)


ROBOT_CONFIG = ROOT / "config" / "robot.yaml"
POSITION_BIAS_CONFIG = ROOT / "config" / "visual_position_bias.json"
OUTPUT_ROOT = ROOT / "data" / "visual_cross_place"


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


def require_motion_completed(
    arm: Any,
    label: str,
    result: Any,
    target_pose: list[float],
    position_tolerance_mm: float = 1.0,
    angle_tolerance_deg: float = 2.0,
) -> None:
    code = result[0] if isinstance(result, tuple) else result
    print(f"{label}: {result}")
    if code in (0, None):
        return
    if code != 3:
        raise RuntimeError(f"{label} failed with code/result: {result}")

    # Code 3 is a feedback timeout. The controller may still have reached the target.
    for _ in range(5):
        pose_code, actual_pose = arm.get_position(is_radian=False)
        state_code, state = arm.get_state()
        err_code, err_warn = arm.get_err_warn_code()
        position_error = float(np.max(np.abs(np.asarray(actual_pose[:3]) - np.asarray(target_pose[:3]))))
        angle_error = float(
            np.max(
                np.abs(
                    [wrapped_angle_deg(actual_pose[index] - target_pose[index]) for index in range(3, 6)]
                )
            )
        )
        if (
            pose_code == 0
            and state_code == 0
            and state == 2
            and err_code == 0
            and list(err_warn) == [0, 0]
            and position_error <= position_tolerance_mm
            and angle_error <= angle_tolerance_deg
        ):
            print(
                f"WARNING: {label} feedback timed out, but target was reached "
                f"(position error={position_error:.3f} mm, angle error={angle_error:.3f} deg)."
            )
            return
        time.sleep(0.2)
    raise RuntimeError(
        f"{label} timed out and target completion could not be verified; "
        f"last position error={position_error:.3f} mm, angle error={angle_error:.3f} deg, "
        f"state={state}, err/warn={err_warn}"
    )


def offset_detection(item: dict[str, Any], bias_m: np.ndarray) -> dict[str, Any]:
    result = dict(item)
    result["point_base_m"] = (np.asarray(item["point_base_m"], dtype=float) + bias_m).tolist()
    return result


def wrapped_angle_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def pose(point_m: list[float], z_mm: float, orientation: list[float]) -> list[float]:
    point_mm = np.asarray(point_m, dtype=float) * 1000.0
    return [float(point_mm[0]), float(point_mm[1]), float(z_mm), *map(float, orientation)]


def build_cycle_route(
    home_pose: list[float],
    leaf_point_m: list[float],
    ceramic_point_m: list[float],
    place_yaw_offset_deg: float,
    transport_clearance_mm: float,
    approach_clearance_mm: float,
    compression_mm: float,
    release_clearance_mm: float,
    layer_height_mm: float,
) -> list[tuple[str, list[float], str, str | None]]:
    pickup_orientation = list(map(float, home_pose[3:6]))
    place_orientation = pickup_orientation.copy()
    place_orientation[2] = wrapped_angle_deg(place_orientation[2] + place_yaw_offset_deg)
    leaf_mm = np.asarray(leaf_point_m, dtype=float) * 1000.0
    ceramic_mm = np.asarray(ceramic_point_m, dtype=float) * 1000.0
    transport_z = float(max(leaf_mm[2], ceramic_mm[2]) + transport_clearance_mm)

    route: list[tuple[str, list[float], str, str | None]] = [
        ("move above source leaf", pose(leaf_point_m, transport_z, pickup_orientation), "transit", None),
        (
            "descend near source leaf",
            pose(leaf_point_m, float(leaf_mm[2] + approach_clearance_mm), pickup_orientation),
            "vertical",
            None,
        ),
        (
            "compress suction cups on source leaf",
            pose(leaf_point_m, float(leaf_mm[2] - compression_mm), pickup_orientation),
            "contact",
            "acquire",
        ),
        ("lift leaf vertically", pose(leaf_point_m, transport_z, pickup_orientation), "vertical", None),
    ]
    route.extend(
        [
            (
                "carry leaf above ceramic while interpolating yaw",
                pose(ceramic_point_m, transport_z, place_orientation),
                "transit",
                None,
            ),
            (
                "descend above ceramic",
                pose(
                    ceramic_point_m,
                    float(ceramic_mm[2] + approach_clearance_mm + layer_height_mm),
                    place_orientation,
                ),
                "vertical",
                None,
            ),
            (
                "lower leaf to release height",
                pose(
                    ceramic_point_m,
                    float(ceramic_mm[2] + release_clearance_mm + layer_height_mm),
                    place_orientation,
                ),
                "contact",
                "release",
            ),
            ("retract vertically above ceramic", pose(ceramic_point_m, transport_z, place_orientation), "vertical", None),
        ]
    )
    route.append(
        (
            "return home while restoring tool yaw",
            list(map(float, home_pose)),
            "transit",
            None,
        )
    )
    return route


def choose_second_source_leaf(
    detections: list[dict[str, Any]],
    first_source_m: list[float],
    allow_clipping: bool,
    max_shift_m: float,
) -> dict[str, Any]:
    candidates, _ = select_leaf_waypoints(
        detections,
        distinct_distance_m=0.05,
        allow_any_clipping=allow_clipping,
    )
    reference_xy = np.asarray(first_source_m, dtype=float)[:2]
    selected = min(
        candidates,
        key=lambda item: float(np.linalg.norm(np.asarray(item["point_base_m"], dtype=float)[:2] - reference_xy)),
    )
    shift_m = float(np.linalg.norm(np.asarray(selected["point_base_m"], dtype=float)[:2] - reference_xy))
    if shift_m > max_shift_m:
        raise RuntimeError(
            f"Second-scan leaf is {shift_m * 1000.0:.1f} mm from the original source stack; "
            "refusing to risk picking the leaf already placed on the ceramic disk"
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--confirm-no-pressure-sensor", action="store_true")
    parser.add_argument("--allow-validated-clipped-leaf", action="store_true")
    parser.add_argument("--calibration", type=Path, default=latest_global_calibration())
    parser.add_argument("--vision-env", default="winejar-vision")
    parser.add_argument("--serial")
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--tcp-offset-z-mm", type=float, default=100.0)
    parser.add_argument("--first-place-yaw-offset-deg", type=float, default=90.0)
    parser.add_argument("--second-place-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--second-layer-height-mm", type=float, default=3.0)
    parser.add_argument("--max-second-source-shift-mm", type=float, default=80.0)
    parser.add_argument("--transport-clearance-mm", type=float, default=200.0)
    parser.add_argument("--approach-clearance-mm", type=float, default=35.0)
    parser.add_argument("--compression-mm", type=float, default=15.0)
    parser.add_argument("--release-clearance-mm", type=float, default=5.0)
    parser.add_argument("--vacuum-settle-seconds", type=float, default=1.0)
    parser.add_argument("--release-settle-seconds", type=float, default=0.5)
    parser.add_argument("--transit-speed", type=float, default=70.0)
    parser.add_argument("--vertical-speed", type=float, default=35.0)
    parser.add_argument("--contact-speed", type=float, default=15.0)
    parser.add_argument("--acc", type=float, default=240.0)
    parser.add_argument("--min-x-mm", type=float)
    parser.add_argument("--max-x-mm", type=float)
    parser.add_argument("--min-y-mm", type=float)
    parser.add_argument("--max-y-mm", type=float)
    parser.add_argument("--min-z-mm", type=float)
    parser.add_argument("--max-z-mm", type=float)
    return parser.parse_args()


def execute_route(
    arm: Any,
    vacuum: VacuumController,
    route: list[tuple[str, list[float], str, str | None]],
    args: argparse.Namespace,
    holding_state: dict[str, bool],
) -> None:
    speeds = {
        "transit": args.transit_speed,
        "vertical": args.vertical_speed,
        "contact": args.contact_speed,
    }
    for index, (label, target_pose, motion, event) in enumerate(route, start=1):
        result = arm.set_position(
            *target_pose,
            speed=speeds[motion],
            mvacc=args.acc,
            wait=True,
            radius=-1,
            is_radian=False,
        )
        require_motion_completed(
            arm,
            f"move {index:02d}: {label}",
            result,
            target_pose,
        )
        if event == "acquire":
            vacuum.acquire(args.vacuum_settle_seconds)
            holding_state["value"] = True
        elif event == "release":
            vacuum.release(args.release_settle_seconds)
            holding_state["value"] = False


def main() -> int:
    args = parse_args()
    if args.run and not args.confirm_no_pressure_sensor:
        print("ERROR: --run requires --confirm-no-pressure-sensor")
        return 2
    limits = (args.min_x_mm, args.max_x_mm, args.min_y_mm, args.max_y_mm, args.min_z_mm, args.max_z_mm)
    if args.run and any(value is None for value in limits):
        print("ERROR: --run requires explicit min/max X, Y, and Z limits")
        return 3
    if args.calibration is None or not args.calibration.expanduser().exists():
        raise RuntimeError("Global camera calibration is missing")
    if not 0 <= args.compression_mm <= 20:
        raise ValueError("Compression must remain within the stated 0~20 mm cup travel")

    calibration_path = args.calibration.expanduser().resolve()
    base_t_camera = np.asarray(json.loads(calibration_path.read_text())["base_T_camera"], dtype=float)
    position_bias_m = load_bias_mm(POSITION_BIAS_CONFIG) / 1000.0
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_dir = OUTPUT_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    first_frame = capture_rgbd(run_dir / "scan_1", args.serial)
    first_vision = run_vision(first_frame, args.vision_env, args.min_score)
    first_detections = detections_in_base(first_vision.get("detections", []), base_t_camera, args.min_score)
    ceramic = offset_detection(select_ceramic(first_detections), position_bias_m)
    first_leaf = offset_detection(
        select_leaf_waypoints(
            first_detections,
            distinct_distance_m=0.05,
            allow_any_clipping=args.allow_validated_clipped_leaf,
        )[0][0],
        position_bias_m,
    )

    with ROBOT_CONFIG.open("r", encoding="utf-8") as file:
        robot_ip = str(yaml.safe_load(file)["robot_ip"])
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(do_not_open=True, is_radian=False)
    vacuum: VacuumController | None = None
    holding_state = {"value": False}
    records: list[dict[str, Any]] = []
    try:
        require_code("connect", arm.connect(port=robot_ip))
        require_code(
            "set suction-cup TCP offset",
            arm.set_tcp_offset([0, 0, args.tcp_offset_z_mm, 0, 0, 0], is_radian=False, wait=True),
        )
        pose_code, home_pose = arm.get_position(is_radian=False)
        require_code("get camera-clear home pose", (pose_code, home_pose))

        first_route = build_cycle_route(
            home_pose,
            first_leaf["point_base_m"],
            ceramic["point_base_m"],
            args.first_place_yaw_offset_deg,
            args.transport_clearance_mm,
            args.approach_clearance_mm,
            args.compression_mm,
            args.release_clearance_mm,
            0.0,
        )
        routes = [("leaf_1", first_route)]
        print(f"System positioning bias [mm]: {(position_bias_m * 1000).round(2).tolist()}")
        print(f"First source [mm]: {(np.asarray(first_leaf['point_base_m']) * 1000).round(2).tolist()}")
        print(f"Saved ceramic target [mm]: {(np.asarray(ceramic['point_base_m']) * 1000).round(2).tolist()}")

        bound_values = {
            "min_x_mm": args.min_x_mm if args.min_x_mm is not None else -1000.0,
            "max_x_mm": args.max_x_mm if args.max_x_mm is not None else 1000.0,
            "min_y_mm": args.min_y_mm if args.min_y_mm is not None else -1000.0,
            "max_y_mm": args.max_y_mm if args.max_y_mm is not None else 1000.0,
            "min_z_mm": args.min_z_mm if args.min_z_mm is not None else -1000.0,
            "max_z_mm": args.max_z_mm if args.max_z_mm is not None else 1000.0,
        }
        first_safety = safety_report([item[1] for item in first_route], **bound_values)
        if args.run and not first_safety["all_ok"]:
            raise RuntimeError("First-leaf route is outside explicit workspace limits")

        if not args.run:
            print("DRY RUN: first scan and first-leaf route validated; second scan requires the first physical placement.")
            return 0

        require_emergency_stop_released(arm)
        vacuum = VacuumController(arm, VacuumIOConfig())
        vacuum.configure()
        vacuum.safe_shutdown()
        require_code("clean error", arm.clean_error())
        require_code("clean warn", arm.clean_warn())
        require_code("motion enable", arm.motion_enable(enable=True))
        require_code("set mode 0", arm.set_mode(0))
        require_code("set state 0", arm.set_state(0))

        execute_route(arm, vacuum, first_route, args, holding_state)
        records.append({"leaf": 1, "frame_dir": str(first_frame), "route": first_route})

        second_frame = capture_rgbd(run_dir / "scan_2", args.serial)
        second_vision = run_vision(second_frame, args.vision_env, args.min_score)
        second_detections = detections_in_base(second_vision.get("detections", []), base_t_camera, args.min_score)
        second_detections = [offset_detection(item, position_bias_m) for item in second_detections]
        second_leaf = choose_second_source_leaf(
            second_detections,
            first_leaf["point_base_m"],
            args.allow_validated_clipped_leaf,
            args.max_second_source_shift_mm / 1000.0,
        )
        print(f"Second source [mm]: {(np.asarray(second_leaf['point_base_m']) * 1000).round(2).tolist()}")
        second_route = build_cycle_route(
            home_pose,
            second_leaf["point_base_m"],
            ceramic["point_base_m"],
            args.second_place_yaw_offset_deg,
            args.transport_clearance_mm,
            args.approach_clearance_mm,
            args.compression_mm,
            args.release_clearance_mm,
            args.second_layer_height_mm,
        )
        second_safety = safety_report([item[1] for item in second_route], **bound_values)
        if not second_safety["all_ok"]:
            raise RuntimeError("Second-leaf route is outside explicit workspace limits")
        execute_route(arm, vacuum, second_route, args, holding_state)
        records.append({"leaf": 2, "frame_dir": str(second_frame), "route": second_route})
        vacuum.safe_shutdown()

        result_path = run_dir / "result.json"
        result_path.write_text(
            json.dumps(
                {
                    "calibration": str(calibration_path),
                    "position_bias_base_mm": (position_bias_m * 1000).tolist(),
                    "ceramic_target": ceramic,
                    "cycles": records,
                },
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved result: {result_path}")
        return 0
    except Exception:
        if holding_state["value"]:
            print(
                "EMERGENCY NOTE: vacuum may still be holding a leaf. Stop the robot and make the area safe "
                "before manually releasing it.",
                file=sys.stderr,
            )
        elif vacuum is not None:
            vacuum.safe_shutdown()
        raise
    finally:
        arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
