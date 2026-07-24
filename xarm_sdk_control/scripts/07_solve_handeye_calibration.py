#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAMPLES_ROOT = ROOT / "data" / "handeye_samples"
OUTPUT_ROOT = ROOT / "calibration"


HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


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
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    # Common xArm/robotics RPY interpretation: fixed-axis roll/pitch/yaw,
    # equivalent to R = Rz(yaw) * Ry(pitch) * Rx(roll).
    rotation = rz(yaw) @ ry(pitch) @ rx(roll)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0
    return transform


def rt_to_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (np.trace(rotation) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return math.degrees(math.acos(value))


def load_samples(samples_root: Path, max_reprojection_error_px: float | None, excludes: set[str]):
    paths = sorted(samples_root.glob("*/sample.json"))
    samples = []
    skipped = []
    for path in paths:
        if path.parent.name in excludes or str(path) in excludes:
            skipped.append((path, "explicitly excluded"))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        target = data.get("target", {})
        robot = data.get("robot", {})
        if target.get("type") != "chessboard":
            skipped.append((path, "not a chessboard sample"))
            continue
        err = target.get("mean_reprojection_error_px")
        if max_reprojection_error_px is not None and err is not None:
            if float(err) > max_reprojection_error_px:
                skipped.append((path, f"reprojection error {err:.3f}px exceeds limit"))
                continue
        tcp_pose = robot.get("tcp_pose_mm_deg")
        rvec = target.get("camera_T_target_rvec")
        tvec = target.get("camera_T_target_tvec_m")
        if tcp_pose is None or rvec is None or tvec is None:
            skipped.append((path, "missing tcp pose or target pose"))
            continue

        rotation_camera_target, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64).reshape(3, 1))
        camera_t_target = rt_to_transform(
            rotation_camera_target,
            np.array(tvec, dtype=np.float64).reshape(3),
        )
        base_t_tcp = tcp_pose_to_base_t_tcp(tcp_pose)
        samples.append(
            {
                "path": path,
                "timestamp": data.get("timestamp", path.parent.name),
                "tcp_pose_mm_deg": tcp_pose,
                "mean_reprojection_error_px": float(err),
                "base_T_tcp": base_t_tcp,
                "camera_T_target": camera_t_target,
            }
        )
    return samples, skipped


def solve_one_method(method_name: str, samples: list[dict]):
    r_gripper2base = [sample["base_T_tcp"][:3, :3] for sample in samples]
    t_gripper2base = [sample["base_T_tcp"][:3, 3] for sample in samples]
    r_target2cam = [sample["camera_T_target"][:3, :3] for sample in samples]
    t_target2cam = [sample["camera_T_target"][:3, 3] for sample in samples]

    rotation_tcp_camera, translation_tcp_camera = cv2.calibrateHandEye(
        r_gripper2base,
        t_gripper2base,
        r_target2cam,
        t_target2cam,
        method=HAND_EYE_METHODS[method_name],
    )
    tcp_t_camera = rt_to_transform(rotation_tcp_camera, translation_tcp_camera)
    base_t_targets = [
        sample["base_T_tcp"] @ tcp_t_camera @ sample["camera_T_target"] for sample in samples
    ]
    translations = np.stack([transform[:3, 3] for transform in base_t_targets], axis=0)
    mean_translation = translations.mean(axis=0)
    translation_errors = np.linalg.norm(translations - mean_translation, axis=1)

    reference_rotation = base_t_targets[0][:3, :3]
    rotation_errors_deg = np.array(
        [
            rotation_angle_deg(reference_rotation.T @ transform[:3, :3])
            for transform in base_t_targets
        ],
        dtype=np.float64,
    )
    return {
        "method": method_name,
        "tcp_T_camera": tcp_t_camera,
        "base_T_target_mean_translation_m": mean_translation,
        "target_translation_error_m": translation_errors,
        "target_rotation_error_deg": rotation_errors_deg,
        "base_T_target_per_sample": base_t_targets,
    }


def matrix_to_list(transform: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in transform]


def main():
    parser = argparse.ArgumentParser(
        description="Solve eye-in-hand calibration from collected xArm + RealSense samples."
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=SAMPLES_ROOT,
        help="Directory containing timestamped handeye sample folders.",
    )
    parser.add_argument(
        "--max-reprojection-error-px",
        type=float,
        default=1.0,
        help="Discard samples above this solvePnP reprojection error. Use <=0 to disable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Calibration JSON output path. Defaults to xarm_sdk_control/calibration/handeye_<time>.json.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Sample timestamp folder or sample.json path to exclude. Can be repeated.",
    )
    args = parser.parse_args()

    error_limit = args.max_reprojection_error_px
    if error_limit is not None and error_limit <= 0:
        error_limit = None

    samples, skipped = load_samples(args.samples_root, error_limit, set(args.exclude))
    print(f"Loaded valid samples: {len(samples)}")
    if skipped:
        print(f"Skipped samples: {len(skipped)}")
        for path, reason in skipped:
            print(f"  - {path}: {reason}")
    if len(samples) < 8:
        print("Need at least 8 good samples; 10-15 is preferred.")
        return 2

    results = []
    for method_name in HAND_EYE_METHODS:
        try:
            result = solve_one_method(method_name, samples)
        except cv2.error as exc:
            print(f"{method_name}: OpenCV failed: {exc}")
            continue
        results.append(result)
        trans_err_mm = result["target_translation_error_m"] * 1000.0
        rot_err = result["target_rotation_error_deg"]
        camera_xyz_mm = result["tcp_T_camera"][:3, 3] * 1000.0
        camera_rot_deg = rotation_angle_deg(result["tcp_T_camera"][:3, :3])
        print(
            f"{method_name:9s} "
            f"target position std/mean/max = {np.std(trans_err_mm):6.2f}/"
            f"{np.mean(trans_err_mm):6.2f}/{np.max(trans_err_mm):6.2f} mm, "
            f"rotation mean/max = {np.mean(rot_err):6.2f}/{np.max(rot_err):6.2f} deg, "
            f"tcp->camera xyz = [{camera_xyz_mm[0]:.1f}, {camera_xyz_mm[1]:.1f}, {camera_xyz_mm[2]:.1f}] mm, "
            f"rot_angle = {camera_rot_deg:.1f} deg"
        )

    if not results:
        print("No handeye method produced a result.")
        return 3

    best = min(results, key=lambda item: float(np.mean(item["target_translation_error_m"])))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or (OUTPUT_ROOT / f"handeye_{timestamp}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": timestamp,
        "description": "Eye-in-hand calibration. Transform maps RealSense camera coordinates into xArm TCP coordinates.",
        "rpy_convention": "R = Rz(yaw) @ Ry(pitch) @ Rx(roll), xArm pose units are mm/deg",
        "sample_count": len(samples),
        "sample_paths": [str(sample["path"]) for sample in samples],
        "max_reprojection_error_px": error_limit,
        "selected_method": best["method"],
        "tcp_T_camera": matrix_to_list(best["tcp_T_camera"]),
        "tcp_T_camera_translation_m": best["tcp_T_camera"][:3, 3].reshape(3).tolist(),
        "quality": {
            "target_position_error_mean_mm": float(
                np.mean(best["target_translation_error_m"]) * 1000.0
            ),
            "target_position_error_max_mm": float(
                np.max(best["target_translation_error_m"]) * 1000.0
            ),
            "target_position_error_std_mm": float(
                np.std(best["target_translation_error_m"]) * 1000.0
            ),
            "target_rotation_error_mean_deg": float(np.mean(best["target_rotation_error_deg"])),
            "target_rotation_error_max_deg": float(np.max(best["target_rotation_error_deg"])),
        },
        "all_methods": [
            {
                "method": result["method"],
                "tcp_T_camera": matrix_to_list(result["tcp_T_camera"]),
                "target_position_error_mean_mm": float(
                    np.mean(result["target_translation_error_m"]) * 1000.0
                ),
                "target_position_error_max_mm": float(
                    np.max(result["target_translation_error_m"]) * 1000.0
                ),
                "target_position_error_std_mm": float(
                    np.std(result["target_translation_error_m"]) * 1000.0
                ),
                "target_rotation_error_mean_deg": float(
                    np.mean(result["target_rotation_error_deg"])
                ),
                "target_rotation_error_max_deg": float(np.max(result["target_rotation_error_deg"])),
            }
            for result in results
        ],
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nSelected method:", best["method"])
    print("Selected tcp_T_camera:")
    print(np.array2string(best["tcp_T_camera"], precision=6, suppress_small=False))
    print(f"Saved calibration: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
