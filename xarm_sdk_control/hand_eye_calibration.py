"""Geometry and validation for an eye-in-hand chessboard calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from xarm_sdk_control.multiview_calibration import (
    average_rigid_transforms,
    transform_rotation_angle_deg,
)


@dataclass(frozen=True)
class HandEyeCalibration:
    """Flange-to-camera transform and fixed-board consistency diagnostics."""

    transform_flange_from_camera: np.ndarray
    method: str
    accepted_indices: list[int]
    rejected_indices: list[int]
    board_translation_residuals_m: list[float]
    board_rotation_residuals_deg: list[float]
    transform_base_from_board: np.ndarray


def pose_mm_deg_to_transform(pose_mm_deg: Sequence[float]) -> np.ndarray:
    """Convert xArm [x,y,z,roll,pitch,yaw] into a homogeneous transform.

    xArm uses the conventional RPY matrix ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    Translation is converted from millimetres to metres.
    """
    pose = np.asarray(pose_mm_deg, dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("pose_mm_deg must contain six finite values")
    roll, pitch, yaw = np.radians(pose[3:])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    rotation_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rotation_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x
    transform[:3, 3] = pose[:3] / 1000.0
    return transform


def flange_transform_from_tcp_pose(
    base_from_tcp_pose_mm_deg: Sequence[float],
    flange_from_tcp_offset_mm_deg: Sequence[float],
) -> np.ndarray:
    """Recover ``base_T_flange`` from the reported TCP pose and active offset."""
    base_from_tcp = pose_mm_deg_to_transform(base_from_tcp_pose_mm_deg)
    flange_from_tcp = pose_mm_deg_to_transform(flange_from_tcp_offset_mm_deg)
    return base_from_tcp @ np.linalg.inv(flange_from_tcp)


def _board_consistency(
    base_from_flange: Sequence[np.ndarray],
    camera_from_board: Sequence[np.ndarray],
    flange_from_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    board_poses = [
        np.asarray(robot_pose) @ flange_from_camera @ np.asarray(board_pose)
        for robot_pose, board_pose in zip(base_from_flange, camera_from_board)
    ]
    mean_pose = average_rigid_transforms(board_poses)
    translation = np.asarray(
        [np.linalg.norm(pose[:3, 3] - mean_pose[:3, 3]) for pose in board_poses],
        dtype=np.float64,
    )
    rotation = np.asarray(
        [transform_rotation_angle_deg(pose, mean_pose) for pose in board_poses],
        dtype=np.float64,
    )
    return mean_pose, translation, rotation


def _solve_method(
    base_from_flange: Sequence[np.ndarray],
    camera_from_board: Sequence[np.ndarray],
    method: int,
) -> np.ndarray:
    rotation, translation = cv2.calibrateHandEye(
        [pose[:3, :3] for pose in base_from_flange],
        [pose[:3, 3].reshape(3, 1) for pose in base_from_flange],
        [pose[:3, :3] for pose in camera_from_board],
        [pose[:3, 3].reshape(3, 1) for pose in camera_from_board],
        method=method,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _candidate_methods() -> list[tuple[str, int]]:
    return [
        ("TSAI", cv2.CALIB_HAND_EYE_TSAI),
        ("PARK", cv2.CALIB_HAND_EYE_PARK),
        ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
        ("ANDREFF", cv2.CALIB_HAND_EYE_ANDREFF),
        ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS),
    ]


def _best_solution(
    base_from_flange: Sequence[np.ndarray],
    camera_from_board: Sequence[np.ndarray],
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates: list[tuple[float, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for name, method in _candidate_methods():
        try:
            transform = _solve_method(base_from_flange, camera_from_board, method)
            if not np.all(np.isfinite(transform)):
                continue
            board_pose, translation, rotation = _board_consistency(
                base_from_flange, camera_from_board, transform
            )
            score = float(np.median(translation) + np.radians(np.median(rotation)) * 0.05)
            candidates.append((score, name, transform, board_pose, translation, rotation))
        except cv2.error:
            continue
    if not candidates:
        raise ValueError("All OpenCV hand-eye methods failed; collect more varied robot orientations")
    _, name, transform, board_pose, translation, rotation = min(candidates, key=lambda item: item[0])
    return name, transform, board_pose, translation, rotation


def _robust_limit(values: np.ndarray, floor: float) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(floor, median + 3.5 * 1.4826 * mad)


def calibrate_eye_in_hand(
    base_from_flange: Sequence[np.ndarray],
    camera_from_board: Sequence[np.ndarray],
    *,
    min_samples: int = 8,
) -> HandEyeCalibration:
    """Solve ``flange_T_camera`` and reject inconsistent fixed-board samples."""
    if len(base_from_flange) != len(camera_from_board):
        raise ValueError("Robot and chessboard pose lists must have equal lengths")
    if len(base_from_flange) < min_samples:
        raise ValueError(f"At least {min_samples} synchronized samples are required")
    robot = [np.asarray(value, dtype=np.float64) for value in base_from_flange]
    board = [np.asarray(value, dtype=np.float64) for value in camera_from_board]
    if any(value.shape != (4, 4) or not np.all(np.isfinite(value)) for value in [*robot, *board]):
        raise ValueError("Every pose must be a finite 4x4 transform")

    _, _, _, translation, rotation = _best_solution(robot, board)
    translation_limit = _robust_limit(translation, 0.004)
    rotation_limit = _robust_limit(rotation, 0.5)
    accepted = [
        index
        for index, (translation_error, rotation_error) in enumerate(zip(translation, rotation))
        if translation_error <= translation_limit and rotation_error <= rotation_limit
    ]
    if len(accepted) < min_samples:
        accepted = list(range(len(robot)))

    filtered_robot = [robot[index] for index in accepted]
    filtered_board = [board[index] for index in accepted]
    method, transform, board_pose, translation, rotation = _best_solution(filtered_robot, filtered_board)
    rejected = sorted(set(range(len(robot))) - set(accepted))
    return HandEyeCalibration(
        transform_flange_from_camera=transform,
        method=method,
        accepted_indices=accepted,
        rejected_indices=rejected,
        board_translation_residuals_m=translation.tolist(),
        board_rotation_residuals_deg=rotation.tolist(),
        transform_base_from_board=board_pose,
    )


def pose_set_diversity(base_from_flange: Sequence[np.ndarray]) -> dict[str, float]:
    """Summarize robot-pose spread used to diagnose weak calibration motion."""
    poses = [np.asarray(value, dtype=np.float64) for value in base_from_flange]
    if len(poses) < 2:
        return {"max_translation_m": 0.0, "max_rotation_deg": 0.0}
    translations: list[float] = []
    rotations: list[float] = []
    for first_index, first in enumerate(poses[:-1]):
        for second in poses[first_index + 1 :]:
            translations.append(float(np.linalg.norm(first[:3, 3] - second[:3, 3])))
            rotations.append(transform_rotation_angle_deg(first, second))
    return {
        "max_translation_m": max(translations),
        "max_rotation_deg": max(rotations),
    }
