from __future__ import annotations

import math

import numpy as np


def matrix_to_list(transform: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in transform]


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (np.trace(rotation) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return math.degrees(math.acos(value))


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float64)])
    return (transform @ homogeneous.T).T[:, :3]


def solve_rigid_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Return target_T_source that best maps source_points onto target_points."""
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError("source_points and target_points must have the same shape")
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_points and target_points must have shape (N, 3)")
    if source.shape[0] < 3:
        raise ValueError("at least 3 point pairs are required")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    if np.linalg.matrix_rank(source_centered) < 2:
        raise ValueError("source points are degenerate; use non-collinear points")

    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def residuals_m(transform: np.ndarray, source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    predicted = transform_points(transform, source_points)
    target = np.asarray(target_points, dtype=np.float64)
    return np.linalg.norm(predicted - target, axis=1)
