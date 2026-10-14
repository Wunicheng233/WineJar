"""Depth-based initialization and validation for metric rim localization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraHeightEstimate:
    """One camera's robust estimate of the highest object surface."""

    height_m: float
    candidate_mask: np.ndarray
    top_support_mask: np.ndarray
    candidate_count: int
    top_support_count: int
    median_absolute_deviation_m: float


def transform_reference_points_to_workcell(
    points_reference_m: np.ndarray,
    transform_workcell_from_reference: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_reference_m, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(transform_workcell_from_reference, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(
            "transform_workcell_from_reference must be a finite 4x4 matrix"
        )
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def estimate_camera_top_height(
    points_workcell_m: np.ndarray,
    *,
    planar_radius_m: float = 0.075,
    minimum_object_height_m: float = 0.030,
    maximum_object_height_m: float = 0.500,
    top_quantile: float = 0.99,
    top_band_m: float = 0.004,
    minimum_candidate_points: int = 100,
    minimum_top_support_points: int = 20,
) -> CameraHeightEstimate:
    """Estimate the highest stable surface inside a cylinder around the workcell origin."""
    points = np.asarray(points_workcell_m, dtype=np.float64).reshape(-1, 3)
    if (
        not np.all(np.isfinite(points))
        or planar_radius_m <= 0
        or minimum_object_height_m < 0
        or maximum_object_height_m <= minimum_object_height_m
        or not 0.5 < top_quantile < 1.0
        or top_band_m <= 0
        or minimum_candidate_points < 1
        or minimum_top_support_points < 1
    ):
        raise ValueError("Invalid points or top-height estimation parameters")
    planar_distance = np.linalg.norm(points[:, :2], axis=1)
    candidate_mask = (
        (planar_distance <= planar_radius_m)
        & (points[:, 2] >= minimum_object_height_m)
        & (points[:, 2] <= maximum_object_height_m)
    )
    candidate_indices = np.flatnonzero(candidate_mask)
    if len(candidate_indices) < minimum_candidate_points:
        raise ValueError(
            f"Only {len(candidate_indices)} object points remain; need {minimum_candidate_points}"
        )
    candidate_heights = points[candidate_indices, 2]
    threshold = float(np.quantile(candidate_heights, top_quantile))
    supported = candidate_heights >= threshold - top_band_m
    top_indices = candidate_indices[supported]
    if len(top_indices) < minimum_top_support_points:
        raise ValueError(
            f"Only {len(top_indices)} top-surface points remain; need {minimum_top_support_points}"
        )
    top_support_mask = np.zeros(len(points), dtype=bool)
    top_support_mask[top_indices] = True
    top_heights = points[top_indices, 2]
    height = float(np.median(top_heights))
    mad = float(np.median(np.abs(top_heights - height)))
    return CameraHeightEstimate(
        height_m=height,
        candidate_mask=candidate_mask,
        top_support_mask=top_support_mask,
        candidate_count=len(candidate_indices),
        top_support_count=len(top_indices),
        median_absolute_deviation_m=mad,
    )
