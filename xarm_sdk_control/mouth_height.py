"""Direct top/rim-height estimation in a calibrated workcell frame."""

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


@dataclass(frozen=True)
class FusedHeightEstimate:
    """Median fusion of mutually consistent camera height estimates."""

    height_m: float
    accepted_camera_ids: tuple[str, ...]
    rejected_camera_ids: tuple[str, ...]
    camera_spread_m: float


@dataclass(frozen=True)
class RadiusProfileSample:
    """One axial bin of a single camera's visible object surface."""

    lower_height_m: float
    upper_height_m: float
    radius_m: float
    point_count: int


@dataclass(frozen=True)
class CameraNeckBandEstimate:
    """Narrow upper band beginning where the shoulder has converged to the neck."""

    lower_height_m: float
    upper_height_m: float
    body_radius_m: float
    neck_radius_threshold_m: float
    profile: tuple[RadiusProfileSample, ...]


def transform_reference_points_to_workcell(
    points_reference_m: np.ndarray,
    transform_workcell_from_reference: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_reference_m, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(transform_workcell_from_reference, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("transform_workcell_from_reference must be a finite 4x4 matrix")
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


def fuse_camera_heights(
    estimates: dict[str, CameraHeightEstimate],
    *,
    agreement_tolerance_m: float = 0.005,
    minimum_camera_count: int = 2,
) -> FusedHeightEstimate:
    """Reject camera outliers around the median and fuse the remaining estimates."""
    if agreement_tolerance_m <= 0 or minimum_camera_count < 1 or len(estimates) < minimum_camera_count:
        raise ValueError("Invalid camera estimates or fusion parameters")
    camera_ids = list(estimates)
    heights = np.array([estimates[camera_id].height_m for camera_id in camera_ids], dtype=np.float64)
    if not np.all(np.isfinite(heights)):
        raise ValueError("Camera height estimates must be finite")
    initial_median = float(np.median(heights))
    accepted = np.abs(heights - initial_median) <= agreement_tolerance_m
    if int(np.count_nonzero(accepted)) < minimum_camera_count:
        raise ValueError(
            f"Only {int(np.count_nonzero(accepted))} camera heights agree within "
            f"{agreement_tolerance_m * 1000.0:.1f} mm"
        )
    accepted_heights = heights[accepted]
    accepted_ids = tuple(camera_id for camera_id, keep in zip(camera_ids, accepted) if keep)
    rejected_ids = tuple(camera_id for camera_id, keep in zip(camera_ids, accepted) if not keep)
    return FusedHeightEstimate(
        height_m=float(np.median(accepted_heights)),
        accepted_camera_ids=accepted_ids,
        rejected_camera_ids=rejected_ids,
        camera_spread_m=float(np.ptp(accepted_heights)),
    )


def estimate_neck_band(
    points_workcell_m: np.ndarray,
    axis_center_xy_m: np.ndarray,
    top_height_m: float,
    *,
    profile_depth_m: float = 0.100,
    body_window_below_top_m: tuple[float, float] = (0.050, 0.090),
    bin_width_m: float = 0.004,
    radius_percentile: float = 90.0,
    neck_to_body_ratio: float = 0.75,
    maximum_planar_radius_m: float = 0.075,
    minimum_bin_points: int = 80,
    minimum_consecutive_bins: int = 3,
) -> CameraNeckBandEstimate:
    """Locate the stable narrow band between a bottle shoulder and its top.

    This is a one-dimensional surface profile, not a volume reconstruction. The
    body reference is measured below the top, then the first sustained sequence
    narrower than ``neck_to_body_ratio`` marks the lower neck boundary.
    """
    points = np.asarray(points_workcell_m, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(axis_center_xy_m, dtype=np.float64).reshape(2)
    lower_body_offset, upper_body_offset = sorted(body_window_below_top_m)
    if (
        not np.all(np.isfinite(points))
        or not np.all(np.isfinite(center))
        or not np.isfinite(top_height_m)
        or profile_depth_m <= 0
        or lower_body_offset <= 0
        or upper_body_offset <= lower_body_offset
        or bin_width_m <= 0
        or not 50 <= radius_percentile < 100
        or not 0 < neck_to_body_ratio < 1
        or maximum_planar_radius_m <= 0
        or minimum_bin_points < 1
        or minimum_consecutive_bins < 1
    ):
        raise ValueError("Invalid neck-band points or profile parameters")
    radial = np.linalg.norm(points[:, :2] - center, axis=1)
    profile_floor = top_height_m - profile_depth_m
    base_mask = (
        (radial <= maximum_planar_radius_m)
        & (points[:, 2] >= profile_floor)
        & (points[:, 2] <= top_height_m + bin_width_m)
    )
    profile: list[RadiusProfileSample] = []
    for lower in np.arange(profile_floor, top_height_m, bin_width_m):
        upper = min(lower + bin_width_m, top_height_m + bin_width_m)
        selected = base_mask & (points[:, 2] >= lower) & (points[:, 2] < upper)
        count = int(np.count_nonzero(selected))
        if count < minimum_bin_points:
            continue
        profile.append(
            RadiusProfileSample(
                lower_height_m=float(lower),
                upper_height_m=float(upper),
                radius_m=float(np.percentile(radial[selected], radius_percentile)),
                point_count=count,
            )
        )
    if len(profile) < minimum_consecutive_bins + 3:
        raise ValueError("Too few supported height bins for a neck profile")
    body_lower = top_height_m - upper_body_offset
    body_upper = top_height_m - lower_body_offset
    body_radii = [
        sample.radius_m
        for sample in profile
        if sample.lower_height_m >= body_lower and sample.upper_height_m <= body_upper + bin_width_m
    ]
    if len(body_radii) < 3:
        raise ValueError("Too few body-reference bins below the candidate neck")
    body_radius = float(np.median(body_radii))
    threshold = body_radius * neck_to_body_ratio
    narrow = np.array([sample.radius_m <= threshold for sample in profile], dtype=bool)
    start_index = None
    for index in range(0, len(profile) - minimum_consecutive_bins + 1):
        if np.all(narrow[index : index + minimum_consecutive_bins]) and np.mean(narrow[index:]) >= 0.80:
            start_index = index
            break
    if start_index is None:
        raise ValueError("No sustained narrow band exists above the shoulder")
    lower_height = profile[start_index].lower_height_m
    if top_height_m - lower_height < 2.0 * bin_width_m:
        raise ValueError("Detected neck band is too short")
    return CameraNeckBandEstimate(
        lower_height_m=float(lower_height),
        upper_height_m=float(top_height_m),
        body_radius_m=body_radius,
        neck_radius_threshold_m=threshold,
        profile=tuple(profile),
    )
