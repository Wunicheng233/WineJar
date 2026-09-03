"""Explainable surface-of-revolution neck estimate from a multiview visual hull."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RadiusProfileSample:
    """Outer-radius statistics for one axial slice of a visual hull."""

    axial_position_m: float
    radius_mean_m: float
    radius_min_m: float
    radius_max_m: float
    sector_count: int
    point_count: int


@dataclass(frozen=True)
class NeckLocalization:
    """Reference-frame neck geometry and an auditable binding-plane candidate."""

    axis_point_m: np.ndarray
    axis_mouth_to_body: np.ndarray
    mouth_center_m: np.ndarray
    mouth_plane_normal: np.ndarray
    binding_center_m: np.ndarray
    binding_plane_normal: np.ndarray
    binding_min_diameter_m: float
    binding_max_diameter_m: float
    binding_constriction_ratio: float
    profile: list[RadiusProfileSample]
    confidence: float
    failure_reason: str | None


def _orthogonal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    seed = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, seed)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return first, second


def _outer_radius_statistics(
    radial_vectors: np.ndarray,
    basis_first: np.ndarray,
    basis_second: np.ndarray,
    sector_count: int,
) -> tuple[float, float, float, int]:
    radii = np.linalg.norm(radial_vectors, axis=1)
    angles = np.arctan2(radial_vectors @ basis_second, radial_vectors @ basis_first)
    sector_ids = np.floor((angles + np.pi) / (2.0 * np.pi) * sector_count).astype(int) % sector_count
    outer_radii = []
    for sector in range(sector_count):
        values = radii[sector_ids == sector]
        if len(values) >= 3:
            outer_radii.append(float(np.percentile(values, 90)))
    if len(outer_radii) < max(4, sector_count // 3):
        raise ValueError("Too few occupied angular sectors in axial slice")
    return (
        float(np.mean(outer_radii)),
        float(np.min(outer_radii)),
        float(np.max(outer_radii)),
        len(outer_radii),
    )


def fit_neck_from_visual_hull(
    points_reference_m: np.ndarray,
    *,
    mouth_to_body_hint: np.ndarray = np.array([0.0, 0.0, 1.0]),
    slice_width_m: float = 0.008,
    sector_count: int = 16,
    min_slice_points: int = 30,
    min_circularity: float = 0.55,
) -> NeckLocalization:
    """Fit a principal axis and radius profile, then select a stable narrow neck band.

    The result deliberately remains a geometric estimate rather than a learned
    bottle-class model. A caller must inspect confidence and failure_reason
    before treating the binding plane as a production target.
    """
    points = np.asarray(points_reference_m, dtype=np.float64).reshape(-1, 3)
    if len(points) < 300:
        raise ValueError("Visual hull requires at least 300 occupied points")
    if (
        not np.all(np.isfinite(points))
        or slice_width_m <= 0
        or sector_count < 8
        or min_slice_points < 8
        or not 0 < min_circularity <= 1
    ):
        raise ValueError("Invalid visual-hull points or profile parameters")
    hint = np.asarray(mouth_to_body_hint, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(hint)) or np.linalg.norm(hint) == 0:
        raise ValueError("mouth_to_body_hint must be a finite non-zero vector")
    hint /= np.linalg.norm(hint)
    axis_point = np.mean(points, axis=0)
    covariance = np.cov((points - axis_point).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if float(axis @ hint) < 0:
        axis = -axis
    first, second = _orthogonal_basis(axis)
    axial = (points - axis_point) @ axis
    start = float(np.min(axial))
    stop = float(np.max(axial))
    extent = stop - start
    if extent < 4.0 * slice_width_m:
        raise ValueError("Visual hull is too short for a neck radius profile")
    edges = np.arange(start, stop + slice_width_m, slice_width_m)
    profile: list[RadiusProfileSample] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (axial >= lower) & (axial < upper)
        if int(np.count_nonzero(selected)) < min_slice_points:
            continue
        slice_axial = axial[selected]
        radial_vectors = points[selected] - (axis_point + np.outer(slice_axial, axis))
        try:
            radius_mean, radius_min, radius_max, sectors = _outer_radius_statistics(
                radial_vectors,
                first,
                second,
                sector_count,
            )
        except ValueError:
            continue
        profile.append(
            RadiusProfileSample(
                axial_position_m=float((lower + upper) / 2.0),
                radius_mean_m=radius_mean,
                radius_min_m=radius_min,
                radius_max_m=radius_max,
                sector_count=sectors,
                point_count=int(np.count_nonzero(selected)),
            )
        )
    if len(profile) < 5:
        raise ValueError("Too few supported axial slices for neck localization")
    positions = np.array([sample.axial_position_m for sample in profile])
    radii = np.array([sample.radius_mean_m for sample in profile])
    min_position = float(np.min(positions))
    max_position = float(np.max(positions))
    profile_extent = max_position - min_position
    # Avoid treating the physical rim itself as a binding location. Search the
    # upper portion, where a neck should be present before the shoulder/body.
    upper_neck_window = np.nonzero(
        (positions >= min_position + 0.08 * profile_extent)
        & (positions <= min_position + 0.48 * profile_extent)
    )[0]
    if len(upper_neck_window) == 0:
        raise ValueError("No candidate slices remain below the mouth rim")
    circularity_ratio = np.array(
        [sample.radius_min_m / max(sample.radius_max_m, 1e-9) for sample in profile],
        dtype=np.float64,
    )
    supported = np.array([sample.sector_count >= int(np.ceil(0.75 * sector_count)) for sample in profile])
    stable = supported & (circularity_ratio >= min_circularity)
    candidates = upper_neck_window[stable[upper_neck_window]]
    if len(candidates) == 0:
        raise ValueError("No candidate slice has sufficient angular support and circularity")
    scores = []
    for index in candidates:
        neighbours = radii[max(0, index - 1) : min(len(radii), index + 2)]
        local_variation = float(np.ptp(neighbours))
        circularity_penalty = profile[index].radius_max_m - profile[index].radius_min_m
        scores.append(radii[index] + 0.7 * local_variation + 0.3 * circularity_penalty)
    binding_index = int(candidates[int(np.argmin(scores))])
    binding = profile[binding_index]
    mouth_center = axis_point + min_position * axis
    binding_center = axis_point + binding.axial_position_m * axis
    slice_coverage = len(profile) / max(1, len(edges) - 1)
    circularity = circularity_ratio[binding_index]
    circularity_score = float(np.clip((circularity - min_circularity) / max(1e-9, 1.0 - min_circularity), 0.0, 1.0))
    body_indices = np.nonzero(positions >= min_position + 0.55 * profile_extent)[0]
    body_radii = radii[body_indices][stable[body_indices]]
    if len(body_radii) == 0:
        body_radii = radii[body_indices]
    if len(body_radii) == 0:
        raise ValueError("No supported body slices remain for neck constriction assessment")
    constriction_ratio = float(binding.radius_mean_m / np.median(body_radii))
    support = min(1.0, len(points) / 8_000.0)
    constriction_score = float(np.clip((0.90 - constriction_ratio) / 0.30, 0.0, 1.0))
    confidence = float(
        np.clip(0.35 * slice_coverage + 0.30 * circularity_score + 0.20 * support + 0.15 * constriction_score, 0.0, 1.0)
    )
    failure_reason = None
    if constriction_ratio > 0.90:
        failure_reason = "NO_STABLE_BINDING_ZONE"
    elif confidence < 0.55:
        failure_reason = "LOW_VISUAL_HULL_SUPPORT_OR_CIRCULARITY"
    return NeckLocalization(
        axis_point_m=axis_point,
        axis_mouth_to_body=axis,
        mouth_center_m=mouth_center,
        mouth_plane_normal=axis,
        binding_center_m=binding_center,
        binding_plane_normal=axis,
        binding_min_diameter_m=2.0 * binding.radius_min_m,
        binding_max_diameter_m=2.0 * binding.radius_max_m,
        binding_constriction_ratio=constriction_ratio,
        profile=profile,
        confidence=confidence,
        failure_reason=failure_reason,
    )
