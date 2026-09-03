"""Material-independent RGB-D surface scoring for a fixed dual suction cup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent / "config" / "suction_surface_profiles.yaml"


@dataclass(frozen=True)
class SurfaceProfile:
    edge_margin_mm: float
    dark_contrast: int
    max_defect_fraction: float
    max_roughness_mm: float
    max_tilt_deg: float
    min_valid_depth_fraction: float


@dataclass(frozen=True)
class CupEvaluation:
    center_px: tuple[float, float]
    score: float
    clearance_mm: float
    valid_depth_fraction: float
    defect_fraction: float
    roughness_mm: float
    tilt_deg: float


@dataclass(frozen=True)
class SuctionCandidate:
    center_px: tuple[float, float]
    angle_deg: float
    cup_centers_px: tuple[tuple[float, float], tuple[float, float]]
    center_camera_m: tuple[float, float, float]
    score: float
    cups: tuple[CupEvaluation, CupEvaluation]


@dataclass(frozen=True)
class SurfaceAnalysis:
    candidates: list[SuctionCandidate]
    object_mask: np.ndarray
    defect_mask: np.ndarray
    roughness_mm: np.ndarray
    cup_radius_px: float
    cup_spacing_px: float
    reference_depth_m: float


def load_surface_profiles(path: Path = DEFAULT_PROFILE_PATH) -> dict[str, SurfaceProfile]:
    payload = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    profiles: dict[str, SurfaceProfile] = {}
    for name, values in payload.items():
        profile = SurfaceProfile(**values)
        if (
            profile.edge_margin_mm < 0
            or not 0 <= profile.dark_contrast <= 255
            or not 0 < profile.max_defect_fraction <= 1
            or profile.max_roughness_mm <= 0
            or not 0 < profile.max_tilt_deg <= 90
            or not 0 < profile.min_valid_depth_fraction <= 1
        ):
            raise ValueError(f"Invalid suction surface profile: {name}")
        profiles[str(name)] = profile
    return profiles


def _disk_kernel(radius_px: int) -> np.ndarray:
    radius = max(1, int(radius_px))
    size = radius * 2 + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = ((x * x + y * y) <= radius * radius).astype(np.float32)
    return kernel / float(np.sum(kernel))


def _normalized_disk_mean(values: np.ndarray, support: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    support_float = support.astype(np.float32)
    numerator = cv2.filter2D(values.astype(np.float32) * support_float, -1, kernel)
    denominator = cv2.filter2D(support_float, -1, kernel)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-6)


def _prepare_mask(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError("object_mask must be a single-channel image")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def _surface_maps(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    profile: SurfaceProfile,
    focal_px: float,
    reference_depth_m: float,
    cup_radius_px: int,
) -> dict[str, np.ndarray]:
    valid = (depth_m > 0) & np.isfinite(depth_m) & (object_mask > 0)
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    contrast_radius = max(2, int(round(cup_radius_px * 0.75)))
    contrast_size = contrast_radius * 2 + 1
    contrast_sizes = {
        contrast_size,
        max(contrast_size, int(round(cup_radius_px * 2.0)) * 2 + 1),
    }
    dark_contrast = np.zeros_like(gray)
    for size in sorted(contrast_sizes):
        closing = cv2.morphologyEx(
            gray,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )
        dark_contrast = np.maximum(dark_contrast, cv2.subtract(closing, gray))
    defect = ((dark_contrast >= profile.dark_contrast) & (object_mask > 0)).astype(np.uint8)
    continuity_radius = max(1, int(round(cup_radius_px * 0.6)))
    continuity_size = continuity_radius * 2 + 1
    defect = cv2.morphologyEx(
        defect,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (continuity_size, continuity_size),
        ),
    )
    defect = cv2.dilate(defect, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    fill_depth = depth_m.astype(np.float32).copy()
    fill_depth[~valid] = float(reference_depth_m)
    smooth_size = max(5, cup_radius_px * 2 + 1)
    if smooth_size % 2 == 0:
        smooth_size += 1
    smooth_depth = cv2.GaussianBlur(fill_depth, (smooth_size, smooth_size), 0)
    roughness_mm = np.abs(fill_depth - smooth_depth) * 1000.0
    roughness_mm[~valid] = profile.max_roughness_mm * 2.0

    derivative_x = cv2.Sobel(smooth_depth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    derivative_y = cv2.Sobel(smooth_depth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    metric_scale = focal_px / max(reference_depth_m, 1e-6)
    tilt_deg = np.degrees(np.arctan(np.hypot(derivative_x, derivative_y) * metric_scale))
    tilt_deg[~valid] = 90.0
    return {
        "valid": valid.astype(np.uint8),
        "defect": defect,
        "roughness_mm": roughness_mm,
        "tilt_deg": tilt_deg,
        "distance_px": cv2.distanceTransform(object_mask, cv2.DIST_L2, 5),
    }


def _cup_score(
    center_px: tuple[float, float],
    maps: dict[str, np.ndarray],
    filtered: dict[str, np.ndarray],
    profile: SurfaceProfile,
    radius_px: float,
    px_per_mm: float,
) -> CupEvaluation | None:
    x = int(round(center_px[0]))
    y = int(round(center_px[1]))
    height, width = maps["valid"].shape
    if x < 0 or x >= width or y < 0 or y >= height:
        return None
    clearance_mm = max(0.0, (float(maps["distance_px"][y, x]) - radius_px) / px_per_mm)
    valid_fraction = float(filtered["valid_fraction"][y, x])
    defect_fraction = float(filtered["defect_fraction"][y, x])
    roughness = float(filtered["roughness_mm"][y, x])
    tilt = float(filtered["tilt_deg"][y, x])
    if (
        clearance_mm < profile.edge_margin_mm
        or valid_fraction < profile.min_valid_depth_fraction
        or defect_fraction > profile.max_defect_fraction
        or roughness > profile.max_roughness_mm
        or tilt > profile.max_tilt_deg
    ):
        return None
    factors = np.array(
        [
            min(1.0, clearance_mm / max(profile.edge_margin_mm * 2.0, 1e-6)),
            (valid_fraction - profile.min_valid_depth_fraction)
            / max(1.0 - profile.min_valid_depth_fraction, 1e-6),
            1.0 - defect_fraction / profile.max_defect_fraction,
            1.0 - roughness / profile.max_roughness_mm,
            1.0 - tilt / profile.max_tilt_deg,
        ],
        dtype=np.float64,
    )
    weights = np.array([0.20, 0.15, 0.25, 0.25, 0.15], dtype=np.float64)
    score = float(np.dot(np.clip(factors, 0.0, 1.0), weights))
    return CupEvaluation(
        center_px=(float(center_px[0]), float(center_px[1])),
        score=score,
        clearance_mm=clearance_mm,
        valid_depth_fraction=valid_fraction,
        defect_fraction=defect_fraction,
        roughness_mm=roughness,
        tilt_deg=tilt,
    )


def analyze_suction_surface(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    camera_matrix: np.ndarray,
    profile: SurfaceProfile,
    *,
    cup_diameter_mm: float = 8.0,
    cup_spacing_mm: float = 48.0,
    angle_step_deg: float = 15.0,
    preferred_angle_deg: float = 0.0,
    grid_step_px: int = 5,
    max_candidates: int = 20,
) -> SurfaceAnalysis:
    """Return ranked collision-free dual-cup poses in the color image."""
    color = np.asarray(color_bgr)
    depth = np.asarray(depth_m, dtype=np.float32)
    mask = _prepare_mask(object_mask)
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if color.ndim != 3 or color.shape[:2] != depth.shape or depth.shape != mask.shape:
        raise ValueError("color, depth, and object mask must have matching image dimensions")
    if matrix.shape != (3, 3) or min(matrix[0, 0], matrix[1, 1]) <= 0:
        raise ValueError("camera_matrix must be a valid 3x3 intrinsic matrix")
    if min(cup_diameter_mm, cup_spacing_mm, angle_step_deg, grid_step_px, max_candidates) <= 0:
        raise ValueError("Suction geometry and search parameters must be positive")
    if not np.isfinite(preferred_angle_deg):
        raise ValueError("preferred_angle_deg must be finite")
    valid_depth = depth[(mask > 0) & np.isfinite(depth) & (depth > 0)]
    if len(valid_depth) < 100:
        raise ValueError("Object mask contains too little valid depth")
    reference_depth = float(np.median(valid_depth))
    focal_px = float((matrix[0, 0] + matrix[1, 1]) / 2.0)
    px_per_mm = focal_px / (reference_depth * 1000.0)
    radius_px = max(2.0, cup_diameter_mm * 0.5 * px_per_mm)
    spacing_px = cup_spacing_mm * px_per_mm
    kernel = _disk_kernel(int(np.ceil(radius_px)))
    maps = _surface_maps(
        color, depth, mask, profile, focal_px, reference_depth, int(np.ceil(radius_px))
    )
    support = mask > 0
    filtered = {
        "valid_fraction": cv2.filter2D(maps["valid"].astype(np.float32), -1, kernel),
        "defect_fraction": cv2.filter2D(maps["defect"].astype(np.float32), -1, kernel),
        "roughness_mm": _normalized_disk_mean(maps["roughness_mm"], support, kernel),
        "tilt_deg": _normalized_disk_mean(maps["tilt_deg"], support, kernel),
    }

    rows, columns = np.nonzero(mask)
    centroid = np.array([float(np.mean(columns)), float(np.mean(rows))])
    object_scale = max(float(np.ptp(columns)), float(np.ptp(rows)), 1.0)
    candidates: list[SuctionCandidate] = []
    y_min, y_max = int(np.min(rows)), int(np.max(rows))
    x_min, x_max = int(np.min(columns)), int(np.max(columns))
    for y in range(y_min, y_max + 1, grid_step_px):
        for x in range(x_min, x_max + 1, grid_step_px):
            if not mask[y, x]:
                continue
            for angle_deg in np.arange(0.0, 180.0, angle_step_deg):
                angle_rad = np.radians(angle_deg)
                offset = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * spacing_px * 0.5
                first_xy = np.array([x, y], dtype=np.float64) - offset
                second_xy = np.array([x, y], dtype=np.float64) + offset
                first = _cup_score(tuple(first_xy), maps, filtered, profile, radius_px, px_per_mm)
                second = _cup_score(tuple(second_xy), maps, filtered, profile, radius_px, px_per_mm)
                if first is None or second is None:
                    continue
                surface_score = min(first.score, second.score)
                center_distance = np.linalg.norm(np.array([x, y]) - centroid)
                centrality = float(np.exp(-center_distance / (0.35 * object_scale)))
                angle_delta = abs((angle_deg - preferred_angle_deg + 90.0) % 180.0 - 90.0)
                orientation = float(np.exp(-np.square(angle_delta / 25.0)))
                # Surface checks above are hard gates. Among feasible poses, favor
                # a balanced load near the object centre and the default horizontal yaw.
                score = 0.40 * surface_score + 0.40 * centrality + 0.20 * orientation
                z = float(np.median([depth[int(round(first_xy[1])), int(round(first_xy[0]))], depth[int(round(second_xy[1])), int(round(second_xy[0]))]]))
                center_camera = (
                    (x - matrix[0, 2]) * z / matrix[0, 0],
                    (y - matrix[1, 2]) * z / matrix[1, 1],
                    z,
                )
                candidates.append(
                    SuctionCandidate(
                        center_px=(float(x), float(y)),
                        angle_deg=float(angle_deg),
                        cup_centers_px=(tuple(first_xy), tuple(second_xy)),
                        center_camera_m=tuple(map(float, center_camera)),
                        score=float(score),
                        cups=(first, second),
                    )
                )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return SurfaceAnalysis(
        candidates=candidates[:max_candidates],
        object_mask=mask,
        defect_mask=maps["defect"],
        roughness_mm=maps["roughness_mm"],
        cup_radius_px=radius_px,
        cup_spacing_px=spacing_px,
        reference_depth_m=reference_depth,
    )


def masked_center_camera_point(
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[float, float, float]:
    """Return an object's mask centroid with robust central-surface depth."""
    depth = np.asarray(depth_m, dtype=np.float32)
    mask = _prepare_mask(object_mask)
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if depth.ndim != 2 or depth.shape != mask.shape:
        raise ValueError("depth and object mask must have matching image dimensions")
    if matrix.shape != (3, 3) or min(matrix[0, 0], matrix[1, 1]) <= 0:
        raise ValueError("camera_matrix must be a valid 3x3 intrinsic matrix")
    rows, columns = np.nonzero(mask)
    if len(rows) < 25:
        raise ValueError("Object mask is too small to locate its centre")
    center_x = float(np.mean(columns))
    center_y = float(np.mean(rows))
    span = max(6.0, 0.15 * min(float(np.ptp(columns)), float(np.ptp(rows))))
    x_grid = np.arange(mask.shape[1])[None, :]
    y_grid = np.arange(mask.shape[0])[:, None]
    central = (mask > 0) & ((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= span**2)
    valid = central & np.isfinite(depth) & (depth > 0)
    if np.count_nonzero(valid) < 10:
        valid = (mask > 0) & np.isfinite(depth) & (depth > 0)
    if np.count_nonzero(valid) < 10:
        raise ValueError("Object mask contains too little valid central depth")
    z = float(np.median(depth[valid]))
    return (
        float((center_x - matrix[0, 2]) * z / matrix[0, 0]),
        float((center_y - matrix[1, 2]) * z / matrix[1, 1]),
        z,
    )


def candidate_to_dict(candidate: SuctionCandidate) -> dict[str, object]:
    return {
        "center_px": list(candidate.center_px),
        "angle_deg": candidate.angle_deg,
        "cup_centers_px": [list(point) for point in candidate.cup_centers_px],
        "center_camera_m": list(candidate.center_camera_m),
        "score": candidate.score,
        "cups": [
            {
                "center_px": list(cup.center_px),
                "score": cup.score,
                "clearance_mm": cup.clearance_mm,
                "valid_depth_fraction": cup.valid_depth_fraction,
                "defect_fraction": cup.defect_fraction,
                "roughness_mm": cup.roughness_mm,
                "tilt_deg": cup.tilt_deg,
            }
            for cup in candidate.cups
        ],
    }


def draw_suction_candidates(
    color_bgr: np.ndarray,
    analysis: SurfaceAnalysis,
    count: int = 5,
) -> np.ndarray:
    overlay = np.asarray(color_bgr).copy()
    rejected = (analysis.defect_mask > 0) & (analysis.object_mask > 0)
    overlay[rejected] = (0.55 * overlay[rejected] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
    contours, _ = cv2.findContours(analysis.object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 220, 255), 1)
    radius = max(2, int(round(analysis.cup_radius_px)))
    for rank, candidate in enumerate(analysis.candidates[:count], start=1):
        colour = (0, 255, 0) if rank == 1 else (255, 180, 0)
        thickness = 3 if rank == 1 else 1
        points = [tuple(int(round(value)) for value in point) for point in candidate.cup_centers_px]
        cv2.line(overlay, points[0], points[1], colour, thickness)
        for point in points:
            cv2.circle(overlay, point, radius, colour, thickness)
        center = tuple(int(round(value)) for value in candidate.center_px)
        cv2.putText(
            overlay,
            str(rank),
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
        )
    if analysis.candidates:
        best = analysis.candidates[0]
        cv2.putText(
            overlay,
            f"best score={best.score:.2f} angle={best.angle_deg:.0f}deg",
            (10, overlay.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 0),
            2,
        )
    return overlay
