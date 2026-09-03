"""Workcell reference-plane calibration from one fixed RGB-D camera."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PlaneFit:
    """Robust plane estimate in the multiview reference-camera frame."""

    point_m: np.ndarray
    normal_toward_camera: np.ndarray
    offset_m: float
    inlier_mask: np.ndarray
    rms_residual_m: float
    p95_residual_m: float
    maximum_residual_m: float


@dataclass(frozen=True)
class WorkcellFrame:
    """Right-handed workcell frame whose XY plane is the support surface."""

    transform_reference_from_workcell: np.ndarray
    transform_workcell_from_reference: np.ndarray
    plane: PlaneFit
    x_axis_source: str


def deproject_depth_roi_to_reference(
    depth_raw: np.ndarray,
    depth_scale_m: float,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    transform_reference_from_camera: np.ndarray,
    roi_xywh: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Deproject valid aligned-depth pixels in an ROI into the reference frame."""
    depth = np.asarray(depth_raw)
    if depth.ndim != 2 or depth_scale_m <= 0:
        raise ValueError("depth_raw must be a depth image and depth_scale_m must be positive")
    x, y, width, height = roi_xywh
    image_height, image_width = depth.shape
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError(f"ROI {roi_xywh} is outside depth image {image_width}x{image_height}")
    roi_depth = depth[y : y + height, x : x + width]
    valid_rows, valid_columns = np.nonzero(roi_depth > 0)
    valid_fraction = float(len(valid_rows) / roi_depth.size)
    if len(valid_rows) < 3:
        raise ValueError("ROI contains fewer than three valid depth pixels")
    pixels = np.column_stack((valid_columns + x, valid_rows + y)).astype(np.float64)
    normalized = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion_coefficients, dtype=np.float64),
    ).reshape(-1, 2)
    z_camera = roi_depth[valid_rows, valid_columns].astype(np.float64) * depth_scale_m
    points_camera = np.column_stack((normalized * z_camera[:, None], z_camera))
    transform = np.asarray(transform_reference_from_camera, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform_reference_from_camera must be 4x4")
    points_reference = (transform[:3, :3] @ points_camera.T).T + transform[:3, 3]
    return points_reference, pixels.astype(np.int32), valid_fraction


def fit_plane_ransac(
    points_m: np.ndarray,
    *,
    distance_threshold_m: float = 0.002,
    iterations: int = 600,
    minimum_inlier_ratio: float = 0.85,
    toward_point_m: np.ndarray | None = None,
    random_seed: int = 0,
) -> PlaneFit:
    """Fit a plane robustly, orienting its normal toward a known camera position."""
    points = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    if (
        len(points) < 20
        or not np.all(np.isfinite(points))
        or distance_threshold_m <= 0
        or iterations < 1
        or not 0 < minimum_inlier_ratio <= 1
    ):
        raise ValueError("Invalid plane points or RANSAC parameters")
    rng = np.random.default_rng(random_seed)
    best_mask: np.ndarray | None = None
    best_score: tuple[int, float] = (-1, -np.inf)
    for _ in range(iterations):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-10:
            continue
        normal /= norm
        residuals = np.abs((points - sample[0]) @ normal)
        mask = residuals <= distance_threshold_m
        count = int(np.count_nonzero(mask))
        score = (count, -float(np.median(residuals[mask])) if count else -np.inf)
        if score > best_score:
            best_score = score
            best_mask = mask
    if best_mask is None or np.count_nonzero(best_mask) < 3:
        raise ValueError("RANSAC could not find a valid support plane")
    for _ in range(3):
        inliers = points[best_mask]
        point = np.mean(inliers, axis=0)
        _, _, right_vectors = np.linalg.svd(inliers - point, full_matrices=False)
        normal = right_vectors[-1]
        normal /= np.linalg.norm(normal)
        refined_mask = np.abs((points - point) @ normal) <= distance_threshold_m
        if np.array_equal(refined_mask, best_mask):
            break
        best_mask = refined_mask
    inlier_ratio = float(np.count_nonzero(best_mask) / len(points))
    if inlier_ratio < minimum_inlier_ratio:
        raise ValueError(
            f"Plane inlier ratio {inlier_ratio:.3f} is below required {minimum_inlier_ratio:.3f}"
        )
    inliers = points[best_mask]
    point = np.mean(inliers, axis=0)
    _, _, right_vectors = np.linalg.svd(inliers - point, full_matrices=False)
    normal = right_vectors[-1]
    normal /= np.linalg.norm(normal)
    if toward_point_m is not None:
        toward = np.asarray(toward_point_m, dtype=np.float64).reshape(3)
        if float(normal @ (toward - point)) < 0:
            normal = -normal
    signed_residuals = (inliers - point) @ normal
    absolute_residuals = np.abs(signed_residuals)
    return PlaneFit(
        point_m=point,
        normal_toward_camera=normal,
        offset_m=-float(normal @ point),
        inlier_mask=best_mask,
        rms_residual_m=float(np.sqrt(np.mean(signed_residuals**2))),
        p95_residual_m=float(np.percentile(absolute_residuals, 95)),
        maximum_residual_m=float(np.max(absolute_residuals)),
    )


def build_workcell_frame(
    plane: PlaneFit,
    camera_x_axis_reference: np.ndarray,
    *,
    x_axis_source: str = "reference_camera_x_projected",
) -> WorkcellFrame:
    """Build an orthonormal right-handed frame on a fitted support plane."""
    z_axis = np.asarray(plane.normal_toward_camera, dtype=np.float64).reshape(3)
    x_seed = np.asarray(camera_x_axis_reference, dtype=np.float64).reshape(3)
    x_axis = x_seed - float(x_seed @ z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-8:
        raise ValueError("Requested X-axis direction is parallel to the support-plane normal")
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = plane.point_m
    return WorkcellFrame(
        transform_reference_from_workcell=transform,
        transform_workcell_from_reference=np.linalg.inv(transform),
        plane=plane,
        x_axis_source=x_axis_source,
    )


def height_above_workcell_plane(point_reference_m: np.ndarray, frame: WorkcellFrame) -> float:
    """Return signed height above the support plane, positive toward workcell +Z."""
    point = np.asarray(point_reference_m, dtype=np.float64).reshape(3)
    return float(frame.plane.normal_toward_camera @ (point - frame.plane.point_m))
