"""Voxel visual-hull reconstruction from calibrated fixed-camera silhouettes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xarm_sdk_control.multiview_rig import FixedCameraRig


@dataclass(frozen=True)
class AxisAlignedBounds:
    """Axis-aligned reference-frame reconstruction bounds, in metres."""

    minimum_m: np.ndarray
    maximum_m: np.ndarray

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum_m, dtype=np.float64).reshape(3)
        maximum = np.asarray(self.maximum_m, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)) or np.any(maximum <= minimum):
            raise ValueError("Bounds must be finite and have maximum greater than minimum on every axis")
        object.__setattr__(self, "minimum_m", minimum)
        object.__setattr__(self, "maximum_m", maximum)

    @property
    def extent_m(self) -> np.ndarray:
        return self.maximum_m - self.minimum_m


@dataclass(frozen=True)
class VisualHull:
    """Occupied voxel centres retained by every camera silhouette."""

    bounds: AxisAlignedBounds
    voxel_size_m: float
    candidate_voxel_count: int
    occupied_points_reference_m: np.ndarray


def derive_bounds_from_reference_mask(
    mask: np.ndarray,
    depth_raw: np.ndarray,
    depth_scale_m: float,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    *,
    depth_padding_m: float = 0.04,
    lateral_padding_m: float = 0.03,
    max_depth_span_m: float | None = 0.18,
) -> AxisAlignedBounds:
    """Create a conservative local 3D box from one reference silhouette and depth map."""
    silhouette = np.asarray(mask, dtype=np.uint8)
    depth = np.asarray(depth_raw)
    if silhouette.ndim != 2 or depth.shape != silhouette.shape:
        raise ValueError("Mask and depth_raw must be same-size single-channel images")
    if (
        depth_scale_m <= 0
        or depth_padding_m < 0
        or lateral_padding_m < 0
        or (max_depth_span_m is not None and max_depth_span_m <= 0)
    ):
        raise ValueError("Depth scale and padding values must be non-negative, with positive depth scale")
    ys, xs = np.nonzero(silhouette > 0)
    if len(xs) < 10:
        raise ValueError("Reference silhouette contains too few pixels")
    valid_depth = depth[ys, xs].astype(np.float64) * float(depth_scale_m)
    valid_depth = valid_depth[valid_depth > 0]
    if len(valid_depth) < 50:
        raise ValueError("Reference silhouette has too few valid depth pixels for automatic bounds")
    near_m, far_m = np.percentile(valid_depth, [5, 95])
    near_m = max(0.05, float(near_m) - depth_padding_m)
    far_m = float(far_m) + depth_padding_m
    if max_depth_span_m is not None:
        far_m = min(far_m, near_m + max_depth_span_m)
    if far_m <= near_m:
        raise ValueError("Automatic depth bounds are degenerate")
    image_corners = np.array(
        [[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    import cv2

    normalized = cv2.undistortPoints(image_corners, camera_matrix, distortion_coefficients).reshape(-1, 2)
    corner_points = np.vstack(
        [
            np.column_stack((normalized * near_m, np.full(4, near_m))),
            np.column_stack((normalized * far_m, np.full(4, far_m))),
        ]
    )
    lateral_padding = np.array([lateral_padding_m, lateral_padding_m, 0.0], dtype=np.float64)
    return AxisAlignedBounds(
        minimum_m=np.min(corner_points, axis=0) - lateral_padding,
        maximum_m=np.max(corner_points, axis=0) + lateral_padding,
    )


def voxel_centres(bounds: AxisAlignedBounds, voxel_size_m: float, max_voxels: int = 800_000) -> np.ndarray:
    """Return a regular grid of voxel centres inside bounds, with a safety cap."""
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be finite and positive")
    axes = [
        np.arange(lower + voxel_size_m / 2.0, upper, voxel_size_m, dtype=np.float64)
        for lower, upper in zip(bounds.minimum_m, bounds.maximum_m)
    ]
    count = int(np.prod([len(axis) for axis in axes]))
    if count == 0:
        raise ValueError("Bounds are smaller than one voxel")
    if count > max_voxels:
        raise ValueError(
            f"Requested {count} voxels, exceeding safety limit {max_voxels}; use a coarser voxel size or smaller bounds"
        )
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([value.reshape(-1) for value in mesh])


def reconstruct_visual_hull(
    rig: FixedCameraRig,
    silhouettes: dict[str, np.ndarray],
    bounds: AxisAlignedBounds,
    voxel_size_m: float,
    *,
    max_voxels: int = 800_000,
) -> VisualHull:
    """Keep reference-frame voxels that project inside every supplied silhouette."""
    expected_ids = set(rig.cameras)
    if set(silhouettes) != expected_ids:
        raise ValueError(f"Silhouettes must match rig cameras exactly: expected {sorted(expected_ids)}")
    candidates = voxel_centres(bounds, voxel_size_m, max_voxels)
    keep = np.ones(len(candidates), dtype=bool)
    for camera_id, camera in rig.cameras.items():
        silhouette = np.asarray(silhouettes[camera_id], dtype=np.uint8)
        expected_shape = (camera.image_size_px[1], camera.image_size_px[0])
        if silhouette.shape != expected_shape:
            raise ValueError(f"{camera_id} silhouette shape {silhouette.shape} does not match {expected_shape}")
        pixels = rig.project_reference_points(camera_id, candidates)
        columns = np.rint(pixels[:, 0]).astype(np.int32)
        rows = np.rint(pixels[:, 1]).astype(np.int32)
        in_image = (
            (columns >= 0)
            & (columns < camera.image_size_px[0])
            & (rows >= 0)
            & (rows < camera.image_size_px[1])
        )
        visible = np.zeros(len(candidates), dtype=bool)
        valid = np.nonzero(in_image)[0]
        visible[valid] = silhouette[rows[valid], columns[valid]] > 0
        keep &= visible
    return VisualHull(
        bounds=bounds,
        voxel_size_m=float(voxel_size_m),
        candidate_voxel_count=len(candidates),
        occupied_points_reference_m=candidates[keep],
    )


def write_ascii_ply(path: Path, points_m: np.ndarray) -> Path:
    """Write a minimal point-cloud PLY for inspection in standard viewers."""
    resolved = path.expanduser().resolve()
    points = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for point in points:
            handle.write(f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f}\n")
    return resolved
