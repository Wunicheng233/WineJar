"""Observation-derived display clouds; no meshes, target labels or hole filling."""
from dataclasses import dataclass

import numpy as np

from .depth_profile import ProfileSettings, rgb_foreground


@dataclass
class DisplayCloud:
    camera_name: str
    points: np.ndarray
    colors: np.ndarray
    station_mask: np.ndarray


def observation_cloud(camera, rgb, depth, settings=ProfileSettings(), *, foreground_mask=None):
    mask = rgb_foreground(rgb) if foreground_mask is None else np.asarray(foreground_mask, bool)
    if mask.shape != depth.shape:
        raise ValueError('Observation-derived foreground mask must match depth shape')
    valid = mask & np.isfinite(depth) & (depth > 0)
    yy, xx = np.nonzero(valid)
    points = camera.unproject(np.c_[xx, yy], depth[yy, xx])
    station = ((points[:, 2] >= settings.world_z_bounds_m[0]) &
               (points[:, 2] <= settings.world_z_bounds_m[1]) &
               (np.linalg.norm(points[:, :2], axis=1) < settings.world_xy_radius_m))
    return DisplayCloud(camera.name, points, rgb[yy, xx].copy(), station)


def display_samples(clouds, used, crop, source_colors=None, maximum_per_camera=60000):
    """Deterministic thinning only for display; report the pre-thinning count."""
    points, colors, total = [], [], 0
    for cloud in clouds:
        if cloud.camera_name not in used:
            continue
        selected = np.flatnonzero(cloud.station_mask) if crop else np.arange(len(cloud.points))
        total += len(selected)
        stride = max(1, int(np.ceil(len(selected)/maximum_per_camera)))
        selected = selected[::stride]
        points.append(cloud.points[selected])
        colors.append(cloud.colors[selected] if source_colors is None else
                      np.tile(source_colors[cloud.camera_name], (len(selected), 1)))
    xyz = np.concatenate(points) if points else np.empty((0, 3))
    rgb = np.concatenate(colors) if colors else np.empty((0, 3), np.uint8)
    return np.ascontiguousarray(xyz, dtype=np.float32), np.ascontiguousarray(rgb, dtype=np.uint8), total
