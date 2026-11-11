"""Small observation-only baseline for the short-neck scene.

The planar mouth crown supplies an AXIS PROXY, not the target circle. The
target is selected separately from the measured radius profile below it.
This assumes a roughly planar crown parallel to the local neck sections;
it is deliberately a restricted baseline, not a general neck-axis estimator.
No mesh, jar dimensions, target labels or evaluation files are read here.
"""
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull, QhullError

from .geometry import Camera, Estimate, basis


@dataclass
class DepthFrame:
    camera: Camera
    rgb: np.ndarray
    depth_z_m: np.ndarray
    diagnostic_mask: np.ndarray | None = None


@dataclass(frozen=True)
class ProfileSettings:
    # Fixed station volume, not a per-jar target crop.
    world_z_bounds_m: tuple = (.40, .76)
    world_xy_radius_m: float = .28
    plane_top_window_m: float = .035
    plane_seed_top_m: float = .006
    plane_tolerance_m: float = .00018
    plane_trials: int = 450
    profile_depth_m: float = .060
    exclude_crown_m: float = .002
    bin_width_m: float = .0005
    near_min_delta_m: float = .001
    radial_quantile: float = .85
    min_bin_points: int = 8
    maximum_gap_m: float = .0025


class ProfileRejection(ValueError):
    pass


def rgb_foreground(rgb):
    """Simple image extraction for this declared grey-jar/blue-background rig."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 1] < 100) & (hsv[:, :, 2] > 35)).astype(np.uint8)
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return np.zeros(mask.shape, bool)
    largest = 1 + np.argmax(statistics[1:, cv2.CC_STAT_AREA])
    return labels == largest


def _point_cloud(frames, evidence, settings):
    points, mask_counts = [], {}
    for frame in frames:
        if evidence == "mask_diagnostic":
            if frame.diagnostic_mask is None:
                raise ProfileRejection("diagnostic_mask_missing")
            mask = frame.diagnostic_mask.astype(bool)
        elif evidence == "rgb":
            mask = rgb_foreground(frame.rgb)
        else:
            raise ValueError("Unknown evidence condition")
        valid = mask & np.isfinite(frame.depth_z_m) & (frame.depth_z_m > 0)
        yy, xx = np.nonzero(valid)
        cloud = frame.camera.unproject(np.c_[xx, yy], frame.depth_z_m[yy, xx])
        keep = ((cloud[:, 2] >= settings.world_z_bounds_m[0]) &
                (cloud[:, 2] <= settings.world_z_bounds_m[1]) &
                (np.linalg.norm(cloud[:, :2], axis=1) < settings.world_xy_radius_m))
        points.append(cloud[keep])
        mask_counts[frame.camera.name] = int(keep.sum())
    cloud = np.concatenate(points)
    if len(cloud) < 300:
        raise ProfileRejection("insufficient_surface_points_in_fixed_station_volume")
    return cloud, mask_counts


def _mouth_plane(cloud, settings, rng):
    top = cloud[cloud[:, 2] > cloud[:, 2].max() - settings.plane_top_window_m]
    if len(top) < 80:
        raise ProfileRejection("insufficient_upper_surface_support")
    # Seed on the observed upper envelope, then expand the consensus over the
    # wider top region. A dominant plane anywhere on the shoulder is not a
    # mouth detector. This uses observed points, never nominal/true jar height.
    upper = top[top[:, 2] > top[:, 2].max()-settings.plane_seed_top_m]
    sample = upper[rng.choice(len(upper), min(6000, len(upper)), replace=False)]
    best_count, best = 0, None
    for _ in range(settings.plane_trials):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        if normal[2] < np.cos(np.deg2rad(30)):
            continue
        residual = np.abs((sample-a) @ normal)
        count = int(np.count_nonzero(residual < settings.plane_tolerance_m))
        if count > best_count:
            best_count, best = count, (normal, a)
    if best is None or best_count < 60:
        raise ProfileRejection("no_supported_near_upward_mouth_plane")
    normal, origin = best
    for _ in range(4):
        inliers = top[np.abs((top-origin) @ normal) < settings.plane_tolerance_m]
        origin = inliers.mean(axis=0)
        _, _, vectors = np.linalg.svd(inliers-origin, full_matrices=False)
        normal = vectors[-1]
        if normal[2] < 0:
            normal = -normal
    return origin, normal, inliers


def _mouth_center(origin, normal, inliers):
    u, v = basis(normal)
    xy = np.column_stack(((inliers-origin) @ u, (inliers-origin) @ v))
    hull = ConvexHull(xy)
    if hull.volume < .0008:
        raise ProfileRejection("mouth_plane_has_insufficient_spatial_support")
    boundary = xy[hull.vertices]
    matrix = np.c_[2*boundary, np.ones(len(boundary))]
    a, b, c = np.linalg.lstsq(matrix, np.sum(boundary**2, axis=1), rcond=None)[0]
    initial = [a, b, np.sqrt(max(c+a*a+b*b, 1e-8))]
    fit = least_squares(lambda x: np.linalg.norm(boundary-x[:2], axis=1)-x[2],
                        initial, loss="soft_l1", f_scale=.0004)
    if not .025 < fit.x[2] < .12:
        raise ProfileRejection("mouth_radius_outside_declared_broad_prior")
    center = origin + fit.x[0]*u + fit.x[1]*v
    return center, float(fit.x[2])


def _radius_profile(cloud, origin, normal, settings):
    delta = cloud-origin
    axial = delta @ normal
    radius = np.linalg.norm(delta-axial[:, None]*normal, axis=1)
    edges = np.arange(-settings.profile_depth_m, -settings.exclude_crown_m + 1e-9,
                      settings.bin_width_m)
    centres = (edges[:-1]+edges[1:])/2
    bins = np.searchsorted(edges, axial, side="right")-1
    values = np.full(len(centres), np.nan)
    counts = np.zeros(len(centres), dtype=int)
    for i in range(len(centres)):
        points = radius[bins == i]
        counts[i] = len(points)
        if len(points) >= settings.min_bin_points:
            values[i] = np.quantile(points, settings.radial_quantile)
    good = np.isfinite(values)
    if good.sum() < 12:
        raise ProfileRejection("insufficient_radius_profile_support")
    first, last = np.flatnonzero(good)[[0, -1]]
    s = centres[first:last+1]
    values = values[first:last+1]
    good = np.isfinite(values)
    if np.max(np.diff(s[good])) > settings.maximum_gap_m:
        raise ProfileRejection("unsupported_gap_in_radius_profile")
    values = np.interp(s, s[good], values[good])
    smooth = savgol_filter(values, 5, 2)
    index = int(np.argmin(smooth))
    threshold = smooth[index]+settings.near_min_delta_m
    selected = smooth <= threshold
    left, right = index, index
    while left > 0 and selected[left-1]:
        left -= 1
    while right+1 < len(s) and selected[right+1]:
        right += 1
    if left == 0 or right == len(s)-1:
        raise ProfileRejection("near_minimum_interval_touches_observed_search_boundary")
    # Linear threshold crossings, rather than choosing the lowest grid sample.
    low = float(np.interp(threshold, [smooth[left], smooth[left-1]], [s[left], s[left-1]]))
    high = float(np.interp(threshold, [smooth[right], smooth[right+1]], [s[right], s[right+1]]))
    midpoint = (low+high)/2
    return midpoint, float(np.interp(midpoint, s, smooth)), {
        "axial_from_observed_mouth_m": s.tolist(), "radius_m": values.tolist(),
        "smoothed_radius_m": smooth.tolist(), "near_min_interval_from_mouth_m": [low, high],
        "near_min_interval_length_m": high-low, "minimum_bin_radius_m": float(smooth[index]),
        "bin_counts": counts[first:last+1].tolist()}


def estimate_depth_profile(frames, *, evidence="rgb", axis_mode="mouth_proxy",
                           seed=4101, settings=ProfileSettings()):
    start = perf_counter()
    estimate = Estimate(diagnostics={"axis_mode": axis_mode, "evidence": evidence,
        "axis_assumption": "planar mouth crown parallel to neck sections; unverified on real jars",
        "uncertainty": "not_estimated; residual is not absolute accuracy"})
    try:
        cloud, mask_counts = _point_cloud(frames, evidence, settings)
        origin, normal, inliers = _mouth_plane(cloud, settings, np.random.default_rng(seed))
        plane_normal = normal.copy()
        center, mouth_radius = _mouth_center(origin, normal, inliers)
        if axis_mode == "vertical":
            normal = np.array([0., 0., 1.])
        elif axis_mode != "mouth_proxy":
            raise ValueError("Unknown axis baseline")
        position, radius, profile = _radius_profile(cloud, center, normal, settings)
        if not .02 < radius < mouth_radius-.001:
            raise ProfileRejection("candidate_not_narrower_than_observed_raised_lip")
        estimate.center, estimate.normal, estimate.radius = center+position*normal, normal, radius
        estimate.accepted = True
        estimate.residual = float(np.sqrt(np.mean(((inliers-origin) @ plane_normal)**2)))
        estimate.diagnostics.update(profile, point_counts=mask_counts,
                                    observed_mouth_center_m=center.tolist(),
                                    observed_mouth_radius_m=mouth_radius,
                                    observed_mouth_normal=plane_normal.tolist(),
                                    mouth_plane_support=len(inliers))
    except (ProfileRejection, np.linalg.LinAlgError, QhullError) as error:
        estimate.reasons.append(str(error))
    estimate.elapsed_ms = (perf_counter()-start)*1000
    return estimate
