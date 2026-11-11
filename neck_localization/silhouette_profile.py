"""Experimental RGB-only multi-view surface-of-revolution baseline.

This is a small geometric reimplementation inspired by SOR contour literature,
not the original authors' complete bundle-adjustment implementation. A smooth
contour's backprojected tangent plane contains the surface normal. For a given
axis, that normal and the axis define a meridian plane, yielding a 3D contour
generator without depth measurements. Multiple views share a free radial spline.
"""
from dataclasses import asdict, dataclass
from time import perf_counter

import cv2
import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from .geometry import Camera, Estimate
from .stereo_robust import extract_stereo_masks


@dataclass
class SilhouetteImage:
    camera: Camera
    rgb: np.ndarray


@dataclass(frozen=True)
class SilhouetteSettings:
    # Declared station search volume, not a true jar height or radius.
    world_z_bounds_m: tuple = (.40, .76)
    hull_xy_half_extent_m: float = .14
    hull_xy_step_m: float = .004
    hull_z_step_m: float = .002
    origin_below_observed_hull_top_m: float = .020
    contour_below_origin_mm: float = 35.
    contour_above_origin_mm: float = 13.
    maximum_candidate_radius_mm: float = 120.
    maximum_abs_vertical_contour_normal: float = .7
    contour_smoothing_pixels: float = 1.
    knot_spacing_mm: float = 1.5
    spline_second_difference_weight: float = .5
    axis_xy_limit_mm: float = 30.
    axis_slope_limit: float = .4
    maximum_evaluations: int = 150
    near_minimum_delta_mm: float = 1.
    maximum_context_rms_mm: float = 1.


def contour_generator(a, normal, cameras, rays, tangent_normals):
    """Intersect camera ray with the normal/axis meridian plane.

    tangent_normals are world normals to backprojected IMAGE tangent planes.
    Rays need not be unit vectors. The normal sign is immaterial.
    """
    plane_normal = np.cross(normal, tangent_normals)
    denominator = np.sum(plane_normal*rays, axis=1)
    if np.any(abs(denominator) < 1e-9):
        raise ValueError('degenerate_contour_meridian_intersection')
    distance = np.sum(plane_normal*(a-cameras), axis=1)/denominator
    points = cameras+distance[:, None]*rays
    offset = points-a
    axial_mm = offset @ normal*1000
    radial_mm = np.linalg.norm(offset-(offset @ normal)[:, None]*normal, axis=1)*1000
    return axial_mm, radial_mm, points


def observed_hull_origin(images, masks, settings):
    """Coarse cone intersection supplies initialization, not a target estimate."""
    xy = np.arange(-settings.hull_xy_half_extent_m,
                   settings.hull_xy_half_extent_m+settings.hull_xy_step_m/2,
                   settings.hull_xy_step_m)
    xx, yy = np.meshgrid(xy, xy)
    points = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]
    top = None
    zlow, zhigh = settings.world_z_bounds_m
    for z in np.arange(zhigh, zlow, -settings.hull_z_step_m):
        points[:, 2] = z
        valid = np.ones(len(points), bool)
        for image, mask in zip(images, masks):
            pixel = np.rint(image.camera.project(points)).astype(int)
            inside = ((pixel[:, 0] >= 0) & (pixel[:, 0] < image.camera.width) &
                      (pixel[:, 1] >= 0) & (pixel[:, 1] < image.camera.height))
            inside[inside] &= mask[pixel[inside, 1], pixel[inside, 0]]
            valid &= inside
        if valid.sum() < 4:
            continue
        if top is None:
            top = float(z)
        if z < top-.014:
            origin = points[valid].mean(axis=0)
            origin[2] = top-settings.origin_below_observed_hull_top_m
            return origin, top
    raise ValueError('no_rgb_visual_hull_in_declared_station_volume')


def extract_contours(images, masks, settings):
    centers, rays, normals, pixels, ids = [], [], [], [], []
    for index, (image, mask) in enumerate(zip(images, masks)):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError('no_rgb_foreground_contour')
        q = max(contours, key=cv2.contourArea)[:, 0].astype(float)
        q = gaussian_filter1d(q, settings.contour_smoothing_pixels, axis=0, mode='wrap')
        tangent = (np.roll(q, -1, axis=0)-np.roll(q, 1, axis=0))/2
        line = np.c_[-tangent[:, 1], tangent[:, 0],
                     tangent[:, 1]*q[:, 0]-tangent[:, 0]*q[:, 1]]
        world_normal = line @ image.camera.K @ image.camera.T_world_from_camera[:3, :3].T
        norm = np.linalg.norm(world_normal, axis=1)
        good = norm > 1e-9
        q, world_normal, norm = q[good], world_normal[good], norm[good]
        world_normal /= norm[:, None]
        ray = (np.c_[q, np.ones(len(q))] @ np.linalg.inv(image.camera.K).T
               @ image.camera.T_world_from_camera[:3, :3].T)
        centers.append(np.broadcast_to(image.camera.T_world_from_camera[:3, 3], ray.shape))
        rays.append(ray)
        normals.append(world_normal)
        pixels.append(q)
        ids.append(np.full(len(q), index))
    return tuple(np.concatenate(values) for values in [centers, rays, normals, pixels, ids])


class SilhouetteSplineProblem:
    def __init__(self, origin, samples, settings=SilhouetteSettings(), reference_normal=None):
        self.origin, self.samples, self.settings = np.array(origin), samples, settings
        if reference_normal is None:
            reference_normal = np.array([0., 0., 1.])
        axial, _, _ = contour_generator(self.origin, reference_normal, *samples[:3])
        low, high = float(min(axial)), float(max(axial))
        self.knots = np.r_[[low-5]*4, np.arange(low, high, settings.knot_spacing_mm), [high+5]*4]
        count = len(self.knots)-4
        self.penalty = np.diff(np.eye(count), n=2, axis=0)*settings.spline_second_difference_weight

    def evaluate(self, x, full=False, fixed_axis=None):
        origin = self.origin+np.r_[np.asarray(x[:2])/1000, 0.]
        normal = np.r_[x[2:], 1.]
        normal /= np.linalg.norm(normal)
        if fixed_axis is not None:
            origin, normal = fixed_axis
        s, radius, points = contour_generator(origin, normal, *self.samples[:3])
        design = BSpline.design_matrix(s, self.knots, 3, extrapolate=True).toarray()
        augmented = np.r_[design, self.penalty]
        target = np.r_[radius, np.zeros(len(self.penalty))]
        coefficients = np.linalg.lstsq(augmented, target, rcond=None)[0]
        residual = augmented @ coefficients-target
        if full:
            return residual, s, radius, BSpline(self.knots, coefficients, 3), origin, normal, points
        return residual


def estimate_silhouette_profile(images, settings=SilhouetteSettings(), *, diagnostic_axis=None,
                                diagnostic_axial_interval_from_anchor_m=None):
    """RGB+K/T only. Explicit oracle options are for separately labelled diagnosis."""
    started = perf_counter()
    result = Estimate()
    diagnostic = diagnostic_axis is not None or diagnostic_axial_interval_from_anchor_m is not None
    detail = dict(method='silhouette_axis_free_profile', diagnostic_oracle=diagnostic,
                  inputs='RGB_only_and_supplied_K_T; no_depth_no_target_dimensions',
                  settings=asdict(settings), uncertainty='not_calibrated')
    try:
        if len(images) < 2:
            raise ValueError('single_view_scale_and_axis_ambiguity')
        masks, foreground = extract_stereo_masks(images)
        detail['foreground'] = foreground
        origin, top = observed_hull_origin(images, masks, settings)
        normal = np.array([0., 0., 1.])
        if diagnostic_axis is not None:
            point, normal = diagnostic_axis
            normal = np.asarray(normal, float)/np.linalg.norm(normal)
            origin = np.asarray(point)+(origin[2]-point[2])/normal[2]*normal
        samples = extract_contours(images, masks, settings)
        s, radius, points = contour_generator(origin, normal, *samples[:3])
        keep = ((s > -settings.contour_below_origin_mm) & (s < settings.contour_above_origin_mm) &
                (radius > 5) & (radius < settings.maximum_candidate_radius_mm) &
                (abs(samples[2] @ normal) < settings.maximum_abs_vertical_contour_normal))
        if diagnostic_axial_interval_from_anchor_m is not None:
            if diagnostic_axis is None:
                raise ValueError('diagnostic_axial_region_requires_its_axis_anchor')
            from_anchor = s/1000+(origin-diagnostic_axis[0]) @ normal
            keep &= ((from_anchor >= diagnostic_axial_interval_from_anchor_m[0]) &
                     (from_anchor <= diagnostic_axial_interval_from_anchor_m[1]))
        samples = tuple(value[keep] for value in samples)
        detail.update(observed_hull_top_m=top, initial_axis_origin_m=origin.tolist(),
                      fit_pixels=samples[3].tolist(), fit_camera_ids=samples[4].tolist(),
                      fit_camera_names=[image.camera.name for image in images],
                      selected_contour_samples=len(samples[0]),
                      diagnostic_axial_interval_from_anchor_m=(
                          list(diagnostic_axial_interval_from_anchor_m)
                          if diagnostic_axial_interval_from_anchor_m is not None else None))
        if len(samples[0]) < 24 or len(np.unique(samples[4])) < 2:
            raise ValueError('insufficient_local_multiview_silhouette_samples')
        problem = SilhouetteSplineProblem(origin, samples, settings, reference_normal=normal)
        if diagnostic_axis is None:
            xy, slope = settings.axis_xy_limit_mm, settings.axis_slope_limit
            fit = least_squares(problem.evaluate, np.zeros(4),
                                bounds=([-xy, -xy, -slope, -slope], [xy, xy, slope, slope]),
                                max_nfev=settings.maximum_evaluations, diff_step=1e-4)
            x, converged, evaluations = fit.x, bool(fit.success), fit.nfev
            condition = float(np.linalg.cond(fit.jac))
        else:
            x, converged, evaluations, condition = np.zeros(4), True, 0, None
        residual, s, radius, spline, origin, normal, points = problem.evaluate(
            x, True, fixed_axis=(origin, normal) if diagnostic_axis is not None else None)
        lower, upper = np.percentile(s, [2, 98])
        grid = np.arange(lower, upper+.025, .05)
        values = spline(grid)
        minimum = int(np.argmin(values))
        left = right = minimum
        while left > 0 and values[left-1] <= values[minimum]+settings.near_minimum_delta_mm:
            left -= 1
        while right+1 < len(grid) and values[right+1] <= values[minimum]+settings.near_minimum_delta_mm:
            right += 1
        target_s = float((grid[left]+grid[right])/2)
        result.center = origin+target_s/1000*normal
        result.normal, result.radius = normal, float(spline(target_s)/1000)
        rms = float(np.sqrt(np.mean(residual[:len(s)]**2)))
        reasons = []
        if not converged:
            reasons.append('silhouette_optimizer_not_converged')
        if rms > settings.maximum_context_rms_mm:
            reasons.append('multiview_contour_profile_inconsistency')
        if minimum == 0 or minimum == len(grid)-1:
            reasons.append('minimum_at_observed_profile_boundary')
        if result.radius <= 0:
            reasons.append('nonpositive_profile_radius')
        result.accepted, result.reasons, result.residual = not reasons, reasons, rms/1000
        detail.update(observed_hull_top_m=top, axis_origin_m=origin.tolist(),
                      fit_pixels=samples[3].tolist(), fit_camera_ids=samples[4].tolist(),
                      fit_camera_names=[image.camera.name for image in images],
                      reconstructed_contour_points_m=points.tolist(),
                      axial_samples_mm=s.tolist(), radial_samples_mm=radius.tolist(),
                      profile_grid_axial_mm=grid.tolist(), profile_radius_mm=values.tolist(),
                      near_min_interval_from_axis_origin_m=[float(grid[left]/1000), float(grid[right]/1000)],
                      context_rms_mm=rms, nonlinear_jacobian_condition=condition,
                      evaluations=evaluations, optimizer_converged=converged,
                      normal_sign='toward_positive_world_Z',
                      target_definition='midpoint_of_observed_silhouette_near_minimum; physical_groove_not_certified')
    except (ValueError, np.linalg.LinAlgError) as error:
        result.reasons = [str(error)]
    result.diagnostics = detail
    result.elapsed_ms = (perf_counter()-started)*1000
    return result
