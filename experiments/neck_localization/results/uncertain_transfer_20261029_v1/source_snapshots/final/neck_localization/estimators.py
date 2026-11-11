"""Circle estimators using observed pixels/depth and supplied camera calibration.

No function accepts scene ground truth.  The broad radius/tilt limits are research
priors, not a jar specification.  Reported covariance is conditional on fixed
calibration, fixed boundary labels and the selected inliers; it is not an absolute
accuracy guarantee.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from .geometry import Estimate, Observation, basis, normal_positive_z


MIN_RADIUS_M = 0.020
MAX_RADIUS_M = 0.120
MAX_TILT_DEG = 55.0
DEPTH_INLIER_M = 0.003
PIXEL_INLIER_PX = 2.5
MIN_POINTS = 12
MIN_ARC_DEG = 120.0


def _finish(result, start):
    result.elapsed_ms = (perf_counter() - start) * 1000
    return result


def _reject(reason, start, **diagnostics):
    return _finish(Estimate(reasons=[reason], diagnostics=diagnostics), start)


def _depth_points(observations):
    valid = []
    for observation in observations:
        if observation.points_world is None:
            continue
        points = np.asarray(observation.points_world, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Depth points must have shape (N, 3)")
        valid.append(points[np.all(np.isfinite(points), axis=1)])
    return np.concatenate(valid) if valid else np.empty((0, 3))


def _circle_distances(points, center, normal, radius):
    delta = points - center
    height = delta @ normal
    radial = np.sqrt(np.maximum(np.sum(delta * delta, axis=1) - height**2, 0))
    return np.hypot(height, radial - radius)


def _arc_span(points, center, normal):
    """Coverage is 360 degrees minus the largest unsupported angular gap."""
    u, v = basis(normal)
    delta = points - center
    angles = np.sort(np.mod(np.arctan2(delta @ v, delta @ u), 2 * np.pi))
    if len(angles) < 2:
        return 0.0
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    return float(np.degrees(2 * np.pi - np.max(gaps)))


def _plane_circle(points):
    """PCA plane followed by a normalized, algebraic least-squares circle."""
    origin = np.mean(points, axis=0)
    _, singular, vh = np.linalg.svd(points - origin, full_matrices=False)
    if len(singular) != 3 or singular[1] < 1e-7 or singular[1] / singular[0] < 1e-3:
        raise ValueError("Depth support is collinear or spatially degenerate")
    normal = normal_positive_z(vh[-1])
    u, v = basis(normal)
    xy = np.column_stack(((points - origin) @ u, (points - origin) @ v))
    scale = float(np.sqrt(np.mean(np.sum(xy**2, axis=1))))
    if scale < 1e-6:
        raise ValueError("Depth support has negligible spatial extent")
    q = xy / scale
    design = np.column_stack((2 * q, np.ones(len(q))))
    coefficients, _, rank, _ = np.linalg.lstsq(design, np.sum(q**2, axis=1), rcond=None)
    radius2 = float(coefficients[2] + coefficients[:2] @ coefficients[:2])
    if rank < 3 or radius2 <= 0:
        raise ValueError("Depth support does not define a circle")
    center = origin + scale * (coefficients[0] * u + coefficients[1] * v)
    return center, normal, scale * np.sqrt(radius2), float(np.linalg.cond(design))


def _three_point_circle(points):
    a, b, c = points
    ab, ac = b - a, c - a
    cross = np.cross(ab, ac)
    denom = 2 * float(cross @ cross)
    if denom < 1e-16:
        return None
    center = a + (
        (ab @ ab) * np.cross(ac, cross) + (ac @ ac) * np.cross(cross, ab)
    ) / denom
    return center, normal_positive_z(cross), float(np.linalg.norm(center - a))


def _ransac_depth(points):
    # Fixed seed makes rejection and the robust/least-squares ablation repeatable.
    rng = np.random.default_rng(20261021)
    best = None
    for _ in range(256):
        model = _three_point_circle(points[rng.choice(len(points), 3, replace=False)])
        if model is None or not MIN_RADIUS_M <= model[2] <= MAX_RADIUS_M:
            continue
        distance = _circle_distances(points, *model)
        inliers = distance <= DEPTH_INLIER_M
        count = int(np.sum(inliers))
        score = (count, -float(np.median(distance[inliers]))) if count else (0, -np.inf)
        if best is None or score > best[0]:
            best = score, inliers
    if best is None or best[0][0] < MIN_POINTS:
        raise ValueError("RANSAC found insufficient consistent circle support")
    inliers = best[1]
    # Two deterministic refinements remove points inconsistent with the refitted plane.
    for _ in range(2):
        center, normal, radius, _ = _plane_circle(points[inliers])
        proposed = _circle_distances(points, center, normal, radius) <= DEPTH_INLIER_M
        if np.sum(proposed) < MIN_POINTS:
            break
        inliers = proposed
    return inliers


def fit_depth(observations: list[Observation], robust: bool = False) -> Estimate:
    """Fit a 3D ring: PCA plane + plane-local LS circle, optionally RANSAC.

    Invalid/insufficient data and weak arc support produce explicit rejections.
    Residual is 3D orthogonal ring RMSE in metres, evaluated on selected inliers.
    A one-view RGB-D observation has metric scale and is allowed.
    """
    start = perf_counter()
    try:
        points = _depth_points(observations)
        if len(points) < MIN_POINTS:
            return _reject("insufficient_finite_depth_points", start, point_count=len(points))
        inliers = _ransac_depth(points) if robust else np.ones(len(points), dtype=bool)
        center, normal, radius, condition = _plane_circle(points[inliers])
    except (ValueError, np.linalg.LinAlgError) as error:
        return _reject("degenerate_depth_geometry", start, detail=str(error))
    distances = _circle_distances(points, center, normal, radius)
    rms = float(np.sqrt(np.mean(distances[inliers]**2)))
    fraction = float(np.mean(inliers))
    span = _arc_span(points[inliers], center, normal)
    tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1, 1))))
    reasons = []
    if not MIN_RADIUS_M <= radius <= MAX_RADIUS_M:
        reasons.append("radius_outside_prior")
    if tilt > MAX_TILT_DEG:
        reasons.append("normal_tilt_outside_prior")
    if fraction < 0.5:
        reasons.append("insufficient_depth_inlier_fraction")
    if span < MIN_ARC_DEG:
        reasons.append("insufficient_angular_support")
    if rms > DEPTH_INLIER_M:
        reasons.append("depth_ring_residual_too_large")
    if condition > 1e4:
        reasons.append("ill_conditioned_depth_circle")
    result = Estimate(
        center=center, normal=normal, radius=float(radius), accepted=not reasons,
        reasons=reasons, residual=rms, condition=condition,
        diagnostics={
            "method": "depth_ransac_plane_circle" if robust else "depth_pca_plane_circle",
            "point_count": len(points), "inlier_count": int(np.sum(inliers)),
            "inlier_fraction": fraction, "angular_support_deg": span,
            "normal_tilt_deg": tilt, "residual_units": "m",
            "inlier_threshold_m": DEPTH_INLIER_M,
            "uncertainty_scope": "No calibrated uncertainty is returned by the depth baseline",
        },
    )
    return _finish(result, start)


def fit_depth_plane_rays(observations: list[Observation]) -> Estimate:
    """Estimate a depth plane, intersect RGB boundary rays, then fit its circle.

    This diagnostic assumes observed depth support belongs to one local plane.
    It removes per-boundary depth jitter/mixing only when that plane remains
    accurate; a common depth-plane offset is retained, not corrected.  The RGB
    and depth observations are correlated, so no independent-probability claim
    or calibrated covariance is made.
    """
    start = perf_counter()
    try:
        depth = _depth_points(observations)
        if len(depth) < MIN_POINTS:
            return _reject("insufficient_finite_depth_points", start, point_count=len(depth))
        origin = np.mean(depth, axis=0)
        _, singular, vh = np.linalg.svd(depth - origin, full_matrices=False)
        if len(singular) != 3 or singular[1] < 1e-7 or singular[1] / singular[0] < 1e-3:
            return _reject("degenerate_depth_plane", start)
        normal = normal_positive_z(vh[-1])
        plane_rms = float(np.sqrt(np.mean(((depth - origin) @ normal)**2)))
        projected = []
        counts = {}
        total = 0
        for observation in observations:
            pixels = np.asarray(observation.pixels, dtype=float)
            if pixels.ndim != 2 or pixels.shape[1] != 2:
                raise ValueError("Image points must have shape (N, 2)")
            pixels = pixels[np.all(np.isfinite(pixels), axis=1)]
            total += len(pixels)
            camera = observation.camera
            rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(camera.K).T @ camera.T_world_from_camera[:3, :3].T
            camera_origin = camera.T_world_from_camera[:3, 3]
            denominator = rays @ normal
            valid = np.abs(denominator) / np.maximum(np.linalg.norm(rays, axis=1), 1e-12) >= np.sin(np.deg2rad(2))
            distances = np.zeros(len(rays))
            distances[valid] = ((origin - camera_origin) @ normal) / denominator[valid]
            valid &= distances > 0
            reconstructed = camera_origin + distances[valid, None] * rays[valid]
            counts[camera.name] = int(np.sum(valid))
            if len(reconstructed):
                projected.append(Observation(camera, pixels[valid], reconstructed))
        valid_count = sum(counts.values())
        if valid_count < MIN_POINTS:
            return _reject("insufficient_forward_plane_ray_intersections", start, valid_ray_counts=counts)
        result = fit_depth(projected, robust=False)
        if plane_rms > DEPTH_INLIER_M:
            result.reasons.append("depth_support_not_planar")
        if valid_count < .55 * total:
            result.reasons.append("insufficient_plane_ray_fraction")
        result.accepted = not result.reasons
        result.diagnostics.update({
            "method": "depth_plane_rgb_rays_circle",
            "depth_plane_rms_m": plane_rms,
            "depth_plane_origin_m": origin.tolist(),
            "depth_plane_normal": normal.tolist(),
            "valid_ray_counts": counts,
            "valid_ray_fraction": valid_count / max(total, 1),
            "observation_scope": "Observed depth plane and RGB boundary rays; correlated inputs; assumes a planar target feature",
            "uncertainty_scope": "No calibrated covariance; common depth-plane bias remains",
        })
        return _finish(result, start)
    except (ValueError, np.linalg.LinAlgError) as error:
        return _reject("invalid_depth_plane_or_rays", start, detail=str(error))


def _rim_parameters(parameters):
    return parameters[:3], normal_positive_z(np.r_[parameters[3:5], 1.0]), parameters[5]


def _pixel_residuals(parameters, prepared):
    center, normal, radius = _rim_parameters(parameters)
    residuals = []
    for camera, homogeneous, camera_from_world in prepared:
        rotation = camera_from_world[:3, :3]
        center_camera = rotation @ center + camera_from_world[:3, 3]
        normal_camera = rotation @ normal
        distance = normal_camera @ center_camera
        # Intersect a ray x with n.X=d and impose ||X-c||^2=r^2:
        # x.T [d^2 I-d(c n.T+n c.T)+(c.T c-r^2)n n.T] x = 0.
        # This expression avoids an arbitrary in-plane basis (and its branch
        # changes), and avoids inverting a nearly singular plane homography.
        cone = distance**2 * np.eye(3) - distance * (
            np.outer(center_camera, normal_camera) + np.outer(normal_camera, center_camera)
        ) + (center_camera @ center_camera - radius**2) * np.outer(normal_camera, normal_camera)
        inverse_intrinsics = np.linalg.inv(camera.K)
        conic = inverse_intrinsics.T @ cone @ inverse_intrinsics
        conic /= max(float(np.linalg.norm(conic)), 1e-30)
        values = homogeneous @ conic
        denominator = np.maximum(2 * np.linalg.norm(values[:, :2], axis=1), 1e-14)
        errors = np.sum(homogeneous * values, axis=1) / denominator
        if center_camera[2] <= 0:
            errors = errors + 1e4
        residuals.append(errors)
    return residuals


def fit_multiview(
    observations: list[Observation], robust: bool = True, equal_view_weight: bool = True,
    depth_sigma_m: float | None = None,
) -> Estimate:
    """Fit a metric 3D circle to >=2 calibrated views, with depth-only initial pose.

    Input pixels are one physical ring boundary in every view, not arbitrary
    silhouette correspondences.  Depth initializes the optimizer but is absent
    from its RGB objective unless depth_sigma_m enables the explicit fusion
    ablation.  That scale is an assumed weight, not a calibrated independent
    likelihood: boundary RGB and depth can be correlated.  Robust fitting uses per-pixel pseudo-Huber loss
    before view balancing, so duplicating one view cannot change its loss scale.
    Residual is inlier pixel RMSE.  Covariance is a local conditional estimate.
    """
    start = perf_counter()
    if depth_sigma_m is not None and (not np.isfinite(depth_sigma_m) or depth_sigma_m <= 0):
        return _reject("invalid_depth_fusion_scale", start)
    if len(observations) < 2:
        return _reject("monocular_rgb_scale_ambiguity", start)
    prepared = []
    try:
        for observation in observations:
            pixels = np.asarray(observation.pixels, float)
            if pixels.ndim != 2 or pixels.shape[1] != 2:
                raise ValueError("Image points must have shape (N, 2)")
            pixels = pixels[np.all(np.isfinite(pixels), axis=1)]
            if len(pixels) < MIN_POINTS:
                return _reject("insufficient_finite_image_points", start, camera=observation.camera.name)
            camera = observation.camera
            if np.asarray(camera.K).shape != (3, 3) or not np.all(np.isfinite(camera.K)):
                raise ValueError("Invalid camera intrinsics")
            transform = np.linalg.inv(camera.T_world_from_camera)
            if not np.all(np.isfinite(transform)):
                raise ValueError("Invalid camera extrinsics")
            prepared.append((camera, np.column_stack((pixels, np.ones(len(pixels)))), transform))
        if len({entry[0].name for entry in prepared}) < 2:
            return _reject("insufficient_distinct_cameras", start)
        points = _depth_points(observations)
        if len(points) < MIN_POINTS:
            return _reject("depth_initialization_unavailable", start)
        # The robust initializer may reject due to its broader 3D fit quality;
        # its geometric seed remains useful, because RGB is the final objective.
        initial_depth = fit_depth(observations, robust=True)
        if initial_depth.center is None:
            initial_depth = fit_depth(observations, robust=False)
        if initial_depth.center is None:
            return _reject("depth_initialization_degenerate", start)
        center = initial_depth.center
        normal = initial_depth.normal
        radius = np.clip(initial_depth.radius, MIN_RADIUS_M + 1e-5, MAX_RADIUS_M - 1e-5)
    except (ValueError, np.linalg.LinAlgError) as error:
        return _reject("invalid_observation_or_calibration", start, detail=str(error))

    # Same-origin cameras, or virtually coincident viewing directions, do not
    # provide reliable triangulation geometry.  This is a conservative gate.
    pair_angles = []
    pair_baselines = []
    for i, (camera_a, _, _) in enumerate(prepared):
        for camera_b, _, _ in prepared[i + 1:]:
            a, b = camera_a.T_world_from_camera[:3, 3], camera_b.T_world_from_camera[:3, 3]
            ra, rb = center - a, center - b
            if np.linalg.norm(ra) < 1e-8 or np.linalg.norm(rb) < 1e-8:
                continue
            pair_angles.append(float(np.degrees(np.arccos(np.clip(ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb)), -1, 1)))))
            pair_baselines.append(float(np.linalg.norm(a - b)))
    usable_pairs = [angle for angle, baseline in zip(pair_angles, pair_baselines) if 3 <= angle <= 177 and baseline >= 0.005]
    if not usable_pairs:
        return _reject("degenerate_multiview_geometry", start, pair_ray_angles_deg=pair_angles, pair_baselines_m=pair_baselines)

    slope_limit = float(np.tan(np.deg2rad(MAX_TILT_DEG)))
    lower = np.r_[center - .20, -slope_limit, -slope_limit, MIN_RADIUS_M]
    upper = np.r_[center + .20, slope_limit, slope_limit, MAX_RADIUS_M]
    weights = [np.sqrt(64 / len(entry[1])) if equal_view_weight else 1.0 for entry in prepared]
    depth_chunks = []
    if depth_sigma_m is not None:
        for observation in observations:
            if observation.points_world is None:
                continue
            local = np.asarray(observation.points_world, dtype=float)
            local = local[np.all(np.isfinite(local), axis=1)]
            if len(local):
                depth_chunks.append((observation.camera.name, local))
    depth_weights = [np.sqrt(64 / len(chunk)) if equal_view_weight else 1.0 for _, chunk in depth_chunks]

    def robust_residual(errors):
        return errors * np.sqrt(2 / (np.sqrt(1 + errors**2) + 1)) if robust else errors

    def objective(parameters):
        chunks = _pixel_residuals(parameters, prepared)
        # Per-point pseudo-Huber is applied before per-view balancing.
        residuals = [weight * robust_residual(errors) for weight, errors in zip(weights, chunks)]
        if depth_sigma_m is not None:
            geometry = _rim_parameters(parameters)
            residuals.extend(weight * robust_residual(_circle_distances(chunk, *geometry) / depth_sigma_m)
                             for weight, (_, chunk) in zip(depth_weights, depth_chunks))
        return np.concatenate(residuals)

    slopes = normal[:2] / max(normal[2], .2)
    starts = (slopes, np.zeros(2), -slopes + np.array([.08, -.08]))
    candidates = []
    for initial_slope in starts:
        parameters = np.r_[center, initial_slope, radius]
        parameters = np.clip(parameters, lower + 1e-8, upper - 1e-8)
        try:
            fitted = least_squares(objective, parameters, bounds=(lower, upper), x_scale="jac", max_nfev=250,
                                   method="dogbox",
                                   jac="3-point", diff_step=1e-4,
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9)
            if fitted.success and np.all(np.isfinite(fitted.x)):
                candidates.append(fitted)
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not candidates:
        return _reject("multiview_solver_did_not_converge", start)
    fitted = min(candidates, key=lambda item: item.cost)
    center, normal, radius = _rim_parameters(fitted.x)
    raw = _pixel_residuals(fitted.x, prepared)
    inliers = [np.abs(errors) <= PIXEL_INLIER_PX for errors in raw]
    kept = [errors[mask] for errors, mask in zip(raw, inliers) if np.any(mask)]
    rms = float(np.sqrt(np.mean(np.concatenate(kept)**2))) if kept else float("inf")
    column_norm = np.maximum(np.linalg.norm(fitted.jac, axis=0), 1e-12)
    condition = float(np.linalg.cond(fitted.jac / column_norm))
    variance = float(np.sum(fitted.fun**2) / max(1, len(fitted.fun) - len(fitted.x)))
    covariance = np.linalg.pinv(fitted.jac.T @ fitted.jac) * variance
    tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1, 1))))
    reasons = []
    if np.any(np.minimum(fitted.x - lower, upper - fitted.x) / (upper - lower) < 1e-4):
        reasons.append("solution_at_parameter_bound")
    if tilt > MAX_TILT_DEG:
        reasons.append("normal_tilt_outside_prior")
    if not np.isfinite(condition) or condition > 1e5:
        reasons.append("ill_conditioned_multiview_circle")
    if rms > 1.5:
        reasons.append("multiview_pixel_residual_too_large")
    camera_diagnostics = {}
    all_ring_points = []
    for (camera, homogeneous, transform), errors, mask in zip(prepared, raw, inliers):
        count = int(np.sum(mask))
        fraction = float(np.mean(mask))
        if count < MIN_POINTS or fraction < .55:
            reasons.append(f"{camera.name}:insufficient_image_inlier_support")
        # Back-project *observed* inlier rays onto the estimated ring plane for
        # an angular-support check; no cross-view pixel correspondences are used.
        rays = homogeneous[mask] @ np.linalg.inv(camera.K).T @ camera.T_world_from_camera[:3, :3].T
        origin = camera.T_world_from_camera[:3, 3]
        denominator = rays @ normal
        valid = np.abs(denominator) > 1e-7
        distances = np.zeros(len(rays))
        distances[valid] = ((center - origin) @ normal) / denominator[valid]
        valid &= distances > 0
        ring_points = origin + distances[valid, None] * rays[valid]
        if len(ring_points):
            all_ring_points.append(ring_points)
        center_camera = transform[:3, :3] @ center + transform[:3, 3]
        normal_camera = transform[:3, :3] @ normal
        if center_camera[2] <= radius * np.sqrt(max(0, 1 - normal_camera[2]**2)):
            reasons.append(f"{camera.name}:ring_not_fully_in_front_of_camera")
        camera_diagnostics[camera.name] = {
            "point_count": len(errors), "inlier_count": count, "inlier_fraction": fraction,
            "inlier_rmse_px": float(np.sqrt(np.mean(errors[mask]**2))) if count else None,
            "raw_rmse_px": float(np.sqrt(np.mean(errors**2))),
            "raw_p95_abs_px": float(np.percentile(np.abs(errors), 95)),
        }
    span = _arc_span(np.concatenate(all_ring_points), center, normal) if all_ring_points else 0.
    if span < MIN_ARC_DEG:
        reasons.append("insufficient_angular_support")
    result = Estimate(
        center=center, normal=normal, radius=float(radius), accepted=not reasons,
        reasons=reasons, residual=rms, covariance_center=covariance[:3, :3], condition=condition,
        diagnostics={
            "method": "multiview_circle_conic" if depth_sigma_m is None else "multiview_depth_fused_circle",
            "robust": robust,
            "equal_view_weight": equal_view_weight, "view_count": len(prepared),
            "cameras": camera_diagnostics, "angular_support_deg": span,
            "pair_ray_angles_deg": pair_angles, "normal_tilt_deg": tilt,
            "residual_units": "px", "inlier_threshold_px": PIXEL_INLIER_PX,
            "initialization": "observed_depth_circle_only; depth absent from RGB objective" if depth_sigma_m is None else "observed_depth_circle; explicit 3D ring-distance fusion enabled",
            "uncertainty_scope": "Local Jacobian covariance; fixed camera calibration and boundary labels; excludes calibration bias, shape mismatch, and selection uncertainty",
            "solver_cost": float(fitted.cost), "solver_evaluations": int(fitted.nfev),
            "solver_method": "dogbox; three slope initializations; 3-point finite differences",
        },
    )
    if depth_sigma_m is not None:
        result.diagnostics.update({
            "depth_sigma_m": depth_sigma_m,
            "depth_fusion_scope": "Assumed relative weight against 1 px RGB residual scale; RGB/depth boundaries may be correlated; not an independent probabilistic likelihood",
            "depth_cameras": {
                name: {"point_count": len(chunk), "ring_rmse_m": float(np.sqrt(np.mean(_circle_distances(chunk, center, normal, radius)**2)))}
                for name, chunk in depth_chunks
            },
            "uncertainty_scope": "Local weighted Jacobian diagnostic only; correlated RGB/depth inputs and assumed depth scale; excludes calibration bias and shape mismatch; not calibrated probability",
        })
    return _finish(result, start)
