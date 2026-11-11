"""Separable local axis/spline fitting from observations only.

Uses variable projection (O'Leary & Rust): eliminate linear profile
coefficients at EVERY nonlinear axis trial. A local adaptation, not a full
reproduction of Lai/Ueng's general meridian reconstruction algorithm.
All calculations inside the fit use mm; output geometry uses metres.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import qr, solve_triangular
from scipy.optimize import least_squares
from scipy.spatial import QhullError

from .depth_profile import (ProfileSettings, ProfileRejection, rgb_foreground,
                            _mouth_plane, _mouth_center)
from .geometry import Estimate, basis


@dataclass(frozen=True)
class JointSettings:
    fit_depth_mm: float = 28.
    exclude_top_mm: float = 3.
    local_half_span_mm: float = 8.5
    local_context_mm: float = 2.
    facing_cosine_min: float = .15
    max_fit_points_per_camera: int = 2500
    knot_spacing_mm: float = 1.5
    smoothing_mm3: float = .0001
    axis_shift_bound_mm: float = 20.
    normal_tangent_bound: float = .24
    multistart_tangent: float = .06
    max_nfev: int = 70
    maximum_radial_rms_mm: float = .40
    near_min_delta_mm: float = 1.
    support_bin_mm: float = .5
    maximum_target_gap_mm: float = 2.5
    endpoint_support_mm: float = 1.5
    maximum_condition: float = 10000.


class AxisSplineProblem:
    """Exact QR variable projection for a fixed weighted radial LS objective.

    Origin has fixed world Z (gauge); nonlinear variables are dx,dy,eta1,eta2.
    Curve roughness is a fixed second-difference approximation. No point
    normals, sensor-noise covariance, target truth or nominal jar size used.
    """
    def __init__(self, points_mm, origin_mm, normal, camera_ids, settings=JointSettings()):
        self.points, self.origin = np.asarray(points_mm), np.asarray(origin_mm)
        self.n0 = np.asarray(normal)/np.linalg.norm(normal)
        self.U = np.array(basis(self.n0)).T
        self.settings = settings
        initial_s = (self.points-self.origin) @ self.n0
        h = settings.knot_spacing_mm
        lo, hi = np.floor((initial_s.min()-15)/h)*h, np.ceil((initial_s.max()+15)/h)*h
        knots = np.r_[np.repeat(lo, 4), np.arange(lo+h, hi, h), np.repeat(hi, 4)]
        k = len(knots)-4
        self.basis = BSpline(knots, np.eye(k), 3, extrapolate=True)
        self.derivative = self.basis.derivative()
        # Equal total weight per view: engineering balancing, not a noise model.
        camera_ids = np.asarray(camera_ids)
        counts = np.bincount(camera_ids)
        weights = 1/counts[camera_ids]
        weights *= len(weights)/weights.sum()
        self.sqrt_weight = np.sqrt(weights)
        self.regularizer = np.diff(np.eye(k), n=2, axis=0)*np.sqrt(
            len(weights)*settings.smoothing_mm3/h**3)
        self._last = None

    def axis(self, x):
        a = self.origin+np.r_[x[:2], 0.]
        raw_n = self.n0+self.U @ x[2:]
        norm = np.linalg.norm(raw_n)
        n = raw_n/norm
        dn = (np.eye(3)-np.outer(n, n)) @ self.U/norm
        return a, n, dn

    def evaluate(self, x):
        if self._last is not None and np.array_equal(x, self._last[0]):
            return self._last[1]
        a, n, dn = self.axis(x)
        delta = self.points-a
        s = delta @ n
        radial = delta-s[:, None]*n
        rho = np.linalg.norm(radial, axis=1)
        if np.min(rho) < 1e-5:
            raise ProfileRejection("axis_intersects_selected_points")
        e = radial/rho[:, None]
        B, Bprime = self.basis(s), self.derivative(s)
        sw = self.sqrt_weight
        A = np.vstack((sw[:, None]*B, self.regularizer))
        b = np.r_[sw*rho, np.zeros(len(self.regularizer))]
        Q, R = qr(A, mode="economic", check_finite=False)
        if np.min(np.abs(np.diag(R))) < 1e-9:
            raise ProfileRejection("rank_deficient_profile_basis")
        beta = solve_triangular(R, Q.T @ b, check_finite=False)
        residual = b-A @ beta
        ds = np.c_[-np.tile(n[:2], (len(s), 1)), delta @ dn]
        drho = np.c_[-e[:, :2], -s[:, None]*(e @ dn)]
        raw_jacobian = sw[:, None]*(drho-(Bprime @ beta)[:, None]*ds)
        full_jacobian = np.vstack((raw_jacobian, np.zeros((len(self.regularizer), 4))))
        # Both Golub/Pereyra terms, including variable right-hand-side rho(x).
        second = Bprime.T @ ((sw*residual[:len(s)])[:, None]*ds)
        jacobian = (full_jacobian-Q @ (Q.T @ full_jacobian)
                    -Q @ solve_triangular(R.T, second, lower=True, check_finite=False))
        result = dict(residual=residual, jacobian=jacobian, beta=beta,
                      radial_residual=rho-B @ beta, s=s, origin=a, normal=n,
                      data_profile_basis=sw[:, None]*B, data_axis_jacobian=raw_jacobian,
                      objective=float(residual @ residual), regularizer=self.regularizer)
        self._last = (np.array(x).copy(), result)
        return result

    def fun(self, x):
        return self.evaluate(x)["residual"]

    def jac(self, x):
        return self.evaluate(x)["jacobian"]


def observed_initialization(frames, settings, seed):
    clouds, ids, counts = [], [], {}
    station = ProfileSettings()
    for index, frame in enumerate(frames):
        valid = rgb_foreground(frame.rgb) & np.isfinite(frame.depth_z_m) & (frame.depth_z_m > 0)
        yy, xx = np.nonzero(valid)
        p = frame.camera.unproject(np.c_[xx, yy], frame.depth_z_m[yy, xx])
        keep = ((p[:, 2] >= station.world_z_bounds_m[0]) &
                (p[:, 2] <= station.world_z_bounds_m[1]) &
                (np.linalg.norm(p[:, :2], axis=1) < station.world_xy_radius_m))
        p = p[keep]
        clouds.append(p)
        ids.append(np.full(len(p), index))
        counts[frame.camera.name] = len(p)
    cloud, source = np.concatenate(clouds), np.concatenate(ids)
    if len(cloud) < 300:
        raise ProfileRejection("insufficient_surface_points_in_fixed_station_volume")
    plane_origin, normal, inliers = _mouth_plane(cloud, station, np.random.default_rng(seed))
    center, mouth_radius = _mouth_center(plane_origin, normal, inliers)
    d = cloud-center
    s = d @ normal
    radial = d-s[:, None]*normal
    rho = np.linalg.norm(radial, axis=1)
    camera_positions = np.array([f.camera.T_world_from_camera[:3, 3] for f in frames])
    view = camera_positions[source]-cloud
    view -= (view @ normal)[:, None]*normal
    cosine = np.sum(radial*view, axis=1)/np.maximum(rho*np.linalg.norm(view, axis=1), 1e-12)
    keep = ((s >= -settings.fit_depth_mm/1000) & (s <= -settings.exclude_top_mm/1000)
            & (cosine > settings.facing_cosine_min))
    selected, selected_ids = cloud[keep], source[keep]
    if len(selected) < 200:
        raise ProfileRejection("insufficient_camera_facing_local_outer_candidates")
    # Locate a coarse minimum FROM OBSERVATIONS, then avoid forcing the lower
    # shoulder/collar transition onto the same local revolution axis.
    selected_s, selected_rho = s[keep]*1000, rho[keep]*1000
    bins = np.floor(selected_s/settings.support_bin_mm).astype(int)
    occupied, count = np.unique(bins, return_counts=True)
    occupied = occupied[count >= 8]
    if len(occupied) < 8:
        raise ProfileRejection("insufficient_coarse_local_profile_support")
    medians = np.array([np.median(selected_rho[bins == b]) for b in occupied])
    minimum = int(np.argmin(medians))
    left = right = minimum
    threshold = medians[minimum]+settings.near_min_delta_mm
    while (left > 0 and medians[left-1] <= threshold
           and (occupied[left]-occupied[left-1])*settings.support_bin_mm <= settings.maximum_target_gap_mm):
        left -= 1
    while (right+1 < len(occupied) and medians[right+1] <= threshold
           and (occupied[right+1]-occupied[right])*settings.support_bin_mm <= settings.maximum_target_gap_mm):
        right += 1
    low, high = occupied[left]*settings.support_bin_mm, (occupied[right]+1)*settings.support_bin_mm
    coarse_center = (low+high)/2
    half_span = max(settings.local_half_span_mm, (high-low)/2+settings.local_context_mm)
    local = np.abs(selected_s-coarse_center) <= half_span
    selection_info = dict(coarse_minimum_from_initial_axis_m=coarse_center/1000,
        coarse_near_minimum_interval_m=[low/1000, high/1000],
        local_selection_interval_from_initial_axis_m=[(coarse_center-half_span)/1000,
                                                      (coarse_center+half_span)/1000],
        broad_outer_candidates=len(selected))
    selected, selected_ids = selected[local], selected_ids[local]
    if len(selected) < 200:
        raise ProfileRejection("insufficient_local_neck_support")
    fit_indices = []
    for index in range(len(frames)):
        indices = np.flatnonzero(selected_ids == index)
        if len(indices):
            fit_indices.extend(indices[np.linspace(0, len(indices)-1,
                min(len(indices), settings.max_fit_points_per_camera)).astype(int)])
    return selected, selected_ids, np.asarray(fit_indices), center, normal, mouth_radius, counts, selection_info


def supported_interval(problem, solution, all_points_mm, settings):
    a, n = solution["origin"], solution["normal"]
    s = (all_points_mm-a) @ n
    grid = np.arange(s.min()+.25, s.max()-.25, .05)
    radius = problem.basis(grid) @ solution["beta"]
    if not len(grid) or not np.all(np.isfinite(radius)):
        raise ProfileRejection("invalid_profile")
    index = int(np.argmin(radius))
    threshold = radius[index]+settings.near_min_delta_mm
    selected = radius <= threshold
    left, right = index, index
    while left > 0 and selected[left-1]:
        left -= 1
    while right+1 < len(grid) and selected[right+1]:
        right += 1
    if left == 0 or right == len(grid)-1:
        raise ProfileRejection("near_minimum_interval_touches_observed_search_boundary")
    low = float(np.interp(threshold, [radius[left], radius[left-1]], [grid[left], grid[left-1]]))
    high = float(np.interp(threshold, [radius[right], radius[right+1]], [grid[right], grid[right+1]]))
    support = np.sort(s[(s >= low-settings.endpoint_support_mm) & (s <= high+settings.endpoint_support_mm)])
    if len(support) < 40 or np.max(np.diff(support)) > settings.maximum_target_gap_mm:
        raise ProfileRejection("unsupported_target_interval")
    if max(np.min(np.abs(s-low)), np.min(np.abs(s-high))) > settings.endpoint_support_mm:
        raise ProfileRejection("unsupported_interval_endpoint")
    mid = (low+high)/2
    return mid, float(problem.basis(mid) @ solution["beta"]), {
        "near_min_interval_from_axis_origin_m": [low/1000, high/1000],
        "profile_s_m": (grid/1000).tolist(), "profile_radius_m": (radius/1000).tolist(),
        "axis_origin_m": (a/1000).tolist(), "near_min_interval_length_m": (high-low)/1000}


def estimate_joint_profile(frames, *, free_axis=True, seed=4101, settings=JointSettings()):
    start = perf_counter()
    estimate = Estimate(diagnostics={"method": "joint_axis_spline" if free_axis else "spline_fixed_axis",
        "solver": "QR variable projection; analytic reduced Jacobian with both terms",
        "residual_semantics": "weighted radial mm; not exact orthogonal distance; no noise calibration",
        "uncertainty": "not_estimated", "outer_gate": "camera-facing radial side under observed initial axis; no GT"})
    try:
        selected, ids, indices, center, normal, lip_radius, counts, selection_info = observed_initialization(frames, settings, seed)
        estimate.diagnostics.update(selection_info)
        problem = AxisSplineProblem(selected[indices]*1000, center*1000, normal, ids[indices], settings)
        scale = np.array([5., 5., .1, .1])
        bounds = np.array([settings.axis_shift_bound_mm]*2+[settings.normal_tangent_bound]*2)
        starts = [np.zeros(4)]
        if free_axis:
            for k in (2, 3):
                for sign in (-1, 1):
                    x = np.zeros(4); x[k] = sign*settings.multistart_tangent
                    starts.append(x)
        candidates = []
        for x in starts:
            if free_axis:
                fit = least_squares(problem.fun, x, jac=problem.jac, bounds=(-bounds, bounds),
                                    x_scale=scale, max_nfev=settings.max_nfev,
                                    ftol=1e-8, xtol=1e-8, gtol=1e-7)
                x, converged = fit.x, bool(fit.success)
            else:
                converged = True
            state = problem.evaluate(x)
            candidates.append((state["objective"], x.copy(), state, converged))
        candidates.sort(key=lambda item: item[0])
        objective, x, solution, converged = candidates[0]
        initial = problem.evaluate(np.zeros(4))
        rms = float(np.sqrt(np.mean(solution["radial_residual"]**2)))
        # Remove profile nuisance directions using DATA only. The smoothness
        # penalty must not manufacture apparent measurement observability.
        Qdata, values, _ = np.linalg.svd(solution["data_profile_basis"], full_matrices=False)
        rank = int(np.count_nonzero(values > values[0]*1e-8))
        Qdata = Qdata[:, :rank]
        Jdata = solution["data_axis_jacobian"]
        effective = (Jdata-Qdata @ (Qdata.T @ Jdata))*scale
        singular = np.linalg.svd(effective, compute_uv=False)
        estimate.condition = float(singular[0]/max(singular[-1], 1e-15))
        estimate.residual = rms/1000
        estimate.diagnostics.update(point_counts=counts, selected_outer_points=len(selected), fit_points=len(indices),
            initial_mouth_center_m=center.tolist(), initial_mouth_normal=normal.tolist(),
            initial_objective_mm2=initial["objective"], final_objective_mm2=objective,
            condition_semantics="data-only local axis tangent after projecting out profile; scaled [5mm,5mm,.1rad,.1rad]; no covariance",
            effective_axis_singular_values=singular.tolist(),
            axis_parameters=x.tolist(), optimizer_converged=converged,
            multistart=[{"objective_mm2": c[0], "axis_parameters": c[1].tolist(), "converged": c[3]} for c in candidates])
        mid, radius, profile = supported_interval(problem, solution, selected*1000, settings)
        estimate.center = (solution["origin"]+mid*solution["normal"])/1000
        estimate.normal, estimate.radius = solution["normal"], radius/1000
        estimate.diagnostics.update(profile)
        comparable = [candidate for candidate in candidates if candidate[0] <= objective*1.01+1e-8]
        spreads = []
        for _, _, state, _ in comparable:
            try:
                other_mid, _, _ = supported_interval(problem, state, selected*1000, settings)
                other_center = state["origin"]+other_mid*state["normal"]
                spreads.append((float(np.linalg.norm(other_center-estimate.center*1000)),
                                float(np.rad2deg(np.arccos(np.clip(state["normal"] @ estimate.normal, -1, 1))))))
            except ProfileRejection:
                continue
        estimate.diagnostics["near_optimal_start_spread_center_mm"] = max((v[0] for v in spreads), default=0.)
        estimate.diagnostics["near_optimal_start_spread_normal_deg"] = max((v[1] for v in spreads), default=0.)
        if spreads and (max(v[0] for v in spreads) > 2 or max(v[1] for v in spreads) > 2):
            estimate.reasons.append("ambiguous_near_optimal_axes")
        if not converged:
            estimate.reasons.append("optimizer_not_converged")
        if free_axis and np.any(np.abs(x) > .98*bounds):
            estimate.reasons.append("axis_parameter_hits_bound")
        if free_axis and estimate.condition > settings.maximum_condition:
            estimate.reasons.append("poor_local_axis_condition")
        if rms > settings.maximum_radial_rms_mm:
            estimate.reasons.append("local_surface_model_mismatch")
        if estimate.normal[2] < np.cos(np.deg2rad(30)):
            estimate.reasons.append("normal_outside_near_upward_working_range")
        if not .02 < estimate.radius < lip_radius-.001:
            estimate.reasons.append("candidate_not_narrower_than_observed_raised_lip")
        estimate.accepted = not estimate.reasons
    except (ProfileRejection, np.linalg.LinAlgError, QhullError) as error:
        estimate.reasons.append(str(error))
    estimate.elapsed_ms = (perf_counter()-start)*1000
    return estimate
