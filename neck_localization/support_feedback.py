"""Observation-only ablations of camera weights, Huber IRLS and axial support.

The original joint_profile module stays frozen. All fits use the SAME initial
candidate superset. A separate common fixed-axis full-context readout preserves
the evidence for both functional-range endpoints. No target/model metadata read.
"""
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import QhullError

from .depth_profile import ProfileRejection
from .geometry import Estimate
from .joint_profile import (AxisSplineProblem, JointSettings, observed_initialization,
                            supported_interval)


@dataclass(frozen=True)
class FeedbackSettings:
    huber_delta_mm: float = .10  # ideal-depth engineering threshold, not sensor sigma
    maximum_outer_iterations: int = 12
    core_radius_relief_mm: float = .30
    region_taper_mm: float = .50
    region_weight_floor: float = .001
    minimum_core_span_mm: float = 3.
    minimum_core_points: int = 80
    minimum_angular_coverage_deg: float = 90.
    weight_tolerance: float = .015
    axis_shift_tolerance_mm: float = .005
    axis_angle_tolerance_deg: float = .002


def camera_weights(camera_ids, mode, trust=None):
    ids = np.asarray(camera_ids)
    trust = np.ones(int(ids.max())+1) if trust is None else np.asarray(trust, float)
    if np.any(~np.isfinite(trust)) or np.any(trust <= 0):
        raise ValueError('Camera trust must be finite and positive')
    weights = trust[ids]
    if mode == 'balanced':
        weights = weights/np.bincount(ids)[ids]
    elif mode != 'equal_trust':
        raise ValueError(mode)
    return weights/weights.mean()  # both camera rules have total weight N


class WeightedAxisSplineProblem(AxisSplineProblem):
    """Frozen analytic variable projection with explicit fixed data weights."""
    def set_weights(self, weights):
        weights = np.asarray(weights, float)
        if weights.shape != (len(self.points),) or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError('Positive per-point weights required')
        self.sqrt_weight = np.sqrt(weights)
        self._last = None


def huber_weights(residual, delta):
    return np.minimum(1., delta/np.maximum(np.abs(residual), 1e-12))


def huber_cost(residual, delta):
    absolute = np.abs(residual)
    return np.where(absolute <= delta, residual**2, 2*delta*absolute-delta**2)


def region_weights(problem, x, settings, joint_settings):
    """Connected low-radius core from observed bin medians under CURRENT axis.

    Context points are softly downweighted, not deleted. The core is for AXIS
    fitting only; it is deliberately narrower than the +1 mm functional range.
    """
    a, n, _ = problem.axis(x)
    delta = problem.points-a
    s = delta @ n
    radial = delta-s[:, None]*n
    rho = np.linalg.norm(radial, axis=1)
    h = joint_settings.support_bin_mm
    bins = np.floor(s/h).astype(int)
    occupied, counts = np.unique(bins, return_counts=True)
    occupied = occupied[counts >= 8]
    if len(occupied) < 8:
        raise ProfileRejection('insufficient_bins_for_region_feedback')
    medians = np.array([np.median(rho[bins == b]) for b in occupied])
    minimum = int(np.argmin(medians))
    left = right = minimum
    threshold = medians[minimum]+settings.core_radius_relief_mm
    while left > 0 and medians[left-1] <= threshold and occupied[left]-occupied[left-1] <= 2:
        left -= 1
    while right+1 < len(occupied) and medians[right+1] <= threshold and occupied[right+1]-occupied[right] <= 2:
        right += 1
    low, high = occupied[left]*h, (occupied[right]+1)*h
    if left == 0 or right == len(occupied)-1:
        raise ProfileRejection('axis_core_touches_candidate_boundary')
    core = (s >= low) & (s <= high)
    if high-low < settings.minimum_core_span_mm or core.sum() < settings.minimum_core_points:
        raise ProfileRejection('axis_core_has_insufficient_extent')
    # Azimuth in current-axis tangent basis, not the initial-axis tangent basis.
    from .geometry import basis
    u, v = basis(n)
    angles = np.sort(np.mod(np.arctan2(radial[core] @ v, radial[core] @ u), 2*np.pi))
    gaps = np.diff(np.r_[angles, angles[0]+2*np.pi])
    coverage = float(np.degrees(2*np.pi-gaps.max()))
    if coverage < settings.minimum_angular_coverage_deg:
        raise ProfileRejection('axis_core_has_insufficient_angular_coverage')
    distance = np.maximum(np.maximum(low-s, s-high), 0.)
    weights = settings.region_weight_floor+(1-settings.region_weight_floor)*np.exp(
        -.5*(distance/settings.region_taper_mm)**2)
    return weights, dict(core_interval_mm=[float(low), float(high)], core_points=int(core.sum()),
                         angular_coverage_deg=coverage)


def _solve(problem, x, settings):
    bounds = np.array([settings.axis_shift_bound_mm]*2+[settings.normal_tangent_bound]*2)
    return least_squares(problem.fun, x, jac=problem.jac, bounds=(-bounds, bounds),
        x_scale=[5, 5, .1, .1], max_nfev=settings.max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-7)


def estimate_support_feedback(frames, *, mode='fixed', camera_mode='equal_trust',
                              settings=FeedbackSettings(), joint_settings=JointSettings(), seed=4101):
    started = perf_counter()
    name = f'{mode}_{camera_mode}'
    estimate = Estimate(diagnostics=dict(method=name, uncertainty='not_estimated',
        candidate_policy='same frozen observation-selected superset for all ablations',
        residual_semantics='full-context unweighted radial RMS in metres; not uncertainty',
        camera_rule=camera_mode, camera_trust=[1.]*len(frames),
        feedback_settings=asdict(settings)))
    try:
        if mode not in ('fixed', 'huber', 'region'):
            raise ValueError(mode)
        selected, ids, indices, center, normal, lip_radius, counts, selection = observed_initialization(
            frames, joint_settings, seed)
        points, sources = selected[indices]*1000, ids[indices]
        base = camera_weights(sources, camera_mode)
        problem = WeightedAxisSplineProblem(points, center*1000, normal, sources, joint_settings)
        problem.set_weights(base)
        starts = [np.zeros(4)]
        for k in (2, 3):
            for sign in (-1, 1):
                x = np.zeros(4); x[k] = sign*joint_settings.multistart_tangent; starts.append(x)
        initial_fits = [_solve(problem, x, joint_settings) for x in starts]
        fit = min(initial_fits, key=lambda f: problem.evaluate(f.x)['objective'])
        x = fit.x.copy()
        estimate.diagnostics.update(selection)
        estimate.diagnostics.update(initial_mouth_center_m=center.tolist(), initial_mouth_normal=normal.tolist(),
            fit_points=len(indices), selected_outer_points=len(selected), fit_camera_ids=sources.tolist(),
            fit_camera_names=[f.camera.name for f in frames], base_point_weights=base.tolist(),
            fit_point_weights=base.tolist(), axis_parameters=x.tolist())
        history, converged = [], mode == 'fixed'
        previous_weights = np.ones(len(points))
        core_info = None
        for iteration in range(settings.maximum_outer_iterations if mode != 'fixed' else 0):
            state = problem.evaluate(x)
            if mode == 'huber':
                multipliers = huber_weights(state['radial_residual'], settings.huber_delta_mm)
                core_info = None
            else:
                multipliers, core_info = region_weights(problem, x, settings, joint_settings)
                # Preserve total data weight and the scale of the smoothness term.
                multipliers /= np.average(multipliers, weights=base)
            problem.set_weights(base*multipliers)
            old_a, old_n, _ = problem.axis(x)
            fit = _solve(problem, x, joint_settings)
            x = fit.x.copy()
            new_a, new_n, _ = problem.axis(x)
            shift = float(np.linalg.norm(new_a-old_a))
            angle = float(np.degrees(np.arctan2(np.linalg.norm(np.cross(old_n,new_n)), old_n @ new_n)))
            weight_change = float(np.mean(np.abs(multipliers-previous_weights)))
            st = problem.evaluate(x)
            true_cost = (float(base @ huber_cost(st['radial_residual'], settings.huber_delta_mm)
                               + np.linalg.norm(problem.regularizer @ st['beta'])**2)
                         if mode == 'huber' else None)
            history.append(dict(iteration=iteration, axis_parameters=x.tolist(), shift_mm=shift,
                angle_deg=angle, mean_weight_change=weight_change, solver_success=bool(fit.success),
                weighted_ls_surrogate_mm2=st['objective'], huber_objective_mm2=true_cost, core=core_info))
            previous_weights = multipliers.copy()
            estimate.diagnostics.update(feedback_history=history, fit_point_weights=(base*multipliers).tolist(),
                                        axis_parameters=x.tolist(), final_core=core_info)
            if shift < settings.axis_shift_tolerance_mm and angle < settings.axis_angle_tolerance_deg and weight_change < settings.weight_tolerance:
                converged = True; break
        state = problem.evaluate(x)
        a, n, _ = problem.axis(x)
        # All modes use the same full-context, fixed-axis spline readout. Axis
        # support and target endpoint evidence are deliberately distinct.
        readout = WeightedAxisSplineProblem(points, center*1000, normal, sources, joint_settings)
        readout.set_weights(base)
        read_state = readout.evaluate(x)
        rms_full = float(np.sqrt(np.mean(read_state['radial_residual']**2)))
        fit_rms = float(np.sqrt(np.average(state['radial_residual']**2,
                                           weights=problem.sqrt_weight**2)))
        q, sv, _ = np.linalg.svd(state['data_profile_basis'], full_matrices=False)
        q = q[:, sv > sv[0]*1e-8]
        j = state['data_axis_jacobian']; j = (j-q @ (q.T @ j))*[5,5,.1,.1]
        singular = np.linalg.svd(j, compute_uv=False)
        estimate.condition = float(singular[0]/max(singular[-1], 1e-15))
        estimate.residual = rms_full/1000
        estimate.diagnostics.update(selection)
        estimate.diagnostics.update(initial_mouth_center_m=center.tolist(), initial_mouth_normal=normal.tolist(),
            point_counts=counts, selected_outer_points=len(selected), fit_points=len(indices),
            fit_indices=indices.tolist(), fit_camera_ids=sources.tolist(), axis_parameters=x.tolist(),
            fit_point_weights=(problem.sqrt_weight**2).tolist(), base_point_weights=base.tolist(),
            feedback_history=history, feedback_converged=converged, final_core=core_info,
            initial_multistart=[dict(axis_parameters=f.x.tolist(), success=bool(f.success)) for f in initial_fits],
            axis_fit_weighted_rms_mm=fit_rms, full_context_rms_mm=rms_full,
            full_context_readout_objective_mm2=read_state['objective'],
            effective_axis_singular_values=singular.tolist(),
            effective_weighted_point_count=float(np.sum(problem.sqrt_weight**2)**2/np.sum(problem.sqrt_weight**4)),
            condition_semantics='data-only weighted axis Jacobian with profile nuisance removed; no probability')
        mid, radius, profile = supported_interval(readout, read_state, selected*1000, joint_settings)
        estimate.center, estimate.normal, estimate.radius = (a+mid*n)/1000, n, radius/1000
        estimate.diagnostics.update(profile)
        low, high = np.array(profile['near_min_interval_from_axis_origin_m'])*1000
        near_target = ((read_state['s'] >= low-joint_settings.endpoint_support_mm) &
                       (read_state['s'] <= high+joint_settings.endpoint_support_mm))
        target_rms = float(np.sqrt(np.mean(read_state['radial_residual'][near_target]**2)))
        estimate.diagnostics['target_context_rms_mm'] = target_rms
        estimate.diagnostics['target_context_points'] = int(near_target.sum())
        if not fit.success: estimate.reasons.append('optimizer_not_converged')
        if not converged: estimate.reasons.append('feedback_not_converged')
        bounds = np.array([joint_settings.axis_shift_bound_mm]*2+[joint_settings.normal_tangent_bound]*2)
        if np.any(np.abs(x) > .98*bounds): estimate.reasons.append('axis_parameter_hits_bound')
        if estimate.condition > joint_settings.maximum_condition: estimate.reasons.append('poor_local_axis_condition')
        if fit_rms > joint_settings.maximum_radial_rms_mm: estimate.reasons.append('weighted_axis_model_mismatch')
        # Inspect readout around BOTH endpoints, not unrelated lower-shoulder
        # model error. Full-context RMS remains saved and is never hidden.
        if target_rms > joint_settings.maximum_radial_rms_mm: estimate.reasons.append('target_context_readout_model_mismatch')
        if not .02 < radius/1000 < lip_radius-.001: estimate.reasons.append('candidate_not_narrower_than_raised_lip')
        if n[2] < np.cos(np.deg2rad(30)): estimate.reasons.append('normal_outside_working_range')
        estimate.accepted = not estimate.reasons
    except (ProfileRejection, np.linalg.LinAlgError, QhullError) as error:
        estimate.reasons.append(str(error))
    estimate.elapsed_ms = (perf_counter()-started)*1000
    return estimate
