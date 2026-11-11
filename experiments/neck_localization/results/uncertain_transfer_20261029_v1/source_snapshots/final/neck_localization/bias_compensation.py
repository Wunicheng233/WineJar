"""Experimental observation-only bias estimation then frozen geometric fitting.

Single-window self-calibration under EXACTLY additive axial-Z error, circular
local sections and supplied correct K/T. Priors are assumed sensitivity values.
No known radius, target position, injected perturbation or model labels used.
"""
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import QhullError

from .bias_profile import BiasAxisSplineProblem, observation_rays
from .depth_perturbation import axial_depth_bias
from .depth_profile import ProfileRejection
from .reliable_output import _prepare, _initial
from .task_validated_output import estimate_task_validated_output


@dataclass(frozen=True)
class BiasCompensationSettings:
    sigma_data_mm: float = .1  # assumed; no probability coverage claim
    sigma_bias_mm: float | None = 2.  # None = no bias prior; otherwise assumed, not measured
    bias_bound_mm: float = 8.
    maximum_data_condition: float = 10000.
    heldout_degradation_mm: float = .02
    maximum_evaluations: int = 70


def estimate_bias_compensated(frames, *, settings=BiasCompensationSettings()):
    started = perf_counter()
    baseline = estimate_task_validated_output(frames)
    d = baseline.diagnostics
    diagnostic = dict(settings=asdict(settings), semantics='assumed additive axial-depth bias; no calibrated uncertainty',
                      baseline_accepted=baseline.accepted, baseline_reasons=list(baseline.reasons),
                      applied=False, estimated_bias_by_camera_mm={},
                      uncertainty='not_calibrated; local information diagnostic only')
    try:
        depth = d.get('candidate_depth_mm', 28.)
        prep = _prepare(frames, depth)
        base = prep['problem']
        matching = ('fit_point_positions_m' in d and np.array(d['fit_point_positions_m']).shape == base.points.shape
                    and np.allclose(np.array(d['fit_point_positions_m'])*1000, base.points, atol=1e-8))
        if matching:
            x0 = np.array(d['axis_parameters'])
            weights = np.array(d['fit_point_weights'])
            diagnostic['initialization'] = 'frozen_observed_axis_and_weights'
        else:
            fit0 = _initial(base, prep['joint'])
            x0, weights = fit0.x, np.ones(len(base.points))
            diagnostic['initialization'] = 'observed_fixed_weight_axis_no_target_readout_required'
        base.set_weights(weights)
        ids = prep['ids'][prep['train']]
        rays = observation_rays(base.points/1000, ids, frames)
        problem = BiasAxisSplineProblem(base, ids, rays, sigma_data_mm=settings.sigma_data_mm,
                                       sigma_bias_mm=np.inf if settings.sigma_bias_mm is None else settings.sigma_bias_mm)
        x0 = np.r_[x0, np.zeros(problem.ncamera)]
        bounds = np.array([20, 20, .24, .24]+[settings.bias_bound_mm]*problem.ncamera)
        fit = least_squares(problem.fun, x0, jac=problem.jac, bounds=(-bounds, bounds),
                            x_scale=[5,5,.1,.1]+[2]*problem.ncamera,
                            max_nfev=settings.maximum_evaluations, ftol=1e-8, xtol=1e-8, gtol=1e-7)
        before, after = problem.evaluate(x0), problem.evaluate(fit.x)
        information = problem.data_information(fit.x)
        held_ids = prep['ids'][prep['held']]
        held = prep['selected'][prep['held']]*1000
        held_rays = observation_rays(held/1000, held_ids, frames)
        initial_s = (held-before['origin']) @ before['normal']
        order = np.argsort(before['s'])
        held_weights = np.interp(initial_s, before['s'][order], weights[order])
        def validation(x, state):
            corrected = held-x[4:][held_ids, None]*held_rays
            delta = corrected-state['origin']
            s = delta @ state['normal']
            radial = np.linalg.norm(delta-s[:, None]*state['normal'], axis=1)-base.basis(s) @ state['beta']
            return float(np.sqrt(np.average(radial**2, weights=held_weights))), float(np.average(radial, weights=held_weights))
        before_rms, before_mean = validation(x0, before)
        after_rms, after_mean = validation(fit.x, after)
        bias = {frame.camera.name:float(fit.x[4+j]) for j,frame in enumerate(frames)}
        reasons = []
        if not fit.success:
            reasons.append('bias_optimizer_not_converged')
        if np.any(np.abs(fit.x) >= .98*bounds):
            reasons.append('bias_or_axis_parameter_hits_bound')
        if information['scaled_data_condition'] > settings.maximum_data_condition:
            reasons.append('bias_geometry_data_degenerate')
        if after_rms > before_rms+settings.heldout_degradation_mm:
            reasons.append('bias_heldout_degraded')
        diagnostic.update(estimated_bias_by_camera_mm=bias, objective_before=before['objective'], objective_after=after['objective'],
                          heldout_weighted_rms_before_mm=before_rms, heldout_weighted_rms_after_mm=after_rms,
                          heldout_weighted_mean_before_mm=before_mean, heldout_weighted_mean_after_mm=after_mean,
                          data_information=information, compensation_reasons=reasons,
                          fit_evaluations=fit.nfev, fit_point_count=len(base.points))
        if reasons:
            result = baseline
        else:
            corrected_frames = axial_depth_bias(frames, {name:-value/1000 for name,value in bias.items()})
            corrected = estimate_task_validated_output(corrected_frames)
            diagnostic.update(corrected_accepted=corrected.accepted, corrected_reasons=list(corrected.reasons))
            # Keep the original observed output only if correction cannot give
            # an accepted result. No truth-based portfolio selection.
            if corrected.accepted or not baseline.accepted:
                result = corrected
                diagnostic['applied'] = True
            else:
                result = baseline
                diagnostic['fallback'] = 'corrected_geometry_rejected_keep_accepted_baseline'
    except (ProfileRejection, np.linalg.LinAlgError, QhullError, ValueError) as error:
        result = baseline
        diagnostic['compensation_reasons'] = [type(error).__name__+':'+str(error)]
    result.diagnostics['bias_compensation'] = diagnostic
    result.diagnostics['method'] = 'bias_compensated_profile'
    result.elapsed_ms = (perf_counter()-started)*1000
    return result
