"""Task-local surface verification, paired against frozen full-context gates.

Only the RGB validation domain changes: the fitted surface over the requested
3 mm operation span must lie in every foreground. Full-context projection is
retained as an advisory and as the UNCHANGED candidate ranking statistic.
This is necessary image support, not a pose/physical band safety guarantee.
"""
from copy import deepcopy
from dataclasses import asdict
from time import perf_counter

import numpy as np
from scipy.spatial import QhullError

from .band_geometry import DEFAULT_BAND_WIDTH_M
from .depth_profile import ProfileRejection
from .geometry import Estimate
from .masked_geometry_baseline import validate_explicit_masks
from .structure_portfolio import PortfolioSettings, fit_seed, observed_cloud, seed_hypotheses
from .visible_profile_baseline import profile_projection_support, revalidate_candidate


SURFACE_REASON = 'fitted_surface_projects_outside_rgb_foreground'


def task_surface_support(candidate, frames, masks, width_m=DEFAULT_BAND_WIDTH_M):
    """No extrapolation, 0.1 mm axial sampling incl. exact band endpoints."""
    if not np.isfinite(width_m) or width_m <= 0:
        raise ValueError('positive_finite_operation_width_required')
    local=deepcopy(candidate)
    d=local.diagnostics
    grid=np.asarray(d['profile_s_m'],float)
    radius=np.asarray(d['profile_radius_m'],float)
    normal=np.asarray(local.normal,float);normal=normal/np.linalg.norm(normal)
    origin=np.asarray(d['axis_origin_m'],float)
    position=float((local.center-origin)@normal)
    low,high=position-width_m/2,position+width_m/2
    if (len(grid)<2 or radius.shape!=grid.shape or not np.isfinite(np.r_[grid,radius]).all()
            or np.any(np.diff(grid)<=0)):
        raise ProfileRejection('invalid_task_surface_profile')
    if low < grid[0]-1e-12 or high > grid[-1]+1e-12:
        raise ProfileRejection('operation_surface_outside_fitted_support')
    axial=np.unique(np.r_[low,grid[(grid>low)&(grid<high)],high])
    d['profile_s_m']=axial.tolist()
    d['profile_radius_m']=np.interp(axial,grid,radius).tolist()
    support=profile_projection_support(local,frames,masks)
    support.update(validation='one-sided fitted surface containment over requested operation span only',
        operation_width_m=float(width_m),axis_interval_m=[float(low),float(high)],
        context_is_not_validated_as_accurate_surface=True,
        radius_interpolation='linear on saved 0.1 mm profile; exact requested interval endpoints',
        no_extrapolation=True)
    return support


def revalidate_task_candidate(context,frames,masks,settings=PortfolioSettings(),width_m=DEFAULT_BAND_WIDTH_M):
    """Keep every non-RGB gate, the 2px threshold and the original ranking."""
    result=deepcopy(context);d=result.diagnostics
    d['context_surface_projection']=deepcopy(d['surface_projection'])
    d['context_validation']=dict(accepted=context.accepted,reasons=list(context.reasons))
    result.reasons=[r for r in result.reasons if r!=SURFACE_REASON]
    try:
        support=task_surface_support(result,frames,masks,width_m)
        d['task_surface_projection']=support
        if support['maximum_outside_pixels']>settings.validation_contact_pixels:
            result.reasons.append('operation_surface_projects_outside_rgb_foreground')
    except ProfileRejection as error:
        result.reasons.append(str(error));d['task_surface_projection']=dict(error=str(error))
    result.accepted=not result.reasons
    d.update(method='task_surface_validation',
        validation_change='only RGB validation domain; all non-RGB gates and full-context ranking unchanged',
        operation_width_m=float(width_m),
        pose_gate_accepted=not [r for r in result.reasons if r!='insufficient_three_mm_band_span'],
        uncertainty='not_calibrated; target support is necessary, not sufficient for absolute accuracy or physical band use')
    return result


def estimate_paired_surfaces(frames,masks,foreground,*,settings=PortfolioSettings(),width_m=DEFAULT_BAND_WIDTH_M):
    """Fit once/seed, then independently select each validator's candidates.

    no_rgb_gate is an explicit diagnostic ablation, never a default proposal.
    No GT, CAD, learned model or external state is read by this function.
    """
    started=perf_counter();methods=['visible_profile','task_surface','no_rgb_gate']
    candidates={m:[] for m in methods};attempts={m:[] for m in methods}
    masks,foreground=validate_explicit_masks(frames,masks,foreground)
    error=None;count=0
    try:
        cloud,source=observed_cloud(frames,masks,settings);count=len(cloud)
        if count<300:raise ProfileRejection('insufficient_multiview_consistent_observed_cloud')
        seeds=seed_hypotheses(frames,masks,cloud,settings)
        if not seeds:raise ProfileRejection('no_observed_rim_hypothesis')
        for index,seed in enumerate(seeds):
            try:
                frozen=fit_seed(frames,masks,cloud,source,seed,settings)
                context=revalidate_candidate(frozen,frames,masks,settings)
                task=revalidate_task_candidate(context,frames,masks,settings,width_m)
                ungated=deepcopy(context)
                ungated.reasons=[r for r in ungated.reasons if r!=SURFACE_REASON]
                ungated.accepted=not ungated.reasons
                ungated.diagnostics['method']='no_rgb_gate_diagnostic'
                for method,c in zip(methods,[context,task,ungated]):
                    c.diagnostics['candidate_index']=index
                    candidates[method].append(c)
                    attempts[method].append(dict(candidate_index=index,seed=seed['label'],
                        accepted=c.accepted,reasons=list(c.reasons),center_m=c.center.tolist(),
                        normal=c.normal.tolist(),radius_m=c.radius,score=c.diagnostics['selection_score']))
            except (ValueError,np.linalg.LinAlgError,QhullError) as exception:
                for m in methods:attempts[m].append(dict(candidate_index=index,seed=seed['label'],accepted=False,reasons=[str(exception)]))
    except (ValueError,np.linalg.LinAlgError,QhullError) as exception:error=str(exception)
    results={}
    for method in methods:
        result=(min(candidates[method],key=lambda c:(not c.accepted,c.diagnostics['selection_score']))
            if candidates[method] else Estimate(reasons=[error or 'no_valid_structural_candidate']))
        result.diagnostics.update(attempts=attempts[method],foreground=deepcopy(foreground),
            station_consistent_points=count,settings=asdict(settings),operation_width_m=width_m,
            paired_shared_fit=True,all_candidates_rescored_before_selection=True,
            candidate_count=len(candidates[method]),uncertainty='not_calibrated',
            ablation_only=method=='no_rgb_gate')
        result.elapsed_ms=(perf_counter()-started)*1000
        results[method]=result
    return results
