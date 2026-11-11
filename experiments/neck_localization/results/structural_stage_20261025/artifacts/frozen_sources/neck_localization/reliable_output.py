"""Success-oriented offline pipeline, retaining frozen estimators as controls.

Continuous axial kernel profile, damped support feedback, adaptive SOFTWARE
candidate depth, and a separately validated fixed-weight alternative. No model,
target truth, transport or robot state is read. Coordinates metres externally.
"""
from dataclasses import dataclass, replace, asdict
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import QhullError

from .depth_profile import ProfileRejection
from .geometry import Estimate, basis
from .joint_profile import JointSettings, observed_initialization, supported_interval
from .support_feedback import WeightedAxisSplineProblem


@dataclass(frozen=True)
class ReliabilitySettings:
    candidate_depths_mm: tuple = (28., 42., 56.)
    kernel_sigma_mm: float = .60
    profile_grid_mm: float = .25
    core_relief_mm: float = .30
    core_taper_mm: float = .50
    core_weight_floor: float = .001
    damping: float = .5
    maximum_iterations: int = 24
    minimum_core_span_mm: float = 3.
    minimum_core_points: int = 60
    minimum_coverage_deg: float = 90.
    core_rms_limit_mm: float = .4
    target_rms_limit_mm: float = .4
    heldout_mean_limit_mm: float = .20
    position_tolerance_mm: float = .005
    angle_tolerance_deg: float = .002
    weight_tolerance: float = .015


def continuous_core(problem, x, settings=ReliabilitySettings()):
    a,n,_=problem.axis(x)
    delta=problem.points-a
    s=delta@n
    radial=delta-s[:,None]*n
    rho=np.linalg.norm(radial,axis=1)
    # Fixed NUMBER of grid points; continuous moving coordinates, no floor bins.
    count=max(30,int(np.ceil(np.ptp(s)/settings.profile_grid_mm)))
    grid=np.linspace(s.min()+.25,s.max()-.25,count)
    kernel=np.exp(-.5*((grid[:,None]-s[None,:])/settings.kernel_sigma_mm)**2)
    support=kernel.sum(axis=1)
    profile=(kernel@rho)/np.maximum(support,1e-12)
    supported=support>=8
    if supported.sum()<8:raise ProfileRejection('insufficient_continuous_profile_support')
    minimum=int(np.argmin(np.where(supported,profile,np.inf)))
    threshold=profile[minimum]+settings.core_relief_mm
    left=right=minimum
    while left>0 and supported[left-1] and profile[left-1]<=threshold:left-=1
    while right+1<len(grid) and supported[right+1] and profile[right+1]<=threshold:right+=1
    if left==0 or right==len(grid)-1 or not supported[left-1] or not supported[right+1]:
        raise ProfileRejection('continuous_core_touches_candidate_boundary')
    low=float(np.interp(threshold,[profile[left],profile[left-1]],[grid[left],grid[left-1]]))
    high=float(np.interp(threshold,[profile[right],profile[right+1]],[grid[right],grid[right+1]]))
    core=(s>=low)&(s<=high)
    if high-low<settings.minimum_core_span_mm or core.sum()<settings.minimum_core_points:
        raise ProfileRejection('continuous_core_insufficient_extent')
    u,v=basis(n)
    angles=np.sort(np.mod(np.arctan2(radial[core]@v,radial[core]@u),2*np.pi))
    coverage=float(np.degrees(2*np.pi-np.diff(np.r_[angles,angles[0]+2*np.pi]).max()))
    if coverage<settings.minimum_coverage_deg:raise ProfileRejection('continuous_core_insufficient_coverage')
    distance=np.maximum(np.maximum(low-s,s-high),0.)
    weights=settings.core_weight_floor+(1-settings.core_weight_floor)*np.exp(-.5*(distance/settings.core_taper_mm)**2)
    weights/=weights.mean()
    return weights,dict(core_interval_mm=[low,high],core_points=int(core.sum()),angular_coverage_deg=coverage)


def _solve(problem,x,joint):
    bounds=np.array([joint.axis_shift_bound_mm]*2+[joint.normal_tangent_bound]*2)
    return least_squares(problem.fun,x,jac=problem.jac,bounds=(-bounds,bounds),
        x_scale=[5,5,.1,.1],max_nfev=joint.max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-7)


def _prepare(frames,depth):
    joint=replace(JointSettings(),fit_depth_mm=depth)
    selected,ids,indices,origin,normal,lip,counts,selection=observed_initialization(frames,joint,4101)
    # Deterministic interleaved heldout samples; never used in axis/curve objective.
    held=indices[np.arange(len(indices))%4==0]
    train=indices[np.arange(len(indices))%4!=0]
    p=WeightedAxisSplineProblem(selected[train]*1000,origin*1000,normal,ids[train],joint)
    p.set_weights(np.ones(len(train)))
    return dict(problem=p,joint=joint,selected=selected,ids=ids,train=train,held=held,
                origin=origin,normal=normal,lip=lip,counts=counts,selection=selection)


def _initial(problem,joint):
    starts=[np.zeros(4)]
    for k in [2,3]:
        for sign in [-1,1]:
            x=np.zeros(4);x[k]=sign*joint.multistart_tangent;starts.append(x)
    fits=[_solve(problem,x,joint) for x in starts]
    fit=min(fits,key=lambda f:problem.evaluate(f.x)['objective'])
    return fit


def _candidate(prepared,x,weights,frames,method,settings,history,converged,solver_success):
    p,joint=prepared['problem'],prepared['joint']
    result=Estimate(diagnostics=dict(method=method,uncertainty='not_estimated'))
    # Common full-context TRAINING readout; heldout is validation only.
    read=WeightedAxisSplineProblem(p.points,p.origin,p.n0,prepared['ids'][prepared['train']],joint)
    read.set_weights(np.ones(len(p.points)))
    st=read.evaluate(x)
    mid,radius,profile=supported_interval(read,st,prepared['selected']*1000,joint)
    a,n=st['origin'],st['normal']
    result.center,result.normal,result.radius=(a+mid*n)/1000,n,radius/1000
    validation_weights,core=continuous_core(p,x,settings)
    low,high=core['core_interval_mm']
    core_mask=(st['s']>=low)&(st['s']<=high)
    target_low,target_high=np.array(profile['near_min_interval_from_axis_origin_m'])*1000
    target_mask=(st['s']>=target_low-joint.endpoint_support_mm)&(st['s']<=target_high+joint.endpoint_support_mm)
    held=prepared['selected'][prepared['held']]*1000
    delta=held-a;hs=delta@n
    hr=np.linalg.norm(delta-hs[:,None]*n,axis=1)-read.basis(hs)@st['beta']
    target_held=(hs>=target_low-joint.endpoint_support_mm)&(hs<=target_high+joint.endpoint_support_mm)
    core_held=(hs>=low)&(hs<=high)
    if core_held.sum()<20 or target_held.sum()<30:raise ProfileRejection('insufficient_heldout_support')
    rms=lambda values:float(np.sqrt(np.mean(values**2)))
    metrics=dict(training_core_rms_mm=rms(st['radial_residual'][core_mask]),
        training_target_rms_mm=rms(st['radial_residual'][target_mask]),
        heldout_core_rms_mm=rms(hr[core_held]),heldout_target_rms_mm=rms(hr[target_held]),
        heldout_target_mean_mm=float(np.mean(hr[target_held])))
    # Condition on core weights rather than unrelated shoulder leverage.
    p.set_weights(validation_weights)
    axis_state=p.evaluate(x)
    q,sv,_=np.linalg.svd(axis_state['data_profile_basis'],full_matrices=False)
    q=q[:,sv>sv[0]*1e-8];j=axis_state['data_axis_jacobian']
    singular=np.linalg.svd((j-q@(q.T@j))*[5,5,.1,.1],compute_uv=False)
    result.condition=float(singular[0]/max(singular[-1],1e-15))
    result.residual=metrics['heldout_target_rms_mm']/1000
    d=result.diagnostics
    d.update(prepared['selection']);d.update(profile);d.update(metrics)
    d.update(axis_parameters=x.tolist(),initial_mouth_center_m=prepared['origin'].tolist(),
        initial_mouth_normal=prepared['normal'].tolist(),candidate_depth_mm=joint.fit_depth_mm,
        fit_point_positions_m=(p.points/1000).tolist(),fit_camera_ids=prepared['ids'][prepared['train']].tolist(),
        fit_camera_names=[f.camera.name for f in frames],fit_point_weights=weights.tolist(),
        fit_points=len(p.points),heldout_points=len(held),feedback_history=history,feedback_converged=converged,
        final_core=core,full_context_rms_mm=rms(st['radial_residual']),
        quality_semantics='core + target readout + deterministic heldout checks; not calibrated absolute accuracy',
        condition_semantics='data-only core-weighted axis information with spline nuisance removed',
        effective_axis_singular_values=singular.tolist())
    if not converged:result.reasons.append('continuous_feedback_not_converged')
    if not solver_success:result.reasons.append('optimizer_not_converged')
    if max(metrics['training_core_rms_mm'],metrics['heldout_core_rms_mm'])>settings.core_rms_limit_mm:
        result.reasons.append('core_model_mismatch')
    if max(metrics['training_target_rms_mm'],metrics['heldout_target_rms_mm'])>settings.target_rms_limit_mm:
        result.reasons.append('target_readout_mismatch')
    if abs(metrics['heldout_target_mean_mm'])>settings.heldout_mean_limit_mm:result.reasons.append('heldout_systematic_residual')
    if result.condition>joint.maximum_condition:result.reasons.append('poor_local_axis_condition')
    bounds=np.array([joint.axis_shift_bound_mm]*2+[joint.normal_tangent_bound]*2)
    if np.any(np.abs(x)>.98*bounds):result.reasons.append('axis_parameter_hits_bound')
    if n[2]<np.cos(np.deg2rad(30)):result.reasons.append('normal_outside_working_range')
    if not .02<result.radius<prepared['lip']-.001:result.reasons.append('candidate_not_narrower_than_raised_lip')
    result.accepted=not result.reasons
    return result


def _attempt(prepared,frames,settings,method):
    p,joint=prepared['problem'],prepared['joint']
    initial=_initial(p,joint)
    x=initial.x.copy();fixed_x=x.copy()
    prepared['fixed_x'],prepared['fixed_success']=fixed_x,bool(initial.success)
    previous=None;history=[];converged=False
    for iteration in range(settings.maximum_iterations):
        target,core=continuous_core(p,x,settings)
        weights=target if previous is None else (1-settings.damping)*previous+settings.damping*target
        p.set_weights(weights)
        old_a,old_n,_=p.axis(x)
        fit=_solve(p,x,joint);x=fit.x.copy()
        a,n,_=p.axis(x)
        shift=float(np.linalg.norm(a-old_a))
        angle=float(np.degrees(np.arctan2(np.linalg.norm(np.cross(n,old_n)),n@old_n)))
        change=float(np.mean(np.abs(weights-(np.ones(len(weights)) if previous is None else previous))))
        history.append(dict(iteration=iteration,core=core,shift_mm=shift,angle_deg=angle,mean_weight_change=change,
                            axis_parameters=x.tolist(),solver_success=bool(fit.success)))
        previous=weights
        if shift<settings.position_tolerance_mm and angle<settings.angle_tolerance_deg and change<settings.weight_tolerance:
            converged=True;break
    result=_candidate(prepared,x,weights,frames,method,settings,history,converged,bool(fit.success))
    return result,fixed_x,bool(initial.success)


def estimate_reliable_output(frames,*,adaptive=True,settings=ReliabilitySettings()):
    started=perf_counter();method='adaptive_validated' if adaptive else 'stable_local'
    attempts=[];last=None
    depths=settings.candidate_depths_mm if adaptive else settings.candidate_depths_mm[:1]
    for depth in depths:
        prepared=None;fixed_x=None
        try:
            prepared=_prepare(frames,depth)
            result,fixed_x,fixed_success=_attempt(prepared,frames,settings,method)
            attempts.append(dict(depth_mm=depth,path='continuous_feedback',accepted=result.accepted,reasons=result.reasons))
            last=result
            if result.accepted:break
        except (ProfileRejection,np.linalg.LinAlgError,QhullError) as error:
            attempts.append(dict(depth_mm=depth,path='continuous_feedback',accepted=False,reasons=[str(error)]))
            if prepared is not None:
                fixed_x=prepared.get('fixed_x')
                fixed_success=prepared.get('fixed_success',False)
        if adaptive and fixed_x is not None:
            try:
                weights=np.ones(len(prepared['problem'].points))
                fallback=_candidate(prepared,fixed_x,weights,frames,method,settings,[],True,fixed_success)
                fallback.diagnostics['chosen_path']='validated_fixed_weight_alternative'
                attempts.append(dict(depth_mm=depth,path='fixed_alternative',accepted=fallback.accepted,reasons=fallback.reasons))
                last=fallback
                if fallback.accepted:break
            except (ProfileRejection,np.linalg.LinAlgError,QhullError) as error:
                attempts.append(dict(depth_mm=depth,path='fixed_alternative',accepted=False,reasons=[str(error)]))
    if last is None:
        last=Estimate(reasons=attempts[-1]['reasons'] if attempts else ['no_candidate'])
    last.diagnostics.update(method=method,attempts=attempts,reliability_settings=asdict(settings),
        camera_trust=[1.]*len(frames),uncertainty='not_estimated',
        chosen_path=last.diagnostics.get('chosen_path','continuous_feedback' if last.center is not None else 'none'))
    last.elapsed_ms=(perf_counter()-started)*1000
    return last
