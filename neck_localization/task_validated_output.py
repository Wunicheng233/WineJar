"""Additional task-range validation for short necks; v1 remains frozen.

A narrow internal axis core is not required of the alternative. The complete
functional target interval, heldout surface and its data-only information ARE
required. A portfolio failure is still returned if these checks do not pass.
"""
from dataclasses import dataclass,asdict
from time import perf_counter
import numpy as np
from scipy.spatial import QhullError

from .depth_profile import ProfileRejection
from .geometry import Estimate,basis
from .joint_profile import supported_interval
from .support_feedback import WeightedAxisSplineProblem
from .reliable_output import estimate_reliable_output,_prepare,_initial,ReliabilitySettings


@dataclass(frozen=True)
class TaskValidationSettings:
    minimum_target_span_mm: float = 1.5  # three 0.5 mm support bins, not a 3 mm axis core
    minimum_training_target_points: int = 60
    minimum_heldout_target_points: int = 20
    minimum_coverage_deg: float = 90.
    residual_rms_mm: float = .4
    heldout_signed_mean_mm: float = .2
    maximum_condition: float = 10000.
    agreement_center_mm: float = 2.
    agreement_normal_deg: float = 2.


def _validate_task(prep,x,frames,settings,label,solver_success):
    p,joint=prep['problem'],prep['joint']
    p.set_weights(np.ones(len(p.points)))
    st=p.evaluate(x)
    mid,radius,profile=supported_interval(p,st,prep['selected']*1000,joint)
    a,n=st['origin'],st['normal']
    low,high=np.array(profile['near_min_interval_from_axis_origin_m'])*1000
    mask=(st['s']>=low)&(st['s']<=high)
    held=prep['selected'][prep['held']]*1000
    delta=held-a;hs=delta@n
    hr=np.linalg.norm(delta-hs[:,None]*n,axis=1)-p.basis(hs)@st['beta']
    hm=(hs>=low)&(hs<=high)
    if high-low<settings.minimum_target_span_mm or mask.sum()<settings.minimum_training_target_points or hm.sum()<settings.minimum_heldout_target_points:
        raise ProfileRejection('insufficient_task_range_support')
    radial=p.points-a-st['s'][:,None]*n;u,v=basis(n)
    angles=np.sort(np.mod(np.arctan2(radial[mask]@v,radial[mask]@u),2*np.pi))
    coverage=float(np.degrees(2*np.pi-np.diff(np.r_[angles,angles[0]+2*np.pi]).max()))
    if coverage<settings.minimum_coverage_deg:raise ProfileRejection('insufficient_task_range_coverage')
    rms=lambda z:float(np.sqrt(np.mean(z*z)))
    tm=(st['s']>=low-joint.endpoint_support_mm)&(st['s']<=high+joint.endpoint_support_mm)
    hcontext=(hs>=low-joint.endpoint_support_mm)&(hs<=high+joint.endpoint_support_mm)
    metrics=dict(training_target_rms_mm=rms(st['radial_residual'][mask]),heldout_target_rms_mm=rms(hr[hm]),
                 training_context_rms_mm=rms(st['radial_residual'][tm]),heldout_context_rms_mm=rms(hr[hcontext]),
                 heldout_target_mean_mm=float(np.mean(hr[hm])))
    # Only TASK RANGE data contribute information; no shoulder/regularizer leverage.
    B=st['data_profile_basis'][mask];J=st['data_axis_jacobian'][mask]
    q,sv,_=np.linalg.svd(B,full_matrices=False);q=q[:,sv>sv[0]*1e-8]
    singular=np.linalg.svd((J-q@(q.T@J))*[5,5,.1,.1],compute_uv=False)
    condition=float(singular[0]/max(singular[-1],1e-15))
    result=Estimate(center=(a+mid*n)/1000,normal=n,radius=radius/1000,condition=condition,
                    residual=metrics['heldout_target_rms_mm']/1000,diagnostics={})
    if not solver_success:result.reasons.append('alternative_optimizer_not_converged')
    if max(metrics[k] for k in ['training_target_rms_mm','heldout_target_rms_mm','training_context_rms_mm','heldout_context_rms_mm'])>settings.residual_rms_mm:
        result.reasons.append('task_range_validation_mismatch')
    if abs(metrics['heldout_target_mean_mm'])>settings.heldout_signed_mean_mm:result.reasons.append('heldout_systematic_residual')
    # Check each available camera separately as well as pooled RMS.
    per_camera=[]
    held_ids=prep['ids'][prep['held']]
    for i,f in enumerate(frames):
        cm=hm&(held_ids==i)
        if cm.sum()>=6:
            mean=float(np.mean(hr[cm]));crms=rms(hr[cm])
            per_camera.append(dict(camera=f.camera.name,n=int(cm.sum()),rms_mm=crms,mean_mm=mean))
            if crms>settings.residual_rms_mm or abs(mean)>settings.heldout_signed_mean_mm:
                result.reasons.append('cross_camera_task_validation_mismatch')
    if condition>settings.maximum_condition:result.reasons.append('poor_task_axis_information')
    bounds=np.array([joint.axis_shift_bound_mm]*2+[joint.normal_tangent_bound]*2)
    if np.any(np.abs(x)>.98*bounds):result.reasons.append('axis_parameter_hits_bound')
    if n[2]<np.cos(np.deg2rad(30)):result.reasons.append('normal_outside_working_range')
    if not .02<result.radius<prep['lip']-.001:result.reasons.append('candidate_not_narrower_than_lip')
    result.accepted=not result.reasons
    d=result.diagnostics;d.update(prep['selection']);d.update(profile);d.update(metrics)
    d.update(method='task_validated_output',chosen_path=label,candidate_depth_mm=joint.fit_depth_mm,
        initial_mouth_center_m=prep['origin'].tolist(),initial_mouth_normal=prep['normal'].tolist(),
        axis_parameters=x.tolist(),fit_point_positions_m=(p.points/1000).tolist(),fit_point_weights=np.ones(len(p.points)).tolist(),
        fit_camera_ids=prep['ids'][prep['train']].tolist(),fit_camera_names=[f.camera.name for f in frames],
        fit_points=len(p.points),heldout_points=len(held),per_camera_heldout_validation=per_camera,
        task_validation_span_mm=high-low,task_validation_coverage_deg=coverage,
        condition_semantics='task-only DATA axis Jacobian with spline nuisance removed; no regularizer',
        effective_axis_singular_values=singular.tolist(),uncertainty='not_estimated',
        quality_semantics='target-range and surrounding support, heldout and camera-specific checks; not calibrated accuracy')
    return result


def estimate_task_validated_output(frames,*,settings=TaskValidationSettings()):
    started=perf_counter()
    primary=estimate_reliable_output(frames)
    if primary.accepted:
        primary.diagnostics.update(method='task_validated_output',task_validation_settings=asdict(settings))
        primary.elapsed_ms=(perf_counter()-started)*1000
        return primary
    attempts=list(primary.diagnostics.get('attempts',[]));best=None
    for depth in ReliabilitySettings().candidate_depths_mm:
        try:
            prep=_prepare(frames,depth);fit=_initial(prep['problem'],prep['joint'])
        except (ProfileRejection,np.linalg.LinAlgError,QhullError) as error:
            attempts.append(dict(depth_mm=depth,path='task_alternative_preparation',accepted=False,reasons=[str(error)]));continue
        candidates=[]
        for label,x,ok in [('task_validated_joint_alternative',fit.x,bool(fit.success)),
                           ('task_validated_observed_axis',np.zeros(4),True)]:
            try:
                candidate=_validate_task(prep,x,frames,settings,label,ok)
                attempts.append(dict(depth_mm=depth,path=label,accepted=candidate.accepted,reasons=candidate.reasons))
                if candidate.accepted:candidates.append(candidate)
            except (ProfileRejection,np.linalg.LinAlgError,QhullError) as error:
                attempts.append(dict(depth_mm=depth,path=label,accepted=False,reasons=[str(error)]))
        if not candidates:continue
        if len(candidates)>1:
            a,b=candidates[:2]
            gap_center=float(np.linalg.norm(a.center-b.center)*1000)
            gap_normal=float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a.normal,b.normal)),a.normal@b.normal)))
            if gap_center>settings.agreement_center_mm or gap_normal>settings.agreement_normal_deg:
                attempts.append(dict(depth_mm=depth,path='task_alternative_agreement',accepted=False,reasons=['ambiguous_task_alternatives'],center_gap_mm=gap_center,normal_gap_deg=gap_normal));continue
        best=min(candidates,key=lambda c:c.diagnostics['heldout_context_rms_mm'])
        break
    result=best if best is not None else primary
    result.diagnostics.update(method='task_validated_output',attempts=attempts,primary_rejection=primary.reasons,
        task_validation_settings=asdict(settings),uncertainty='not_estimated')
    result.elapsed_ms=(perf_counter()-started)*1000
    return result
