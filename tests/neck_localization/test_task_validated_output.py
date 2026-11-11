"""Task-range validation must pass short observable profiles, reject bad data."""
from types import SimpleNamespace
import numpy as np
from neck_localization.joint_profile import JointSettings
from neck_localization.support_feedback import WeightedAxisSplineProblem
from neck_localization.task_validated_output import TaskValidationSettings,_validate_task
from test_reliable_output import surface


def prepare(p,offsets):
    train=p.points.copy();n=len(train);held=train.copy()
    held[:,:2]*=(1+offsets/np.linalg.norm(held[:,:2],axis=1))[:,None]
    camera_ids=np.arange(n)%2
    prep=dict(problem=p,joint=JointSettings(),selected=np.r_[train,held]/1000,
        ids=np.r_[camera_ids,camera_ids],train=np.arange(n),held=np.arange(n,2*n),
        origin=p.origin/1000,normal=p.n0,lip=.070,selection={})
    frames=[SimpleNamespace(camera=SimpleNamespace(name=name)) for name in ['a','b']]
    return prep,frames


def test_observable_short_functional_range_does_not_require_wide_internal_core():
    p=surface();s=(p.points-p.origin)@p.n0
    rho=np.linalg.norm(p.points[:,:2],axis=1)
    p.points[:,:2]*=((52+.8*(s+13)**2)/rho)[:,None]
    prep,frames=prepare(p,np.zeros(len(s)))
    result=_validate_task(prep,np.zeros(4),frames,TaskValidationSettings(),'diagnostic',True)
    assert result.accepted,result.reasons
    assert 1.5<result.diagnostics['task_validation_span_mm']<3.


def test_heldout_bias_cannot_be_bypassed_by_task_alternative():
    p=surface();prep,frames=prepare(p,np.full(len(p.points),.8))
    result=_validate_task(prep,np.zeros(4),frames,TaskValidationSettings(),'diagnostic',True)
    assert not result.accepted
    assert 'heldout_systematic_residual' in result.reasons


def test_opposite_camera_biases_do_not_cancel_in_pooled_validation():
    p=surface();offsets=np.where(np.arange(len(p.points))%2==0,.3,-.3)
    prep,frames=prepare(p,offsets)
    result=_validate_task(prep,np.zeros(4),frames,TaskValidationSettings(),'diagnostic',True)
    assert not result.accepted
    assert 'cross_camera_task_validation_mismatch' in result.reasons
