"""Core continuity and task-validation checks for the success-rate pipeline."""
import numpy as np
import pytest
from neck_localization.depth_profile import ProfileRejection
from neck_localization.joint_profile import JointSettings
from neck_localization.support_feedback import WeightedAxisSplineProblem
from neck_localization.reliable_output import ReliabilitySettings,continuous_core,_candidate


def surface():
    s=np.repeat(np.linspace(-25,-2,80),32)
    phi=np.tile(np.linspace(0,2*np.pi,32,endpoint=False),80)
    radius=52+.15*np.maximum(np.abs(s+13)-5,0)**2
    origin=np.array([0.,0.,600.])
    points=origin+np.c_[radius*np.cos(phi),radius*np.sin(phi),s]
    p=WeightedAxisSplineProblem(points,origin,np.array([0.,0.,1.]),np.zeros(len(points),int))
    p.set_weights(np.ones(len(points)))
    return p


def test_small_axis_perturbation_does_not_jump_core_by_a_bin():
    p=surface()
    _,a=continuous_core(p,np.zeros(4))
    _,b=continuous_core(p,np.array([.001,-.001,1e-5,-1e-5]))
    assert np.max(np.abs(np.array(a['core_interval_mm'])-b['core_interval_mm']))<.02
    assert a['core_points']>100


def test_constant_radius_without_observed_ends_is_still_rejected():
    p=surface();s=(p.points-p.origin)@p.n0
    p.points[:,:2]*=(52/np.linalg.norm(p.points[:,:2],axis=1))[:,None]
    with pytest.raises(ProfileRejection,match='boundary'):
        continuous_core(p,np.zeros(4))


def test_heldout_bias_is_not_accepted_by_a_good_training_fit():
    p=surface();train=p.points.copy()
    held=train.copy();held[:,:2]*=(1+.8/np.linalg.norm(held[:,:2],axis=1))[:,None]
    selected=np.r_[train,held]/1000
    n=len(train)
    prep=dict(problem=p,joint=JointSettings(),selected=selected,ids=np.zeros(2*n,int),
        train=np.arange(n),held=np.arange(n,2*n),origin=p.origin/1000,normal=p.n0,
        lip=.070,selection={})
    from types import SimpleNamespace
    result=_candidate(prep,np.zeros(4),np.ones(n),[SimpleNamespace(camera=SimpleNamespace(name='a'))],
                      'diagnostic',ReliabilitySettings(),[],True,True)
    assert not result.accepted
    assert 'heldout_systematic_residual' in result.reasons
    assert 'target_readout_mismatch' in result.reasons


def test_missing_depth_is_not_forced_to_success_by_expansion_or_fallback():
    from neck_localization.depth_profile import DepthFrame
    from neck_localization.geometry import look_at
    from neck_localization.reliable_output import estimate_reliable_output
    camera=look_at('a',[.8,0,.7],[0,0,.59],width=40,height=30)
    frame=DepthFrame(camera,np.full((30,40,3),120,np.uint8),np.zeros((30,40)))
    result=estimate_reliable_output([frame])
    assert not result.accepted
    assert result.center is None
    assert len(result.diagnostics['attempts'])==3
