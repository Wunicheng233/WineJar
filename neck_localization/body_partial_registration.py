"""Partial-view ICP ablation: observed subset queries the instance template.

Uses the same point-plane loss, Huber scale, distance and iteration limits as
the forward ICP control. Reciprocal matching is a separate ablation. Neither
variant receives geometric labels or an exact body model.
"""
import numpy as np
from scipy.spatial import cKDTree
from .station_tying import pose


def register_observed_subset(source,target,normals,initial,direction,*,reciprocal=False):
    T=np.array(initial,float).copy();p=np.asarray(source,float);q=np.asarray(target,float)
    if min(len(p),len(q))<50:
        return dict(accepted=False,reason='insufficient_body_observations',transform=T)
    axis=np.array(direction,float);axis/=np.linalg.norm(axis)
    target_tree=cKDTree(q)
    for iteration in range(35):
        moved=p @ T[:3,:3].T+T[:3,3]
        distance,ids=cKDTree(moved).query(q)
        keep=distance<.012
        if reciprocal:
            _,reverse=target_tree.query(moved)
            keep &= reverse[ids]==np.arange(len(q))
        if keep.sum()<50:return dict(accepted=False,reason='insufficient_overlap',transform=T)
        n=normals[keep];residual=np.sum((moved[ids[keep]]-q[keep])*n,axis=1)
        J=n @ axis
        scale=max(.0002,1.4826*float(np.median(abs(residual-np.median(residual)))))
        weight=np.minimum(1.,1.345*scale/np.maximum(abs(residual),1e-12))
        information=float(np.sum(weight*J**2))
        if information<1e-6:return dict(accepted=False,reason='unobservable_conveyor_translation',transform=T)
        delta=-float(np.sum(weight*J*residual)/information)
        T=pose(translation=axis*delta) @ T
        if abs(delta)<1e-7:break
    return dict(accepted=True,reason='registration_candidate_not_process_release',transform=T,
        point_plane_rms_m=float(np.sqrt(np.mean(residual**2))),
        overlap_fraction=float(keep.mean()),iterations=iteration+1,
        scalar_information=information,reciprocal=reciprocal)
