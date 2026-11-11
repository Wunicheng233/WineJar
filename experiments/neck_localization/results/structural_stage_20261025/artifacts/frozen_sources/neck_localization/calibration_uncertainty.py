"""First-order COMMON world-reference SE(3) uncertainty propagation.

xi=(dt_world [metres], dtheta_world [radians]), left perturbation about world
origin: p' = Exp(dtheta) p + dt. This term is shared by all observations; it
must not shrink with point count. This module does not estimate the prior.
"""
import numpy as np


def skew(vector):
    x,y,z=np.asarray(vector,float)
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])


def common_pose_covariance(center,normal,covariance_pose):
    c,n=np.asarray(center,float),np.asarray(normal,float)
    covariance=np.asarray(covariance_pose,float)
    if c.shape!=(3,) or n.shape!=(3,) or covariance.shape!=(6,6):
        raise ValueError('Expected centre/normal and 6x6 common-pose covariance')
    if not np.all(np.isfinite(np.r_[c,n,covariance.ravel()])) or np.linalg.norm(n)<1e-12:
        raise ValueError('Finite inputs and nonzero normal required')
    if not np.allclose(covariance,covariance.T,atol=1e-15) or np.linalg.eigvalsh(covariance).min() < -1e-15:
        raise ValueError('Covariance must be symmetric positive semidefinite')
    n=n/np.linalg.norm(n)
    Jc=np.c_[np.eye(3),-skew(c)]
    Jn=np.c_[np.zeros((3,3)),-skew(n)]
    J=np.vstack((Jc,Jn))
    propagated=J@covariance@J.T
    return dict(joint_center_normal_covariance=propagated,center_covariance=propagated[:3,:3],
                normal_covariance=propagated[3:,3:],center_normal_cross_covariance=propagated[:3,3:],jacobian=J)
