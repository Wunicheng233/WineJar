"""Variable identified-corner observations. No imputation of invisible corners."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .station_tying import pose
from .fixed_camera_calibration import projection,rotation_excitation
from .poe_camera_calibration import CameraPOE


def initialize(frames,K):
    robot=np.array([r['B_T_F_reported'] for r in frames]);measured=[]
    if not rotation_excitation(robot)['independent_axes']:raise ValueError('insufficient rotation excitation')
    for f in frames:
        ok,rv,tv=cv2.solvePnP(np.array(f['points_P_m'],float),np.array(f['identified_pixels'],float),K,None,flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:raise ValueError('partial board PnP failed')
        measured.append(pose(cv2.Rodrigues(rv)[0],tv.ravel()))
    inverse=np.linalg.inv(robot)
    r,t=cv2.calibrateHandEye(list(inverse[:,:3,:3]),list(inverse[:,:3,3]),
        [x[:3,:3] for x in measured],[x[:3,3] for x in measured],method=cv2.CALIB_HAND_EYE_PARK)
    X=pose(r,t.ravel());mounts=inverse @X @np.array(measured)
    Y=pose(Rotation.from_matrix(mounts[:,:3,:3]).mean().as_matrix(),mounts[:,:3,3].mean(0))
    return X,Y


def fit_partial_board(frames,K,nominal,*,kinematic=False):
    X0,Y0=initialize(frames,K)
    def matrices(v):
        X=pose(Rotation.from_rotvec(v[3:6]).as_matrix() @X0[:3,:3],X0[:3,3]+v[:3])
        Y=pose(Rotation.from_rotvec(v[9:12]).as_matrix() @Y0[:3,:3],Y0[:3,3]+v[6:9])
        return X,Y
    def pixel_residual(v):
        X,Y=matrices(v)
        return np.concatenate([(projection(np.linalg.inv(X) @f['B_T_F_reported'] @Y,f['points_P_m'],K)-f['identified_pixels']).ravel() for f in frames])
    fit=least_squares(pixel_residual,np.zeros(12),loss='huber',f_scale=.5,x_scale='jac',max_nfev=100,
        ftol=1e-9,xtol=1e-10,gtol=1e-7)
    X,Y=matrices(fit.x)
    if not kinematic:
        return dict(B_T_C=X,F_T_P=Y,accepted=bool(fit.success),pixel_rms=float(np.sqrt(np.mean(pixel_residual(fit.x)**2))),
                    evaluations=fit.nfev,reason='partial_board_calibration_candidate')
    R=X[:3,:3].T
    axes=np.array(nominal['axes_B']) @R.T
    points=(np.array(nominal['points_B'])-X[:3,3]) @R.T
    points-=np.sum(points*axes,axis=1)[:,None]*axes
    model=CameraPOE(axes,points,np.linalg.inv(X) @nominal['B_T_F0'] @Y)
    q=np.array([f['q_reported'] for f in frames]);weights=np.ones(len(frames));v=np.zeros(30);evaluations=fit.nfev
    def blocks(v):
        T=model.forward(q,v)
        return [(projection(t,f['points_P_m'],K)-f['identified_pixels']).ravel()/.15 for t,f in zip(T,frames)]
    for outer in range(7):
        if outer:
            weights=np.array([1/(1+np.mean(b*b)/4) for b in blocks(v)])
        def residual(v):return np.r_[np.concatenate([b*np.sqrt(w) for b,w in zip(blocks(v),weights)]),v[6:]]
        result=least_squares(residual,v,max_nfev=100,x_scale='jac',ftol=1e-9,xtol=1e-9,gtol=1e-7,
            loss='soft_l1' if outer==0 else 'linear',f_scale=2)
        v=result.x;evaluations+=result.nfev
    model.parameters=v
    return dict(B_T_C=X,F_T_P=Y,camera_poe=model.as_dict(),accepted=bool(result.success),
        pixel_rms=float(np.sqrt(np.mean(np.concatenate(blocks(v))**2))*.15),view_weights=weights,evaluations=evaluations,
        reason='partial_board_camera_joint_mapping_candidate',initial_X_Y_not_runtime_mapping=True)
