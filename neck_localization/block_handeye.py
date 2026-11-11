"""A simple fixed-robot hand-eye model with the same whole-view Cauchy loss.

Supplementary engineering control, not a robot-kinematics calibration.
Uses the reported poses and identified pixels only.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .fixed_camera_calibration import solve_eye_to_hand
from .station_tying import pose


def solve_block_handeye(robot_poses, image_points, board_points, K):
    A, uv, points = map(np.asarray, (robot_poses, image_points, board_points))
    initial = solve_eye_to_hand(A, uv, points, K, method='pixel_huber')
    if not initial['accepted']:
        return initial
    X0, Y0 = initial['B_T_C'], initial['F_T_P']
    def unpack(v):
        return (pose(Rotation.from_rotvec(v[3:6]*.1).as_matrix() @ X0[:3, :3], X0[:3, 3]+v[:3]*.01),
                pose(Rotation.from_rotvec(v[9:12]*.1).as_matrix() @ Y0[:3, :3], Y0[:3, 3]+v[6:9]*.01))
    def blocks(v):
        X, Y = unpack(v)
        T = np.linalg.inv(X) @ A @ Y
        p = np.einsum('nij,kj->nki', T[:, :3, :3], points)+T[:, None, :3, 3]
        pixels = p[:, :, :2]/p[:, :, 2, None]*[K[0, 0], K[1, 1]]+K[:2, 2]
        return ((pixels-uv)/.15).reshape(len(A), -1)
    v = np.zeros(12)
    evaluations = initial['evaluations']
    history = []
    for iteration in range(6):
        weights = 1/(1+np.mean(blocks(v)**2, axis=1)/4.)
        history.append(weights.copy())
        fit = least_squares(lambda x: (blocks(x)*np.sqrt(weights[:, None])).ravel(), v,
                            max_nfev=70, ftol=1e-9, xtol=1e-9, gtol=1e-7, x_scale='jac')
        change = np.linalg.norm(fit.x-v)
        v = fit.x
        evaluations += fit.nfev
        if change < 1e-5:
            break
    X, Y = unpack(v)
    return dict(B_T_C=X, F_T_P=Y, accepted=bool(fit.success),
                reason='fixed_robot_whole_view_robust_calibration_candidate',
                evaluations=evaluations, view_weights=weights, view_weight_history=history,
                pixel_rms=float(np.sqrt(np.mean(blocks(v)**2))*.15),
                reported_robot_poses_fixed=True, covariance_not_calibrated=True)
