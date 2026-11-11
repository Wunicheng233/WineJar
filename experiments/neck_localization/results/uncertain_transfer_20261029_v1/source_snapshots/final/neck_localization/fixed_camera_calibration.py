"""Offline eye-to-hand calibration, T_A_B maps B into A, metres/radians.

Static camera C, robot base B, flange F, rigid board P:
    B_T_C @ C_T_P[i] == B_T_F[i] @ F_T_P.
The board mount F_T_P is distinct from the gun mount F_T_G.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .station_tying import pose


def transform_points(T, points):
    return np.asarray(points) @ T[:3, :3].T+T[:3, 3]


def projection(T_C_P, points, K, distortion=None):
    uv, _ = cv2.projectPoints(np.asarray(points, float), cv2.Rodrigues(T_C_P[:3, :3])[0],
                            T_C_P[:3, 3], np.asarray(K, float), distortion)
    return uv.reshape(-1, 2)


def rotation_excitation(robot_poses):
    base = np.asarray(robot_poses)[0, :3, :3]
    rotvecs = np.array([Rotation.from_matrix(base.T @ t[:3, :3]).as_rotvec() for t in robot_poses[1:]])
    s = np.linalg.svd(rotvecs, compute_uv=False)
    return dict(singular_values_rad=s, independent_axes=bool(len(s) >= 2 and s[1] > .01),
                practical_secondary_excitation_floor_rad=.01)


def solve_eye_to_hand(robot_poses, image_points, board_points, K, *, method='park', distortion=None):
    """Observation-only solver: no actual camera/board/robot truth arguments.

    Pixel refinement holds reported robot poses and intrinsics fixed. It is
    therefore NOT a full errors-in-variables calibration of robot kinematics.
    """
    robot = np.asarray(robot_poses, float)
    if not hasattr(cv2, 'calibrateHandEye'):
        raise RuntimeError('This OpenCV build lacks calibrateHandEye; use the existing winejar-mujoco environment (OpenCV 4.13).')
    images, points = np.asarray(image_points, float), np.asarray(board_points, float)
    if len(robot) < 4 or images.shape != (len(robot), len(points), 2):
        raise ValueError('paired robot poses and identified board pixels required')
    excitation = rotation_excitation(robot)
    if not excitation['independent_axes']:
        return dict(accepted=False, reason='insufficient_nonparallel_rotation', excitation=excitation)
    measured = []
    for pixels in images:
        ok, rv, tv = cv2.solvePnP(points, pixels, K, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return dict(accepted=False, reason='pnp_failed', excitation=excitation)
        measured.append(pose(cv2.Rodrigues(rv)[0], tv.ravel()))
    # OpenCV's eye-to-hand equation uses F_T_B, i.e. inverse reported B_T_F.
    inverses = np.linalg.inv(robot)
    flag = cv2.CALIB_HAND_EYE_HORAUD if method == 'horaud' else cv2.CALIB_HAND_EYE_PARK
    r, t = cv2.calibrateHandEye(list(inverses[:, :3, :3]), list(inverses[:, :3, 3]),
                              [x[:3, :3] for x in measured], [x[:3, 3] for x in measured], method=flag)
    if not np.isfinite(r).all() or not np.isfinite(t).all():
        return dict(accepted=False, reason='nonfinite_closed_form', excitation=excitation)
    X = pose(r, t.ravel())
    mounts = inverses @ X @ np.array(measured)
    Y = pose(Rotation.from_matrix(mounts[:, :3, :3]).mean().as_matrix(), np.mean(mounts[:, :3, 3], axis=0))
    def pack(T):
        return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()]
    def unpack(a):
        return pose(Rotation.from_rotvec(a[3:]).as_matrix(), a[:3])
    def residual(v):
        xi, yi = unpack(v[:6]), unpack(v[6:])
        Ti = np.linalg.inv(xi) @ robot @ yi
        return np.concatenate([(projection(T, points, K, distortion)-pixels).ravel() for T, pixels in zip(Ti, images)])
    x0 = np.r_[pack(X), pack(Y)]
    optimized = None
    if method in ('pixel_ls', 'pixel_huber'):
        optimized = least_squares(residual, x0, loss='huber' if method == 'pixel_huber' else 'linear',
            f_scale=.5, x_scale='jac', max_nfev=70, ftol=1e-9, xtol=1e-10, gtol=1e-7)
        X, Y, x0 = unpack(optimized.x[:6]), unpack(optimized.x[6:]), optimized.x
    errors = residual(x0)
    return dict(accepted=optimized is None or bool(optimized.success), reason='ok' if optimized is None or optimized.success else 'optimizer_budget',
        B_T_C=X, F_T_P=Y, pixel_rms=float(np.sqrt(np.mean(errors**2))),
        evaluations=0 if optimized is None else int(optimized.nfev), excitation=excitation,
        robot_poses_treated_as_fixed=True, no_real_accuracy_or_confidence_claim=True)
