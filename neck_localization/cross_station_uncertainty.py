"""Measurement-level station uncertainty. No hardware or operational vision.

SE(3) composition follows Barfoot/Furgale; uncertainty assignment/propagation
follows GUM/JCGM 101. Numerical priors MUST come from the supplied protocol;
the included protocol contains assumption scans, not measured device errors.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .station_tying import pose


def perturb(translation_m, rotation_rad):
    """Rigid transform with explicit translation and rotation-vector coordinates.

    This is a coordinate chart, NOT a claim that translation equals the
    translational Lie-algebra component of an exact coupled SE(3) exponential.
    """
    return pose(Rotation.from_rotvec(rotation_rad).as_matrix(), translation_m)


def fit_rigid_points(source, measured, sigma_m, *, robust=False):
    """Weighted point registration; known artifact points are calibration data.

    Optimisation pivot is the source centroid. Returns a 6x6 LOCAL noise-only
    covariance in (centroid translation, left rotation) coordinates; fixed
    survey bias/artifact error is NOT magically included in that covariance.
    """
    p, q = np.asarray(source, float), np.asarray(measured, float)
    sigma = np.broadcast_to(np.asarray(sigma_m, float), q.shape)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape != q.shape or len(p) < 4:
        raise ValueError('at least four paired 3D calibration points required')
    if not np.isfinite(np.r_[p.ravel(), q.ravel(), sigma.ravel()]).all() or np.any(sigma <= 0):
        raise ValueError('finite measurements and positive noise scales required')
    if np.linalg.matrix_rank(p-p.mean(axis=0), tol=1e-10) < 2:
        raise ValueError('collinear calibration points do not determine a rigid pose')
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    u, _, vt = np.linalg.svd((p-pc).T @ (q-qc))
    r0 = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
    def decode(x):
        r = Rotation.from_rotvec(x[3:]).as_matrix() @ r0
        return r, qc+x[:3]
    def residual(x):
        r, centroid = decode(x)
        return (((p-pc) @ r.T+centroid-q)/sigma).ravel()
    fit = least_squares(residual, np.zeros(6), loss='huber' if robust else 'linear', f_scale=1.345,
                        xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=180)
    r, centroid = decode(fit.x)
    T = pose(r, centroid-r @ pc)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac)
    return dict(transform=T, normalized_rms=float(np.sqrt(np.mean(residual(fit.x)**2))),
                source_centroid_m=pc, local_noise_only_covariance=covariance,
                information_condition=float((singular[0]/singular[-1])**2), converged=bool(fit.success),
                robust=robust, covariance_excludes_fixed_survey_and_artifact_bias=True)


def reference_points(layout='wide'):
    # A virtual external coordinate survey, NOT robot probing beyond its reach.
    x = [0., .8, 1.6, 2.4] if layout == 'wide' else [0., .03, .06, .09]
    return np.array([[a, b, c] for a in x for b, c in [(-.2, .05), (.2, .05), (-.2, .65), (.2, .65)]])


def tool_points():
    return np.array([[x, y, z] for x in (-.08, .08) for y in (-.06, .06) for z in (-.015, .035)])


def encoder_reading(s_m, *, scale_error, cyclic_amplitude_m, cyclic_period_m, phase_rad, tick_m):
    if tick_m <= 0 or cyclic_period_m <= 0 or abs(scale_error) >= 1:
        raise ValueError('invalid encoder parameter')
    analog = np.asarray(s_m)*(1+scale_error)+cyclic_amplitude_m*np.sin(2*np.pi*np.asarray(s_m)/cyclic_period_m+phase_rad)
    return np.rint(analog/tick_m)*tick_m


def fit_encoder_scale(readout_m, surveyed_m):
    """A through-origin scale for DIFFERENCES, so counter zero offsets cancel."""
    x, y = np.asarray(readout_m, float), np.asarray(surveyed_m, float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 3 or x @ x <= 1e-12:
        raise ValueError('independent calibration displacement pairs required')
    return float(x @ y/(x @ x))


def executed_circle(G0, B_T_W_true, B_T_W_estimated, F_T_G_true, F_T_G_estimated,
                    measured_displacement, measured_direction=(1, 0, 0)):
    """Actual guide pose after commanding a flange through estimated calibration.

    Robot tracks the commanded flange exactly in this stage. It can be extended
    with externally measured robot FK residuals; no made-up robot accuracy is
    hidden inside the camera or encoder noise.
    """
    moved = np.asarray(G0).copy()
    moved[:3, 3] += measured_displacement*np.asarray(measured_direction)
    B_T_F_command = B_T_W_estimated @ moved @ np.linalg.inv(F_T_G_estimated)
    return np.linalg.inv(B_T_W_true) @ B_T_F_command @ F_T_G_true


def relabel_world(B_T_W, M_W, W_T_G, H_new_from_old):
    """One shared gauge relabels ALL world-referenced quantities consistently."""
    H = H_new_from_old
    return B_T_W @ np.linalg.inv(H), H @ M_W @ np.linalg.inv(H), H @ W_T_G


def evaluate_arrays(centers, normals, radii, truth_centers, truth_normals, neck_lengths, width_m=.003):
    d = centers-truth_centers
    dot = np.clip(np.sum(normals*truth_normals, axis=-1), -1, 1)
    sine = np.linalg.norm(np.cross(normals, truth_normals), axis=-1)
    axial = np.sum(d*truth_normals, axis=-1)
    margin = neck_lengths/2-abs(axial)-width_m/2*abs(dot)-radii*sine
    return dict(center_mm=np.linalg.norm(d, axis=-1)*1000, axial_mm=axial*1000,
                tangent_mm=np.linalg.norm(d-axial[..., None]*truth_normals, axis=-1)*1000,
                xyz_mm=d*1000, normal_deg=np.rad2deg(np.arctan2(sine, dot)),
                band_margin_mm=margin*1000, band_contained=margin >= -1e-12)
