"""Experimental joint local-axis/profile and per-camera axial bias model.

No target/model labels or known radii. Variables: XY shift [mm], tangent normal
coordinates, then camera biases [mm]. A positive bias is SUBTRACTED along the
camera pixel ray. Profile coefficients are eliminated with exact QR variable
projection. Calibration prior and assumed noise are explicit, not measured.
"""
import numpy as np
from scipy.linalg import qr, solve_triangular

from .depth_profile import ProfileRejection
from .support_feedback import WeightedAxisSplineProblem


def observation_rays(points_m, camera_ids, frames):
    """dp_world/dz_camera from OBSERVED points and supplied camera parameters."""
    points_m, camera_ids = np.asarray(points_m), np.asarray(camera_ids)
    rays = np.empty_like(points_m, dtype=float)
    for j, frame in enumerate(frames):
        mask = camera_ids == j
        camera = frame.camera
        local = (points_m[mask]-camera.T_world_from_camera[:3, 3]) @ camera.T_world_from_camera[:3, :3]
        if np.any(local[:, 2] <= 0):
            raise ValueError('Point behind measurement camera')
        rays[mask] = (points_m[mask]-camera.T_world_from_camera[:3, 3])/local[:, 2, None]
    return rays


class BiasAxisSplineProblem:
    def __init__(self, base, camera_ids, rays, *, sigma_data_mm=.1, sigma_bias_mm=np.inf):
        self.base = base
        self.ids = np.asarray(camera_ids, int)
        self.rays = np.asarray(rays, float)
        self.ncamera = int(self.ids.max())+1
        self.sigma_data_mm = float(sigma_data_mm)
        self.sigma_bias_mm = float(sigma_bias_mm)
        if not self.sigma_data_mm > 0 or not self.sigma_bias_mm > 0:
            raise ValueError('Positive noise/prior scales required')
        if self.rays.shape != self.base.points.shape or not np.all(np.isfinite(self.rays)):
            raise ValueError('Invalid observation rays')
        self.assignment = np.eye(self.ncamera)[self.ids]
        self._last = None

    def evaluate(self, x):
        x = np.asarray(x)
        if self._last is not None and np.array_equal(x, self._last[0]):
            return self._last[1]
        base = self.base
        points = base.points-x[4:][self.ids, None]*self.rays
        a, n, dn = base.axis(x[:4])
        delta = points-a
        s = delta @ n
        radial = delta-s[:, None]*n
        rho = np.linalg.norm(radial, axis=1)
        if np.min(rho) < 1e-5:
            raise ProfileRejection('bias_corrected_axis_intersects_points')
        e = radial/rho[:, None]
        B, Bprime = base.basis(s), base.derivative(s)
        sw = base.sqrt_weight
        A = np.vstack((sw[:, None]*B, base.regularizer))
        b = np.r_[sw*rho, np.zeros(len(base.regularizer))]
        Q, R = qr(A, mode='economic', check_finite=False)
        if np.min(np.abs(np.diag(R))) < 1e-9:
            raise ProfileRejection('bias_profile_rank_deficient')
        beta = solve_triangular(R, Q.T @ b, check_finite=False)
        residual = b-A @ beta
        ds = np.c_[-np.tile(n[:2], (len(s), 1)), delta @ dn,
                   -(self.rays @ n)[:, None]*self.assignment]
        drho = np.c_[-e[:, :2], -s[:, None]*(e @ dn),
                     -np.sum(e*self.rays, axis=1)[:, None]*self.assignment]
        raw = sw[:, None]*(drho-(Bprime @ beta)[:, None]*ds)
        full = np.vstack((raw, np.zeros((len(base.regularizer), len(x)))))
        second = Bprime.T @ ((sw*residual[:len(s)])[:, None]*ds)
        jac = full-Q @ (Q.T @ full)-Q @ solve_triangular(R.T, second, lower=True, check_finite=False)
        # In residual-mm units; data sigma cancels in fit without calibration
        # prior. This is an assumed prior about SENSOR parameters, not jar size.
        if np.isfinite(self.sigma_bias_mm):
            strength = self.sigma_data_mm/self.sigma_bias_mm
            residual = np.r_[residual, strength*x[4:]]
            jac = np.vstack((jac, np.c_[np.zeros((self.ncamera, 4)), strength*np.eye(self.ncamera)]))
        result = dict(residual=residual, jacobian=jac, beta=beta, s=s, normal=n, origin=a,
                      corrected_points_mm=points, radial_residual=rho-B @ beta,
                      data_profile_basis=sw[:, None]*B, data_joint_jacobian=raw,
                      objective=float(residual @ residual))
        self._last = x.copy(), result
        return result

    def fun(self, x):
        return self.evaluate(x)['residual']

    def jac(self, x):
        return self.evaluate(x)['jacobian']

    def data_information(self, x):
        """Profile-nuisance-projected DATA only, no smoothness/bias prior rows.

        Singular values use fixed parameter scales to avoid comparing mm with
        radians directly. Full numerical rank does not imply noise robustness.
        """
        state = self.evaluate(x)
        B, J = state['data_profile_basis'], state['data_joint_jacobian']
        U, sv, _ = np.linalg.svd(B, full_matrices=False)
        Q = U[:, sv > sv[0]*1e-8]
        projected = J-Q @ (Q.T @ J)
        scales = np.array([5, 5, .1, .1]+[2]*self.ncamera)
        _, singular, vt = np.linalg.svd(projected*scales, full_matrices=False)
        return dict(scaled_data_singular_values=singular.tolist(),
                    scaled_data_condition=float(singular[0]/max(singular[-1], 1e-15)),
                    parameter_scales=scales.tolist(), weakest_scaled_direction=vt[-1].tolist(),
                    information_semantics='local weighted data Jacobian, profile nuisance projected, no priors; not calibrated uncertainty')
