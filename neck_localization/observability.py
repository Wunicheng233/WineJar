"""Local linearizations for analysis, not a locator or calibrated uncertainty model."""
import numpy as np

from .geometry import basis


def unprojection_jacobian(u, v, z, K):
    return np.array([[z/K[0, 0], 0, (u-K[0, 2])/K[0, 0]],
                     [0, z/K[1, 1], (v-K[1, 2])/K[1, 1]], [0, 0, 1.]])


def radial_linearization(points, origin, normal, profile, derivative, profile_basis):
    """g=rho-r(s), with ambient origin, 2D tangent-normal, profile coefficients.

    The ambient origin block includes axial gauge freedom. Actual optimization
    must fix an axis anchor convention, e.g. a_z=z_ref. No normals of cloud
    points are needed. This is not an orthogonal-distance fit for large errors.
    """
    normal = np.asarray(normal, float)
    normal = normal/np.linalg.norm(normal)
    delta = np.asarray(points)-origin
    s = delta @ normal
    radial = delta-s[:, None]*normal
    rho = np.linalg.norm(radial, axis=1)
    if np.any(rho < 1e-10):
        raise ValueError("Axis points have no defined radial direction")
    e = radial/rho[:, None]
    slope = derivative(s)
    v = e-slope[:, None]*normal
    U = np.array(basis(normal)).T
    return rho-profile(s), {
        "origin": -v,
        "normal_tangent": -(s+rho*slope)[:, None]*(e @ U),
        "profile": -profile_basis(s),
        "point": v,
        "normal_basis": U,
    }


def circle_information(angles):
    """Unit radial variance, unknown 2D centre AND radius; no visibility model."""
    angles = np.asarray(angles)
    J = -np.c_[np.cos(angles), np.sin(angles), np.ones(len(angles))]
    return J.T @ J
