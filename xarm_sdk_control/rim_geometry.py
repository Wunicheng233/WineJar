"""Calibrated multiview rim fitting in metric coordinates, without row-height lookup."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from xarm_sdk_control.multiview_rig import FixedCameraRig


@dataclass(frozen=True)
class RimObservation:
    camera_id: str
    pixels: np.ndarray
    inner: bool = False
    horizontal_extent_px: tuple[float, float] | None = None


@dataclass(frozen=True)
class RimFit:
    center_m: np.ndarray
    normal: np.ndarray
    inner_radius_m: float
    outer_radius_m: float
    residuals_px: dict[str, np.ndarray]
    parameter_std: np.ndarray
    condition_number: float
    at_bound: bool


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    seed = np.eye(3)[np.argmin(np.abs(n))]
    u = np.cross(seed, n)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def circle_points(
    center: np.ndarray, normal: np.ndarray, radius: float, count: int = 360
) -> np.ndarray:
    u, v = plane_basis(normal)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return center + radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)


def project_workcell(
    rig: FixedCameraRig,
    camera_id: str,
    transform_reference_from_workcell: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    transform = np.asarray(transform_reference_from_workcell)
    reference = np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]
    return rig.project_reference_points(camera_id, reference)


def pixel_rays_workcell(
    rig: FixedCameraRig,
    camera_id: str,
    transform_workcell_from_reference: np.ndarray,
    pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera = rig.camera(camera_id)
    normalized = cv2.undistortPoints(
        np.asarray(pixels, float).reshape(-1, 1, 2),
        camera.camera_matrix,
        camera.distortion_coefficients,
    ).reshape(-1, 2)
    transform = (
        transform_workcell_from_reference @ camera.transform_reference_from_camera
    )
    rays = np.column_stack((normalized, np.ones(len(normalized)))) @ transform[:3, :3].T
    return transform[:3, 3], rays


def intersect_height(
    rig: FixedCameraRig,
    camera_id: str,
    transform_workcell_from_reference: np.ndarray,
    pixels: np.ndarray,
    height_m: float,
) -> np.ndarray:
    origin, rays = pixel_rays_workcell(
        rig, camera_id, transform_workcell_from_reference, pixels
    )
    if np.any(np.abs(rays[:, 2]) < 1e-6):
        raise ValueError("Viewing ray is parallel to the requested height plane")
    distances = (height_m - origin[2]) / rays[:, 2]
    if np.any(distances <= 0):
        raise ValueError("Requested plane lies behind the camera")
    return origin + distances[:, None] * rays


def _parameters_to_rim(
    parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    normal = np.r_[parameters[3:5], 1.0]
    normal /= np.linalg.norm(normal)
    return parameters[:3], normal, parameters[5], parameters[5] + parameters[6]


def fit_multiview_rim(
    rig: FixedCameraRig,
    transform_reference_from_workcell: np.ndarray,
    observations: list[RimObservation],
    initial_center_m: np.ndarray,
    initial_inner_radius_m: float,
    initial_wall_m: float = 0.012,
    max_tilt_deg: float = 30,
) -> RimFit:
    """Fit concentric inner/outer rim circles in a shared, possibly tilted plane.

    Image residuals use the projected circle conic and its gradient (Sampson
    distance), with robust least squares. Side-view spans constrain outer radius.
    No depth observation is treated as an exact metric surface measurement.
    """
    if len({o.camera_id for o in observations}) < 3 or not any(
        o.inner for o in observations
    ):
        raise ValueError(
            "A top aperture and two independent side observations are required"
        )
    prepared = []
    for obs in observations:
        xy = np.asarray(obs.pixels, float).reshape(-1, 2)
        if len(xy) < 8 or not np.all(np.isfinite(xy)):
            raise ValueError(f"{obs.camera_id}: insufficient finite rim observations")
        camera = rig.camera(obs.camera_id)
        xy_undistorted = cv2.undistortPoints(
            xy[:, None],
            camera.camera_matrix,
            camera.distortion_coefficients,
            P=camera.camera_matrix,
        ).reshape(-1, 2)
        camera_from_reference = camera.transform_camera_from_reference
        tcw = camera_from_reference @ transform_reference_from_workcell
        rvec, _ = cv2.Rodrigues(camera_from_reference[:3, :3])
        prepared.append((
            obs, np.column_stack((xy_undistorted, np.ones(len(xy)))), tcw,
            camera, rvec, camera_from_reference[:3, 3], np.sqrt(80 / len(xy)),
        ))

    # Fixed within one fit, including its three initializations. Keep caches local
    # so a later call with changed calibration always prepares fresh geometry.
    circle_conic = np.diag([1.0, 1.0, -1.0])
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    circle_cos, circle_sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
    has_extents = any(obs.horizontal_extent_px is not None for obs in observations)
    transform = np.asarray(transform_reference_from_workcell)

    def residuals(parameters: np.ndarray, weighted: bool = True) -> list[np.ndarray]:
        center, normal, inner, outer = _parameters_to_rim(parameters)
        u, v = plane_basis(normal)
        if weighted and has_extents:
            # All side views project the same outer circle. Preserve the original
            # world -> reference -> camera operation order for numerical parity.
            circle = center + outer * (circle_cos * u + circle_sin * v)
            reference_circle = circle @ transform[:3, :3].T + transform[:3, 3]
        residual = []
        for obs, homogeneous, tcw, camera, rvec, translation, weight in prepared:
            radius = inner if obs.inner else outer
            center_camera = tcw[:3, :3] @ center + tcw[:3, 3]
            homography = camera.camera_matrix @ np.column_stack(
                (tcw[:3, :3] @ u * radius, tcw[:3, :3] @ v * radius, center_camera)
            )
            inverse = np.linalg.pinv(homography)
            conic = inverse.T @ circle_conic @ inverse
            conic_points = homogeneous @ conic
            gradient = 2 * conic_points
            errors = np.sum(homogeneous * conic_points, axis=1) / np.maximum(
                np.linalg.norm(gradient[:, :2], axis=1), 1e-12
            )
            if np.any(center_camera[2] <= 0):
                errors += 1e4
            residual.append(errors * weight if weighted else errors)
            if weighted and obs.horizontal_extent_px is not None:
                projected, _ = cv2.projectPoints(
                    reference_circle, rvec, translation,
                    camera.camera_matrix, camera.distortion_coefficients,
                )
                projected = projected.reshape(-1, 2)
                residual.append(
                    2
                    * (
                        np.array([projected[:, 0].min(), projected[:, 0].max()])
                        - obs.horizontal_extent_px
                    )
                )
        return residual

    center = np.asarray(initial_center_m, float).reshape(3)
    if (
        not np.all(np.isfinite(center))
        or initial_inner_radius_m <= 0
        or not 0 < max_tilt_deg < 60
    ):
        raise ValueError("Invalid initial rim geometry")
    slope = np.tan(np.deg2rad(max_tilt_deg))
    lower = np.r_[
        center - [0.12, 0.12, 0.08],
        -slope,
        -slope,
        initial_inner_radius_m * 0.65,
        0.001,
    ]
    upper = np.r_[
        center + [0.12, 0.12, 0.08], slope, slope, initial_inner_radius_m * 1.4, 0.04
    ]
    candidates = []
    for tilt in ([0.04, 0.04], [-0.04, -0.04], [0.12, -0.12]):
        start = np.r_[center, tilt, initial_inner_radius_m, initial_wall_m]
        start = np.clip(start, lower + 1e-8, upper - 1e-8)
        fit = least_squares(
            lambda p: np.concatenate(residuals(p)),
            start,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            x_scale="jac",
            max_nfev=350,
        )
        if (
            fit.success
            and np.all(np.isfinite(fit.x))
            and np.linalg.norm(fit.x[3:5]) <= slope
        ):
            candidates.append(fit)
    if not candidates:
        raise ValueError("Multiview rim fit did not converge")
    fit = min(candidates, key=lambda f: f.cost)
    center, normal, inner, outer = _parameters_to_rim(fit.x)
    residual_values = residuals(fit.x, weighted=False)
    errors = {
        entry[0].camera_id: error for entry, error in zip(prepared, residual_values)
    }
    # Conditional fit uncertainty assumes calibrations and boundary labels are exact.
    variance = float(np.sum(fit.fun**2) / max(1, len(fit.fun) - len(fit.x)))
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * variance
    column_norm = np.maximum(np.linalg.norm(fit.jac, axis=0), 1e-12)
    condition = float(np.linalg.cond(fit.jac / column_norm))
    bound = bool(
        np.any(np.minimum(fit.x - lower, upper - fit.x) / (upper - lower) < 0.001)
    )
    return RimFit(
        center,
        normal,
        inner,
        outer,
        errors,
        np.sqrt(np.maximum(0, np.diag(covariance))),
        condition,
        bound,
    )


def surface_point_at_silhouette(
    rig: FixedCameraRig,
    camera_id: str,
    transform_workcell_from_reference: np.ndarray,
    pixels: np.ndarray,
    rim: RimFit,
) -> tuple[np.ndarray, np.ndarray]:
    """Find ray/axis closest approach at a cylinder silhouette, retaining ray miss.

    Unlike row lookup this uses both pixel coordinates and the actual viewing
    direction. The shortest distance must agree with the fitted outer radius.
    """
    origin, rays = pixel_rays_workcell(
        rig, camera_id, transform_workcell_from_reference, pixels
    )
    points, misses = [], []
    for ray in rays:
        system = np.column_stack((ray, -rim.normal))
        distance, axial = np.linalg.lstsq(system, rim.center_m - origin, rcond=None)[0]
        if distance <= 0:
            raise ValueError("Silhouette ray lies behind camera")
        surface = origin + ray * distance
        axis_point = rim.center_m + rim.normal * axial
        points.append(surface)
        misses.append(np.linalg.norm(surface - axis_point) - rim.outer_radius_m)
    return np.asarray(points), np.asarray(misses)
