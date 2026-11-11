"""Virtual board calibration from synthetic corner coordinates, without target truth.

This isolates calibration geometry and corner noise. It is not an end-to-end
rendered chessboard experiment: visibility, identity and corner extraction are
idealized, including the availability of identified corners on either side of
the virtual board. Camera 0 fixes the world gauge; its intrinsics are estimated.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import Camera


def _board_points() -> np.ndarray:
    x, y = np.meshgrid(np.arange(9), np.arange(6))
    points = np.c_[x.ravel() - 4, y.ravel() - 2.5, np.zeros(x.size)]
    return points.astype(np.float64) * 0.020


def _common_view_center(cameras: list[Camera]) -> np.ndarray:
    """Used only by the measurement simulator to place boards in shared view."""
    a, b = np.zeros((3, 3)), np.zeros(3)
    for camera in cameras:
        forward = camera.T_world_from_camera[:3, 2]
        projector = np.eye(3) - np.outer(forward, forward)
        a += projector
        b += projector @ camera.T_world_from_camera[:3, 3]
    if np.linalg.cond(a) > 1e8:
        camera = cameras[0]
        return camera.T_world_from_camera[:3, 3] + 0.6 * camera.T_world_from_camera[:3, 2]
    return np.linalg.solve(a, b)


def _simulate_boards(cameras, rng, count, physical_points):
    """Generate synchronously observed corners; true board poses stay here."""
    center = _common_view_center(cameras)
    reference_rotation = cameras[0].T_world_from_camera[:3, :3]
    observations = []
    for _ in range(5000):
        angles = rng.uniform([-45, -45, -170], [45, 45, 170])
        rotation = reference_rotation @ Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        translation = center + reference_rotation @ rng.uniform([-0.055, -0.055, -0.070], [0.055, 0.055, 0.070])
        world_points = physical_points @ rotation.T + translation
        pixels_by_camera = []
        visible = True
        for camera in cameras:
            points_camera = (world_points - camera.T_world_from_camera[:3, 3]) @ camera.T_world_from_camera[:3, :3]
            pixels = camera.project(world_points)
            if not (np.all(points_camera[:, 2] > 0.10)
                    and np.all(pixels[:, 0] > 5) and np.all(pixels[:, 0] < camera.width - 6)
                    and np.all(pixels[:, 1] > 5) and np.all(pixels[:, 1] < camera.height - 6)):
                visible = False
                break
            # Reject almost edge-on boards: corner identity is idealized, but
            # planar calibration still needs a measurable projected area.
            normal_camera = rotation[:, 2] @ camera.T_world_from_camera[:3, :3]
            if abs(normal_camera[2]) < 0.16:
                visible = False
                break
            pixels_by_camera.append(pixels)
        if visible:
            observations.append(pixels_by_camera)
            if len(observations) == count:
                return observations
    raise ValueError("virtual_calibration_insufficient_shared_board_visibility")


def _pnp(points, pixels, k):
    success, rvec, tvec = cv2.solvePnP(
        points.astype(np.float64), pixels.astype(np.float64), k, np.zeros(5),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise ValueError("virtual_calibration_pnp_failed")
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = tvec.ravel()
    return transform


def _rms(errors):
    return float(np.sqrt(np.mean(np.sum(np.asarray(errors) ** 2, axis=-1))))


def _estimate_rig(corner_observations, nominal_points, image_sizes):
    """Estimator boundary: receives nominal board coordinates and pixels only."""
    intrinsics, training_rms, board_poses = [], [], []
    flags = (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1
             | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    for camera_index, size in enumerate(image_sizes):
        object_points = [nominal_points.astype(np.float32) for _ in corner_observations]
        image_points = [pose[camera_index].astype(np.float32) for pose in corner_observations]
        rms, k, _, _, _ = cv2.calibrateCamera(
            object_points, image_points, size, None, None, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 150, 1e-10),
        )
        intrinsics.append(k)
        training_rms.append(float(rms))
        board_poses.append([_pnp(nominal_points, pose[camera_index], k) for pose in corner_observations])
    relative = [np.eye(4)]
    for index in range(1, len(image_sizes)):
        transforms = [pose @ np.linalg.inv(reference) for pose, reference in zip(board_poses[index], board_poses[0])]
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_matrix(np.stack([t[:3, :3] for t in transforms])).mean().as_matrix()
        transform[:3, 3] = np.mean([t[:3, 3] for t in transforms], axis=0)
        relative.append(transform)
    return intrinsics, relative, training_rms


def virtual_calibrate(cameras: list[Camera], seed=20261021, corner_sigma_px=0.1,
                      board_scale_error=0.0, board_warp_m=0.0):
    """Return estimated cameras and calibration diagnostics.

    ``board_scale_error`` is the fractional physical spacing error relative to
    the assumed 20 mm board; +0.01 means a true spacing of 20.2 mm. ``board_warp_m``
    is a deterministic quadratic departure from planarity, zero at the board
    center and reaching the stated displacement at the extreme corners. The
    solver always receives the nominal flat board. No ring parameters enter.

    Measurement simulation uses true camera parameters. Only reference camera
    0's world transform is passed to the gauge placement step after calibration.
    True camera parameters reappear solely in diagnostics labeled evaluation.
    """
    if len(cameras) < 2:
        raise ValueError("virtual_calibration_requires_at_least_two_cameras")
    if corner_sigma_px < 0 or board_scale_error <= -1:
        raise ValueError("invalid_virtual_calibration_noise_or_scale")
    points = _board_points()
    physical_points = points * (1 + board_scale_error)
    normalized_xy = points[:, :2] / np.max(np.abs(points[:, :2]), axis=0)
    physical_points[:, 2] = board_warp_m * np.sum(normalized_xy ** 2, axis=1) / 2
    rng = np.random.default_rng(seed)
    training_clean = _simulate_boards(cameras, rng, 28, physical_points)
    training = [[pixels + rng.normal(0, corner_sigma_px, pixels.shape) for pixels in pose]
                for pose in training_clean]
    heldout_rng = np.random.default_rng(seed + 100003)
    heldout_clean = _simulate_boards(cameras, heldout_rng, 10, physical_points)
    heldout = [[pixels + heldout_rng.normal(0, corner_sigma_px, pixels.shape) for pixels in pose]
               for pose in heldout_clean]
    sizes = [(c.width, c.height) for c in cameras]
    intrinsics, relative, training_rms = _estimate_rig(training, points, sizes)
    reference_gauge = cameras[0].T_world_from_camera.copy()
    estimated = [Camera(c.name, k, reference_gauge @ np.linalg.inv(t), c.width, c.height)
                 for c, k, t in zip(cameras, intrinsics, relative)]

    heldout_independent, heldout_rig = [[] for _ in cameras], [[] for _ in cameras]
    for pose in heldout:
        reference_from_board = _pnp(points, pose[0], intrinsics[0])
        for index, k in enumerate(intrinsics):
            own_from_board = _pnp(points, pose[index], k)
            own_points = points @ own_from_board[:3, :3].T + own_from_board[:3, 3]
            own_projected = own_points @ k.T
            heldout_independent[index].append(own_projected[:, :2] / own_projected[:, 2:] - pose[index])
            camera_from_board = relative[index] @ reference_from_board
            camera_points = points @ camera_from_board[:3, :3].T + camera_from_board[:3, 3]
            projected = camera_points @ k.T
            heldout_rig[index].append(projected[:, :2] / projected[:, 2:] - pose[index])

    evaluation = []
    for true_camera, recovered in zip(cameras, estimated):
        true_relative = np.linalg.inv(true_camera.T_world_from_camera) @ reference_gauge
        recovered_relative = np.linalg.inv(recovered.T_world_from_camera) @ reference_gauge
        delta_rotation = recovered_relative[:3, :3] @ true_relative[:3, :3].T
        evaluation.append({
            "name": true_camera.name,
            "fx_error_px": float(recovered.K[0, 0] - true_camera.K[0, 0]),
            "fy_error_px": float(recovered.K[1, 1] - true_camera.K[1, 1]),
            "principal_point_error_px": float(np.linalg.norm(recovered.K[:2, 2] - true_camera.K[:2, 2])),
            "relative_rotation_error_deg": float(np.rad2deg(Rotation.from_matrix(delta_rotation).magnitude())),
            "relative_translation_error_mm": float(np.linalg.norm(recovered_relative[:3, 3] - true_relative[:3, 3]) * 1000),
            "true_baseline_mm": float(np.linalg.norm(true_relative[:3, 3]) * 1000),
            "estimated_baseline_mm": float(np.linalg.norm(recovered_relative[:3, 3]) * 1000),
        })
    diagnostics = {
        "kind": "synthetic_projected_board_corners",
        "not_end_to_end_rendered_corner_detection": True,
        "world_gauge": "camera0_true_world_transform_only",
        "board": {"inner_corners": [9, 6], "assumed_square_m": 0.020,
                  "physical_spacing_fractional_error": float(board_scale_error),
                  "physical_quadratic_warp_m": float(board_warp_m),
                  "identified_corners_both_board_sides": "idealized"},
        "seed": int(seed), "corner_sigma_px": float(corner_sigma_px),
        "training_board_poses": len(training), "heldout_board_poses": len(heldout),
        "training_reprojection_rms_px": training_rms,
        "heldout_per_camera_pnp_reprojection_rms_px": [_rms(e) for e in heldout_independent],
        "heldout_rig_reference_pnp_reprojection_rms_px": [_rms(e) for e in heldout_rig],
        "evaluation_truth_only": evaluation,
    }
    return estimated, diagnostics
