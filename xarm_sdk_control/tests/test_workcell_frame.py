from __future__ import annotations

import cv2
import numpy as np
import pytest

from xarm_sdk_control.workcell_frame import (
    build_workcell_frame,
    deproject_depth_roi_to_reference,
    fit_plane_ransac,
    height_above_workcell_plane,
)


def test_deproject_depth_roi_uses_intrinsics_and_depth_scale() -> None:
    depth = np.zeros((5, 6), dtype=np.uint16)
    depth[2, 3] = 1000
    depth[2, 4] = 1000
    depth[3, 3] = 1000
    matrix = np.array([[100.0, 0.0, 3.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]])
    points, pixels, fraction = deproject_depth_roi_to_reference(
        depth,
        0.001,
        matrix,
        np.zeros(5),
        np.eye(4),
        (3, 2, 2, 2),
    )
    assert fraction == pytest.approx(0.75)
    assert pixels.tolist() == [[3, 2], [4, 2], [3, 3]]
    np.testing.assert_allclose(points, [[0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [0.0, 0.01, 1.0]])


def test_ransac_plane_and_workcell_height_reject_outliers() -> None:
    rng = np.random.default_rng(8)
    x, y = np.meshgrid(np.linspace(-0.15, 0.15, 25), np.linspace(-0.10, 0.10, 20))
    z = 0.60 + 0.04 * x - 0.03 * y + rng.normal(0.0, 0.00025, x.shape)
    plane_points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    outliers = rng.uniform([-0.15, -0.10, 0.50], [0.15, 0.10, 0.70], size=(40, 3))
    points = np.vstack((plane_points, outliers))
    fit = fit_plane_ransac(
        points,
        distance_threshold_m=0.001,
        minimum_inlier_ratio=0.85,
        toward_point_m=np.zeros(3),
    )
    expected_normal = np.array([0.04, -0.03, -1.0])
    expected_normal /= np.linalg.norm(expected_normal)
    np.testing.assert_allclose(fit.normal_toward_camera, expected_normal, atol=0.003)
    assert np.mean(fit.inlier_mask) > 0.90
    assert fit.rms_residual_m < 0.0004

    frame = build_workcell_frame(fit, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(frame.transform_reference_from_workcell[:3, :3].T @ frame.transform_reference_from_workcell[:3, :3], np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(frame.transform_reference_from_workcell[:3, :3]), 1.0, atol=1e-10)
    point_above = fit.point_m + 0.125 * fit.normal_toward_camera
    assert height_above_workcell_plane(point_above, frame) == pytest.approx(0.125)
