from __future__ import annotations

import math
import unittest

import numpy as np

from conftest import load_script_module
from xarm_sdk_control.calibration_utils import (
    residuals_m,
    solve_rigid_transform,
    transform_points,
)


COLLECTOR = load_script_module(
    "global_camera_collector",
    "xarm_sdk_control/scripts/11_collect_global_camera_sample.py",
)


class FakeDepthFrame:
    def __init__(self, depth_m):
        self.depth_m = np.asarray(depth_m, dtype=float)

    def get_width(self):
        return self.depth_m.shape[1]

    def get_height(self):
        return self.depth_m.shape[0]

    def get_distance(self, x, y):
        return float(self.depth_m[y, x])


def rz(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


class GlobalCameraCalibrationTest(unittest.TestCase):
    def test_depth_statistics_keeps_clicked_foreground_layer(self):
        depth = np.full((9, 9), 0.60)
        depth[3:6, 3:6] = 0.30
        median_m, mad_m, count = COLLECTOR.depth_statistics(FakeDepthFrame(depth), 4, 4, 4)
        self.assertAlmostEqual(median_m, 0.30)
        self.assertAlmostEqual(mad_m, 0.0)
        self.assertEqual(count, 9)

    def test_depth_statistics_rejects_missing_center_instead_of_background(self):
        depth = np.full((9, 9), 0.60)
        depth[4, 4] = 0.0
        median_m, mad_m, count = COLLECTOR.depth_statistics(FakeDepthFrame(depth), 4, 4, 4)
        self.assertEqual(median_m, 0.0)
        self.assertTrue(np.isinf(mad_m))
        self.assertEqual(count, 0)

    def test_marker_offset_is_transformed_by_tcp_orientation(self):
        tcp_pose = [300.0, 100.0, 500.0, 0.0, 0.0, 90.0]
        point = COLLECTOR.marker_point_in_base_m(tcp_pose, np.array([20.0, 0.0, 0.0]))
        np.testing.assert_allclose(point, [0.30, 0.12, 0.50], atol=1e-12)

    def test_solve_rigid_transform_recovers_known_transform(self):
        camera_points = np.array(
            [
                [0.10, 0.20, 0.50],
                [0.25, 0.10, 0.55],
                [0.05, -0.15, 0.60],
                [0.30, -0.05, 0.48],
                [0.18, 0.22, 0.70],
            ],
            dtype=np.float64,
        )
        expected = np.eye(4, dtype=np.float64)
        expected[:3, :3] = rz(math.radians(32.0))
        expected[:3, 3] = np.array([0.31, -0.12, 0.08], dtype=np.float64)
        base_points = transform_points(expected, camera_points)

        actual = solve_rigid_transform(camera_points, base_points)

        np.testing.assert_allclose(actual, expected, atol=1e-10)
        np.testing.assert_allclose(residuals_m(actual, camera_points, base_points), 0.0, atol=1e-10)

    def test_solve_rigid_transform_rejects_degenerate_points(self):
        camera_points = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3]])
        base_points = camera_points + np.array([0.1, 0.2, 0.3])

        with self.assertRaisesRegex(ValueError, "degenerate"):
            solve_rigid_transform(camera_points, base_points)


if __name__ == "__main__":
    unittest.main()
