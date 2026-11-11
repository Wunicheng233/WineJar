"""CPU checks for observation-only preprocessing and geometric bookkeeping."""
import unittest

import numpy as np

from neck_localization.point_restoration_baseline import (
    average_patch_predictions, normalize_unit_sphere, observation_cloud, patch_indices,
)


class PointRestorationAdapterTests(unittest.TestCase):
    def test_mature_foreground_preserves_dark_neck_pixels(self):
        from neck_localization.depth_profile import DepthFrame, rgb_foreground
        from neck_localization.geometry import Camera
        from neck_localization.stereo_robust import blue_screen_foreground
        rgb = np.full((40, 40, 3), [0, 0, 255], dtype=np.uint8)
        rgb[8:32, 10:30] = 8
        camera = Camera("cam_a", np.array([[100., 0., 20.], [0., 100., 20.], [0., 0., 1.]]),
                        np.eye(4), 40, 40)
        frame = DepthFrame(camera, rgb, np.full((40, 40), .6))
        points, camera_ids, uv, metadata = observation_cloud([frame])
        self.assertEqual(len(points), 24 * 20)
        self.assertEqual(int(rgb_foreground(rgb).sum()), 0)
        self.assertEqual(len(points), int(blue_screen_foreground(rgb).sum()))
        self.assertEqual(len(metadata["foreground"]["cameras"]), 1)
        self.assertEqual(len(metadata["foreground_settings_sha256"]), 64)

    def test_unit_sphere_reversible_with_metric_scale(self):
        points = np.array([[0.1, -0.2, 0.6], [0.2, 0.1, 0.64], [-0.1, 0.15, 0.55]])
        normalized, center, scale = normalize_unit_sphere(points)
        np.testing.assert_allclose(normalized * scale + center, points, atol=1e-14)
        self.assertAlmostEqual(float(np.linalg.norm(normalized, axis=1).max()), 1.)
        with self.assertRaises(ValueError):
            normalize_unit_sphere(np.ones((3, 3)))

    def test_repeated_small_input_preserves_point_identity(self):
        points = np.array([[0., 0., 0.], [0.1, 0., 0.], [0., 0.1, 0.]])
        indices = patch_indices(points, patch_size=2048)
        output, counts = average_patch_predictions(points, indices, points[indices])
        self.assertEqual(indices.shape, (1, 2048))
        self.assertTrue(np.all(counts > 0))
        np.testing.assert_allclose(output, points)

    def test_uncovered_points_remain_and_overlaps_average(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        indices = np.array([[0, 1], [1, 0]])
        predictions = np.array([[[0., 0., 0.], [1., 2., 0.]],
                                [[1., 4., 0.], [0., 0., 0.]]])
        output, counts = average_patch_predictions(points, indices, predictions)
        np.testing.assert_allclose(output, [[0., 0., 0.], [1., 3., 0.], [2., 0., 0.]])
        np.testing.assert_array_equal(counts, [2, 2, 0])


if __name__ == "__main__":
    unittest.main()
