"""Displayed station points must match the frozen estimator's actual input."""
import unittest

import numpy as np

from neck_localization.depth_profile import DepthFrame, ProfileSettings, _point_cloud
from neck_localization.geometry import Camera
from neck_localization.point_cloud_display import observation_cloud, display_samples


class ObservationCloudTests(unittest.TestCase):
    def test_saved_observation_mask_can_preserve_dark_neck_without_changing_rgb(self):
        camera = Camera('a', np.eye(3), np.eye(4), 4, 3)
        rgb = np.zeros((3, 4, 3), np.uint8)
        depth = np.full((3, 4), .5)
        mask = np.zeros((3, 4), bool)
        mask[1, 2] = True
        cloud = observation_cloud(camera, rgb, depth, foreground_mask=mask)
        np.testing.assert_allclose(cloud.points, camera.unproject([[2, 1]], [.5]))
        np.testing.assert_array_equal(cloud.colors, [[0, 0, 0]])
        np.testing.assert_array_equal(rgb, np.zeros_like(rgb))
        with self.assertRaises(ValueError):
            observation_cloud(camera, rgb, depth, foreground_mask=np.zeros((2, 4), bool))

    def test_station_cloud_matches_estimator_without_truth_or_completion(self):
        pose = np.diag([1., -1., -1., 1.]); pose[:3, 3] = [0, 0, 1]
        camera = Camera("a", np.array([[80., 0, 19.5], [0, 80., 19.5], [0, 0, 1]]), pose, 40, 40)
        rgb = np.full((40, 40, 3), 120, np.uint8)
        rgb[:2, :, :] = [0, 0, 200]
        depth = np.full((40, 40), .5)
        depth[3, 0:3] = [0, np.nan, 1.]
        settings = ProfileSettings()
        cloud = observation_cloud(camera, rgb, depth, settings)
        expected, _ = _point_cloud([DepthFrame(camera, rgb, depth)], "rgb", settings)
        np.testing.assert_array_equal(cloud.points[cloud.station_mask], expected)
        sampled, _, total = display_samples([cloud], {"a"}, True, maximum_per_camera=100)
        self.assertEqual(total, len(expected))
        self.assertLessEqual(len(sampled), 100)
        # Display thinning is a subset, never newly generated/interpolated data.
        stride = int(np.ceil(len(expected)/100))
        np.testing.assert_allclose(sampled, expected[::stride], atol=2e-8)


if __name__ == "__main__":
    unittest.main()
