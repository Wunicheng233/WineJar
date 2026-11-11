"""CPU tests for paired cloud reprojection; no network, renderer or device."""
import unittest

import numpy as np

from neck_localization.depth_profile import DepthFrame
from neck_localization.geometry import Camera
from scripts.evaluate_capability_cloud import source_camera_zbuffers, read_cloud


class CloudReadoutTests(unittest.TestCase):
    def frames(self):
        K = np.array([[100., 0., 9.], [0., 100., 7.], [0., 0., 1.]])
        return [DepthFrame(Camera(name, K.copy(), np.eye(4), 20, 16),
                           np.zeros((16, 20, 3), np.uint8), None) for name in ("cam_a", "cam_b", "cam_c")]

    def test_source_view_only_zbuffer_quantization_and_no_hole_filling(self):
        points = np.array([[0., 0., .8004], [0., 0., .9001], [0., 0., 1.2001],
                           [1., 0., .8], [0., 0., -.8]])
        original = points.copy()
        projected, metadata = source_camera_zbuffers(points, np.array([0, 0, 1, 0, 0]), self.frames())
        self.assertAlmostEqual(float(projected[0].depth_z_m[7, 9]), .800, places=6)
        self.assertAlmostEqual(float(projected[1].depth_z_m[7, 9]), 1.200, places=6)
        self.assertEqual(np.count_nonzero(projected[0].depth_z_m), 1)
        self.assertEqual(np.count_nonzero(projected[1].depth_z_m), 1)
        self.assertEqual(np.count_nonzero(projected[2].depth_z_m), 0)
        self.assertEqual(metadata[0]["pixel_collisions"], 1)
        self.assertEqual(metadata[0]["behind_camera"], 1)
        np.testing.assert_array_equal(points, original)

    def test_invalid_cloud_or_source_ids_fail_explicitly(self):
        for points, ids in (([[0., 0., np.nan]], [0]), ([[0., 0., 1.]], [3]),
                            ([[0., 0., 1.]], [.5]), ([[0., 0., 1.]], [])):
            with self.subTest(points=points, ids=ids), self.assertRaises(ValueError):
                source_camera_zbuffers(np.array(points), np.array(ids), self.frames())

    def test_prepared_cloud_is_not_reported_as_inference(self):
        with self.assertRaisesRegex(ValueError, "restoration_not_completed"):
            read_cloud(dict(status="prepared_not_inferred"), {})


if __name__ == "__main__":
    unittest.main()
