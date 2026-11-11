import unittest
import numpy as np

from neck_localization.depth_profile import DepthFrame
from neck_localization.depth_perturbation import axial_depth_bias
from neck_localization.geometry import look_at


class DepthBiasTests(unittest.TestCase):
    def frame(self, name='a'):
        return DepthFrame(look_at(name, [1, -.7, .8], [0, 0, .6]),
                          np.zeros((2, 3, 3), np.uint8),
                          np.array([[.5, 0, np.nan], [.001, -.1, np.inf]], np.float32))

    def test_invalid_pixels_and_inputs_preserved(self):
        frame = self.frame()
        original = frame.depth_z_m.copy()
        changed = axial_depth_bias([frame], {'a': -.002})[0]
        np.testing.assert_array_equal(frame.depth_z_m, original)
        self.assertAlmostEqual(changed.depth_z_m[0, 0], .498)
        self.assertEqual(changed.depth_z_m[0, 1], 0)
        self.assertTrue(np.isnan(changed.depth_z_m[0, 2]))
        self.assertEqual(changed.depth_z_m[1, 0], 0)
        self.assertEqual(changed.depth_z_m[1, 1], original[1, 1])
        self.assertTrue(np.isinf(changed.depth_z_m[1, 2]))
        self.assertIs(changed.rgb, frame.rgb)
        self.assertIs(changed.camera, frame.camera)

    def test_only_named_camera_changes_and_zero_control_is_exact(self):
        frames = [self.frame('a'), self.frame('b')]
        result = axial_depth_bias(frames, {'a': .002})
        np.testing.assert_array_equal(result[1].depth_z_m, frames[1].depth_z_m)
        self.assertEqual(result[1].depth_z_m.dtype, frames[1].depth_z_m.dtype)
        np.testing.assert_array_equal(axial_depth_bias(frames, {})[0].depth_z_m, frames[0].depth_z_m)

    def test_world_displacement_follows_pixel_ray_not_global_z(self):
        frame = self.frame()
        pixels = np.array([[100, 140], [450, 340]])
        depth = np.array([.7, .9])
        bias = .003
        camera = frame.camera
        expected = (np.c_[pixels, np.ones(2)] @ np.linalg.inv(camera.K).T)*bias
        expected = expected @ camera.T_world_from_camera[:3, :3].T
        actual = camera.unproject(pixels, depth+bias)-camera.unproject(pixels, depth)
        np.testing.assert_allclose(actual, expected, atol=1e-15)
        self.assertFalse(np.allclose(actual[0], actual[1]))
        self.assertFalse(np.allclose(actual[:, :2], 0))

    def test_invalid_camera_or_nonfinite_bias_is_error(self):
        with self.assertRaises(ValueError):
            axial_depth_bias([self.frame()], {'unknown': .001})
        with self.assertRaises(ValueError):
            axial_depth_bias([self.frame()], {'a': np.nan})


if __name__ == '__main__':
    unittest.main()
