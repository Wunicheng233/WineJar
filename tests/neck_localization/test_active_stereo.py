"""CPU-only geometry/official-SDK checks; no renderer or hardware discovery.

These tests exercise independent analytic surfaces, not the sensor's geometry
truth buffers. They intentionally do not import or initialize SAPIEN rendering.
"""
import importlib.util
import unittest

import numpy as np

from neck_localization.active_stereo import (
    CV_FROM_ROS,
    OfflineSpatialFilter,
    StereoSettings,
    aligned_source_z_to_rgb_z,
    pose_cv_to_sapien,
    pose_sapien_to_cv,
    quantize_depth,
)
from neck_localization.geometry import Camera, look_at


def camera(name="depth", transform=None):
    return Camera(name, np.array([[80., 0., 31.5], [0., 80., 23.5], [0., 0., 1.]]),
                  np.eye(4) if transform is None else transform, 64, 48)


def y_rotation(angle_deg):
    c, s = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


class ActiveStereoGeometryTests(unittest.TestCase):
    def test_pose_conversion_preserves_world_rays_and_does_not_mutate(self):
        original = look_at("view", [.44, -.23, .78], [.02, .03, .56]).T_world_from_camera
        saved = original.copy()
        converted = pose_cv_to_sapien(original)
        # SAPIEN forward/left/up must equal OpenCV forward/left/up in world.
        np.testing.assert_allclose(converted[:3, 0], original[:3, 2], atol=1e-14)
        np.testing.assert_allclose(converted[:3, 1], -original[:3, 0], atol=1e-14)
        np.testing.assert_allclose(converted[:3, 2], -original[:3, 1], atol=1e-14)
        np.testing.assert_array_equal(converted[:3, 3], original[:3, 3])
        sample_sapien = np.array([.72, -.12, .08])
        sample_cv = CV_FROM_ROS @ sample_sapien
        np.testing.assert_allclose(converted[:3, :3] @ sample_sapien + converted[:3, 3],
                                   original[:3, :3] @ sample_cv + original[:3, 3], atol=1e-14)
        np.testing.assert_allclose(pose_sapien_to_cv(converted), original, atol=1e-14)
        np.testing.assert_array_equal(original, saved)

    def test_quantization_invalid_values_range_and_no_input_mutation(self):
        depth = np.array([[0., -1., np.nan, np.inf, -np.inf],
                          [.0001, .0016, .8004, 65.535, 70.]])
        saved = depth.copy()
        depth.setflags(write=False)
        result = quantize_depth(depth, .001)
        np.testing.assert_array_equal(result, np.array([[0, 0, 0, 0, 0],
                                                       [1, 2, 800, 0, 0]], np.uint16))
        np.testing.assert_array_equal(depth, saved)
        self.assertEqual(result.dtype, np.uint16)

    def test_quantization_rejects_nonphysical_units(self):
        for unit in [0., -.001, np.inf, np.nan]:
            with self.subTest(unit=unit), self.assertRaises(ValueError):
                quantize_depth(np.ones((2, 2)), unit)

    def test_registered_source_z_conversion_for_noncoaxial_cameras(self):
        # An infinite plane has left-camera axial Z = 1.25 m. Rotate and move
        # RGB: its axial Z is not 1.25 m, and varies across the output image.
        world_left = np.eye(4)
        world_left[:3, :3] = y_rotation(-11.)
        world_left[:3, 3] = [.3, -.2, .1]
        left_rgb = np.eye(4)
        left_rgb[:3, :3] = y_rotation(17.)
        left_rgb[:3, 3] = [.12, .01, .08]
        left = camera("left", world_left)
        rgb = camera("rgb", world_left @ left_rgb)
        source_z = np.full((48, 64), 1.25)
        actual = aligned_source_z_to_rgb_z(source_z, left, rgb)
        yy, xx = np.indices(source_z.shape)
        # Independent ray/plane intersection in RGB coordinates.
        plane_normal_rgb = left_rgb[:3, :3].T @ np.array([0., 0., 1.])
        rays_rgb = np.stack(((xx-31.5)/80., (yy-23.5)/80., np.ones_like(xx)), axis=-1)
        expected = (1.25-.08) / np.einsum("...i,i->...", rays_rgb, plane_normal_rgb)
        np.testing.assert_allclose(actual, expected, atol=1.2e-7)
        self.assertGreater(float(np.ptp(actual)), .2)
        pixels = np.column_stack((xx.ravel(), yy.ravel()))
        world_points = rgb.unproject(pixels, actual.ravel())
        points_left = (world_points-world_left[:3, 3]) @ world_left[:3, :3]
        np.testing.assert_allclose(points_left[:, 2], 1.25, atol=1.2e-7)
        np.testing.assert_array_equal(source_z, np.full((48, 64), 1.25))

    def test_registered_conversion_masks_invalid_or_backward_rays(self):
        left, rgb = camera(), camera("rgb")
        source = np.full((48, 64), 1.)
        source[0, :4] = [0., -1., np.nan, np.inf]
        actual = aligned_source_z_to_rgb_z(source, left, rgb)
        np.testing.assert_array_equal(actual[0, :4], np.zeros(4))
        np.testing.assert_allclose(actual[1:], 1.)
        rgb.T_world_from_camera[:3, :3] = y_rotation(180.)
        np.testing.assert_array_equal(aligned_source_z_to_rgb_z(source, left, rgb),
                                      np.zeros_like(source))


@unittest.skipUnless(importlib.util.find_spec("pyrealsense2"), "offline SDK not installed")
class OfflineRealSenseFilterTests(unittest.TestCase):
    def test_spatial_preserves_invalid_points_and_readonly_input(self):
        processor = OfflineSpatialFilter(camera(), .05, StereoSettings())
        try:
            depth = np.full((48, 64), .8, np.float32)
            depth[12:22, 24:40] = 0.
            depth[30, 30] = np.nan
            depth[31, 30] = np.inf
            depth[32, 30] = -1.
            saved = depth.copy()
            depth.setflags(write=False)
            raw, filtered = processor.process(depth)
            np.testing.assert_array_equal(depth, saved)
            np.testing.assert_array_equal(filtered[raw == 0], np.zeros(np.count_nonzero(raw == 0)))
            np.testing.assert_allclose(filtered[raw > 0], .8, atol=1e-7)
            np.testing.assert_allclose(raw[raw > 0], .8, atol=1e-7)
            self.assertFalse(np.shares_memory(raw, depth))
            self.assertFalse(np.shares_memory(filtered, depth))
        finally:
            processor.close()

    def test_registered_plane_uses_current_frame_and_noncoaxial_rgb_axial_z(self):
        left, rgb = camera(), camera("rgb")
        rgb.T_world_from_camera[:3, :3] = y_rotation(3.)
        rgb.T_world_from_camera[:3, 3] = [.012, 0., .006]
        processor = OfflineSpatialFilter(left, .05, StereoSettings(), rgb_camera=rgb)
        color = np.zeros((48, 64, 3), np.uint8)
        color.setflags(write=False)
        yy, xx = np.indices((48, 64))
        plane_normal_rgb = rgb.T_world_from_camera[:3, :3].T @ np.array([0., 0., 1.])
        rays_rgb = np.stack(((xx-31.5)/80., (yy-23.5)/80., np.ones_like(xx)), axis=-1)
        denominator = np.einsum("...i,i->...", rays_rgb, plane_normal_rgb)
        try:
            for distance in [.8, 1.1]:
                processor.process(np.full((48, 64), distance))
                expected = (distance-.006)/denominator
                for filtered in [False, True]:
                    with self.subTest(distance=distance, filtered=filtered):
                        registered = processor.register(color, filtered=filtered)
                        good = registered > 0
                        self.assertGreater(np.count_nonzero(good), 48*64*.85)
                        np.testing.assert_allclose(registered[good], expected[good], atol=2e-7)
            np.testing.assert_array_equal(color, np.zeros_like(color))
        finally:
            processor.close()


if __name__ == "__main__":
    unittest.main()
