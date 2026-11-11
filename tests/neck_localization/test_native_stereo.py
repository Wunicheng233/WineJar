"""Analytic native-point projection/color/visibility tests; no GPU or devices."""
from types import SimpleNamespace
import unittest

import numpy as np

from neck_localization.geometry import Camera
from neck_localization.native_stereo import map_rgb_onto_native
from scripts.evaluate_native_stereo import MODES, summarize


def camera(name, width, height, focal=10., transform=None):
    return Camera(name, np.array([[focal, 0., (width-1)/2], [0., focal, (height-1)/2], [0., 0., 1.]]),
                  np.eye(4) if transform is None else transform, width, height)


class NativeStereoMappingTests(unittest.TestCase):
    def test_identity_mapping_copies_exact_rgb_and_keeps_native_depth(self):
        cam = camera("same", 7, 5)
        color = np.full((5, 7, 3), [19, 33, 57], np.uint8)
        color[1:4, 2:5] = [100, 97, 94]
        color[2, 3] = [5, 4, 3]  # black neck remains foreground
        depth = np.full((5, 7), .8)
        original_rgb, original_z = color.copy(), depth.copy()
        color.setflags(write=False)
        depth.setflags(write=False)
        frame, mask, valid, meta = map_rgb_onto_native(SimpleNamespace(camera=cam, rgb=color), cam, depth)
        np.testing.assert_array_equal(frame.rgb, color)
        np.testing.assert_array_equal(frame.depth_z_m, depth)
        np.testing.assert_array_equal(valid, np.ones_like(valid))
        expected = np.zeros_like(mask)
        expected[1:4, 2:5] = True
        np.testing.assert_array_equal(mask, expected)
        np.testing.assert_array_equal(color, original_rgb)
        np.testing.assert_array_equal(depth, original_z)
        self.assertEqual(meta["foreground_points"], 9)

    def test_offset_camera_samples_original_rgb_at_analytic_projection(self):
        native = camera("ir", 3, 3, focal=4.)
        transform = np.eye(4)
        transform[0, 3] = .25
        rgb_camera = camera("rgb", 7, 7, focal=4., transform=transform)
        yy, xx = np.indices((7, 7))
        color = np.stack([70+xx, 80+yy, np.full_like(xx, 60)], axis=-1).astype(np.uint8)
        depth = np.ones((3, 3))
        frame, mask, valid, _ = map_rgb_onto_native(SimpleNamespace(camera=rgb_camera, rgb=color), native, depth)
        # x_ir=(u-1)/4 at Z=1; x_rgb=x_ir-.25;
        # u_rgb=4*x_rgb+3=u_ir+1, v_rgb=v_ir+2.
        np.testing.assert_array_equal(frame.rgb, color[2:5, 1:4])
        self.assertTrue(mask.all())
        self.assertTrue(valid.all())
        # Native geometry is retained, never RGB-grid depth/reprojection.
        xyz = frame.camera.unproject([[0, 0], [2, 2]], [1., 1.])
        np.testing.assert_allclose(xyz, [[-.25, -.25, 1.], [.25, .25, 1.]])

    def test_invalid_behind_and_outside_points_are_excluded_explicitly(self):
        native = camera("ir", 3, 2)
        color = np.full((2, 3, 3), 120, np.uint8)
        depth = np.array([[0., np.nan, -1.], [np.inf, 1., 1.]])
        rgb = SimpleNamespace(camera=native, rgb=color)
        _, mask, valid, _ = map_rgb_onto_native(rgb, native, depth)
        expected = np.array([[False, False, False], [False, True, True]])
        np.testing.assert_array_equal(mask, expected)
        np.testing.assert_array_equal(valid, expected)
        reverse = np.eye(4)
        reverse[:3, :3] = np.diag([-1., 1., -1.])
        backward_rgb = SimpleNamespace(camera=camera("back", 3, 2, transform=reverse), rgb=color)
        _, mask, valid, _ = map_rgb_onto_native(backward_rgb, native, depth)
        self.assertFalse(mask.any())
        self.assertFalse(valid.any())
        moved = np.eye(4)
        moved[0, 3] = 100.
        outside_rgb = SimpleNamespace(camera=camera("out", 3, 2, transform=moved), rgb=color)
        _, mask, valid, _ = map_rgb_onto_native(outside_rgb, native, depth)
        self.assertFalse(mask.any())
        self.assertFalse(valid.any())

    def test_native_point_zbuffer_rejects_farther_collisions(self):
        native = camera("ir", 3, 1, focal=100.)
        rgb = SimpleNamespace(camera=camera("rgb", 1, 1, focal=1.),
                              rgb=np.full((1, 1, 3), 120, np.uint8))
        _, mask, valid, meta = map_rgb_onto_native(rgb, native, np.array([[2., 1., 3.]]))
        np.testing.assert_array_equal(mask, [[False, True, False]])
        np.testing.assert_array_equal(valid, [[False, True, False]])
        self.assertEqual(meta["projected_inside_rgb_points"], 3)
        self.assertEqual(meta["native_visible_points"], 1)

    def test_native_depth_shape_must_match_intrinsic_grid(self):
        cam = camera("a", 3, 2)
        with self.assertRaises(ValueError):
            map_rgb_onto_native(SimpleNamespace(camera=cam, rgb=np.zeros((2, 3, 3), np.uint8)), cam, np.ones((3, 2)))

    def test_summary_handles_rejected_records_without_fabricated_candidates(self):
        rows = [dict(condition_name=mode, accepted=False, truth_pass=False,
                     reasons=["no_candidate"], runtime_ms=1.) for mode in MODES]
        for row in summarize(rows):
            self.assertEqual(row["candidates"], 0)
            self.assertEqual(row["accepted"], 0)
            self.assertEqual(row["reasons"], {"no_candidate":1})


if __name__ == "__main__":
    unittest.main()
