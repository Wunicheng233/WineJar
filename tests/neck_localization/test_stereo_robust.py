"""CPU-only extraction and frozen-selection equivalence checks."""
from types import SimpleNamespace
import unittest

import numpy as np

from neck_localization.depth_profile import rgb_foreground
from neck_localization.joint_profile import JointSettings, observed_initialization
from neck_localization.stereo_robust import (
    _observed_initialization, blue_screen_foreground, extract_stereo_masks,
)


def image_with_dark_neck():
    rgb = np.full((40, 40, 3), [19, 33, 57], np.uint8)
    rgb[15:38, 8:32] = [140, 137, 133]
    rgb[7:16, 16:24] = [12, 11, 10]  # below the old V>35 threshold
    rgb[4:8, 12:28] = [83, 81, 77]
    rgb[5:7, 17:23] = 0  # dark mouth interior
    return rgb


class StereoForegroundTests(unittest.TestCase):
    def test_dark_neck_keeps_lip_connected_without_a_brightness_threshold(self):
        rgb = image_with_dark_neck()
        old, new = rgb_foreground(rgb), blue_screen_foreground(rgb)
        self.assertFalse(old[10, 20])
        self.assertFalse(old[4, 13])
        self.assertTrue(new[10, 20])
        self.assertTrue(new[4, 13])
        self.assertTrue(new[25, 20])
        self.assertFalse(new[0, 0])

    def test_fills_internal_hole_and_ignores_disconnected_small_distractor(self):
        rgb = image_with_dark_neck()
        rgb[0:2, 0:2] = [150, 150, 150]
        mask = blue_screen_foreground(rgb)
        self.assertTrue(mask[5, 20])
        self.assertFalse(mask[0, 0])
        self.assertFalse(mask[3, 20])

    def test_empty_blue_backdrop_produces_no_object(self):
        rgb = np.full((30, 40, 3), [19, 33, 57], np.uint8)
        self.assertFalse(blue_screen_foreground(rgb).any())

    def test_uses_rgb_only_and_does_not_change_readonly_input(self):
        rgb = image_with_dark_neck()
        saved = rgb.copy()
        rgb.setflags(write=False)
        # There is deliberately no depth, mask, geometry or truth attribute.
        frame = SimpleNamespace(rgb=rgb, camera=SimpleNamespace(name="a"))
        masks, details = extract_stereo_masks([frame])
        np.testing.assert_array_equal(rgb, saved)
        self.assertEqual(details["cameras"][0]["pixels"], np.count_nonzero(masks[0]))
        self.assertEqual(masks[0].dtype, np.bool_)

    def test_rejects_unsupported_image_format(self):
        with self.assertRaises(ValueError):
            blue_screen_foreground(np.ones((3, 4, 3), float))

    def test_explicit_mask_selection_matches_frozen_geometry_exactly(self):
        # A point-lookup camera isolates the SELECTION layer from projection,
        # which has separate analytic camera tests. Identical masks must yield
        # identical candidates, axis seed, indices and coarse profile region.
        theta = np.linspace(0, 2*np.pi, 120, endpoint=False)
        points = []
        for radius in [.060, .062, .064, .066]:
            points.append(np.c_[radius*np.cos(theta), radius*np.sin(theta), np.full(len(theta), .600)])
        for height in np.linspace(.565, .596, 64):
            radius = .050+12*(height-.586)**2
            points.append(np.c_[radius*np.cos(theta), radius*np.sin(theta), np.full(len(theta), height)])
        cloud = np.concatenate(points)
        width, height = len(cloud)//2, 2

        class LookupCamera:
            def __init__(self, name, angle):
                self.name = name
                self.T_world_from_camera = np.eye(4)
                self.T_world_from_camera[:3, 3] = [.6*np.cos(angle), .6*np.sin(angle), .8]

            def unproject(self, pixels, depth):
                return cloud[pixels[:, 1]*width+pixels[:, 0]].copy()

        frames = [SimpleNamespace(camera=LookupCamera(str(i), a),
                                  rgb=np.full((height, width, 3), 140, np.uint8),
                                  depth_z_m=np.ones((height, width)))
                  for i, a in enumerate([0., 2*np.pi/3, 4*np.pi/3])]
        masks = [blue_screen_foreground(f.rgb) for f in frames]
        for frame, mask in zip(frames, masks):
            np.testing.assert_array_equal(mask, rgb_foreground(frame.rgb))
        frozen = observed_initialization(frames, JointSettings(), 4101)
        adapted = _observed_initialization(frames, masks, JointSettings(), 4101)
        for old, new in zip(frozen[:6], adapted[:6]):
            np.testing.assert_array_equal(old, new)
        self.assertEqual(frozen[6], adapted[6])
        for key, value in frozen[7].items():
            self.assertEqual(value, adapted[7][key])


if __name__ == "__main__":
    unittest.main()
