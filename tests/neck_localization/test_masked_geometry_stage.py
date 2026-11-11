"""Foreground provenance must bind a mask to the actual prepared RGB frame."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from scripts.run_masked_geometry_capability_stage import load_foreground


class CachedForegroundTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        observations = self.root / 'observations'
        observations.mkdir()
        self.rgb = np.full((8, 10, 3), [50, 75, 100], np.uint8)
        rgb_path = observations / 'cam_a_rgb.png'
        cv2.imwrite(str(rgb_path), cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR))
        self.mask = np.zeros((8, 10), bool)
        self.mask[1:7, 2:8] = True
        self.mask_path = self.root / 'mask.png'
        cv2.imwrite(str(self.mask_path), self.mask.astype(np.uint8) * 255)
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        self.entry = dict(layout='close', case='example', camera='cam_a', status='ok',
            rgb_path=str(rgb_path), rgb_sha256=sha(rgb_path), mask_path=str(self.mask_path),
            mask_sha256=sha(self.mask_path),
            mask_pixels_sha256=hashlib.sha256(self.mask.astype(np.uint8).tobytes()).hexdigest(),
            prompt_box_xyxy=[0, 0, 9, 7], source_coarse_mask_bbox_xyxy=[2, 1, 7, 6],
            predicted_scores=[.9], selected_index=0)
        self.cache = self.root / 'manifest.json'
        self.write_cache()
        self.pairs = [dict(layout='close', case='example', camera='cam_a', scene_dir=str(self.root))]
        self.frames = [SimpleNamespace(camera=SimpleNamespace(name='cam_a'), rgb=self.rgb.copy())]

    def write_cache(self):
        self.cache.write_text(json.dumps(dict(schema=1, inputs_and_adapter_unchanged=True,
            no_ground_truth_read=True, no_depth_or_CAD_read=True,
            protocol={'name': 'fixed_test_cache'}, model={}, entries=[self.entry])))

    def test_mask_is_bound_to_RGB_and_has_auditable_pixels(self):
        tracked = {}
        masks, foreground = load_foreground(self.cache, self.pairs, self.frames, tracked)
        np.testing.assert_array_equal(masks[0], self.mask)
        self.assertEqual(foreground['cameras'][0]['mask_sha256'], self.entry['mask_pixels_sha256'])
        self.assertIn(str(self.cache.resolve()), tracked)
        self.assertIn(str(self.mask_path.resolve()), tracked)

    def test_changed_mask_file_is_rejected_before_geometry(self):
        cv2.imwrite(str(self.mask_path), np.zeros((8, 10), np.uint8))
        with self.assertRaisesRegex(ValueError, 'foreground_mask_file_sha256_mismatch'):
            load_foreground(self.cache, self.pairs, self.frames, {})

    def test_prepared_frame_must_equal_cached_RGB_even_if_files_match(self):
        self.frames[0].rgb[2, 3, 0] += 1
        with self.assertRaisesRegex(ValueError, 'prepared_frame_differs_from_cache_RGB_input'):
            load_foreground(self.cache, self.pairs, self.frames, {})


if __name__ == '__main__':
    unittest.main()
