"""Keep the original MuJoCo replay tied to actual saved observations."""
import copy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from neck_localization.depth_profile import DepthFrame
from neck_localization.geometry import Camera
from scripts.export_capability_structural_viewer import (
    check_fit_subset, exported_masks, file_hash, load_foreground_cache, preferred_method, viewer_record,
)
from scripts import view_structural_study


class CapabilityStructuralExportTests(unittest.TestCase):
    def test_viewer_routing_keeps_rejected_candidate_and_saved_diagnostics(self):
        row = dict(backend='portfolio', accepted=False, reasons=['unchanged_gate'],
                   estimate=dict(center_m=[0, 0, .6], radius_m=.05),
                   diagnostics=dict(fit_point_positions_m=[[0, 0, .6]]))
        frozen = copy.deepcopy(row)
        result = viewer_record(row, 'foundation_stereo', '/tmp/records.json.gz')
        self.assertEqual(row, frozen)
        self.assertEqual(result['method'], 'portfolio')
        self.assertEqual(result['evidence'], 'rgb')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['diagnostics'], row['diagnostics'])
        self.assertEqual(result['estimate'], row['estimate'])

    def test_fit_cloud_must_be_a_subset_of_replayed_depth(self):
        camera = Camera('cam_a', np.array([[100., 0, 1], [0, 100., 1], [0, 0, 1]]),
                        np.eye(4), 3, 3)
        frame = DepthFrame(camera, np.zeros((3, 3, 3), np.uint8), np.ones((3, 3)))
        points = camera.unproject(np.array([[1, 1], [0, 2]]), np.ones(2))
        row = dict(diagnostics=dict(fit_point_positions_m=points.tolist(), fit_camera_ids=[0, 0],
                                   fit_point_weights=[1., .5], fit_camera_names=['cam_a']))
        self.assertEqual(check_fit_subset(row, [frame])['points'], 2)
        row['diagnostics']['fit_point_positions_m'][0][2] = 1.01
        with self.assertRaisesRegex(ValueError, 'differ_from_registered_depth'):
            check_fit_subset(row, [frame])

    def test_default_prefers_new_bundle_and_explicit_old_bundle_remains_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = root/'old', root/'new'
            old.mkdir(); new.mkdir()
            original = dict(condition='close/stereo_raw', label='old', method='portfolio')
            modern = dict(condition='close/foundation_stereo/clean', label='new', method='portfolio')
            (old/'datasets.json').write_text(json.dumps([original]))
            (new/'datasets.json').write_text(json.dumps([modern]))
            with patch.object(view_structural_study, 'DEFAULT', old), \
                 patch.object(view_structural_study, 'CAPABILITY_DEFAULT', new):
                self.assertEqual(view_structural_study.select_dataset()[2], modern)
                self.assertEqual(view_structural_study.select_dataset(old)[2], original)
                (new/'datasets.json').unlink()
                self.assertEqual(view_structural_study.select_dataset()[2], original)

    def test_preferred_method_uses_correct_readout_without_changing_legacy_bundle(self):
        self.assertEqual(preferred_method(['portfolio', 'standard_sor', 'visible_profile']), 'visible_profile')
        self.assertEqual(preferred_method(['portfolio', 'standard_sor']), 'standard_sor')
        self.assertEqual(preferred_method(['portfolio']), 'portfolio')

    def test_explicit_sam_bundle_prefers_existing_twelve_case_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            confirmation = dict(condition='close/foundation_stereo_confirm_sam/clean',
                                label='new eight', method='visible_profile')
            original = dict(condition='close/foundation_stereo_sam/clean',
                            label='paired twelve', method='visible_profile')
            (root/'datasets.json').write_text(json.dumps([confirmation, original]))
            self.assertEqual(view_structural_study.select_dataset(root)[2], original)

    def test_explicit_segmentation_cache_is_used_and_all_hashes_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'scene/observations').mkdir(parents=True)
            rgb_path = root/'scene/observations/cam_a_rgb.png'
            cv2.imwrite(str(rgb_path), np.zeros((3, 3, 3), np.uint8))
            mask_path = root/'mask.png'
            mask = np.ones((3, 3), bool)
            cv2.imwrite(str(mask_path), mask.astype(np.uint8)*255)
            entry = dict(layout='close', case='example', camera='cam_a',
                rgb_path=str(rgb_path), rgb_sha256=file_hash(rgb_path), mask_path=str(mask_path),
                mask_sha256=file_hash(mask_path),
                mask_pixels_sha256=sha256(mask.astype(np.uint8).tobytes()).hexdigest())
            (root/'manifest.json').write_text(json.dumps(dict(entries=[entry])))
            camera = Camera('cam_a', np.eye(3), np.eye(4), 3, 3)
            frame = DepthFrame(camera, np.zeros((3, 3, 3), np.uint8), np.ones((3, 3)))
            pair = dict(layout='close', case='example', camera='cam_a', scene_dir=str(root/'scene'))
            tracked = {}
            cache = load_foreground_cache(root, tracked)
            self.assertEqual(load_foreground_cache(root/'manifest.json', {}), cache)
            result, metadata = exported_masks([frame], [pair], cache, tracked)
            np.testing.assert_array_equal(result[0], mask)
            self.assertEqual(metadata['method'], 'explicit_saved_RGB_segmentation_cache')
            self.assertIn(str(mask_path), tracked)
            for field, expected_error in [('mask_sha256', 'mask_file_sha256'),
                    ('mask_pixels_sha256', 'mask_pixel_sha256'), ('rgb_sha256', 'RGB_sha256')]:
                corrupted = copy.deepcopy(cache)
                corrupted[('close', 'example', 'cam_a')][field] = '0'*64
                with self.assertRaisesRegex(ValueError, expected_error):
                    exported_masks([frame], [pair], corrupted, {})
            with self.assertRaisesRegex(ValueError, 'missing_camera'):
                exported_masks([frame], [pair], {}, {})

    def test_fit_points_cannot_be_displayed_as_supported_by_a_different_mask(self):
        camera = Camera('cam_a', np.eye(3), np.eye(4), 3, 3)
        frame = DepthFrame(camera, np.zeros((3, 3, 3), np.uint8), np.ones((3, 3)))
        row = dict(diagnostics=dict(fit_point_positions_m=[[1., 1., 1.]], fit_camera_ids=[0],
            fit_point_weights=[1.], fit_camera_names=['cam_a']))
        with self.assertRaisesRegex(ValueError, 'outside_exported_foreground'):
            check_fit_subset(row, [frame], masks={'cam_a': np.zeros((3, 3), bool)})


if __name__ == '__main__':
    unittest.main()
