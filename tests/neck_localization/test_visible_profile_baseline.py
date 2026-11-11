"""CPU checks of correct silhouette geometry and per-seed reselection."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from neck_localization.geometry import Estimate, circle_points, look_at
from neck_localization.structure_portfolio import target_contact
from neck_localization.visible_profile_baseline import (
    estimate_visible_profile, profile_projection_support, revalidate_candidate,
)


def setup_view():
    camera = look_at('cam_a', [.5, -.5, .68], [0, 0, .60], width=400, height=300)
    outer = np.concatenate([circle_points([0, 0, z], [0, 0, 1], .09, 720)
                            for z in [.52, .69]])
    polygon = cv2.convexHull(camera.project(outer).astype(np.float32))
    mask = np.zeros((camera.height, camera.width), np.uint8)
    cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 1)
    frame = SimpleNamespace(camera=camera, rgb=np.zeros((*mask.shape, 3), np.uint8))
    return frame, mask.astype(bool)


def candidate(*, center_x=0., rms_mm=1., old_score=100., span=.006):
    return Estimate(center=np.array([center_x, 0., .6]), normal=np.array([0., 0., 1.]),
        radius=.04, residual=rms_mm/1000, accepted=False,
        reasons=['candidate_inconsistent_with_rgb_silhouette'],
        diagnostics=dict(axis_origin_m=[center_x, 0., .6],
            profile_s_m=np.linspace(-.01, .01, 21).tolist(),
            profile_radius_m=np.full(21, .04).tolist(),
            heldout_rms_mm=rms_mm, heldout_mean_mm=0., selection_score=old_score,
            near_min_interval_length_m=span, target_contact_rms_pixels=50.))


class VisibleProfileTests(unittest.TestCase):
    def test_exact_internal_section_does_not_have_to_touch_whole_silhouette(self):
        frame, mask = setup_view()
        original = candidate()
        contact, _ = target_contact([frame], [mask], original.center*1000,
                                    original.normal, original.radius*1000)
        self.assertGreater(float(np.sqrt(np.mean(contact**2))), 2.)
        corrected = revalidate_candidate(original, [frame], [mask])
        self.assertTrue(corrected.accepted)
        self.assertEqual(corrected.diagnostics['surface_projection']['maximum_outside_pixels'], 0.)
        self.assertGreater(corrected.diagnostics['surface_projection']['views'][0]
                           ['locally_visible_sample_count'], 0)
        self.assertFalse(original.accepted)
        self.assertEqual(original.reasons, ['candidate_inconsistent_with_rgb_silhouette'])

    def test_surface_crossing_foreground_is_rejected_not_merely_old_reason_removed(self):
        frame, mask = setup_view()
        wrong = candidate(center_x=.20)
        corrected = revalidate_candidate(wrong, [frame], [mask])
        self.assertFalse(corrected.accepted)
        self.assertIn('fitted_surface_projects_outside_rgb_foreground', corrected.reasons)
        self.assertGreater(corrected.diagnostics['surface_projection']['maximum_outside_pixels'], 2.)

    def test_checks_whole_local_surface_not_only_target_circle(self):
        frame, mask = setup_view()
        original = candidate()
        original.diagnostics['profile_radius_m'][0] = .18
        self.assertEqual(original.radius, .04)
        corrected = revalidate_candidate(original, [frame], [mask])
        self.assertFalse(corrected.accepted)
        self.assertIn('fitted_surface_projects_outside_rgb_foreground', corrected.reasons)

    def test_no_new_visibility_gate_and_original_noncontact_gates_remain(self):
        frame, mask = setup_view()
        original = candidate(span=.002)
        original.reasons += ['insufficient_three_mm_band_span', 'spatial_block_depth_mismatch']
        corrected = revalidate_candidate(original, [frame], [mask])
        self.assertFalse(corrected.accepted)
        self.assertFalse(corrected.diagnostics['pose_gate_accepted'])
        self.assertFalse(corrected.diagnostics['band_usable'])
        self.assertEqual(corrected.reasons,
                         ['insufficient_three_mm_band_span', 'spatial_block_depth_mismatch'])
        original.reasons.remove('spatial_block_depth_mismatch')
        corrected = revalidate_candidate(original, [frame], [mask])
        self.assertFalse(corrected.accepted)
        self.assertTrue(corrected.diagnostics['pose_gate_accepted'])

    def test_all_candidates_are_revalidated_and_sorted_with_new_score(self):
        frame, mask = setup_view()
        old_favorite = candidate(rms_mm=4., old_score=0.)
        new_favorite = candidate(center_x=.004, rms_mm=1., old_score=999.)
        seeds = [dict(label='first', arc_rms=0.), dict(label='second', arc_rms=0.)]
        with patch('neck_localization.visible_profile_baseline.extract_stereo_masks',
                   return_value=([mask], {})), \
             patch('neck_localization.visible_profile_baseline.observed_cloud',
                   return_value=(np.zeros((300, 3)), np.zeros(300, int))), \
             patch('neck_localization.visible_profile_baseline.seed_hypotheses',
                   return_value=seeds), \
             patch('neck_localization.visible_profile_baseline.fit_seed',
                   side_effect=[old_favorite, new_favorite]):
            result = estimate_visible_profile([frame])
        self.assertTrue(result.accepted)
        np.testing.assert_array_equal(result.center, new_favorite.center)
        self.assertEqual(len(result.diagnostics['attempts']), 2)
        self.assertEqual(result.diagnostics['frozen_validation']['selection_score'], 999.)
        self.assertLess(result.diagnostics['selection_score'], 1.)

    def test_invalid_profile_cannot_be_used_as_valid_silhouette_evidence(self):
        frame, mask = setup_view()
        original = candidate()
        original.diagnostics['profile_radius_m'][1] = np.nan
        with self.assertRaisesRegex(ValueError, 'invalid_fitted_profile'):
            profile_projection_support(original, [frame], [mask])


if __name__ == '__main__':
    unittest.main()
