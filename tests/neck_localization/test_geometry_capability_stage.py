"""Geometry-audit evaluation contracts, using synthetic truth and mocked IO.

No saved audit objects, meshes, sensor files or evaluation labels are opened.
In particular these tests do not inspect the newly generated eight objects.
"""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from neck_localization.geometry import Estimate
from scripts import run_geometry_capability_stage as stage


def synthetic_estimate(*, accepted=True, center_offset_m=(0., 0., 0.),
                       radius=.05, normal=(0., 0., 1.), observed_half_width_m=.004):
    center = np.array([0., 0., .6])+np.asarray(center_offset_m)
    return Estimate(center=center, normal=np.asarray(normal, float), radius=radius,
        accepted=accepted, reasons=[] if accepted else ['synthetic_native_rejection'],
        diagnostics=dict(axis_origin_m=center.tolist(),
            near_min_interval_from_axis_origin_m=[-observed_half_width_m, observed_half_width_m]))


class GeometryCapabilityEvaluationTests(unittest.TestCase):
    def fake_worker(self, estimates, *, failure_method=None):
        """Exercise real worker, score and band logic without disk/estimation."""
        methods = tuple(estimates)
        events = []
        pairs = [dict(camera=name, pair_id='synthetic/'+name, layout='close',
                      case='synthetic_contract_only', condition='clean', split='synthetic_unit_test')
                 for name in stage.shared.ALL_CAMERAS]
        truth = dict(center_m=[0., 0., .6], normal=[0., 0., 1.], radius_m=.05,
                     s_m=0., near_min_interval_s_m=[-.006, .006], search_interval_s_m=[-.006, .006])

        def prepare_frame(pair, depth, tracked):
            return SimpleNamespace(camera=SimpleNamespace(name=pair['camera'])), None, {}

        def get_backend(method):
            def infer(frames, *, settings):
                self.assertEqual(tuple(f.camera.name for f in frames), stage.shared.ALL_CAMERAS)
                events.append(('backend', method))
                if method == failure_method:
                    raise RuntimeError('synthetic_backend_failure')
                return estimates[method]
            return infer, None

        def load_truth(pair, tracked):
            self.assertEqual(events, [('backend', method) for method in methods])
            events.append(('truth', None))
            return truth, {}

        with ExitStack() as stack:
            stack.enter_context(patch.object(stage.shared, 'load_prediction_index', return_value={}))
            stack.enter_context(patch.object(stage.shared, 'read_prediction', return_value=(np.ones((2, 2)), {})))
            stack.enter_context(patch.object(stage.shared, 'prepare_frame', side_effect=prepare_frame))
            stack.enter_context(patch.object(stage, 'backend', side_effect=get_backend))
            stack.enter_context(patch.object(stage.shared, '_load_truth', side_effect=load_truth))
            records, checks, tracked = stage.worker((pairs, '/synthetic/nonexistent/predictions', methods))
        self.assertEqual(events[-1], ('truth', None))
        return records

    def test_center_axis_and_diameter_thresholds_are_separate(self):
        # Exact threshold tests avoid floating round-off from world conversion.
        at_preferred = stage.pose_fields(dict(center_mm=3., normal_deg=2., diameter_mm=100.))
        self.assertTrue(at_preferred['center_pass_3mm'])
        self.assertTrue(at_preferred['pose_pass_3mm_2deg'])
        at_maximum = stage.pose_fields(dict(center_mm=4., normal_deg=2., diameter_mm=0.))
        self.assertFalse(at_maximum['center_pass_3mm'])
        self.assertTrue(at_maximum['pose_pass_4mm_2deg'])
        poor_axis = stage.pose_fields(dict(center_mm=2., normal_deg=2.001, diameter_mm=0.))
        self.assertTrue(poor_axis['center_pass_3mm'])
        self.assertFalse(poor_axis['axis_pass_2deg'])
        self.assertFalse(poor_axis['pose_pass_4mm_2deg'])
        missing = stage.pose_fields(dict(center_mm=None, normal_deg=None))
        self.assertFalse(any(missing.values()))

        rows = self.fake_worker({'standard_sor':synthetic_estimate(center_offset_m=(.002, 0., 0.), radius=.055)})
        row = rows[0]
        self.assertAlmostEqual(row['diameter_mm'], 10.)
        self.assertTrue(row['pose_pass_3mm_2deg'])
        self.assertFalse(row['truth_pass'])
        summary = stage.aggregate(rows)
        self.assertEqual(summary['accepted_pose_pass_3mm_2deg'], 1)
        self.assertEqual(summary['full_circle_correct_output'], 0)
        self.assertEqual(summary['full_circle_wrong_accepted'], 1)

    def test_native_acceptance_is_preserved_when_common_band_gate_rejects(self):
        rows = self.fake_worker({
            'visible_profile':synthetic_estimate(observed_half_width_m=.004),
            'standard_sor':synthetic_estimate(observed_half_width_m=.001)})
        broad, narrow = rows
        self.assertTrue(broad['accepted'])
        self.assertTrue(broad['accepted_with_common_band_gate'])
        self.assertTrue(narrow['accepted'])
        self.assertFalse(narrow['observed_band_usable'])
        self.assertFalse(narrow['accepted_with_common_band_gate'])
        summary = stage.aggregate(rows)
        self.assertEqual(summary['accepted'], 2)
        self.assertEqual(summary['common_band_accepted'], 1)
        self.assertEqual(summary['accepted_pose_pass_3mm_2deg'], 2)
        self.assertEqual(summary['common_band_accepted_pose_pass_3mm_2deg'], 1)

    def test_native_rejection_is_not_overridden_by_good_geometry_or_band(self):
        rows = self.fake_worker({'standard_sor':synthetic_estimate(accepted=False)})
        row = rows[0]
        self.assertTrue(row['truth_pass'])
        self.assertTrue(row['observed_band_usable'])
        self.assertFalse(row['accepted'])
        self.assertFalse(row['accepted_with_common_band_gate'])
        summary = stage.aggregate(rows)
        self.assertEqual(summary['candidate_geometry_pass'], 1)
        self.assertEqual(summary['full_circle_correct_output'], 0)
        self.assertEqual(summary['rejected'], 1)
        self.assertEqual(summary['candidate_axial_band_contained'], 1)
        self.assertEqual(summary['accepted_axial_band_contained'], 0)

    def test_axial_containment_cannot_be_read_as_physical_task_success(self):
        # A ring displaced sideways by 100 mm still has the correct axial
        # envelope. This decisive counterexample requires separate naming.
        rows = self.fake_worker({'standard_sor':synthetic_estimate(center_offset_m=(.100, 0., 0.))})
        row = rows[0]
        self.assertAlmostEqual(row['center_mm'], 100.)
        self.assertFalse(row['center_pass_4mm'])
        self.assertFalse(row['truth_pass'])
        self.assertTrue(row['band_evaluation']['full_annotated_neck']['contained'])
        self.assertEqual(row['band_evaluation']['full_annotated_neck']['scope'],
                         'axial_geometry_only_no_contact_collision_or_material_model')
        summary = stage.aggregate(rows)
        self.assertEqual(summary['accepted_axial_band_contained'], 1)
        self.assertEqual(summary['full_circle_correct_output'], 0)
        self.assertFalse(any('success' in key for key in summary))

    def test_both_backend_calls_finish_before_truth_and_errors_stay_in_denominator(self):
        rows = self.fake_worker({
            'visible_profile':synthetic_estimate(),
            'standard_sor':synthetic_estimate()}, failure_method='visible_profile')
        self.assertEqual(rows[0]['status'], 'backend_error')
        self.assertFalse(rows[0]['accepted'])
        self.assertFalse(rows[0]['center_pass_4mm'])
        self.assertEqual(rows[1]['status'], 'ok')
        self.assertTrue(rows[1]['truth_pass'])
        summary = stage.aggregate(rows)
        self.assertEqual(summary['n'], 2)
        self.assertEqual(summary['input_or_backend_errors'], 1)
        self.assertEqual(summary['full_circle_correct_output'], 1)
        self.assertEqual(summary['full_circle_correct_output_rate'], .5)


if __name__ == '__main__':
    unittest.main()
