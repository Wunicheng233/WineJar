import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from neck_localization.cross_station_uncertainty import (
    fit_rigid_points, reference_points, encoder_reading, fit_encoder_scale,
    executed_circle, relabel_world, evaluate_arrays, perturb)
from neck_localization.station_tying import frame, pose, jaw_arcs, GunSpec
from neck_localization.tool_clearance import slab_radial_envelope
from neck_localization.task_circle_fusion import pose_radius_control


class RegistrationTests(unittest.TestCase):
    def test_exact_registration_recovers_rotation_and_translation(self):
        p = reference_points()
        r = Rotation.from_euler('xyz', [10, -20, 35], degrees=True).as_matrix()
        q = p @ r.T+[.2, -.4, .7]
        result = fit_rigid_points(p, q, [.00005, .00005, .0001])
        np.testing.assert_allclose(result['transform'], pose(r, [.2, -.4, .7]), atol=1e-11)

    def test_common_measurement_bias_can_have_zero_fitting_residual(self):
        p = reference_points()
        bias = [.0003, -.0002, .0005]
        result = fit_rigid_points(p, p+bias, 1e-5)
        self.assertLess(result['normalized_rms'], 1e-8)
        np.testing.assert_allclose(result['transform'][:3, 3], bias, atol=1e-12)
        self.assertTrue(result['covariance_excludes_fixed_survey_and_artifact_bias'])

    def test_robust_fit_beats_plain_fit_on_independent_points_with_outlier(self):
        rng = np.random.default_rng(71)
        p = reference_points()
        q = p+rng.normal(0, .00002, p.shape)
        q[-1] += [.005, -.003, .004]
        plain = fit_rigid_points(p, q, .00003)['transform']
        robust = fit_rigid_points(p, q, .00003, robust=True)['transform']
        heldout = np.array([[2.3, -.2, .6], [.1, .1, .5]])
        error = lambda t: np.linalg.norm(heldout @ t[:3, :3].T+t[:3, 3]-heldout)
        self.assertLess(error(robust), error(plain)/5)

    def test_collinear_calibration_is_rejected(self):
        p = np.c_[np.arange(5), np.zeros((5, 2))]
        with self.assertRaises(ValueError):
            fit_rigid_points(p, p, .0001)


class TransferPhysicsTests(unittest.TestCase):
    def test_true_calibration_and_actual_encoder_preserve_original_visual_error(self):
        B = pose(Rotation.from_euler('z', 70, degrees=True).as_matrix(), [-.3, -.4, -.2])
        F = pose(np.diag([1, -1, -1]), [.19, 0, .07])
        G0 = pose(frame([.01, -.02, 1]), [.003, -.002, .59])
        actual = executed_circle(G0, B, B, F, F, 2.4007)
        expected = G0.copy()
        expected[:3, 3] += [2.4007, 0, 0]
        np.testing.assert_allclose(actual, expected, atol=1e-15)

    def test_shared_world_gauge_cancels(self):
        B = perturb([.2, -.4, -.1], [.1, -.2, .3])
        M = perturb([2.4, 0, 0], [0, 0, 0])
        G = pose(frame([.02, .03, 1]), [.001, -.002, .6])
        H = perturb([3, -2, .4], [.3, .1, -.6])
        b, m, g = relabel_world(B, M, G, H)
        np.testing.assert_allclose(b @ m @ g, B @ M @ G, atol=2e-15)

    def test_horizontal_stop_error_is_tangential_for_upright_neck(self):
        metric = evaluate_arrays(np.array([[.002, 0, .6]]), np.array([[0, 0, 1]]), np.array([.055]),
            np.array([[0, 0, .6]]), np.array([[0, 0, 1]]), np.array([.006]))
        self.assertAlmostEqual(metric['tangent_mm'][0], 2)
        self.assertAlmostEqual(metric['axial_mm'][0], 0)
        self.assertAlmostEqual(metric['band_margin_mm'][0], 1.5)

    def test_encoder_endpoint_difference_cancels_counter_zero(self):
        kw = dict(scale_error=.0005, cyclic_amplitude_m=0, cyclic_period_m=.125, phase_rad=.7, tick_m=.0001)
        read = encoder_reading(np.array([.12345, 2.52415]), **kw)
        actual = 2.52415-.12345
        corrected = (read[1]-read[0])/(1+kw['scale_error'])
        self.assertLess(abs(corrected-actual), kw['tick_m'])

    def test_periodic_error_cancels_after_integer_period(self):
        kw = dict(scale_error=0, cyclic_amplitude_m=.0002, cyclic_period_m=.125, phase_rad=.4, tick_m=1e-12)
        delta = encoder_reading(.07+2.5, **kw)-encoder_reading(.07, **kw)
        self.assertAlmostEqual(delta, 2.5, places=10)

    def test_encoder_scale_calibration_uses_separate_measured_displacements(self):
        actual = np.linspace(0, 2.4, 13)
        read = actual*1.0005
        scale = fit_encoder_scale(read, actual)
        self.assertAlmostEqual(scale*2.4007*1.0005, 2.4007)

    def test_outlier_scenario_keeps_all_other_latents_paired(self):
        from scripts.run_cross_station_study import calibration_session
        root = Path(__file__).resolve().parents[2]
        protocol = json.loads((root/'experiments/neck_localization/configs/cross_station_20261028_v1.json').read_text())
        station = json.loads((root/'experiments/neck_localization/configs/three_station_tying_20261028_v1.json').read_text())
        a = calibration_session(np.random.default_rng(31), 'combined', protocol, station)
        b = calibration_session(np.random.default_rng(31), 'combined_survey_outlier', protocol, station)
        np.testing.assert_array_equal(a['Braw'], b['Braw'])
        np.testing.assert_array_equal(a['Fraw'], b['Fraw'])
        self.assertEqual(a['encoder'], b['encoder'])
        np.testing.assert_allclose(b['calibration_measurements']['wide'][-1]-a['calibration_measurements']['wide'][-1], protocol['assumed_bounds']['survey_outlier_m'])
        np.testing.assert_array_equal(a['calibration_measurements']['wide'][:-1], b['calibration_measurements']['wide'][:-1])
        self.assertEqual(protocol['scenario_seed_ids']['combined'], protocol['scenario_seed_ids']['combined_survey_outlier'])


class ToolAndFusionTests(unittest.TestCase):
    def fixture(self, radius):
        return dict(case='one', condition='clean', camera_set='all', accepted=True, reasons=[],
            estimate=dict(center_m=[0, 0, .6], normal=[0, 0, 1], radius_m=radius),
            diagnostics=dict(axis_origin_m=[0, 0, .61], near_min_interval_from_axis_origin_m=[-.013, -.007]))

    def test_composition_keeps_pose_and_changes_only_radius(self):
        a, b = self.fixture(.059), self.fixture(.055)
        b['estimate']['center_m'][0] = .002
        result = pose_radius_control(a, b)
        self.assertEqual(result['estimate']['center_m'], a['estimate']['center_m'])
        self.assertEqual(result['estimate']['normal'], a['estimate']['normal'])
        self.assertEqual(result['estimate']['radius_m'], .055)
        self.assertEqual(a['estimate']['radius_m'], .059)

    def test_failed_channel_is_not_silently_accepted(self):
        a, b = self.fixture(.059), self.fixture(.055)
        b.update(accepted=False, reasons=['bad_radius_channel'])
        result = pose_radius_control(a, b)
        self.assertFalse(result['accepted'])
        self.assertIn('bad_radius_channel', result['reasons'])

    def test_different_exposures_cannot_be_composed(self):
        a, b = self.fixture(.059), self.fixture(.055)
        b['condition'] = 'different'
        with self.assertRaises(ValueError):
            pose_radius_control(a, b)

    def test_refit_uses_FS_points_in_actual_pose_slab(self):
        a, b = self.fixture(.059), self.fixture(.057)
        theta = np.linspace(0, 2*np.pi, 101)[:-1]
        good = np.c_[.055*np.cos(theta), .055*np.sin(theta), np.full(100, .6)]
        irrelevant = good+[.05, 0, .01]
        b['diagnostics']['fit_point_positions_m'] = np.r_[good, irrelevant].tolist()
        result = pose_radius_control(a, b, refit=True)
        self.assertAlmostEqual(result['estimate']['radius_m'], .055, places=10)
        self.assertEqual(result['diagnostics']['channel_composition']['slab_points'], 100)

    def test_clipping_checks_triangles_crossing_slab_with_no_inside_vertex(self):
        triangles = np.array([[[2, 0, -1], [0, 0, 1], [0, 1, 1]]], float)
        result = slab_radial_envelope(triangles, -.1, .1)
        self.assertGreater(result, 1.)
        self.assertLess(result, 1.3)

    def test_outward_open_jaws_never_intrude_into_nominal_bore(self):
        gun = GunSpec()
        for angle in np.linspace(0, 60, 61):
            for arc in jaw_arcs(gun, angle):
                self.assertGreaterEqual(np.linalg.norm(arc[:, :2], axis=1).min(), gun.arc_radius_m-1e-15)


if __name__ == '__main__':
    unittest.main()
