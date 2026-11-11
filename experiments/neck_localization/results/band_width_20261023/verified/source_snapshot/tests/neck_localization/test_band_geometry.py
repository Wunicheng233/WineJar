import unittest
import numpy as np

from neck_localization.band_geometry import plan_band, evaluate_band


class BandGeometryTests(unittest.TestCase):
    def fixture(self, length=.010, width=.005):
        estimate = dict(center_m=[0, 0, .6], normal=[0, 0, 1], radius_m=.05)
        diagnostics = dict(axis_origin_m=[0, 0, .61],
                           near_min_interval_from_axis_origin_m=[-.01-length/2, -.01+length/2])
        truth = dict(estimate, s_m=.6)
        return plan_band(estimate, diagnostics, width), truth

    def test_width_erodes_interval_not_radius(self):
        band, truth = self.fixture()
        np.testing.assert_allclose(band['allowed_center_shift_interval_m'], [-.0025, .0025])
        self.assertEqual(band['radius_m'], .05)
        score = evaluate_band(band, truth, [.595, .605])
        self.assertAlmostEqual(score['minimum_margin_mm'], 2.5)

    def test_short_interval_and_exact_boundary(self):
        short, truth = self.fixture(.004)
        self.assertFalse(short['nominal_compatible'])
        self.assertIsNone(short['allowed_center_shift_interval_m'])
        self.assertFalse(evaluate_band(short, truth, [.598, .602])['ideal_true_axis_center_can_fit'])
        equal, truth = self.fixture(.005)
        self.assertTrue(equal['nominal_compatible'])
        self.assertTrue(evaluate_band(equal, truth, [.5975, .6025])['contained'])

    def test_center_error_causes_false_compatibility(self):
        band, truth = self.fixture(.006)
        band['center_m'][2] += .001
        score = evaluate_band(band, truth, [.597, .603])
        self.assertTrue(score['false_compatible'])
        self.assertAlmostEqual(score['max_axial_overhang_mm'], .5)
        self.assertTrue(score['ideal_true_axis_center_can_fit'])

    def test_tilt_envelope_matches_dense_whole_strip(self):
        band, truth = self.fixture()
        angle = .09
        n = np.array([np.sin(angle), 0, np.cos(angle)])
        band['normal'] = n.tolist()
        band['center_m'][2] += .0008
        score = evaluate_band(band, truth, [.595, .605])
        u, v = np.array([np.cos(angle), 0, -np.sin(angle)]), np.array([0, 1, 0])
        phase = np.linspace(0, 2*np.pi, 20001)
        ring = .05*(np.cos(phase)[:, None]*u+np.sin(phase)[:, None]*v)
        points = np.concatenate([ring+q*n+band['center_m'] for q in np.linspace(-.0025, .0025, 11)])
        axial = (points-np.array(truth['center_m'])) @ np.array(truth['normal'])*1000
        np.testing.assert_allclose(score['actual_strip_axial_envelope_relative_truth_center_mm'],
                                   [axial.min(), axial.max()], atol=1e-10)

    def test_normal_sign_does_not_change_strip(self):
        band, truth = self.fixture()
        first = evaluate_band(band, truth, [.595, .605])
        band['normal'] = [0, 0, -1]
        second = evaluate_band(band, truth, [.595, .605])
        self.assertEqual(first, second)

    def test_invalid_width_and_missing_observed_range(self):
        estimate = dict(center_m=[0, 0, 0], normal=[0, 0, 1], radius_m=.05)
        with self.assertRaises(ValueError):
            plan_band(estimate, {}, -.001)
        with self.assertRaises(KeyError):
            plan_band(estimate, {}, .005)


if __name__ == '__main__':
    unittest.main()
