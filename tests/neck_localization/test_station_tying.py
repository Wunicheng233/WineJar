import copy
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from neck_localization.jar_model import JarSpec, target_metadata
from neck_localization.station_tying import GunSpec, frame, jaw_arcs, jaw_metrics, move_circle, plan_tying, pose, timeline
from neck_localization.tying_evaluation import factory_filter, evaluate_transfer


ROOT = Path(__file__).resolve().parents[2]


class StationContractTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'experiments/neck_localization/configs/three_station_tying_20261028_v1.json').read_text())
        self.circle = dict(center_m=[.01, -.02, .592], normal=[0, 0, 1], radius_m=.0525)
        self.obs = dict(jar_id='fixed_id', time_s=0, calibration_id='test', accepted=True, estimate=self.circle,
                        mouth_seed_m=[.01, -.02, .605])

    def test_factory_endpoints_and_total_angle(self):
        for length in (.006, .010):
            self.assertTrue(factory_filter(JarSpec(neck_length_m=length, neck_tilt_deg=(0, 15)))['retained'])
        self.assertFalse(factory_filter(JarSpec(neck_length_m=.01002))['retained'])
        self.assertFalse(factory_filter(JarSpec(neck_length_m=.00599))['retained'])
        # Both components satisfy 15, but the actual normal does not.
        self.assertFalse(factory_filter(JarSpec(neck_tilt_deg=(12, 12)))['retained'])

    def test_inner_aperture_and_opening_clearance(self):
        gun = GunSpec()
        metrics = jaw_metrics(gun)
        self.assertAlmostEqual(metrics['closed_inner_diameter_m'], .15)
        self.assertAlmostEqual(metrics['open_tip_clearance_m'], .15)
        closed = jaw_arcs(gun, 0)
        self.assertTrue(np.allclose(np.linalg.norm(closed[0], axis=1)-gun.head_tube_radius_m, .075))
        # The two arcs retain the same rear hinge, even while open.
        arcs = jaw_arcs(gun, 60)
        np.testing.assert_allclose(arcs[0][-1], arcs[1][-1], atol=1e-15)
        self.assertLess(metrics['polygon_sagitta_m'], .000025)

    def test_base_flange_chain_closes(self):
        plan = plan_tying(self.obs, self.config)
        gun = GunSpec(**self.config['gun'])
        W_T_G = np.linalg.inv(np.array(self.config['B_T_W'])) @ np.array(plan['B_T_F']) @ gun.T_F_G
        np.testing.assert_allclose(W_T_G, plan['W_T_G_t2'], atol=1e-14)
        np.testing.assert_allclose(plan['circle_t2']['center_m'], [2.41, -.02, .592], atol=1e-14)
        self.assertEqual(plan['jar_id'], 'fixed_id')
        self.assertEqual(plan['post_cover_neck_observations'], 0)

    def test_tilted_descent_is_not_world_vertical(self):
        normal = Rotation.from_euler('y', 15, degrees=True).apply([0, 0, 1])
        self.obs['estimate']['normal'] = normal.tolist()
        plan = plan_tying(self.obs, self.config)
        d = np.array(plan['W_T_above'])[:3, 3]-np.array(plan['W_T_G_t2'])[:3, 3]
        np.testing.assert_allclose(d/np.linalg.norm(d), normal, atol=1e-14)
        self.assertGreater(abs(d[0]), .01)
        r = np.array(plan['W_T_G_t2'])[:3, :3]
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-14)

    def test_common_rigid_transform_preserves_errors(self):
        truth = self.circle
        predicted = copy.deepcopy(truth)
        predicted['center_m'] = (np.array(truth['center_m'])+[.001, -.002, .003]).tolist()
        predicted['normal'] = Rotation.from_euler('x', 2, degrees=True).apply(truth['normal']).tolist()
        T = pose(Rotation.from_euler('xyz', [25, -11, 67], degrees=True).as_matrix(), [1.7, -.8, .4])
        a, b = move_circle(truth, T), move_circle(predicted, T)
        self.assertAlmostEqual(np.linalg.norm(np.array(a['center_m'])-b['center_m']), np.sqrt(14)*.001)
        self.assertAlmostEqual(np.rad2deg(np.arccos(np.array(a['normal']) @ b['normal'])), 2, places=10)

    def test_blocked_is_never_converted_to_motion(self):
        plan = plan_tying(self.obs | dict(accepted=False, reasons=['test_gate']), self.config)
        self.assertEqual(plan['status'], 'blocked_localization')
        self.assertEqual(plan['waypoints'], [])
        self.assertFalse(timeline(self.config, plan, 25)['tool_visible'])
        self.assertFalse(plan['execution_authorized'])
        self.assertIsNone(plan['fastening_success'])

    def test_cover_precedes_tying_no_observation_replacement(self):
        plan = plan_tying(self.obs, self.config)
        self.assertFalse(timeline(self.config, plan, 7)['covered'])
        self.assertTrue(timeline(self.config, plan, 12)['covered'])
        closed = timeline(self.config, plan, 24)
        self.assertTrue(closed['covered'])
        self.assertAlmostEqual(closed['displacement_m'][0], 2.4)
        self.assertEqual(self.obs['estimate'], plan['circle_t0'])

    def test_band_envelope_rejects_large_tilt_despite_exact_center(self):
        truth = target_metadata(JarSpec(neck_length_m=.006))
        circle = {k: truth[k] for k in ('center_m', 'normal', 'radius_m')}
        obs = self.obs | dict(estimate=circle, mouth_seed_m=(np.array(circle['center_m'])+[0, 0, .015]).tolist())
        plan = plan_tying(obs, self.config)
        evaluation = evaluate_transfer(plan, truth, truth['search_interval_s_m'], {'normal_deg': 0}, self.config)
        self.assertTrue(evaluation['physical_band']['contained'])
        obs = copy.deepcopy(obs)
        obs['estimate']['normal'] = Rotation.from_euler('y', 3, degrees=True).apply([0, 0, 1]).tolist()
        evaluation = evaluate_transfer(plan_tying(obs, self.config), truth, truth['search_interval_s_m'], {'normal_deg': 3}, self.config)
        self.assertFalse(evaluation['physical_band']['contained'])

    def test_truth_fields_do_not_affect_planner(self):
        baseline = plan_tying(self.obs, self.config)
        poisoned = self.obs | dict(evaluation_only={'center_m': [999, 999, 999]}, target={'normal': [1, 0, 0]})
        self.assertEqual(baseline, plan_tying(poisoned, self.config))

    def test_invalid_frame_and_tool_are_rejected(self):
        with self.assertRaises(ValueError):
            GunSpec(head_tube_radius_m=-.001)
        with self.assertRaises(ValueError):
            jaw_arcs(GunSpec(), 61)
        with self.assertRaises(ValueError):
            pose(np.diag([1, 1, -1]))


if __name__ == '__main__':
    unittest.main()
