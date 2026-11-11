import unittest
import numpy as np
from neck_localization.bounded_transport import bounded_translation,gaussian_shrinkage
from neck_localization.poe_camera_calibration import CameraPOE,solve_camera_target
from neck_localization.station_tying import pose
from scipy.spatial.transform import Rotation


class BoundedTransportTests(unittest.TestCase):
    def test_interval_contains_truth_under_arbitrarily_correlated_bounded_errors(self):
        for a,b in [(.0005,.003),(.002,.0005),(.002,.003)]:
            for truth in np.linspace(-a,a,13):
                for bias in np.linspace(-b,b,17):
                    r=bounded_translation(truth+bias,a,b);lo,hi=r['interval_m']
                    self.assertLessEqual(lo-1e-14,truth);self.assertGreaterEqual(hi+1e-14,truth)
                    self.assertLessEqual(abs(r['delta_m']-truth),(hi-lo)/2+1e-14)

    def test_inconsistent_observation_preserves_nominal_output_without_process_release(self):
        r=bounded_translation(.020,.0005,.001)
        self.assertEqual(r['delta_m'],0.);self.assertEqual(r['status'],'nominal_fallback_inconsistent_bounds')
        self.assertFalse(r['process_release'])

    def test_missing_observation_is_not_reused_as_last_jar(self):
        r=bounded_translation(np.nan,.002,.001)
        self.assertEqual(r['delta_m'],0.);self.assertEqual(r['status'],'nominal_fallback_no_observation')

    def test_unknown_bias_variance_reduces_shrinkage_gain(self):
        a=gaussian_shrinkage(.003,.0005,.003);b=gaussian_shrinkage(.003,.0005,.0001)
        self.assertLess(a['gain'],b['gain']);self.assertLess(a['delta_m'],.0001)


class CameraPOETests(unittest.TestCase):
    def model(self):
        axes=np.array([[0,0,1],[0,1,0],[0,1,0],[1,0,0],[0,1,0],[0,0,1.]])
        points=np.array([[0,0,0],[0,0,.2],[.1,0,.2],[.2,0,.2],[.2,0,.25],[.2,0,.3]])
        return CameraPOE(axes,points,pose(translation=[.2,0,.4]))

    def test_batched_forward_and_home_pose(self):
        m=self.model();q=np.random.default_rng(25).uniform(-.2,.2,(5,6))
        np.testing.assert_allclose(m.forward(np.zeros(6)),m.home,atol=1e-14)
        np.testing.assert_allclose(m.forward(q),np.array([m.forward(v) for v in q]),atol=1e-14)

    def test_corrected_inverse_reaches_same_independent_pose(self):
        m=self.model();q=np.array([.1,-.2,.15,-.1,.1,.2]);target=m.forward(q)
        r=solve_camera_target(m,target,q+.02,np.tile([-1.,1.],(6,1)))
        self.assertTrue(r['passed']);np.testing.assert_allclose(m.forward(r['q_rad']),target,atol=2e-6)

    def test_axis_line_chart_remains_unit_and_transverse(self):
        m=self.model();n,p,H=m.decode(np.random.default_rng(28).normal(0,.1,30))
        np.testing.assert_allclose(np.linalg.norm(n,axis=1),1,atol=1e-14)
        np.testing.assert_allclose(np.sum(n*p,axis=1),0,atol=1e-14)


if __name__=='__main__':unittest.main()
