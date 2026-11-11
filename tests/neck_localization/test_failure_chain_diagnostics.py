import sys
import unittest
from unittest.mock import patch

import numpy as np

from neck_localization.geometry import basis
from neck_localization.joint_profile import JointSettings
from scripts.diagnose_structural_failure_chain import axis_at_z, read_curve, replay_capture


class FailureChainDiagnosticTests(unittest.TestCase):
    def test_exact_cylinder_readout_for_inclined_axis(self):
        n = np.array([.11, -.17, 1.]); n /= np.linalg.norm(n)
        a = np.array([3., -4., 600.]); u,v = basis(n)
        ss,angles = np.meshgrid(np.linspace(-20,-2,31),np.linspace(0,2*np.pi,21,endpoint=False))
        s,t=ss.ravel(),angles.ravel()
        points=a+s[:,None]*n+51*(np.cos(t)[:,None]*u+np.sin(t)[:,None]*v)
        estimate,d=read_curve(points,np.zeros(len(points),int),a,n,JointSettings(),
            np.ones(len(points)),(-18.,-4.))
        self.assertAlmostEqual(estimate.radius*1000,51.,places=7)
        np.testing.assert_allclose(estimate.center*1000,a+(-11.05)*n,atol=1e-8)
        self.assertFalse(estimate.accepted)
        self.assertTrue(d['boundary'])  # Flat interval touches both supplied bounds.

    def test_oracle_axis_gauge_keeps_same_line(self):
        n=np.array([.2,.1,1.]);n/=np.linalg.norm(n)
        truth=dict(normal=n.tolist(),center_m=[.01,-.02,.61])
        a,returned=axis_at_z(truth,625.)
        self.assertAlmostEqual(a[2],625.)
        delta=a-np.array(truth['center_m'])*1000
        np.testing.assert_allclose(delta-(delta@n)*n,0,atol=1e-12)
        np.testing.assert_array_equal(returned,n)

    def test_profile_hook_restored_if_estimator_raises(self):
        previous=sys.getprofile()
        with patch('scripts.diagnose_structural_failure_chain.portfolio.estimate_portfolio',side_effect=RuntimeError('intentional')):
            with self.assertRaises(RuntimeError):replay_capture([])
        self.assertIs(sys.getprofile(),previous)


if __name__=='__main__':unittest.main()
