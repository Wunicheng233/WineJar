import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from neck_localization.calibration_uncertainty import common_pose_covariance


class CommonCalibrationUncertaintyTests(unittest.TestCase):
    def test_shared_pose_jacobian_matches_se3_finite_difference(self):
        c=np.array([.02,-.01,.6]);n=np.array([.1,0,.99]);n/=np.linalg.norm(n)
        result=common_pose_covariance(c,n,np.eye(6))
        numeric=np.empty((6,6))
        def state(x):
            R=Rotation.from_rotvec(x[3:]).as_matrix()
            return np.r_[R@c+x[:3],R@n]
        for k in range(6):
            x=np.zeros(6);x[k]=1e-6
            numeric[:,k]=(state(x)-state(-x))/2e-6
        np.testing.assert_allclose(result['jacobian'],numeric,atol=1e-10)

    def test_translation_and_rotation_have_distinct_effects(self):
        C=np.zeros((6,6));C[:3,:3]=np.eye(3)*1e-6
        result=common_pose_covariance([0,0,.6],[0,0,1],C)
        np.testing.assert_array_equal(result['center_covariance'],np.eye(3)*1e-6)
        np.testing.assert_array_equal(result['normal_covariance'],0)
        C=np.zeros((6,6));C[4,4]=1e-6
        result=common_pose_covariance([0,0,.6],[0,0,1],C)
        self.assertAlmostEqual(result['center_covariance'][0,0],.36e-6)
        self.assertAlmostEqual(result['center_normal_cross_covariance'][0,0],.6e-6)

    def test_invalid_covariance_is_rejected(self):
        C=np.eye(6);C[0,0]=-1
        with self.assertRaises(ValueError):common_pose_covariance([0,0,.6],[0,0,1],C)


if __name__=='__main__':unittest.main()
