import unittest
import cv2
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from neck_localization.uncertain_handeye import HandEyeResidual,solve_uncertain_eye_to_hand
from neck_localization.fixed_camera_calibration import projection
from neck_localization.station_tying import pose
from scipy.spatial.transform import Rotation


@unittest.skipUnless(hasattr(cv2,'calibrateHandEye'),'OpenCV 4 hand-eye API required')
class UncertainHandEyeTests(unittest.TestCase):
    def data(self,noise=False):
        rng=np.random.default_rng(62);X=pose(Rotation.from_euler('xyz',[.2,.1,-.3]).as_matrix(),[.4,.2,.1]);Y=pose(translation=[.03,.01,.08])
        a,b=np.meshgrid(np.arange(5)*.02,np.arange(4)*.02);points=np.c_[a.ravel(),b.ravel(),np.zeros(a.size)]
        K=np.array([[700,0,319.5],[0,700,239.5],[0,0,1.]])
        P=[pose(Rotation.from_rotvec(rng.normal(0,.2,3)).as_matrix(),[0,0,.6]+rng.normal(0,.02,3)) for i in range(12)]
        A=np.array([X @t @np.linalg.inv(Y) for t in P]);pixels=np.array([projection(t,points,K) for t in P])
        if noise:pixels+=rng.normal(0,.1,pixels.shape)
        return A,pixels,points,K,X,Y

    def test_analytic_jacobian_including_nonzero_pose_corrections(self):
        A,uv,p,K,X,Y=self.data();m=HandEyeResidual(A,uv,p,K,X,Y,[.15,.001,.002])
        x=np.random.default_rng(6).normal(0,.08,m.size)
        a=m.evaluate(x,True);b=approx_derivative(m.evaluate,x,method='3-point')
        self.assertLess(np.linalg.norm(a-b)/np.linalg.norm(b),1e-7)

    def test_ideal_camera_and_board_mount(self):
        A,uv,p,K,X,Y=self.data();r=solve_uncertain_eye_to_hand(A,uv,p,K,variance_components=False)
        self.assertTrue(r['accepted']);np.testing.assert_allclose(r['B_T_C'],X,atol=2e-7)
        np.testing.assert_allclose(r['F_T_P'],Y,atol=2e-7)

    def test_covariance_is_joint_psd_and_not_fixed_bias_guarantee(self):
        A,uv,p,K,X,Y=self.data(True);r=solve_uncertain_eye_to_hand(A,uv,p,K,max_outer=3)
        covariance=np.array(r['joint_X_Y_covariance'])
        self.assertEqual(covariance.shape,(12,12));self.assertGreater(np.linalg.norm(covariance[:6,6:]),0)
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(),-1e-12)
        self.assertTrue(r['covariance_excludes_fixed_bias_and_model_mismatch'])


if __name__=='__main__':unittest.main()
