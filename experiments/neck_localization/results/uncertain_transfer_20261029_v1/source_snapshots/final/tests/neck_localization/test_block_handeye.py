import unittest
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from neck_localization.block_handeye import solve_block_handeye
from neck_localization.fixed_camera_calibration import projection
from neck_localization.station_tying import pose


@unittest.skipUnless(hasattr(cv2,'calibrateHandEye'),'OpenCV 4 hand-eye API required')
class BlockHandEyeTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(109)
        X=pose(Rotation.from_euler('xyz',[.2,-.15,.3]).as_matrix(),[.4,.2,.1])
        Y=pose(Rotation.from_euler('xyz',[-.1,.15,.1]).as_matrix(),[.03,.01,.08])
        a,b=np.meshgrid(np.arange(5)*.02,np.arange(4)*.02)
        points=np.c_[a.ravel(),b.ravel(),np.zeros(a.size)]
        K=np.array([[700,0,319.5],[0,700,239.5],[0,0,1.]])
        board=[pose(Rotation.from_rotvec(rng.normal(0,.25,3)).as_matrix(),[0,0,.6]+rng.normal(0,.03,3)) for _ in range(18)]
        A=np.array([X @ T @ np.linalg.inv(Y) for T in board])
        pixels=np.array([projection(T,points,K) for T in board])
        return A,pixels,points,K,X,Y

    def test_recovers_independent_camera_and_mount_in_ideal_case(self):
        A,uv,p,K,X,Y=self.data()
        r=solve_block_handeye(A,uv,p,K)
        self.assertTrue(r['accepted'])
        np.testing.assert_allclose(r['B_T_C'],X,atol=2e-7)
        np.testing.assert_allclose(r['F_T_P'],Y,atol=2e-7)

    def test_coherent_bad_view_is_downweighted_without_true_index_input(self):
        A,uv,p,K,X,Y=self.data()
        uv+=np.random.default_rng(39).normal(0,.15,uv.shape)
        uv[8]+=[4,-3]
        r=solve_block_handeye(A,uv,p,K)
        self.assertTrue(r['accepted'])
        self.assertLess(r['view_weights'][8],np.median(r['view_weights'])*.05)
        # Test on a separate camera-space point, rather than on fitted pixels.
        query=np.array([.04,-.03,.7,1.])
        self.assertLess(np.linalg.norm((r['B_T_C'] @ query-X @ query)[:3]),.002)


if __name__=='__main__':unittest.main()
