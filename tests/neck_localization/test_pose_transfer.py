import unittest
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from neck_localization.station_tying import pose,move_circle
from neck_localization.body_relocalization import register_body,voxel_points
from neck_localization.fixed_camera_calibration import solve_eye_to_hand


class BodyTransferTests(unittest.TestCase):
    def cloud(self):
        rng=np.random.default_rng(813)
        u=rng.normal(size=(1800,3));u/=np.linalg.norm(u,axis=1,keepdims=True)
        a=np.array([.17,.12,.23]);p=u*a
        n=p/a**2;n/=np.linalg.norm(n,axis=1,keepdims=True)
        return p,n

    def test_axis_recovers_horizontal_translation(self):
        p,n=self.cloud();delta=np.array([.002,0,0])
        r=register_body(p,p+delta,np.eye(4),mode='axis',normals=n)
        self.assertTrue(r['accepted'])
        np.testing.assert_allclose(r['transform'][:3,3],delta,atol=1e-6)

    def test_rigid_recovers_small_general_motion(self):
        p,n=self.cloud();R=Rotation.from_euler('xyz',[.1,.3,-.2],degrees=True).as_matrix();t=np.array([.001,.002,-.001])
        r=register_body(p,p @ R.T+t,np.eye(4),mode='rigid',normals=n @ R.T)
        np.testing.assert_allclose(r['transform'],pose(R,t),atol=1e-5)

    def test_motion_constraint_does_not_invent_missing_vertical_measurement(self):
        p,n=self.cloud();delta=np.array([0,0,.003])
        r=register_body(p,p+delta,np.eye(4),mode='axis',normals=n)
        self.assertEqual(r['transform'][2,3],0)

    def test_cylinder_geometry_has_two_unobservable_directions(self):
        a,z=np.meshgrid(np.linspace(0,2*np.pi,120,endpoint=False),np.linspace(-.15,.15,15))
        p=np.c_[.15*np.cos(a.ravel()),.15*np.sin(a.ravel()),z.ravel()]
        n=np.c_[np.cos(a.ravel()),np.sin(a.ravel()),np.zeros(a.size)]
        r=register_body(p,p,np.eye(4),mode='rigid',normals=n)
        eigen=np.array(r['information_eigenvalues'])
        self.assertLess(abs(eigen[0]),1e-12);self.assertLess(abs(eigen[1]),1e-12)
        self.assertTrue(r['information_is_not_calibrated_covariance'])

    def test_unobserved_body_is_rejected(self):
        r=register_body(np.empty((0,3)),np.ones((80,3)),np.eye(4))
        self.assertFalse(r['accepted']);self.assertEqual(r['reason'],'insufficient_body_observations')

    def test_shared_template_frame_change_preserves_transferred_circle(self):
        circle=dict(center_m=[.01,.02,.58],normal=[.02,0,np.sqrt(1-.02**2)],radius_m=.05)
        H=pose(Rotation.from_euler('y',1,degrees=True).as_matrix(),[.002,-.001,.003])
        T=pose(translation=[2.4,0,0])
        a=move_circle(circle,T);b=move_circle(move_circle(circle,H),T @ np.linalg.inv(H))
        np.testing.assert_allclose(a['center_m'],b['center_m'],atol=1e-14)
        np.testing.assert_allclose(a['normal'],b['normal'],atol=1e-14)

    def test_partial_view_queries_template_without_hidden_surface_labels(self):
        from neck_localization.body_partial_registration import register_observed_subset
        p,n=self.cloud();visible=p[:,0]>.08;delta=np.array([.002,0,0])
        for reciprocal in (False,True):
            r=register_observed_subset(p,p[visible]+delta,n[visible],np.eye(4),[1,0,0],reciprocal=reciprocal)
            self.assertTrue(r['accepted'])
            np.testing.assert_allclose(r['transform'][:3,3],delta,atol=1e-6)


@unittest.skipUnless(hasattr(cv2,'calibrateHandEye'),'OpenCV 4.13 winejar-vision required for hand-eye API')
class FixedCameraTests(unittest.TestCase):
    def data(self,layout='diverse',condition='ideal'):
        from scripts.run_fixed_camera_calibration_study import dataset
        from neck_localization.experiment_io import read
        from pathlib import Path
        protocol=read(Path(__file__).resolve().parents[2]/'experiments/neck_localization/configs/pose_transfer_20261028_v1.json')
        return dataset(protocol,0,layout,condition)

    def test_noiseless_eye_to_hand_direction_and_unknown_board_mount(self):
        d=self.data()
        for method in ['park','horaud','pixel_ls','pixel_huber']:
            r=solve_eye_to_hand(d['robot'][:18],d['pixels'][:18],d['board'],d['K'],method=method)
            self.assertTrue(r['accepted'])
            np.testing.assert_allclose(r['B_T_C'],d['truth_X'],atol=2e-6)
            np.testing.assert_allclose(r['F_T_P'],d['truth_Y'],atol=2e-6)

    def test_one_axis_robot_poses_rejected_before_solver(self):
        d=self.data('one_axis')
        r=solve_eye_to_hand(d['robot'][:18],d['pixels'][:18],d['board'],d['K'])
        self.assertFalse(r['accepted']);self.assertEqual(r['reason'],'insufficient_nonparallel_rotation')

    def test_base_frame_relabel_is_not_physical_error(self):
        d=self.data(condition='gauge_relabel')
        r=solve_eye_to_hand(d['robot'][:18],d['pixels'][:18],d['board'],d['K'],method='pixel_huber')
        np.testing.assert_allclose(np.linalg.inv(d['gauge_relabel']) @ r['B_T_C'],d['truth_X'],atol=2e-6)


if __name__=='__main__':unittest.main()
