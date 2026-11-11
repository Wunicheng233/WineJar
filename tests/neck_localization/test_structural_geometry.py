import unittest
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import BSpline

from neck_localization.geometry import circle_points, look_at
from neck_localization.structural_localization import projected_circle_conic, conic_pixel_residual, top_arc_pixels
from neck_localization.ray_profile import ray_axis_coordinates
from neck_localization.structure_portfolio import radial_variance, PortfolioSettings
from neck_localization.disparity_calibration import DisparityCorrection, DisparitySettings


class StructuralGeometryTests(unittest.TestCase):
    def test_projected_circle_conic_matches_perspective_points(self):
        camera = look_at('a', [.7, -.4, .8], [0, 0, .59], width=848, height=480)
        camera.K[0, 2] += 7.3; camera.K[1, 2] -= 3.1
        n = np.array([.12, -.16, 1.]); n /= np.linalg.norm(n)
        center = np.array([.003, -.006, .604])
        contour = camera.project(circle_points(center, n, .061, 123))
        conic = projected_circle_conic(camera, center*1000, n, 61.)
        np.testing.assert_allclose(conic_pixel_residual(contour, conic), 0., atol=1e-9)

    def test_circle_projection_is_scale_invariant(self):
        camera = look_at('a', [.7, -.4, .8], [0, 0, .59])
        n=np.array([0., 0., 1.]); c=np.array([0., 0., 600.])
        conic=projected_circle_conic(camera,c,n,60)
        p=camera.project(circle_points(c/1000,n,.06,32))
        np.testing.assert_allclose(conic_pixel_residual(p,conic*1000),conic_pixel_residual(p,conic),atol=1e-8)

    def test_upper_arc_is_between_foreground_background_samples(self):
        mask=np.zeros((30,30),bool);mask[10:25,5:25]=True;mask[9:25,10:20]=True
        p=top_arc_pixels(mask,0.)
        np.testing.assert_array_equal(p[:,0],np.arange(10,20))
        np.testing.assert_array_equal(p[:,1],np.full(10,8.5))

    def test_ray_axis_contact_for_exact_cylinder(self):
        radius=50.; normal=np.array([0.,0.,1.]);origin=np.zeros(3)
        # Analytic tangent from C=(300,0,20) to radius50 cylinder.
        C=np.array([[300.,0.,20.]])
        contact=np.array([[radius**2/300, radius*np.sqrt(1-(radius/300)**2),20.]])
        ray=contact-C
        aa,t,rho2,vn,cn=ray_axis_coordinates(origin,normal,C,ray)
        self.assertAlmostEqual(float(t[0]),1.,places=12)
        self.assertAlmostEqual(float(np.sqrt(rho2[0])),radius,places=11)
        self.assertAlmostEqual(float(cn[0]+t[0]*vn[0]),20.,places=12)

    def test_measurement_variance_matches_monte_carlo(self):
        rng=np.random.default_rng(71)
        camera=look_at('a',[.8,0,.61],[0,0,.59],width=848,height=480)
        frame=SimpleNamespace(camera=camera)
        point=np.array([[.04,.03,.59]])
        normal=np.array([.03,-.06,1.]);normal/=np.linalg.norm(normal)
        a=np.array([0,0,590.]);slope=.15; settings=PortfolioSettings(radial_sigma_floor_mm=0.)
        predicted=radial_variance(point*1000,np.array([0]),[frame],a,normal,np.array([slope]),settings)[0]
        uv=camera.project(point)[0]
        local=(point-camera.T_world_from_camera[:3,3])@camera.T_world_from_camera[:3,:3]
        z=local[0,2]
        noisy_uv=uv+rng.normal(0,settings.pixel_sigma,(30000,2))
        sigma_z=settings.optical_depth_sigma_mm_at_800mm*(z/.8)**2/1000
        noisy_z=z+rng.normal(0,sigma_z,30000)
        q=camera.unproject(noisy_uv,noisy_z)*1000-a
        axial=q@normal;g=np.linalg.norm(q-axial[:,None]*normal,axis=1)-slope*axial
        self.assertLess(abs(np.var(g)/predicted-1),.035)

    def test_disparity_correction_preserves_invalid_pixels(self):
        model=DisparityCorrection([.1,0.],21.5,DisparitySettings(harmonics=0))
        z=np.array([[0.,np.nan,.8]],np.float32); result=model.correct(z)
        self.assertEqual(result[0,0],0.)
        self.assertTrue(np.isnan(result[0,1]))
        self.assertAlmostEqual(float(result[0,2]),21.5/(21.5/.8+.1),places=6)


if __name__=='__main__':unittest.main()
