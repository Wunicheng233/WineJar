"""Analytic geometry checks, not evidence of factory localization accuracy."""
from dataclasses import replace
import unittest

import numpy as np
from scipy.optimize import least_squares

from neck_localization.geometry import basis, look_at
from neck_localization.silhouette_profile import (
    SilhouetteSettings, SilhouetteSplineProblem, contour_generator,
)


def smooth_observations(axis_point, axis_normal):
    u, v = basis(axis_normal)
    centers, rays, normals, world = [], [], [], []
    for angle in [0., 2.1, 4.2]:
        camera = np.array([.8*np.cos(angle), .8*np.sin(angle), .72])
        q = camera-axis_point
        qperp = q-(q @ axis_normal)*axis_normal
        phi = np.arctan2(qperp @ v, qperp @ u)
        for s in np.linspace(-.02, .02, 31):
            radius, slope = .05+2*s*s, 4*s
            cosine = (radius+slope*(axis_normal @ q-s))/np.linalg.norm(qperp)
            for sign in [-1, 1]:
                direction = phi+sign*np.arccos(cosine)
                e = np.cos(direction)*u+np.sin(direction)*v
                point = axis_point+s*axis_normal+radius*e
                normal = e-slope*axis_normal
                centers.append(camera)
                rays.append(point-camera)
                normals.append(normal)
                world.append(point)
    return tuple(np.array(x) for x in [centers, rays, normals]), np.array(world)


class SilhouetteGeometryTests(unittest.TestCase):
    def test_tangent_condition_and_reconstruction(self):
        point = np.array([.005, -.004, .6])
        normal = np.array([.03, -.07, 1.])
        normal /= np.linalg.norm(normal)
        samples, world = smooth_observations(point, normal)
        centers, rays, tangent_normals = samples
        np.testing.assert_allclose(np.sum(tangent_normals*rays, axis=1), 0, atol=1e-14)
        _, _, recovered = contour_generator(point, normal, *samples)
        np.testing.assert_allclose(recovered, world, atol=2e-14)
        _, _, opposite = contour_generator(point, normal, centers, rays, -tangent_normals)
        np.testing.assert_allclose(opposite, world, atol=2e-14)

    def test_multiview_axis_fit_without_radius_prior(self):
        point = np.array([.005, -.004, .6])
        normal = np.array([.03, -.07, 1.])
        normal /= np.linalg.norm(normal)
        samples, _ = smooth_observations(point, normal)
        # Input is exact analytic contour/ray/tangent data. Neither radius nor
        # axis is passed to the fitted problem. This is an ideal unit test only.
        origin = np.array([0., 0., .6])
        settings = replace(SilhouetteSettings(), spline_second_difference_weight=0.)
        problem = SilhouetteSplineProblem(origin, samples, settings)
        fit = least_squares(problem.evaluate, np.zeros(4), max_nfev=100,
                            xtol=1e-12, ftol=1e-12, gtol=1e-12)
        self.assertTrue(fit.success)
        self.assertLess(np.sqrt(np.mean(problem.evaluate(fit.x)**2)), 1e-5)
        np.testing.assert_allclose(fit.x[:2], [5, -4], atol=1e-3)
        np.testing.assert_allclose(fit.x[2:], normal[:2]/normal[2], atol=1e-5)

    def test_single_camera_scale_ambiguity(self):
        camera = look_at('example', [.8, 0, .75], [0, 0, .6])
        points = np.array([[0., 0., .6], [.04, -.02, .61], [-.03, .05, .59]])
        center = camera.T_world_from_camera[:3, 3]
        scaled = center+1.15*(points-center)
        np.testing.assert_allclose(camera.project(points), camera.project(scaled), atol=1e-12)

    def test_image_tangent_plane_backprojection(self):
        point = np.array([.005, -.004, .6])
        normal = np.array([.03, -.07, 1.])
        normal /= np.linalg.norm(normal)
        u, v = basis(normal)
        camera = look_at('example', [.8, 0, .75], [0, 0, .6])
        center = camera.T_world_from_camera[:3, 3]
        q = center-point
        perpendicular = q-(q @ normal)*normal
        phi = np.arctan2(perpendicular @ v, perpendicular @ u)
        surface = []
        for s in [-1e-5, 0., 1e-5]:
            radius, slope = .05+2*s*s, 4*s
            angle = phi+np.arccos((radius+slope*(normal @ q-s))/np.linalg.norm(perpendicular))
            surface.append(point+s*normal+radius*(np.cos(angle)*u+np.sin(angle)*v))
        pixels = camera.project(surface)
        line = np.cross(np.r_[pixels[0], 1], np.r_[pixels[2], 1])
        tangent_normal = line @ camera.K @ camera.T_world_from_camera[:3, :3].T
        _, _, recovered = contour_generator(point, normal, center[None],
                                             (np.array(surface[1])-center)[None], tangent_normal[None])
        np.testing.assert_allclose(recovered[0], surface[1], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
