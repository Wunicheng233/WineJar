import unittest

import numpy as np

from neck_localization.geometry import basis
from neck_localization.sor_baseline import SorProblem, SORSettings, _readout


def sample_surface(noisy=False):
    rng = np.random.default_rng(6201)
    normal = np.array([.09, -.06, 1.]); normal /= np.linalg.norm(normal)
    u, v = basis(normal)
    origin = np.array([3., -2., 650.])
    # Three separated camera-facing arcs, no complete synthetic rings.
    ids = np.repeat(np.arange(3), 180)
    angle = ids*2*np.pi/3+rng.uniform(-.65, .65, len(ids))
    s = rng.uniform(-22., -2., len(ids))
    radius = 50.+.045*(s+12.)**2
    points = origin+s[:, None]*normal+radius[:, None]*(np.cos(angle)[:, None]*u+np.sin(angle)[:, None]*v)
    if noisy:
        points += rng.normal(0, .08, points.shape)
        outliers = rng.choice(len(points), 40, replace=False)
        points[outliers] += rng.normal(0, 5., (len(outliers), 3))
    return points, ids, origin, normal


class StandardSorTests(unittest.TestCase):
    def test_joint_axis_and_curve_jacobian(self):
        points, ids, origin, normal = sample_surface()
        problem = SorProblem(points, origin+[2., -1., 0.], [0., 0., 1.], ids)
        x = problem.initial()
        x[:4] = [.7, -.8, .02, -.03]
        analytic = problem.jac(x)
        numerical = np.empty_like(analytic)
        for j in range(len(x)):
            h = 1e-5 if j < 2 or j >= 4 else 1e-7
            plus, minus = x.copy(), x.copy()
            plus[j] += h; minus[j] -= h
            numerical[:, j] = (problem.fun(plus)-problem.fun(minus))/(2*h)
        self.assertLess(np.max(abs(analytic-numerical)), 2e-5)

    def test_recovers_tilted_axis_and_functional_section_from_partial_arcs(self):
        points, ids, origin, normal = sample_surface()
        problem = SorProblem(points, origin+[2., -1., 0.], [0., 0., 1.], ids)
        fit, state = problem.fit()
        middle, radius, *_ = _readout(problem, state, points, problem.settings)
        center = state['origin']+middle*state['normal']
        self.assertTrue(fit.success)
        self.assertLess(np.linalg.norm(center-(origin-12*normal)), .15)
        self.assertLess(np.degrees(np.arccos(np.clip(normal @ state['normal'], -1, 1))), .05)
        self.assertLess(abs(radius-50.), .05)

    def test_huber_remains_stable_with_sparse_large_outliers(self):
        points, ids, origin, normal = sample_surface(noisy=True)
        problem = SorProblem(points, origin+[2., -1., 0.], [0., 0., 1.], ids)
        fit, state = problem.fit()
        middle, radius, *_ = _readout(problem, state, points, problem.settings)
        center = state['origin']+middle*state['normal']
        self.assertTrue(fit.success)
        self.assertLess(np.linalg.norm(center-(origin-12*normal)), .7)
        self.assertLess(np.degrees(np.arccos(np.clip(normal @ state['normal'], -1, 1))), .5)
        self.assertLess(abs(radius-50.), .3)

    def test_huber_only_applies_to_measurement_residuals(self):
        points, ids, origin, normal = sample_surface()
        problem = SorProblem(points, origin, normal, ids)
        z = np.full(len(points)+len(problem.regularizer), 9.)
        value = problem.loss(z)
        np.testing.assert_allclose(value[:, :len(points)], np.array([[5.], [1/3], [-1/54]])*np.ones((1, len(points))))
        np.testing.assert_allclose(value[:, len(points):], np.array([[9.], [1.], [0.]])*np.ones((1, len(problem.regularizer))))


if __name__ == '__main__':
    unittest.main()
