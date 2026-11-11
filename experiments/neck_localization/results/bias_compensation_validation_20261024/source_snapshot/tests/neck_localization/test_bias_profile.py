import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative

from neck_localization.bias_profile import BiasAxisSplineProblem, observation_rays
from neck_localization.depth_profile import DepthFrame
from neck_localization.geometry import look_at
from neck_localization.joint_profile import JointSettings
from neck_localization.support_feedback import WeightedAxisSplineProblem


def problem_fixture(sigma_bias_mm=np.inf):
    rng = np.random.default_rng(8841)
    frames = [DepthFrame(look_at(str(i), p, [0, 0, .6]), None, None)
              for i, p in enumerate([[.7, -.4, .75], [.7, .4, .75], [-.8, 0, .72]])]
    ids = np.repeat(np.arange(3), 120)
    phase = rng.uniform(-np.pi, np.pi, len(ids))
    z = rng.uniform(-7, 7, len(ids))
    radius = 50+.01*z*z
    points = np.c_[radius*np.cos(phase), radius*np.sin(phase), 600+z]
    rays = observation_rays(points/1000, ids, frames)
    base = WeightedAxisSplineProblem(points, [0, 0, 610], [0, 0, 1], ids, JointSettings())
    base.set_weights(np.ones(len(ids)))
    return BiasAxisSplineProblem(base, ids, rays, sigma_bias_mm=sigma_bias_mm), frames


class BiasProfileTests(unittest.TestCase):
    def test_analytic_variable_projection_jacobian(self):
        p, _ = problem_fixture(2)
        x = np.array([.1, -.2, .015, -.01, .8, -1.2, .3])
        actual = p.jac(x)
        # Tangent-normal coordinates move points much farther than mm-valued
        # translations; convergence scan selects a smaller angular FD step.
        step = np.array([1e-4, 1e-4, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4])
        expected = approx_derivative(p.fun, x, method='3-point', abs_step=step)
        np.testing.assert_allclose(actual, expected, atol=4e-7, rtol=2e-5)

    def test_zero_bias_matches_existing_profile_problem(self):
        p, _ = problem_fixture()
        x = np.array([.1, -.2, .015, -.01])
        old = p.base.evaluate(x)
        new = p.evaluate(np.r_[x, [0, 0, 0]])
        np.testing.assert_array_equal(old['residual'], new['residual'])
        np.testing.assert_allclose(old['jacobian'], new['jacobian'][:, :4], atol=1e-12)

    def test_bias_correction_moves_only_own_camera_along_ray(self):
        p, _ = problem_fixture()
        x = np.array([0, 0, 0, 0, 2, 0, 0.])
        state = p.evaluate(x)
        changed = p.base.points-state['corrected_points_mm']
        np.testing.assert_allclose(changed[p.ids==0], 2*p.rays[p.ids==0], atol=1e-12)
        np.testing.assert_array_equal(changed[p.ids!=0], 0)

    def test_data_information_does_not_include_prior(self):
        a, _ = problem_fixture(np.inf)
        b, _ = problem_fixture(.2)
        x = np.zeros(7)
        self.assertEqual(a.data_information(x), b.data_information(x))
        self.assertEqual(len(a.data_information(x)['scaled_data_singular_values']), 7)


if __name__ == '__main__':
    unittest.main()
