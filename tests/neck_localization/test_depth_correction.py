"""Check author GP equations and calibration/inference separation, not accuracy claims."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from neck_localization.depth_correction import (GPSettings, GaussianProcessDepthCorrection,
    depth_features, rbf_kernel, calibration_samples, load_calibration_manifest)


class GPDepthCorrectionTests(unittest.TestCase):
    def model(self):
        rng = np.random.default_rng(19)
        X = rng.uniform([-.2, -.1, .5, 25], [.2, .1, 1.2, 25], (40, 4))
        target = .002+.004*X[:, 0]-.003*(X[:, 2]-.8)**2
        return GaussianProcessDepthCorrection(GPSettings(optimize=False)).fit(X, target)

    def test_kernel_and_posterior_match_independent_dense_equations(self):
        model = self.model()
        q = model.X[::5]+[.001, -.001, .002, 0]
        difference = (model.X[:, None, :]-q[None, :, :])/model.length_scale
        Ks = model.signal_std_m**2*np.exp(-.5*(difference*difference).sum(axis=2))
        C = rbf_kernel(model.X, model.X, model.length_scale, model.signal_std_m)
        C += np.eye(len(model.X))*(model.noise_std_m**2+1e-14)
        mean, std = model.predict(q, return_std=True)
        target = .002+.004*model.X[:, 0]-.003*(model.X[:, 2]-.8)**2
        np.testing.assert_allclose(mean, Ks.T@np.linalg.solve(C, target-target.mean())+target.mean(), atol=1e-13)
        var = model.signal_std_m**2-np.sum(Ks*np.linalg.solve(C, Ks), axis=0)
        np.testing.assert_allclose(std**2, var, atol=1e-15)

    def test_optical_axis_not_range_and_no_half_pixel_shift(self):
        K = np.array([[100., 0., 1.], [0., 100., 1.], [0., 0., 1.]])
        features = depth_features(np.full((3, 3), .8), K, 26, [[2, 0]])
        np.testing.assert_allclose(features, [[.008, -.008, .8, 26.]])

    def test_invalid_pixels_preserved_roundtrip_and_intrinsic_guard(self):
        model = self.model()
        model.K = np.eye(3)
        depth = np.array([[.8, 0., np.nan], [.9, -.1, np.inf]], np.float32)
        untouched = depth.copy()
        corrected, diagnostics = model.correct_depth(depth, np.eye(3), return_diagnostics=True)
        np.testing.assert_array_equal(depth, untouched)
        np.testing.assert_array_equal(corrected[~(np.isfinite(depth) & (depth > 0))], depth[~(np.isfinite(depth) & (depth > 0))])
        self.assertEqual(diagnostics['valid_input_pixels'], 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.npz'
            model.save(path)
            restored = GaussianProcessDepthCorrection.load(path)
            np.testing.assert_array_equal(restored.correct_depth(depth, np.eye(3)), corrected)
        with self.assertRaises(ValueError):
            model.correct_depth(depth, np.diag([2, 2, 1]))

    def test_plane_sampling_uses_only_explicit_reference_mask(self):
        depth = np.ones((16, 20))
        reference = depth+.004
        mask = np.zeros(depth.shape, bool)
        mask[6:10, 8:12] = True
        X, y = calibration_samples(depth, reference, np.eye(3), mask=mask)
        self.assertGreaterEqual(len(y), 8)
        np.testing.assert_allclose(y, .004)
        self.assertTrue(np.all((X[:, 0] >= 8) & (X[:, 0] < 12)))

    def test_manifest_disallows_same_input_in_two_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root/'d.npy', np.ones((4, 4)))
            f = dict(frame_id='a', camera_id='cam_a', split='train', measured_depth='d.npy',
                     reference_depth='d.npy', K=np.eye(3).tolist())
            manifest = dict(schema_version='winejar-depth-calibration-v1', reference_source='known_plane',
                            frames=[f, dict(f, frame_id='b', split='test')])
            path = root/'manifest.json'
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_calibration_manifest(path)

    def test_smooth_spatial_depth_error_generalizes_to_new_positions(self):
        rng = np.random.default_rng(8)
        X = rng.uniform([-.25, -.2, .5, 25], [.25, .2, 1.3, 25], (120, 4))
        delta = lambda q: .003+.008*q[:, 0]-.006*q[:, 1]+.005*(q[:, 2]-.8)**2
        model = GaussianProcessDepthCorrection(GPSettings(optimize=True, max_iterations=50)).fit(X, delta(X))
        q = rng.uniform([-.2, -.15, .6, 25], [.2, .15, 1.2, 25], (30, 4))
        error = model.predict(q)-delta(q)
        self.assertLess(np.sqrt(np.mean(error*error)), .0001)
        self.assertFalse(model.metadata['temperature_varied_in_training'])


if __name__ == '__main__':
    unittest.main()
