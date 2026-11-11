"""Offline GP depth correction adapted from Heindl et al. (ICMV 2017).

The author's ARD squared-exponential GP maps measured camera XYZ and temperature
to ``reference_z - measured_z``. This module ports that model to NumPy/SciPy;
it is not a reproduction of Basso's joint intrinsic/extrinsic calibration.
All depth values are optical-axis Z in metres. Inference has no reference input.

Adapted from cheind/rgbd-correction, commit
3b2eb5d3dea0e34a3e37a15cea376e1e061c3bc4,
sensor_correction/gp_cpu.py and apps/train.py.

BSD 3-Clause License
Copyright (c) 2017, Christoph Heindl. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize


UPSTREAM_COMMIT = '3b2eb5d3dea0e34a3e37a15cea376e1e061c3bc4'


@dataclass(frozen=True)
class GPSettings:
    # Initial values are settings, not measurements of our D435i.
    length_scale: tuple = (.5, .5, .5, 10.)
    signal_std_m: float = .005
    noise_std_m: float = .002
    optimize: bool = True
    max_iterations: int = 70
    max_training_points: int = 512
    samples_per_frame: int = 64
    seed: int = 20261024
    batch_size: int = 2048

    def __post_init__(self):
        if self.max_training_points < 8 or self.samples_per_frame < 1 or self.batch_size < 1 or self.max_iterations < 1:
            raise ValueError('Positive GP batch/iteration/sample limits and at least 8 total training points required')


def depth_features(depth_m, K, temperature_c=25., pixels=None):
    """Return measured XYZ,T; pixel coordinates are integer (u,v), no +0.5."""
    depth = np.asarray(depth_m, dtype=float)
    K = np.asarray(K, dtype=float)
    if depth.ndim != 2 or K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError('2D depth and finite 3x3 K required')
    if not np.isfinite(temperature_c):
        raise ValueError('Finite temperature required')
    if pixels is None:
        yy, xx = np.indices(depth.shape)
        pixels = np.column_stack((xx.ravel(), yy.ravel()))
    pixels = np.asarray(pixels, dtype=int)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError('pixels must have shape Nx2')
    z = depth[pixels[:, 1], pixels[:, 0]]
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(K).T
    return np.column_stack((rays*z[:, None], np.full(len(z), temperature_c)))


def rbf_kernel(A, B, length_scale, signal_std_m):
    """Author's ARD squared-exponential covariance, avoiding NxMx4 storage."""
    A = np.asarray(A, float)/np.asarray(length_scale)
    B = np.asarray(B, float)/np.asarray(length_scale)
    squared = np.maximum(np.sum(A*A, 1)[:, None] + np.sum(B*B, 1)[None, :] - 2*A@B.T, 0.)
    return signal_std_m**2*np.exp(-.5*squared)


class GaussianProcessDepthCorrection:
    """Exact GP mean correction; posterior std is conditional, not full accuracy.

    Hyperparameters use the author's sklearn-branch convention: diagonal noise
    is noise_std**2. No correction is made for invalid depths; holes remain holes.
    """
    def __init__(self, settings=GPSettings()):
        self.settings = settings
        self.metadata = {}

    def fit(self, features, corrections_m, *, K=None, metadata=None):
        X = np.asarray(features, dtype=float)
        y = np.asarray(corrections_m, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[1] != 4 or len(X) != len(y) or len(y) < 8:
            raise ValueError('At least 8 finite XYZ,T samples with matching targets required')
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError('GP training input contains invalid values')
        if len(y) > self.settings.max_training_points:
            raise ValueError('Training samples exceed configured maximum')
        scales = np.asarray(self.settings.length_scale, float)
        if scales.shape != (4,) or np.any(scales <= 0):
            raise ValueError('Four positive ARD length scales required')
        if self.settings.signal_std_m <= 0 or self.settings.noise_std_m <= 0:
            raise ValueError('Positive signal/noise standard deviations required')
        self.X = X.copy()
        self.y_mean = float(y.mean())
        yc = y-self.y_mean
        self.K = np.asarray(K, float).copy() if K is not None else None
        self.metadata = dict(metadata or {})
        self.feature_bounds = np.stack((X.min(axis=0), X.max(axis=0)))
        theta0 = np.log(np.r_[scales, self.settings.signal_std_m, self.settings.noise_std_m])

        def objective(theta, gradient=True):
            ls, signal, noise = np.exp(theta[:4]), np.exp(theta[4]), np.exp(theta[5])
            Csignal = rbf_kernel(X, X, ls, signal)
            C = Csignal + np.eye(len(y))*(noise**2+1e-14)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                return (1e30, np.zeros(6)) if gradient else 1e30
            alpha = cho_solve((L, True), yc, check_finite=False)
            loss = .5*yc@alpha+np.log(np.diag(L)).sum()+.5*len(y)*np.log(2*np.pi)
            if not gradient:
                return float(loss)
            common = cho_solve((L, True), np.eye(len(y)), check_finite=False)-np.outer(alpha, alpha)
            grad = np.empty(6)
            for axis in range(4):
                delta = (X[:, axis, None]-X[None, :, axis])/ls[axis]
                grad[axis] = .5*np.sum(common*Csignal*delta**2)
            grad[4] = np.sum(common*Csignal)
            grad[5] = noise**2*np.trace(common)
            return float(loss), grad

        if self.settings.optimize:
            bounds = [(np.log(.002), np.log(20.))]*3+[(np.log(.05), np.log(100.)),
                     (np.log(1e-6), np.log(.1)), (np.log(1e-6), np.log(.05))]
            solved = minimize(objective, theta0, method='L-BFGS-B', jac=True, bounds=bounds,
                              options={'maxiter': self.settings.max_iterations, 'ftol': 1e-10})
            # Keep the unoptimized model if numeric optimization worsens likelihood.
            theta = solved.x if solved.fun <= objective(theta0, False) else theta0
            self.metadata['optimizer'] = dict(success=bool(solved.success), message=str(solved.message),
                iterations=int(solved.nit), evaluations=int(solved.nfev), negative_log_likelihood=float(objective(theta, False)))
        else:
            theta = theta0
            self.metadata['optimizer'] = dict(success=True, message='fixed hyperparameters', iterations=0)
        self.length_scale = np.exp(theta[:4])
        self.signal_std_m, self.noise_std_m = float(np.exp(theta[4])), float(np.exp(theta[5]))
        C = rbf_kernel(X, X, self.length_scale, self.signal_std_m)
        C.flat[::len(X)+1] += self.noise_std_m**2+1e-14
        self.L = np.linalg.cholesky(C)
        self.alpha = cho_solve((self.L, True), yc, check_finite=False)
        self.metadata['temperature_varied_in_training'] = bool(np.ptp(X[:, 3]) > 1e-9)
        return self

    def predict(self, features, *, return_std=False):
        X = np.asarray(features, dtype=float)
        if X.ndim != 2 or X.shape[1] != 4 or not np.all(np.isfinite(X)):
            raise ValueError('Finite Nx4 query required')
        means, stds = [], []
        for start in range(0, len(X), self.settings.batch_size):
            Ks = rbf_kernel(self.X, X[start:start+self.settings.batch_size], self.length_scale, self.signal_std_m)
            means.append(Ks.T@self.alpha+self.y_mean)
            if return_std:
                v = solve_triangular(self.L, Ks, lower=True, check_finite=False)
                stds.append(np.sqrt(np.maximum(self.signal_std_m**2-np.sum(v*v, axis=0), 0.)))
        mean = np.concatenate(means) if means else np.empty(0)
        return (mean, np.concatenate(stds) if stds else np.empty(0)) if return_std else mean

    def correct_depth(self, depth_m, K, temperature_c=25., *, return_diagnostics=False):
        """Add predicted Z correction. K must match calibration; no resampling."""
        depth = np.asarray(depth_m)
        if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.floating):
            raise ValueError('2D floating-point optical-Z depth in metres required')
        if self.K is not None and not np.allclose(K, self.K, rtol=0, atol=1e-9):
            raise ValueError('Intrinsics differ from calibration; model cannot correct K')
        good = np.isfinite(depth) & (depth > 0)
        yy, xx = np.nonzero(good)
        features = depth_features(depth, K, temperature_c, np.column_stack((xx, yy)))
        corrections = self.predict(features)
        result = depth.copy()
        result[good] += corrections.astype(result.dtype)
        became_invalid = good & (~np.isfinite(result) | (result <= 0))
        result[became_invalid] = 0.
        outside = np.any((features < self.feature_bounds[0]-1e-9) | (features > self.feature_bounds[1]+1e-9), axis=1)
        diag = dict(valid_input_pixels=int(good.sum()), invalidated_pixels=int(became_invalid.sum()),
                    correction_mm_median=float(np.median(corrections)*1000) if len(corrections) else None,
                    correction_mm_max_abs=float(np.max(np.abs(corrections))*1000) if len(corrections) else None,
                    feature_box_extrapolation_fraction=float(outside.mean()) if len(outside) else None,
                    uncertainty='GP posterior is conditional on fixed K/model/reference; not full output accuracy',
                    temperature_varied_in_training=self.metadata['temperature_varied_in_training'])
        return (result, diag) if return_diagnostics else result

    def save(self, path):
        metadata = dict(self.metadata, schema_version='heindl-gp-depth-adapter-v1',
                        upstream_commit=UPSTREAM_COMMIT, settings=asdict(self.settings))
        np.savez_compressed(path, X=self.X, L=self.L, alpha=self.alpha, y_mean=self.y_mean,
            K=np.asarray(self.K) if self.K is not None else np.empty((0, 0)),
            feature_bounds=self.feature_bounds, length_scale=self.length_scale,
            signal_std_m=self.signal_std_m, noise_std_m=self.noise_std_m,
            metadata_json=json.dumps(metadata, sort_keys=True))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata_json']))
            if metadata['schema_version'] != 'heindl-gp-depth-adapter-v1':
                raise ValueError('Unknown depth correction schema')
            model = cls(GPSettings(**metadata.pop('settings')))
            model.metadata = metadata
            for name in ['X', 'L', 'alpha', 'feature_bounds', 'length_scale']:
                setattr(model, name, data[name].copy())
            for name in ['y_mean', 'signal_std_m', 'noise_std_m']:
                setattr(model, name, float(data[name]))
            model.K = data['K'].copy() if data['K'].size else None
        return model


def calibration_samples(depth, reference, K, *, temperature_c=25., mask=None, count=64):
    """Author-style image grid, restricted to externally identified plane pixels."""
    depth, reference = np.asarray(depth), np.asarray(reference)
    if depth.shape != reference.shape or depth.ndim != 2:
        raise ValueError('Measured/reference arrays must share shape')
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(reference) & (reference > 0)
    if mask is not None:
        if np.asarray(mask).shape != depth.shape:
            raise ValueError('Calibration mask shape differs')
        valid &= np.asarray(mask, bool)
    side = max(2, int(np.ceil(np.sqrt(count))))
    x = np.linspace(0, depth.shape[1]-1, side).astype(int)
    y = np.linspace(0, depth.shape[0]-1, side).astype(int)
    xx, yy = np.meshgrid(x, y)
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    pixels = pixels[valid[pixels[:, 1], pixels[:, 0]]][:count]
    # Small calibration board or missing stereo margins may miss a regular grid.
    if len(pixels) < min(8, int(valid.sum())):
        vy, vx = np.nonzero(valid)
        choice = np.linspace(0, len(vx)-1, min(count, len(vx))).astype(int)
        pixels = np.column_stack((vx[choice], vy[choice]))
    features = depth_features(depth, K, temperature_c, pixels)
    corrections = reference[pixels[:, 1], pixels[:, 0]]-depth[pixels[:, 1], pixels[:, 0]]
    return features, corrections


def load_calibration_manifest(path):
    """Load plane-only calibration manifest; refuse repeated IDs across splits."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 'winejar-depth-calibration-v1':
        raise ValueError('Expected winejar-depth-calibration-v1 manifest')
    if not manifest.get('reference_source'):
        raise ValueError('Independent calibration reference_source must be recorded')
    frames, identifiers, measured_paths = [], set(), set()
    for item in manifest['frames']:
        frame = dict(item)
        identifier = (frame['camera_id'], frame['frame_id'])
        if identifier in identifiers or frame['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('Duplicate frame ID or unsupported split')
        identifiers.add(identifier)
        for field in ['measured_depth', 'reference_depth', 'valid_mask']:
            if field in frame:
                source = (path.parent/Path(frame[field])).resolve()
                if field == 'measured_depth':
                    if source in measured_paths:
                        raise ValueError('Measured frame reused, possibly across train/test')
                    measured_paths.add(source)
                frame[field+'_path'] = str(source)
                frame[field] = np.load(source, allow_pickle=False)
        frame['K'] = np.asarray(frame['K'], dtype=float)
        frame['temperature_c'] = float(frame.get('temperature_c', 25.))
        frames.append(frame)
    return manifest, frames
