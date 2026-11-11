"""Small periodic disparity correction learned from independent known planes.

Engineering baseline motivated by subpixel nonlinearity, not official RealSense
calibration code. No jar geometry is used. Cannot repair boundary mismatches,
holes or an unobservable common world-coordinate calibration error.
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class DisparitySettings:
    harmonics: int = 2
    points_per_frame: int = 128
    seed: int = 2026102507
    robust_scale_pixels: float = .04
    ridge: float = .02
    correction_limit_pixels: float = .6


def features(disparity, harmonics):
    d = np.asarray(disparity)
    return np.stack([np.ones_like(d), (d-30)/15]+[f(2*np.pi*k*d) for k in range(1, harmonics+1) for f in [np.sin, np.cos]], axis=-1)


class DisparityCorrection:
    def __init__(self, coefficients, fb, settings=DisparitySettings()):
        self.coefficients = np.asarray(coefficients); self.fb = float(fb); self.settings = settings

    def correct(self, depth):
        original = np.asarray(depth); result = original.copy()
        good = np.isfinite(original) & (original > 0)
        disparity = self.fb/original[good]
        delta = features(disparity, self.settings.harmonics)@self.coefficients
        delta = np.clip(delta, -self.settings.correction_limit_pixels, self.settings.correction_limit_pixels)
        result[good] = self.fb/(disparity+delta)
        return result

    def save(self, path):
        Path(path).write_text(json.dumps(dict(coefficients=self.coefficients.tolist(), fb=self.fb,
            settings=asdict(self.settings)), indent=2)+'\n')

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d['coefficients'], d['fb'], DisparitySettings(**d['settings']))


def fit_correction(frames, fb, settings=DisparitySettings()):
    rng = np.random.default_rng(settings.seed)
    measured, reference = [], []
    for f in frames:
        if f['split'] != 'train': continue
        mask = f['valid_mask'] & (f['measured_depth'] > 0) & (f['reference_depth'] > 0)
        ids = np.flatnonzero(mask)
        ids = rng.choice(ids, min(len(ids), settings.points_per_frame), replace=False)
        measured.extend(fb/f['measured_depth'].ravel()[ids])
        reference.extend(fb/f['reference_depth'].ravel()[ids])
    d, target = np.array(measured), np.array(reference)-np.array(measured)
    X = features(d, settings.harmonics)
    n = X.shape[1]
    fit = least_squares(lambda b:np.r_[X@b-target, settings.ridge*b], np.zeros(n),
        loss='soft_l1', f_scale=settings.robust_scale_pixels, max_nfev=100)
    return DisparityCorrection(fit.x, fb, settings), dict(training_points=len(d), success=bool(fit.success),
        disparity_range=[float(d.min()), float(d.max())], training_disparity_rmse=float(np.sqrt(np.mean((X@fit.x-target)**2))))
