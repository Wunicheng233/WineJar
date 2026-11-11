"""Observation-only conversions for the external-baseline capability audit."""
from hashlib import sha256
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def shared_ir_noise(image, seed, speckle=1., thermal=1.):
    """Upstream SimSense noise distribution, CPU RNG, not its CUDA samples.

    The saved uint8 input is the result supplied to BOTH matchers. Intensity
    units are 0..255, not normalized 0..1. No depth or geometry is consulted.
    """
    image = np.asarray(image)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError('Expected uint8 grayscale IR')
    if speckle < 0 or thermal < 0:
        raise ValueError('Noise levels must be nonnegative')
    rng = np.random.default_rng(seed)
    out = image.astype(np.float64)
    if speckle:
        out *= rng.gamma(1333.33 / speckle, speckle / 1333.33, image.shape)
    if thermal:
        out += rng.normal(0., .25 * thermal, image.shape)
    return np.ascontiguousarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


def disparity_to_depth(disparity_px, focal_px, baseline_m, min_depth=.2, max_depth=10.):
    disparity = np.asarray(disparity_px, dtype=np.float32)
    if disparity.ndim != 2 or focal_px <= 0 or baseline_m <= 0:
        raise ValueError('Positive focal/baseline and a 2D disparity image required')
    valid = np.isfinite(disparity) & (disparity > 0)
    depth = np.zeros(disparity.shape, np.float32)
    np.divide(focal_px * baseline_m, disparity, out=depth, where=valid)
    xx = np.arange(disparity.shape[1])[None, :]
    valid &= (xx - disparity >= 0) & (depth >= min_depth) & (depth < max_depth)
    depth[~valid] = 0
    return depth, valid


def verify_pair(record):
    got = [file_sha256(record[key]) for key in ('left_path', 'right_path')]
    if got != record['input_sha256']:
        raise ValueError('IR input changed after audit manifest was frozen')
