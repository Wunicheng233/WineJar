"""Observation-only sensor perturbations, metres and OpenCV axial Z depth.

Constant additive Z bias is a local sensitivity model, not a fitted RealSense
model or constant Euclidean range/world-Z shift. Invalid depths stay invalid.
"""
from dataclasses import replace
import numpy as np


def axial_depth_bias(frames, bias_by_camera_m):
    names = {f.camera.name for f in frames}
    if not set(bias_by_camera_m).issubset(names):
        raise ValueError('Unknown camera in bias specification')
    if not all(np.isfinite(b) for b in bias_by_camera_m.values()):
        raise ValueError('Depth bias must be finite')
    result = []
    for frame in frames:
        bias = float(bias_by_camera_m.get(frame.camera.name, 0.))
        # Keep the zero-control dtype/values EXACT; all other changes in float64
        # to avoid adding an implicit depth-quantisation experiment.
        depth = frame.depth_z_m.copy() if bias == 0 else np.array(frame.depth_z_m, float, copy=True)
        if bias != 0:
            valid = np.isfinite(depth) & (depth > 0)
            depth[valid] += bias
            depth[valid & (depth <= 0)] = 0  # never invent a positive measurement
        result.append(replace(frame, depth_z_m=depth))
    return result
