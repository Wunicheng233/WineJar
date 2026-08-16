"""Shared pose-composition and angle-wrapping helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def wrapped_angle_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def wrapped_delta_deg(target: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Return shortest signed Euler/joint deltas in degrees."""
    return (np.asarray(target, dtype=float) - np.asarray(start, dtype=float) + 180.0) % 360.0 - 180.0


def undirected_axis_delta_deg(start_deg: float, end_deg: float) -> float:
    """Shortest signed difference between two axes whose direction is modulo 180 degrees."""
    return (float(end_deg) - float(start_deg) + 90.0) % 180.0 - 90.0


def compose_pose_mm(
    point_mm: np.ndarray | Sequence[float],
    z_mm: float,
    orientation: Sequence[float],
) -> list[float]:
    point = np.asarray(point_mm, dtype=float)
    return [float(point[0]), float(point[1]), float(z_mm), *map(float, orientation)]


def compose_pose_from_m(
    point_m: np.ndarray | Sequence[float],
    z_mm: float,
    orientation: Sequence[float],
) -> list[float]:
    return compose_pose_mm(np.asarray(point_m, dtype=float) * 1000.0, z_mm, orientation)
