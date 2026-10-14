"""Image evidence for open-rim localization; learned masks remain coarse proposals."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class ApertureObservation:
    pixels: np.ndarray
    center_px: np.ndarray
    radius_px: float
    ellipse_error_px: float


@dataclass(frozen=True)
class EdgeTransition:
    pixel: np.ndarray
    fit_error_px: float
    slope_change: float


def detect_aperture(color: np.ndarray, object_mask: np.ndarray) -> ApertureObservation:
    """Find the enclosed dark opening, fitting its entire contour rather than its highest pixels."""
    raw_mask = np.uint8(np.asarray(object_mask) != 0) * 255
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Empty object mask")
    filled = np.zeros_like(raw_mask)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    mask = filled != 0
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    rows, columns = np.nonzero(mask)
    if len(rows) < 300:
        raise ValueError("Object mask is too small to locate an aperture")
    center_hint = np.array([np.median(columns), np.median(rows)])
    size = max(np.ptp(rows), np.ptp(columns))
    candidates = []
    thresholds = np.unique(np.percentile(gray[mask], [5, 8, 12, 16, 20, 25, 30]))
    for threshold in thresholds:
        dark = np.uint8(mask & (gray <= threshold)) * 255
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if len(contour) < 30 or not 0.01 * len(rows) < area < 0.30 * len(rows):
                continue
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            a, b = np.array(axes) / 2
            if (
                min(a, b) / max(a, b) < 0.65
                or np.linalg.norm(np.array(center) - center_hint) > size * 0.25
            ):
                continue
            theta = np.deg2rad(angle)
            rotation = np.array(
                [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
            )
            q = (contour.reshape(-1, 2) - center) @ rotation.T
            radial_error = np.abs(
                np.sqrt((q[:, 0] / a) ** 2 + (q[:, 1] / b) ** 2) - 1
            ) * np.sqrt(a * b)
            error = float(np.median(radial_error))
            fill = area / (np.pi * a * b)
            if fill < 0.8:
                continue
            candidates.append(
                (error, -area, contour, np.array(center), float(np.sqrt(a * b)))
            )
    if not candidates:
        raise ValueError("No enclosed, approximately round dark aperture found")
    error, _, contour, center, radius = min(candidates, key=lambda value: value[:2])
    xy = contour.reshape(-1, 2).astype(float)
    # Refine the dark-to-bright edge radially at subpixel positions.
    unit = xy - center
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    offsets = np.linspace(-2, 3, 21)
    coordinates = xy[:, None, :] + unit[:, None, :] * offsets[None, :, None]
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.6).astype(np.float32)
    values = cv2.remap(
        blurred,
        coordinates[:, :, 0].astype(np.float32),
        coordinates[:, :, 1].astype(np.float32),
        cv2.INTER_LINEAR,
    )
    gradient = np.gradient(values, offsets, axis=1)
    peaks = np.argmax(gradient, axis=1)
    refined = xy + unit * offsets[peaks, None]
    return ApertureObservation(
        refined[:: max(1, len(refined) // 240)], center, radius, error
    )


def silhouette_edges(
    mask: np.ndarray, y_min: int, y_max: int, center_x: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the left and right edges of the run nearest the predicted jar axis."""
    left, right = [], []
    for y in range(max(0, y_min), min(mask.shape[0], y_max)):
        xs = np.flatnonzero(mask[y])
        if len(xs) < 5:
            continue
        runs = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
        run = min(runs, key=lambda r: abs((r[0] + r[-1]) / 2 - center_x))
        if len(run) < 5:
            continue
        left.append([float(run[0]) - 0.5, y])
        right.append([float(run[-1]) + 0.5, y])
    if len(left) < 10:
        raise ValueError("Too few supported side-profile rows")
    return np.asarray(left), np.asarray(right)


def refine_horizontal_edges(
    color: np.ndarray, edge: np.ndarray, search_px: float = 2
) -> np.ndarray:
    """Refine mask edges using a local image-gradient maximum, without long morphology kernels."""
    gray = cv2.GaussianBlur(
        cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), (3, 3), 0.6
    ).astype(np.float32)
    offsets = np.arange(-search_px, search_px + 0.125, 0.25)
    x = edge[:, 0, None] + offsets
    y = np.broadcast_to(edge[:, 1, None], x.shape)
    values = cv2.remap(
        gray, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR
    )
    gradient = np.abs(np.gradient(values, offsets, axis=1))
    # A small proximity cost prevents a stronger texture edge from replacing the boundary.
    scores = (
        gradient
        - np.max(gradient, axis=1, keepdims=True)
        * 0.15
        * (offsets / max(search_px, 0.1)) ** 2
    )
    peaks = np.argmax(scores, axis=1)
    refined = edge.copy()
    refined[:, 0] += offsets[peaks]
    return refined


def fit_shoulder_transition(
    edge: np.ndarray, outward_sign: float, minimum_slope_change: float = 0.3
) -> EdgeTransition:
    """Find a collar-to-shoulder kink by continuous two-line regression.

    The corner is fitted locally. No body-width ratio or requested neck length is
    used. A near-vertical upper edge and a widening lower edge are required.
    """
    xy = np.asarray(edge, float)
    if len(xy) < 12:
        raise ValueError("Insufficient rows to resolve a shoulder transition")
    y = xy[:, 1]
    x = gaussian_filter1d(xy[:, 0] * outward_sign, 0.7)
    best = None
    for knot in np.arange(y[3], y[-5], 0.25):
        design = np.column_stack((np.ones(len(y)), y - knot, np.maximum(0, y - knot)))
        coefficients = np.linalg.lstsq(design, x, rcond=None)[0]
        # The upper branch must be a collar, not the widening shoulder itself.
        if abs(coefficients[1]) > 0.8 or coefficients[2] < minimum_slope_change:
            continue
        errors = x - design @ coefficients
        score = float(np.mean(np.minimum(errors**2, 9)))
        if best is None or score < best[0]:
            best = score, knot, coefficients
    if best is None:
        raise ValueError("No supported collar-to-shoulder slope change")
    score, knot, coefficients = best
    return EdgeTransition(
        np.array([coefficients[0] * outward_sign, knot]),
        np.sqrt(score),
        float(coefficients[2]),
    )


def top_arc(
    mask: np.ndarray, x_bounds: tuple[float, float], y_bounds: tuple[float, float]
) -> np.ndarray:
    """Keep actual top boundaries inside a geometric window; do not invent edges at the crop."""
    pixels = []
    for x in range(
        max(0, int(np.ceil(x_bounds[0]))), min(mask.shape[1], int(x_bounds[1]))
    ):
        rows = np.flatnonzero(mask[:, x])
        if len(rows) and y_bounds[0] < rows[0] < y_bounds[1]:
            pixels.append([x, float(rows[0]) - 0.5])
    if len(pixels) < 8:
        raise ValueError("Too few visible top-rim pixels; the top may be occluded")
    return np.asarray(pixels)
