"""Classical fixed-background silhouette extraction for the neck-localization rig."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ForegroundSegmentation:
    """Foreground-difference diagnostics and one filled object silhouette."""

    difference_u8: np.ndarray
    raw_mask: np.ndarray
    silhouette_mask: np.ndarray
    bounding_box_xywh: tuple[int, int, int, int]
    silhouette_area_px: int
    threshold_u8: float


def _odd_kernel_size(value: int, name: str) -> int:
    if value < 1 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer")
    return value


def _validate_roi(roi_xywh: tuple[int, int, int, int] | None, image_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if roi_xywh is None:
        return None
    x, y, width, height = roi_xywh
    image_height, image_width = image_shape
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError(f"ROI {roi_xywh} is outside image size {image_width}x{image_height}")
    return roi_xywh


def segment_fixed_background_foreground(
    foreground_bgr: np.ndarray,
    background_bgr: np.ndarray,
    *,
    roi_xywh: tuple[int, int, int, int] | None = None,
    blur_kernel: int = 5,
    close_kernel: int = 15,
    min_component_area_px: int = 250,
    threshold_scale: float = 1.0,
) -> ForegroundSegmentation:
    """Extract and fill the strongest changed object inside an optional fixed-workspace ROI.

    The adaptive Otsu threshold is intentionally computed from the current
    foreground/background pair. ``threshold_scale`` permits a calibrated
    per-camera correction when one view has lower object/background contrast.
    """
    foreground = np.asarray(foreground_bgr)
    background = np.asarray(background_bgr)
    if foreground.ndim != 3 or foreground.shape[2] != 3 or foreground.shape != background.shape:
        raise ValueError("Foreground and background must be same-size BGR images")
    blur_kernel = _odd_kernel_size(blur_kernel, "blur_kernel")
    close_kernel = _odd_kernel_size(close_kernel, "close_kernel")
    if threshold_scale <= 0:
        raise ValueError("threshold_scale must be positive")
    roi = _validate_roi(roi_xywh, foreground.shape[:2])
    foreground_lab = cv2.cvtColor(foreground, cv2.COLOR_BGR2LAB).astype(np.float32)
    background_lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = np.linalg.norm(foreground_lab - background_lab, axis=2)
    scale = 255.0 / max(float(np.percentile(delta, 99.5)), 1.0)
    difference = np.clip(delta * scale, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(difference, (blur_kernel, blur_kernel), 0)
    otsu_threshold, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(np.clip(otsu_threshold * threshold_scale, 0.0, 255.0))
    _, raw_mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    if roi is not None:
        x, y, width, height = roi
        workspace = np.zeros_like(raw_mask)
        workspace[y : y + height, x : x + width] = 255
        raw_mask = cv2.bitwise_and(raw_mask, workspace)
    closed = cv2.morphologyEx(
        raw_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    candidates = [
        (int(stats[index, cv2.CC_STAT_AREA]), index)
        for index in range(1, component_count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_component_area_px
    ]
    if not candidates:
        raise ValueError("No foreground component exceeded min_component_area_px")
    _, selected_label = max(candidates)
    component_mask = np.where(labels == selected_label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Selected foreground component has no external contour")
    contour = max(contours, key=cv2.contourArea)
    silhouette = np.zeros_like(component_mask)
    cv2.drawContours(silhouette, [contour], -1, 255, thickness=cv2.FILLED)
    x, y, width, height = cv2.boundingRect(contour)
    return ForegroundSegmentation(
        difference_u8=difference,
        raw_mask=raw_mask,
        silhouette_mask=silhouette,
        bounding_box_xywh=(int(x), int(y), int(width), int(height)),
        silhouette_area_px=int(np.count_nonzero(silhouette)),
        threshold_u8=float(threshold),
    )


def draw_silhouette_overlay(image_bgr: np.ndarray, segmentation: ForegroundSegmentation) -> np.ndarray:
    """Overlay a filled-silhouette contour and bounding box for human inspection."""
    display = np.asarray(image_bgr).copy()
    contours, _ = cv2.findContours(segmentation.silhouette_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(display, contours, -1, (0, 255, 0), 2)
    x, y, width, height = segmentation.bounding_box_xywh
    cv2.rectangle(display, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(
        display,
        f"area={segmentation.silhouette_area_px}px threshold={segmentation.threshold_u8:.1f}",
        (max(6, x), max(22, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 0),
        2,
    )
    return display
