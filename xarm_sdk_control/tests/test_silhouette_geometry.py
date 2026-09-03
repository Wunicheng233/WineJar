from __future__ import annotations

import cv2
import numpy as np

from xarm_sdk_control.silhouette_geometry import segment_fixed_background_foreground


def test_segmentation_fills_a_transparent_object_outline() -> None:
    background = np.zeros((100, 120, 3), dtype=np.uint8)
    foreground = background.copy()
    cv2.circle(foreground, (70, 55), 20, (255, 255, 255), thickness=3)

    result = segment_fixed_background_foreground(
        foreground,
        background,
        roi_xywh=(35, 20, 70, 70),
        close_kernel=9,
        min_component_area_px=50,
    )

    assert result.silhouette_area_px > 900
    assert result.silhouette_mask[55, 70] == 255
    assert result.bounding_box_xywh[0] <= 50
    assert result.bounding_box_xywh[0] + result.bounding_box_xywh[2] >= 90


def test_segmentation_uses_workspace_roi_to_ignore_larger_change() -> None:
    background = np.zeros((100, 120, 3), dtype=np.uint8)
    foreground = background.copy()
    cv2.rectangle(foreground, (4, 4), (40, 70), (255, 255, 255), thickness=cv2.FILLED)
    cv2.rectangle(foreground, (75, 35), (98, 65), (255, 255, 255), thickness=cv2.FILLED)

    result = segment_fixed_background_foreground(
        foreground,
        background,
        roi_xywh=(65, 25, 45, 50),
        close_kernel=5,
        min_component_area_px=50,
    )

    x, y, width, height = result.bounding_box_xywh
    assert x >= 70
    assert y >= 30
    assert width <= 30
    assert height <= 40
