import cv2
import numpy as np
import pytest

from xarm_sdk_control.neck_observation import (
    detect_aperture,
    fit_shoulder_transition,
    top_arc,
)


def test_opening_center_is_not_biased_toward_a_bright_rim_patch():
    color = np.full((240, 300, 3), 160, np.uint8)
    mask = np.zeros(color.shape[:2], np.uint8)
    cv2.circle(mask, (150, 120), 95, 255, -1)
    cv2.circle(color, (145, 123), 39, (3, 3, 3), -1)
    cv2.circle(color, (145, 82), 5, (255, 255, 255), -1)
    # SAM2 may exclude the actual hole from the material mask.
    cv2.circle(mask, (145, 123), 39, 0, -1)
    aperture = detect_aperture(color, mask)
    np.testing.assert_allclose(aperture.center_px, [145, 123], atol=1)


def test_shoulder_fit_uses_local_change_in_slope_not_body_size():
    rows = np.arange(20.0, 40.0)
    columns = 80 + 0.05 * (rows - 27) + 1.8 * np.maximum(0, rows - 27)
    transition = fit_shoulder_transition(np.column_stack((columns, rows)), 1)
    assert transition.pixel[1] == pytest.approx(27, abs=0.6)
    assert transition.fit_error_px < 0.2


def test_straight_shoulder_without_collar_is_rejected():
    rows = np.arange(20.0, 40.0)
    with pytest.raises(ValueError, match="No supported"):
        fit_shoulder_transition(np.column_stack((80 + 1.5 * rows, rows)), 1)


def test_pipe_connected_to_sam_mask_does_not_create_artificial_top_at_crop():
    mask = np.zeros((100, 100), np.uint8)
    mask[35:90, 20:80] = 255
    mask[:35, 45:55] = 255
    arc = top_arc(mask, (20, 80), (25, 50))
    assert len(arc) == 50
    assert not np.any((arc[:, 0] >= 45) & (arc[:, 0] < 55))
    assert np.all(arc[:, 1] == 34.5)
