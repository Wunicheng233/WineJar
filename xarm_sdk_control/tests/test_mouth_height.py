from __future__ import annotations

import numpy as np
import pytest

from xarm_sdk_control.mouth_height import (
    CameraHeightEstimate,
    estimate_camera_top_height,
    estimate_neck_band,
    fuse_camera_heights,
    transform_reference_points_to_workcell,
)


def test_top_height_uses_stable_upper_band_not_single_outlier() -> None:
    rng = np.random.default_rng(11)
    body = np.column_stack(
        (
            rng.uniform(-0.03, 0.03, 1200),
            rng.uniform(-0.03, 0.03, 1200),
            rng.uniform(0.03, 0.19, 1200),
        )
    )
    top = np.column_stack(
        (
            rng.uniform(-0.02, 0.02, 250),
            rng.uniform(-0.02, 0.02, 250),
            rng.normal(0.207, 0.0004, 250),
        )
    )
    outliers = np.array([[0.0, 0.0, 0.30], [0.01, 0.01, 0.28]])
    estimate = estimate_camera_top_height(np.vstack((body, top, outliers)), top_quantile=0.98)
    assert estimate.height_m == pytest.approx(0.207, abs=0.001)
    assert estimate.top_support_count >= 200


def test_height_fusion_rejects_one_camera_outlier() -> None:
    def estimate(height: float) -> CameraHeightEstimate:
        return CameraHeightEstimate(height, np.ones(1, bool), np.ones(1, bool), 100, 30, 0.0005)

    fused = fuse_camera_heights(
        {"overhead": estimate(0.207), "side_a": estimate(0.209), "side_b": estimate(0.235)},
        agreement_tolerance_m=0.005,
    )
    assert fused.height_m == pytest.approx(0.208)
    assert fused.accepted_camera_ids == ("overhead", "side_a")
    assert fused.rejected_camera_ids == ("side_b",)


def test_reference_to_workcell_transform_is_applied() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, -0.5]
    result = transform_reference_points_to_workcell([[0.1, 0.2, 0.7]], transform)
    np.testing.assert_allclose(result, [[0.1, 0.2, 0.2]])


def test_neck_band_starts_after_radius_converges_from_shoulder() -> None:
    rng = np.random.default_rng(17)
    points = []
    for height in np.arange(0.108, 0.208, 0.004):
        if height < 0.164:
            radius = 0.038
        elif height < 0.184:
            radius = 0.038 - (height - 0.164) * 0.65
        else:
            radius = 0.024
        angles = rng.uniform(0.0, 2.0 * np.pi, 180)
        noisy_radius = rng.normal(radius, 0.0003, len(angles))
        points.extend(
            np.column_stack(
                (noisy_radius * np.cos(angles), noisy_radius * np.sin(angles), np.full(len(angles), height))
            )
        )
    result = estimate_neck_band(np.asarray(points), np.zeros(2), 0.208)
    assert result.lower_height_m == pytest.approx(0.184, abs=0.004)
    assert result.upper_height_m == pytest.approx(0.208)
    assert result.body_radius_m > result.neck_radius_threshold_m
