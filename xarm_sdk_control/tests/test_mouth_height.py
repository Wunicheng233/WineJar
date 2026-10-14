from __future__ import annotations

import numpy as np
import pytest

from xarm_sdk_control.mouth_height import (
    estimate_camera_top_height,
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
    estimate = estimate_camera_top_height(
        np.vstack((body, top, outliers)), top_quantile=0.98
    )
    assert estimate.height_m == pytest.approx(0.207, abs=0.001)
    assert estimate.top_support_count >= 200


def test_reference_to_workcell_transform_is_applied() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, -0.5]
    result = transform_reference_points_to_workcell([[0.1, 0.2, 0.7]], transform)
    np.testing.assert_allclose(result, [[0.1, 0.2, 0.2]])
