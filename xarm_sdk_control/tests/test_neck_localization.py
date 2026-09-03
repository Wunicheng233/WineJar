from __future__ import annotations

import numpy as np

from xarm_sdk_control.neck_localization import fit_neck_from_visual_hull


def test_surface_of_revolution_fit_recovers_vertical_axis_and_narrow_neck() -> None:
    points = []
    for height in np.linspace(0.0, 0.22, 30):
        radius = 0.020 if height < 0.09 else 0.040
        for angle in np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False):
            points.append([radius * np.cos(angle), radius * np.sin(angle), height])
    result = fit_neck_from_visual_hull(
        np.asarray(points),
        mouth_to_body_hint=np.array([0.0, 0.0, 1.0]),
        slice_width_m=0.012,
    )

    assert result.axis_mouth_to_body[2] > 0.99
    assert len(result.profile) >= 10
    assert result.binding_max_diameter_m < 0.050
    assert result.binding_constriction_ratio < 0.60
    assert result.failure_reason is None
    assert result.confidence >= 0.55
