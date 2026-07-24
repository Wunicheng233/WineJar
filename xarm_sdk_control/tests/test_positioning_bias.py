from __future__ import annotations

import json

import numpy as np

from xarm_sdk_control.positioning_bias import append_bias_sample, load_bias_mm


def test_missing_bias_defaults_to_zero(tmp_path) -> None:
    assert np.array_equal(load_bias_mm(tmp_path / "missing.json"), np.zeros(3))


def test_samples_are_aggregated_with_component_median(tmp_path) -> None:
    path = tmp_path / "bias.json"
    append_bias_sample(path, {"correction_base_mm": [8.0, -3.0, 0.0]})
    result = append_bias_sample(path, {"correction_base_mm": [10.0, -5.0, 0.0]})
    result = append_bias_sample(path, {"correction_base_mm": [9.0, -4.0, 0.0]})

    assert result["bias_base_mm"] == [9.0, -4.0, 0.0]
    assert result["sample_range_base_mm"] == [2.0, 2.0, 0.0]
    assert np.array_equal(load_bias_mm(path), np.array([9.0, -4.0, 0.0]))
    assert json.loads(path.read_text())["sample_count"] == 3
