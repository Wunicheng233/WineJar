from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE = SourceFileLoader(
    "visual_waypoint_validation",
    str(ROOT / "xarm_sdk_control" / "scripts" / "14_visual_waypoint_validation.py"),
).load_module()


def detection(label: str, score: float, point_base_m: list[float]) -> dict:
    return {"label": label, "score": score, "point_base_m": point_base_m}


def test_transform_point_applies_fixed_camera_extrinsic() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [0.30, -0.10, 0.20]
    actual = MODULE.transform_point(transform, [0.10, 0.20, 0.30])
    np.testing.assert_allclose(actual, [0.40, 0.10, 0.50])


def test_two_spatially_distinct_leaves_are_selected_separately() -> None:
    detections = [
        detection("bamboo leaf", 0.80, [0.20, 0.10, 0.30]),
        detection("bamboo leaf", 0.70, [0.35, 0.10, 0.30]),
    ]
    selected, reused = MODULE.select_leaf_waypoints(detections, distinct_distance_m=0.05)
    assert not reused
    assert selected[0]["point_base_m"] != selected[1]["point_base_m"]


def test_single_visible_leaf_reuses_stack_center() -> None:
    detections = [detection("bamboo leaf", 0.80, [0.20, 0.10, 0.30])]
    selected, reused = MODULE.select_leaf_waypoints(detections, distinct_distance_m=0.05)
    assert reused
    assert selected[0]["point_base_m"] == selected[1]["point_base_m"]


def test_clipped_ceramic_is_rejected() -> None:
    item = detection("ceramic disk", 0.90, [0.30, -0.10, 0.02])
    item["touches_image_edge"] = True

    try:
        MODULE.select_ceramic([item])
    except RuntimeError as exc:
        assert "touches the image edge" in str(exc)
    else:
        raise AssertionError("Clipped ceramic detection should be rejected")


def test_leaf_clipped_symmetrically_along_long_axis_is_usable() -> None:
    item = detection("bamboo leaf", 0.80, [0.36, 0.06, -0.02])
    item.update(
        touches_image_edge=True,
        edge_sides=["top", "bottom"],
        box_xyxy=[250.0, 0.0, 440.0, 479.0],
    )

    selected, reused = MODULE.select_leaf_waypoints([item], distinct_distance_m=0.05)

    assert reused
    assert selected[0] == item


def test_leaf_clipped_on_only_one_end_is_rejected() -> None:
    item = detection("bamboo leaf", 0.80, [0.36, 0.06, -0.02])
    item.update(
        touches_image_edge=True,
        edge_sides=["bottom"],
        box_xyxy=[250.0, 80.0, 440.0, 479.0],
    )

    try:
        MODULE.select_leaf_waypoints([item], distinct_distance_m=0.05)
    except RuntimeError as exc:
        assert "touches the image edge" in str(exc)
    else:
        raise AssertionError("One-sided clipping should be rejected")


def test_safety_report_rejects_one_out_of_bounds_waypoint() -> None:
    report = MODULE.safety_report(
        [[300, 0, 400, 180, 0, 0], [510, 0, 400, 180, 0, 0]],
        min_x_mm=100,
        max_x_mm=500,
        min_y_mm=-300,
        max_y_mm=300,
        min_z_mm=200,
        max_z_mm=700,
    )
    assert not report["all_ok"]
    assert report["waypoints"][0]["x_ok"]
    assert not report["waypoints"][1]["x_ok"]
