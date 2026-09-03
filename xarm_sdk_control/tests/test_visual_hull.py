from __future__ import annotations

import numpy as np

from xarm_sdk_control.multiview_rig import FixedCameraModel, FixedCameraRig
from xarm_sdk_control.visual_hull import (
    AxisAlignedBounds,
    derive_bounds_from_reference_mask,
    reconstruct_visual_hull,
)


def _simple_rig() -> FixedCameraRig:
    matrix = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    reference = FixedCameraModel(
        "overhead", "reference", matrix, np.zeros(5), (100, 100), np.eye(4), __file__
    )
    target = FixedCameraModel(
        "side_a", "target", matrix, np.zeros(5), (100, 100), np.eye(4), __file__
    )
    return FixedCameraRig("overhead", {"overhead": reference, "side_a": target}, ())


def test_derive_bounds_uses_silhouette_and_valid_depth() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 45:55] = 255
    depth = np.zeros((100, 100), dtype=np.uint16)
    depth[mask > 0] = 1000
    bounds = derive_bounds_from_reference_mask(
        mask,
        depth,
        0.001,
        np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
        depth_padding_m=0.02,
        lateral_padding_m=0.01,
    )
    assert np.all(bounds.minimum_m < [0.0, 0.0, 1.0])
    assert np.all(bounds.maximum_m > [0.0, 0.0, 1.0])


def test_automatic_bounds_cap_depth_span_when_mask_contains_background_depth() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 45:55] = 255
    depth = np.zeros((100, 100), dtype=np.uint16)
    depth[mask > 0] = 1000
    depth[45:55, 47:53] = 1600
    bounds = derive_bounds_from_reference_mask(
        mask,
        depth,
        0.001,
        np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
        depth_padding_m=0.02,
        max_depth_span_m=0.12,
    )
    assert bounds.extent_m[2] <= 0.1200001


def test_visual_hull_keeps_only_voxels_inside_every_mask() -> None:
    rig = _simple_rig()
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[45:56, 45:56] = 255
    hull = reconstruct_visual_hull(
        rig,
        {"overhead": mask, "side_a": mask},
        AxisAlignedBounds(np.array([-0.1, -0.1, 1.0]), np.array([0.1, 0.1, 1.1])),
        0.05,
    )
    assert hull.candidate_voxel_count == 32
    assert 0 < len(hull.occupied_points_reference_m) < hull.candidate_voxel_count
