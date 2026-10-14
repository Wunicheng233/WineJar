import cv2
import numpy as np
import pytest

from xarm_sdk_control.suction_surface import (
    SurfaceProfile,
    _cup_score_maps,
    analyze_suction_surface,
    masked_center_camera_point,
)


PROFILE = SurfaceProfile(
    edge_margin_mm=5.0,
    dark_contrast=15,
    max_defect_fraction=0.10,
    max_roughness_mm=1.0,
    max_tilt_deg=15.0,
    min_valid_depth_fraction=0.90,
)


def synthetic_surface(with_crack: bool = False):
    color = np.full((240, 320, 3), (120, 180, 160), dtype=np.uint8)
    depth = np.full((240, 320), 0.4, dtype=np.float32)
    mask = np.zeros((240, 320), dtype=np.uint8)
    cv2.rectangle(mask, (30, 30), (290, 210), 255, -1)
    if with_crack:
        cv2.line(color, (160, 40), (160, 200), (10, 10, 10), 5)
    camera_matrix = np.array([[600.0, 0, 160.0], [0, 600.0, 120.0], [0, 0, 1]])
    return color, depth, mask, camera_matrix


def test_dual_cup_candidates_obey_physical_spacing_and_edges() -> None:
    color, depth, mask, matrix = synthetic_surface()
    analysis = analyze_suction_surface(color, depth, mask, matrix, PROFILE)
    best = analysis.candidates[0]
    spacing = np.linalg.norm(np.subtract(best.cup_centers_px[0], best.cup_centers_px[1]))
    assert np.isclose(spacing, analysis.cup_spacing_px)
    assert np.isclose(analysis.cup_spacing_px, 72.0)
    assert all(cup.clearance_mm >= PROFILE.edge_margin_mm for cup in best.cups)
    assert best.angle_deg == 0.0
    assert np.linalg.norm(np.subtract(best.center_px, (160.0, 120.0))) <= 8.0


def test_dark_crack_is_detected_and_avoided_by_both_cups() -> None:
    color, depth, mask, matrix = synthetic_surface(with_crack=True)
    analysis = analyze_suction_surface(color, depth, mask, matrix, PROFILE)
    assert np.count_nonzero(analysis.defect_mask[:, 157:164]) > 500
    best = analysis.candidates[0]
    assert all(abs(cup.center_px[0] - 160.0) > analysis.cup_radius_px for cup in best.cups)
    assert all(cup.defect_fraction <= PROFILE.max_defect_fraction for cup in best.cups)


def test_depth_hole_rejects_cup_footprint() -> None:
    color, depth, mask, matrix = synthetic_surface()
    depth[90:150, 135:185] = 0.0
    analysis = analyze_suction_surface(color, depth, mask, matrix, PROFILE)
    best = analysis.candidates[0]
    assert all(not (135 <= cup.center_px[0] <= 185 and 90 <= cup.center_px[1] <= 150) for cup in best.cups)


def test_broken_dark_ridge_is_joined_into_a_continuous_defect() -> None:
    color, depth, mask, matrix = synthetic_surface()
    for y in range(50, 190, 14):
        cv2.line(color, (160, y), (160, y + 8), (5, 5, 5), 3)
    analysis = analyze_suction_surface(color, depth, mask, matrix, PROFILE)
    centre_column = analysis.defect_mask[55:186, 160]
    assert np.mean(centre_column > 0) > 0.90


def test_masked_center_camera_point_uses_central_surface_depth() -> None:
    depth = np.zeros((80, 100), dtype=np.float32)
    mask = np.zeros_like(depth, dtype=np.uint8)
    cv2.circle(mask, (60, 30), 20, 255, -1)
    depth[mask > 0] = 0.50
    depth[28:33, 58:63] = 0.80
    camera_matrix = np.array([[500.0, 0.0, 50.0], [0.0, 500.0, 40.0], [0.0, 0.0, 1.0]])

    point = masked_center_camera_point(depth, mask, camera_matrix)

    assert np.allclose(point, (0.01, -0.01, 0.50), atol=1e-4)


def test_batch_scores_preserve_each_hard_gate_and_its_boundary() -> None:
    # First six centres are feasible (including equality at every threshold).
    # The remaining five each violate exactly one constraint.
    maps = {"distance_px": np.full((1, 11), 20.0)}
    filtered = {
        "valid_fraction": np.ones((1, 11)),
        "defect_fraction": np.zeros((1, 11)),
        "roughness_mm": np.zeros((1, 11)),
        "tilt_deg": np.zeros((1, 11)),
    }
    maps["distance_px"][0, 1] = 7.0  # radius 2 + margin 5
    filtered["valid_fraction"][0, 2] = PROFILE.min_valid_depth_fraction
    filtered["defect_fraction"][0, 3] = PROFILE.max_defect_fraction
    filtered["roughness_mm"][0, 4] = PROFILE.max_roughness_mm
    filtered["tilt_deg"][0, 5] = PROFILE.max_tilt_deg
    maps["distance_px"][0, 6] = 6.99
    filtered["valid_fraction"][0, 7] = PROFILE.min_valid_depth_fraction - .001
    filtered["defect_fraction"][0, 8] = PROFILE.max_defect_fraction + .001
    filtered["roughness_mm"][0, 9] = PROFILE.max_roughness_mm + .001
    filtered["tilt_deg"][0, 10] = PROFILE.max_tilt_deg + .001
    scores, _ = _cup_score_maps(maps, filtered, PROFILE, 2.0, 1.0)
    assert np.all(np.isfinite(scores[0, :6]))
    assert np.all(np.isneginf(scores[0, 6:]))
    assert scores[0, 0] == pytest.approx(1.0)


def test_out_of_image_cups_do_not_wrap_around_to_valid_pixels() -> None:
    color, depth, mask, matrix = synthetic_surface()
    mask[:] = 255
    analysis = analyze_suction_surface(
        color, depth, mask, matrix, PROFILE, cup_spacing_mm=1000.0,
    )
    assert analysis.candidates == []


def test_top_k_preserves_full_ranking_with_non_default_search() -> None:
    args = (*synthetic_surface(True), PROFILE)
    options = dict(grid_step_px=9, angle_step_deg=22.5, preferred_angle_deg=137.0)
    full = analyze_suction_surface(*args, max_candidates=100000, **options)
    top = analyze_suction_surface(*args, max_candidates=7, **options)
    assert len(full.candidates) > 7
    assert top.candidates == full.candidates[:7]
    assert all(a.score >= b.score for a, b in zip(full.candidates, full.candidates[1:]))
