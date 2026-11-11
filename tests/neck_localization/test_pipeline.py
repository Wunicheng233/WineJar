"""Pipeline invariants: no hardware, target-truth leakage or success filtering."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from copy import deepcopy
import cv2
import mujoco
import numpy as np
import pytest

from neck_localization.estimators import fit_depth, fit_multiview
from neck_localization.evaluation import score, summarize
from neck_localization.geometry import Camera, Estimate, Observation, circle_points, look_at
from neck_localization.perturbation import perturb
from neck_localization.rendering import Rendered, extract_observations, render


@pytest.fixture
def analytic_observations():
    center = np.array([.02, -.015, .4])
    normal = np.array([.15, -.1, 1.])
    normal /= np.linalg.norm(normal)
    radius = .055
    points = circle_points(center, normal, radius, count=100)
    cameras = [look_at("cam0", [.4, -.5, .8], [0, 0, .4]),
               look_at("cam1", [-.5, .2, .7], [0, 0, .4]),
               look_at("cam2", [.1, .1, 1.1], [0, 0, .4])]
    observations = [Observation(c, c.project(points), points.copy()) for c in cameras]
    truth = {"center_m": center.tolist(), "normal": normal.tolist(), "radius_m": radius}
    return observations, truth


def _camera_depth(observation):
    camera = observation.camera
    return ((observation.points_world - camera.T_world_from_camera[:3, 3])
            @ camera.T_world_from_camera[:3, :3])[:, 2]


def test_false_accept_and_success_rates_use_all_attempts():
    truth = {"center_m": [0, 0, .4], "normal": [0, 0, 1], "radius_m": .055}
    rows = []
    for accepted, error_mm in [(True, 0), (True, 5), (False, 12), (False, None)]:
        estimate = Estimate(accepted=accepted, elapsed_ms=2)
        if error_mm is not None:
            estimate.center = np.array([error_mm / 1000, 0, .4])
            estimate.normal = np.array([0., 0., 1.])
            estimate.radius = .055
        row = dict(split="test", geometry="coupon", evidence="rgb", factor="clean", method="fixture")
        row.update(score(estimate, truth))
        rows.append(row)
    summary, = summarize(rows)
    assert summary["n"] == 4
    assert summary["accepted"] == 2
    assert summary["success_rate"] == .25
    assert summary["rejection_rate"] == .5
    assert summary["false_accept_rate"] == .25
    assert summary["false_among_accepted"] == .5
    # Solved rejections remain in error quantiles. Unsolved cases remain in rate
    # denominators and have no fabricated numeric geometric error.
    assert summary["center_mm_median"] == pytest.approx(5)
    assert summary["center_mm_max"] == pytest.approx(12)


def test_score_axis_components_and_positive_z_normal_convention():
    truth = {"center_m": [0, 0, .4], "normal": [0, 0, -1], "radius_m": .055}
    estimate = Estimate(center=np.array([.004, 0, .403]), normal=np.array([0., 0., 1.]),
                        radius=.0565, accepted=True, covariance_center=np.eye(3) * 1e-6)
    row = score(estimate, truth)
    assert row["normal_deg"] == 0
    assert row["normal_component_mm"] == pytest.approx(3)
    assert row["tangent_mm"] == pytest.approx(4)
    assert row["center_mm"] == pytest.approx(5)
    assert row["diameter_mm"] == pytest.approx(3)
    assert not row["truth_pass"]
    assert row["coverage95"] is False
    np.testing.assert_allclose(row["center_std_mm"], [1, 1, 1])


def test_common_frame_bias_is_low_residual_false_accept(analytic_observations):
    observations, truth = analytic_observations
    shift = np.array([.005, 0, 0])
    changed = perturb(observations, {"common_translation_m": shift.tolist()}, seed=42)
    for original, altered in zip(observations, changed):
        np.testing.assert_array_equal(altered.pixels, original.pixels)
        np.testing.assert_allclose(altered.points_world, original.points_world + shift, atol=1e-14)
    for estimator in (fit_depth, lambda obs: fit_multiview(obs, robust=False)):
        result = estimator(changed)
        assert result.accepted, result.reasons
        assert result.residual < 1e-5
        np.testing.assert_allclose(result.center, np.asarray(truth["center_m"]) + shift, atol=1e-7)
        measured = score(result, truth)
        assert measured["center_mm"] == pytest.approx(5, abs=1e-4)
        assert not measured["truth_pass"]


def test_changed_calibration_reconstructs_original_measured_depth(analytic_observations):
    observations, _ = analytic_observations
    original = observations[1]
    changed_camera = deepcopy(original.camera)
    changed_camera.K[0, 0] *= 1.02
    changed_camera.K[1, 1] *= 1.01
    changed_camera.K[:2, 2] += [1.5, -.7]
    changed_camera.T_world_from_camera[:3, 3] += [.003, -.002, .001]
    changed, = perturb([original], {}, seed=7, calibrated_cameras=[changed_camera])
    expected = changed_camera.unproject(original.pixels, _camera_depth(original))
    np.testing.assert_allclose(changed.points_world, expected, atol=1e-14)
    np.testing.assert_allclose(_camera_depth(changed), _camera_depth(original), atol=1e-14)
    assert np.linalg.norm(changed.points_world - original.points_world) > .005
    np.testing.assert_array_equal(changed.pixels, original.pixels)
    # Input arrays and calibration remain reusable by other factor branches.
    np.testing.assert_array_equal(original.camera.K, observations[1].camera.K)
    assert original.camera is not changed.camera


def test_depth_scale_and_offset_act_on_axial_z(analytic_observations):
    observations, _ = analytic_observations
    changed = perturb(observations, {"depth_scale_fraction": .005, "depth_offset_m": .003}, seed=8)
    for original, altered in zip(observations, changed):
        np.testing.assert_allclose(_camera_depth(altered), _camera_depth(original) * 1.005 + .003, atol=1e-14)
        np.testing.assert_array_equal(altered.pixels, original.pixels)


def test_fixed_seed_repeats_noise_holes_and_outliers_without_mutating_inputs(analytic_observations):
    observations, _ = analytic_observations
    saved = deepcopy(observations)
    factor = {"edge_sigma_px": .5, "disparity_sigma_px": .1, "outlier_fraction": .15, "hole_fraction": .3}
    first = perturb(observations, factor, seed=903)
    second = perturb(observations, factor, seed=903)
    different = perturb(observations, factor, seed=904)
    for a, b, original, before in zip(first, second, observations, saved):
        np.testing.assert_array_equal(a.pixels, b.pixels)
        np.testing.assert_array_equal(a.points_world, b.points_world)
        np.testing.assert_array_equal(original.pixels, before.pixels)
        np.testing.assert_array_equal(original.points_world, before.points_world)
        np.testing.assert_array_equal(original.camera.K, before.camera.K)
        np.testing.assert_array_equal(original.camera.T_world_from_camera, before.camera.T_world_from_camera)
        assert 0 < len(a.pixels) < len(original.pixels)
    assert any(a.pixels.shape != b.pixels.shape or not np.array_equal(a.pixels, b.pixels)
               for a, b in zip(first, different))


def test_rgb_extraction_does_not_use_target_truth_or_segmentation():
    width, height = 320, 240
    camera = look_at("cam0", [0, 0, 1], [0, 0, 0], width, height)
    rgb = np.zeros((height, width, 3), np.uint8)
    cv2.ellipse(rgb, (160, 120), (65, 35), 17, 0, 360, (255, 255, 255), 5)
    scene = Rendered([camera], [rgb], [np.full((height, width), .7, np.float32)],
                     [np.zeros((height, width), np.uint8)], {}, "")
    first = extract_observations(scene, "coupon", evidence="rgb")
    assert len(first) == 1
    # These objects raise if treated as mappings/arrays; RGB extraction should
    # not inspect either evaluation truth or a diagnostic segmentation mask.
    scene.truth = object()
    scene.masks = [object()]
    second = extract_observations(scene, "coupon", evidence="rgb")
    np.testing.assert_array_equal(first[0].pixels, second[0].pixels)
    np.testing.assert_array_equal(first[0].points_world, second[0].points_world)


def test_exported_mujoco_cameras_match_declared_opencv_transforms():
    scene = render({"center_m": [.01, -.01, .4], "rotation_deg": [13, -7, 31],
                    "radius_m": .055, "geometry": "coupon", "layout": "wide",
                    "resolution": [320, 240], "segments": 64, "offsamples": 0})
    model = mujoco.MjModel.from_xml_string(scene.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for camera, color, depth in zip(scene.cameras, scene.rgb, scene.depth):
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera.name)
        rotation = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
        np.testing.assert_allclose(rotation, camera.T_world_from_camera[:3, :3], atol=1e-9)
        np.testing.assert_allclose(data.cam_xpos[camera_id], camera.T_world_from_camera[:3, 3], atol=1e-9)
        assert color.shape[:2] == depth.shape == (camera.height, camera.width)
        assert depth.dtype == np.float32
