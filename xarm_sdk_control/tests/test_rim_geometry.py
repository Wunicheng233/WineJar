from pathlib import Path
import json

import numpy as np
import pytest

from xarm_sdk_control.multiview_rig import (
    FixedCameraModel,
    FixedCameraRig,
    assert_rig_valid_for_live_capture,
    rig_geometry_fingerprint,
)
from xarm_sdk_control.rim_geometry import (
    RimFit,
    RimObservation,
    circle_points,
    fit_multiview_rim,
    plane_basis,
    project_workcell,
    surface_point_at_silhouette,
)


def synthetic_rig():
    poses = {}
    for cid, origin in [
        ("overhead", [0, 0, 1.2]),
        ("side_a", [1.25, 0, 0.62]),
        ("side_b", [0, 1.0, 0.26]),
    ]:
        origin = np.asarray(origin, float)
        z = np.array([0, 0, 0.48]) - origin
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.0]) if cid != "overhead" else np.array([1.0, 0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        poses[cid] = np.eye(4)
        poses[cid][:3, :3] = np.column_stack((x, y, z))
        poses[cid][:3, 3] = origin
    reference_from_world = np.linalg.inv(poses["overhead"])
    cameras = {
        cid: FixedCameraModel(
            cid,
            cid,
            np.array([[610.0, 0, 320], [0, 610.0, 240], [0, 0, 1]]),
            np.array([0.02, -0.03, 0.001, -0.001, 0.005]),
            (640, 480),
            reference_from_world @ pose,
            Path("synthetic.json"),
        )
        for cid, pose in poses.items()
    }
    return FixedCameraRig("overhead", cameras, ()), reference_from_world


@pytest.mark.parametrize("tilt", [0.0, 0.18])
@pytest.mark.parametrize("extent_views", [(), ("side_a",), ("side_a", "side_b")])
def test_multiview_rim_recovers_metric_geometry_despite_tilt_and_outliers(tilt, extent_views):
    rig, transform = synthetic_rig()
    center = np.array([0.012, -0.018, 0.487])
    normal = np.array([tilt, -0.07, 1.0])
    normal /= np.linalg.norm(normal)
    rng = np.random.default_rng(74)
    observations = []
    for cid in rig.cameras:
        inner = cid == "overhead"
        points = circle_points(center, normal, 0.05 if inner else 0.062, 200)
        pixels = project_workcell(rig, cid, transform, points)
        span = tuple(np.percentile(pixels[:, 0], [0, 100])) if cid in extent_views else None
        if not inner:
            pixels = pixels[pixels[:, 1] <= np.median(pixels[:, 1])]
        pixels += rng.normal(0, 0.15, pixels.shape)
        pixels[::24] += rng.normal(0, 5.0, pixels[::24].shape)
        observations.append(RimObservation(cid, pixels, inner, span))
    result = fit_multiview_rim(
        rig, transform, observations, center + [0.005, -0.004, -0.012], 0.052
    )
    assert np.linalg.norm(result.center_m - center) < 0.0015
    assert np.degrees(np.arccos(np.clip(result.normal @ normal, -1, 1))) < 3
    assert result.outer_radius_m == pytest.approx(0.062, abs=0.001)
    assert not result.at_bound


def test_fit_refreshes_projection_geometry_after_calibration_change():
    rig, transform = synthetic_rig()
    center = np.array([0.012, -0.018, 0.487])
    normal = np.array([0.12, -0.07, 1.0])
    normal /= np.linalg.norm(normal)
    for changed in (False, True):
        if changed:
            camera = rig.camera("side_b")
            camera.transform_reference_from_camera[0, 3] += .04
            camera.camera_matrix[0, 0] += 25
        observations = []
        for cid in rig.cameras:
            inner = cid == "overhead"
            pixels = project_workcell(
                rig, cid, transform,
                circle_points(center, normal, .05 if inner else .062, 180),
            )
            extent = None if inner else (pixels[:, 0].min(), pixels[:, 0].max())
            observations.append(RimObservation(cid, pixels, inner, extent))
        fitted = fit_multiview_rim(rig, transform, observations, center + [.005, 0, 0], .052)
        np.testing.assert_allclose(fitted.center_m, center, atol=1e-5)
        np.testing.assert_allclose(fitted.normal, normal, atol=1e-4)
        assert fitted.outer_radius_m == pytest.approx(.062, abs=1e-5)


def test_silhouette_ray_recovers_same_neck_length_from_high_and_low_cameras():
    rig, transform = synthetic_rig()
    center = np.array([0.012, -0.018, 0.487])
    normal = np.array([0.10, -0.05, 1.0])
    normal /= np.linalg.norm(normal)
    radius, length = 0.062, 0.008
    rim = RimFit(center, normal, 0.05, radius, {}, np.zeros(7), 1.0, False)
    u, v = plane_basis(normal)
    base = center - length * normal
    for cid in ["side_a", "side_b"]:
        camera_world = (
            np.linalg.inv(transform) @ rig.camera(cid).transform_reference_from_camera
        )
        offset = camera_world[:3, 3] - base
        planar = np.array([offset @ u, offset @ v])
        angle = np.arctan2(planar[1], planar[0])
        tangent_angles = angle + np.array([-1.0, 1.0]) * np.arccos(
            radius / np.linalg.norm(planar)
        )
        surface = base + radius * (
            np.cos(tangent_angles)[:, None] * u + np.sin(tangent_angles)[:, None] * v
        )
        pixels = project_workcell(rig, cid, transform, surface)
        recovered, misses = surface_point_at_silhouette(
            rig, cid, np.linalg.inv(transform), pixels, rim
        )
        np.testing.assert_allclose(-(recovered - center) @ normal, length, atol=1e-5)
        np.testing.assert_allclose(misses, 0, atol=1e-5)


def test_fit_requires_multiple_independent_camera_views():
    rig, transform = synthetic_rig()
    with pytest.raises(ValueError, match="two independent side"):
        fit_multiview_rim(rig, transform, [], np.array([0.0, 0, 0.5]), 0.05)


def test_small_tilt_limit_has_feasible_initial_guesses():
    rig, transform = synthetic_rig()
    center = np.array([0.012, -0.018, 0.487])
    observations = []
    for cid in rig.cameras:
        inner = cid == "overhead"
        pixels = project_workcell(
            rig,
            cid,
            transform,
            circle_points(center, [0, 0, 1], 0.05 if inner else 0.062, 80),
        )
        observations.append(RimObservation(cid, pixels, inner))
    fitted = fit_multiview_rim(
        rig, transform, observations, center, 0.05, max_tilt_deg=1
    )
    np.testing.assert_allclose(fitted.center_m, center, atol=1e-5)
    assert np.degrees(np.arccos(fitted.normal[2])) <= 1


def test_camera_move_invalidation_blocks_live_capture_but_preserves_offline_geometry(
    tmp_path,
):
    rig, _ = synthetic_rig()
    fingerprint = rig_geometry_fingerprint(rig)
    state = tmp_path / "scene_status.json"
    state.write_text(json.dumps({"invalidated_geometry_fingerprints": [fingerprint]}))
    with pytest.raises(ValueError, match="invalidated"):
        assert_rig_valid_for_live_capture(rig, state)
    assert rig_geometry_fingerprint(rig) == fingerprint
    rig.camera("side_a").transform_reference_from_camera[0, 3] += 0.01
    with pytest.raises(ValueError, match="not accepted"):
        assert_rig_valid_for_live_capture(rig, state)


@pytest.mark.parametrize("mismatched_source", ["capture", "workcell"])
def test_pipeline_rejects_mixed_calibrations_before_loading_images(
    tmp_path, monkeypatch, mismatched_source
):
    from xarm_sdk_control import mouth_height_pipeline as pipeline

    rig, _ = synthetic_rig()
    monkeypatch.setattr(pipeline, "load_fixed_camera_rig", lambda path: rig)
    fingerprint = rig_geometry_fingerprint(rig)
    capture = {
        "type": "fixed_multiview_rgbd_capture",
        "rig_geometry_fingerprint": fingerprint,
    }
    plane = {
        "type": "fixed_workcell_support_plane",
        "status": "ok",
        "rig_geometry_fingerprint": fingerprint,
    }
    payload = capture if mismatched_source == "capture" else plane
    payload["rig_geometry_fingerprint"] = "a-different-rig"
    (tmp_path / "capture.json").write_text(json.dumps(capture))
    plane_path = tmp_path / "workcell.json"
    plane_path.write_text(json.dumps(plane))
    with pytest.raises(ValueError, match=f"{mismatched_source}: recorded calibration"):
        pipeline.localize_jar(tmp_path / "rig.json", plane_path, tmp_path, {})
