"""Static scene export checks; no localization, noise trial, or device access."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
import pytest

from neck_localization.geometry import Camera
from neck_localization.scene_preparation import prepare_scene_bundle


@pytest.fixture(scope="module")
def prepared_scene(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    config = deepcopy(json.loads((root / "experiments/neck_localization/configs/preparation_v2.json").read_text()))
    # A smaller export exercises the complete asset/camera path without running
    # a benchmark or changing the review configuration.
    config["scene"]["resolution"] = [320, 240]
    config["angular_segments"] = 48
    config["vertical_samples"] = 64
    config["cases"] = [item for item in config["cases"]
                       if item["name"] in ("reference_shallow", "tilted_neck")]
    output = tmp_path_factory.mktemp("scene_preparation") / "bundle"
    manifest = prepare_scene_bundle(config, output)
    return output, manifest


def _camera_metadata(folder):
    return json.loads((folder / "observations/camera_true_parameters.json").read_text())


def test_fixed_camera_rig_does_not_follow_individual_necks(prepared_scene):
    output, manifest = prepared_scene
    folders = [output / case["relative_dir"] for case in manifest["cases"]]
    assert len(folders) == 2
    first = _camera_metadata(folders[0])
    assert first == _camera_metadata(folders[1])
    assert first["kind"] == "true_simulation_camera_parameters_not_virtual_calibration"
    targets = [json.loads((folder / "evaluation_only/target.json").read_text()) for folder in folders]
    assert not np.allclose(targets[0]["normal"], targets[1]["normal"])
    assert not np.allclose(targets[0]["center_m"], targets[1]["center_m"])
    for folder in folders:
        model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for item in first["cameras"]:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, item["name"])
            actual = np.eye(4)
            actual[:3, :3] = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
            actual[:3, 3] = data.cam_xpos[camera_id]
            np.testing.assert_allclose(actual, item["T_world_from_camera"], atol=1e-9)
        jar_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jar")
        np.testing.assert_array_equal(data.xpos[jar_id], [0, 0, 0])
        np.testing.assert_allclose(data.xmat[jar_id].reshape(3, 3), np.eye(3), atol=1e-15)
        assert model.njnt == 0


def test_evaluation_truth_separate_and_no_rendered_target_marker(prepared_scene):
    output, manifest = prepared_scene
    assert manifest["stage"] == "scene_preparation_only_no_localization_or_noise_benchmark"
    for case in manifest["cases"]:
        folder = output / case["relative_dir"]
        target = json.loads((folder / "evaluation_only/target.json").read_text())
        assert target["evaluation_only"] is True
        assert target["observability_verified"] is False
        assert target["policy_is_synthetic_assumption"] is True
        low, high = target["near_min_interval_s_m"]
        assert low <= target["s_m"] <= high
        observations = folder / "observations"
        assert not any("target" in path.name or "profile" in path.name for path in observations.iterdir())
        scene = ET.parse(folder / "scene.xml").getroot()
        assert not scene.findall(".//site")
        for geom in scene.findall(".//geom"):
            assert not any(marker in geom.get("name", "").lower() for marker in ("target", "band", "marker"))
        # Exterior and cavity may use different materials/mesh assets. They
        # remain parts of the same fixed jar, without a target-specific geom.
        model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
        jar_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jar")
        mesh_ids = np.flatnonzero(model.geom_type == int(mujoco.mjtGeom.mjGEOM_MESH))
        assert len(mesh_ids) > 0
        assert np.all(model.geom_bodyid[mesh_ids] == jar_body)
        for camera in _camera_metadata(folder)["cameras"]:
            assert (observations / (camera["name"] + "_mask_diagnostic.png")).exists()


def test_rgb_depth_and_diagnostic_images_share_grid_and_raw_rgb(prepared_scene):
    output, manifest = prepared_scene
    for case in manifest["cases"]:
        folder = output / case["relative_dir"]
        cameras = _camera_metadata(folder)["cameras"]
        model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        jar_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jar")
        jar_geoms = np.flatnonzero(model.geom_bodyid == jar_body)
        with mujoco.Renderer(model, height=cameras[0]["height"], width=cameras[0]["width"]) as renderer:
            for item in cameras:
                observations = folder / "observations"
                rgb = cv2.imread(str(observations / (item["name"] + "_rgb.png")))
                mask = cv2.imread(str(observations / (item["name"] + "_mask_diagnostic.png")), cv2.IMREAD_GRAYSCALE)
                with np.load(observations / (item["name"] + "_depth_z_m.npz")) as archive:
                    assert archive.files == ["depth_z_m"]
                    depth = archive["depth_z_m"]
                expected_shape = (item["height"], item["width"])
                assert rgb.shape == (*expected_shape, 3)
                assert mask.shape == depth.shape == expected_shape
                assert depth.dtype == np.float32
                assert np.all(np.isfinite(depth))
                assert np.count_nonzero(mask) > 100
                assert np.all(depth[mask > 0] > 0)
                # Re-rendering verifies that the delivered RGB has no truth
                # circle, interval overlay, or post-render visual annotation.
                renderer.update_scene(data, camera=item["name"])
                expected_rgb = renderer.render()
                np.testing.assert_array_equal(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), expected_rgb)
                renderer.enable_depth_rendering()
                expected_depth = renderer.render()
                renderer.disable_depth_rendering()
                # Check the float depth survives export without millimetre
                # quantization or a conversion to Euclidean range.
                np.testing.assert_array_equal(depth, expected_depth)
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render()
                renderer.disable_segmentation_rendering()
                expected_mask = (np.isin(segmentation[:, :, 0], jar_geoms) &
                                 (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
                np.testing.assert_array_equal(mask > 0, expected_mask)


def test_rendered_depth_matches_mesh_raycast_in_camera_coordinates(prepared_scene):
    output, manifest = prepared_scene
    # Ray casting is a rendering-coordinate diagnostic, never estimator input.
    for case in manifest["cases"]:
        folder = output / case["relative_dir"]
        model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        jar_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jar")
        jar_geoms = np.flatnonzero(model.geom_bodyid == jar_body)
        for item in _camera_metadata(folder)["cameras"]:
            camera = Camera(item["name"], np.asarray(item["K"]), np.asarray(item["T_world_from_camera"]),
                            item["width"], item["height"])
            observations = folder / "observations"
            with np.load(observations / (item["name"] + "_depth_z_m.npz")) as archive:
                depth = archive["depth_z_m"]
            mask = cv2.imread(str(observations / (item["name"] + "_mask_diagnostic.png")), cv2.IMREAD_GRAYSCALE)
            interior = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
            yy, xx = np.nonzero(interior)
            indices = np.linspace(0, len(yy) - 1, 9, dtype=int)
            origin = camera.T_world_from_camera[:3, 3].copy()
            for index in indices:
                x, y = int(xx[index]), int(yy[index])
                ray = camera.unproject([[x, y]], [1.])[0] - origin
                ray_length = np.linalg.norm(ray)
                geom_id = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(model, data, origin, ray / ray_length, None, 1, -1, geom_id)
                assert geom_id[0] in jar_geoms
                assert distance > 0
                # The camera ray before normalization has local Z=1. A rendered
                # range would disagree on off-axis pixels; axial depth agrees.
                expected_axial_z = distance / ray_length
                assert abs(float(depth[y, x]) - expected_axial_z) < 3e-5
