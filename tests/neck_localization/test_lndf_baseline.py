"""Meaningful unit/pose and compatibility checks for the external adapter."""
import numpy as np
from scipy.spatial.transform import Rotation
from neck_localization.lndf_baseline import (ObservedSimilarity, apply_pose_numpy,
    function_pose, pose_list, collar_query)


def test_similarity_roundtrip_preserves_pose_and_transformed_query():
    rng = np.random.default_rng(18)
    cloud = rng.normal(size=(100, 3))*.2 + [.3, -.4, .6]
    sim = ObservedSimilarity.from_points(cloud)
    query = rng.normal(size=(30, 3))*.01
    pose = function_pose([.31, -.42, .72], [.05, -.02, 1.])
    world = query @ pose[:3, :3].T + pose[:3, 3]
    model_pose = sim.pose_to_model(pose)
    transformed = (query*sim.scale) @ model_pose[:3, :3].T + model_pose[:3, 3]
    np.testing.assert_allclose(sim.points_to_model(world), transformed, atol=1e-14)
    np.testing.assert_allclose(sim.points_to_world(sim.points_to_model(cloud)), cloud, atol=1e-14)
    np.testing.assert_allclose(sim.pose_to_world(model_pose), pose, atol=1e-14)
    assert np.max(abs(sim.points_to_model(cloud))) <= .04000001


def test_pose_conversion_preserves_column_transform_and_normal():
    rng = np.random.default_rng(19)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.random(random_state=rng).as_matrix()
    pose[:3, 3] = [.12, -.17, .55]
    points = rng.normal(size=(20, 3))*.1
    np.testing.assert_allclose(apply_pose_numpy(points, pose_list(pose)),
        points @ pose[:3, :3].T + pose[:3, 3], atol=1e-14)
    normal = np.array([.05, .1, 1.]); normal /= np.linalg.norm(normal)
    frame = function_pose([0, 0, .6], normal)
    np.testing.assert_allclose(frame[:3, :3].T @ frame[:3, :3], np.eye(3), atol=1e-14)
    np.testing.assert_allclose(frame[:3, 2], normal, atol=1e-14)
    assert np.linalg.det(frame[:3, :3]) > .999999


def test_query_is_deterministic_shared_geometry():
    a = collar_query(.01, .001, 1000, seed=10)
    np.testing.assert_array_equal(a, collar_query(.01, .001, 1000, seed=10))
    radius = np.linalg.norm(a[:, :2], axis=1)
    assert radius.min() >= .008 and radius.max() <= .012
    assert abs(a[:, 2]).max() <= .001


def test_scatter_compat_matches_reference_with_empty_negative_and_duplicates():
    import torch
    from neck_localization.lndf_baseline import _torch_scatter_compat
    src = torch.tensor([[[-3., -2., 4., -1.], [8., -4., 2., -2.]]])
    index = torch.tensor([[[0, 0, 2, 2]]])
    compat = _torch_scatter_compat()
    mean = compat.scatter_mean(src, index, dim_size=4)
    maximum, unused = compat.scatter_max(src, index, dim_size=4)
    torch.testing.assert_close(mean, torch.tensor([[[-2.5, 0, 1.5, 0], [2., 0, 0, 0]]]))
    torch.testing.assert_close(maximum, torch.tensor([[[-2., 0, 4., 0], [8., 0, 2., 0]]]))
    assert unused is None


def test_audit_manifest_requires_only_ordinary_config_and_observation_paths(tmp_path):
    import json
    from scripts.run_lndf_capability import build_manifest
    source = tmp_path / "source"
    folder = source / "close/render_inputs"
    folder.mkdir(parents=True)
    (source / "close/input_source.json").write_text(json.dumps({"render_directory": "render_inputs"}))
    (folder / "effective_config.json").write_text(json.dumps({"cases": [{"name": "case_a"}]}))
    # There are deliberately no geometry, evaluation_only, or target files.
    manifest = build_manifest(source, tmp_path / "dev", ["close"], ["stereo_raw"], "cam_a")
    assert len(manifest) == 1
    assert manifest[0]["observations"] == str((folder / "stereo_raw/scenes/case_a/observations").resolve())
    assert manifest[0]["split"] == "previously_viewed_capability_audit"
