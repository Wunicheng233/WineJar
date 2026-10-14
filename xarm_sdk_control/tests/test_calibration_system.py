import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from xarm_sdk_control.calibration_data import (
    bind_intrinsics,
    file_sha256,
    fingerprint,
    intrinsic_health,
    intrinsic_snapshot,
    load_intrinsics,
    load_pair_intrinsics,
    write_new_json,
)
from xarm_sdk_control.calibration_sampling import pair_frames_fresh
from xarm_sdk_control.calibration_target import CalibrationTarget
from xarm_sdk_control.calibration_validation import (
    check_acceptance_reports,
    evaluate_observations,
)
from xarm_sdk_control.multiview_calibration import (
    BoardObservation,
    calibrate_chessboard_intrinsics,
    chessboard_object_points,
    orient_chessboard_corners,
)
from xarm_sdk_control.multiview_rig import (
    assert_rig_valid_for_live_capture,
    build_fixed_camera_rig,
    load_fixed_camera_rig,
    rig_geometry_fingerprint,
    write_fixed_camera_rig,
)


def intrinsic(serial, name):
    return {
        "type": "fixed_camera_chessboard_intrinsics",
        "camera_id": name,
        "camera": {"serial": serial},
        "capture": {"width": 640, "height": 480},
        "camera_matrix": [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        "distortion_coefficients": [0.0] * 5,
        "chessboard": {
            "inner_corners_cols": 11,
            "inner_corners_rows": 8,
            "square_size_mm": 20.0,
        },
    }


@pytest.fixture
def bound_pair(tmp_path):
    reference, target = intrinsic("ref", "overhead"), intrinsic("side", "side_b")
    ref_path, target_path = tmp_path / "ref.json", tmp_path / "target.json"
    write_new_json(ref_path, reference)
    write_new_json(target_path, target)
    transform = np.eye(4)
    transform[0, 3] = 0.2
    pair = {
        "schema_version": 2,
        "type": "fixed_camera_pair_chessboard_extrinsics",
        "scene_id": "scene",
        "reference_camera": {
            "camera_id": "overhead",
            "serial": "ref",
            "intrinsics_path": "ref.json",
        },
        "target_camera": {
            "camera_id": "side_b",
            "serial": "side",
            "intrinsics_path": "target.json",
        },
        "chessboard": reference["chessboard"],
        "capture": {"width": 640, "height": 480, "fps": 30},
        "transform_reference_from_target": transform.tolist(),
    }
    bind_intrinsics(pair["reference_camera"], reference)
    bind_intrinsics(pair["target_camera"], target)
    path = tmp_path / "pair.json"
    write_new_json(path, pair)
    return path, pair, ref_path, target_path


def test_old_bad_intrinsics_fail_full_field_health():
    matrix = np.array(
        [[605.13784, 0.0, 326.24968], [0.0, 605.77875, 244.82254], [0.0, 0.0, 1.0]]
    )
    result = intrinsic_health(
        matrix, [-0.000559, 1.140673, -0.001715, 0.000560, -4.198553], (640, 480)
    )
    assert not result["passed"]
    assert result["minimum_radial_derivative"] < 0
    assert result["max_roundtrip_px"] > 40
    assert intrinsic_health(matrix, np.zeros(5), (640, 480))["passed"]


def test_brown4_fits_projection_with_k3_fixed():
    matrix = np.array(intrinsic("a", "a")["camera_matrix"])
    distortion = np.array([-0.1, 0.02, 0.001, -0.002, 0.0])
    objects = chessboard_object_points(11, 8, 0.02)
    points = [
        cv2.projectPoints(
            objects,
            np.array([0.06 * i - 0.15, 0.1 - 0.04 * i, 0.03 * i]),
            np.array([-0.1 + 0.02 * i, -0.07 + 0.01 * i, 0.6 + 0.025 * i]),
            matrix,
            distortion,
        )[0]
        for i in range(8)
    ]
    result = calibrate_chessboard_intrinsics(
        points, (640, 480), 11, 8, 0.02, model="brown4"
    )
    assert result.distortion_coefficients.reshape(-1)[4] == 0.0
    np.testing.assert_allclose(result.camera_matrix, matrix, atol=0.05)
    assert intrinsic_health(
        result.camera_matrix, result.distortion_coefficients, (640, 480)
    )["passed"]
    with pytest.raises(ValueError, match="model"):
        calibrate_chessboard_intrinsics(
            points, (640, 480), 11, 8, 0.02, model="unknown"
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("camera_matrix", [[1]]),
        ("distortion_coefficients", [float("nan")] * 5),
        ("distortion_coefficients", [0] * 6),
    ],
)
def test_intrinsic_loader_rejects_bad_numerics(field, value):
    payload = intrinsic("a", "a")
    payload[field] = value
    with pytest.raises(ValueError):
        intrinsic_snapshot(payload)


def test_intrinsic_identity_ignores_board_and_json_format(bound_pair):
    _, _, ref, _ = bound_pair
    payload = json.loads(ref.read_text())
    original = fingerprint(intrinsic_snapshot(payload))
    payload["chessboard"]["square_size_mm"] = 50
    ref.write_text(json.dumps(payload, indent=4))
    _, _, loaded = load_intrinsics(ref, "ref", (640, 480))
    assert fingerprint(intrinsic_snapshot(loaded)) == original


def test_pair_rejects_intrinsics_changed_in_place(bound_pair):
    path, pair, _, target = bound_pair
    data = json.loads(target.read_text())
    data["camera_matrix"][0][0] += 1
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changed"):
        load_pair_intrinsics(pair, path)


def test_rig_rejects_different_reference_intrinsics(bound_pair, tmp_path):
    path, _, reference, _ = bound_pair
    data = json.loads(reference.read_text())
    data["camera_matrix"][0][0] *= 1.05
    other = write_new_json(tmp_path / "other.json", data)
    with pytest.raises(ValueError, match="differ"):
        build_fixed_camera_rig(other, [path], scene_id="scene")


def test_legacy_pair_requires_explicit_offline_opt_in(bound_pair):
    path, pair, reference, _ = bound_pair
    for role in ("reference_camera", "target_camera"):
        pair[role].pop("intrinsics_fingerprint")
        pair[role].pop("intrinsics_snapshot")
    path.write_text(json.dumps(pair))
    with pytest.raises(ValueError, match="Legacy"):
        build_fixed_camera_rig(reference, [path])
    load_pair_intrinsics(pair, path, allow_legacy=True)


def test_rig_roundtrip_and_mutated_intrinsics_rejected(bound_pair, tmp_path):
    path, _, ref, target = bound_pair
    rig = build_fixed_camera_rig(ref, [path], scene_id="scene")
    result = write_fixed_camera_rig(rig, tmp_path / "rig.json")
    assert rig_geometry_fingerprint(
        load_fixed_camera_rig(result)
    ) == rig_geometry_fingerprint(rig)
    data = json.loads(target.read_text())
    data["camera_matrix"][0][0] += 2
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changed"):
        load_fixed_camera_rig(result)


def test_rig_rejects_transform_different_from_its_sources(bound_pair, tmp_path):
    path, _, ref, _ = bound_pair
    result = write_fixed_camera_rig(
        build_fixed_camera_rig(ref, [path], scene_id="scene"), tmp_path / "rig.json"
    )
    data = json.loads(result.read_text())
    data["cameras"][1]["transform_reference_from_camera"][0][3] += 0.01
    result.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="differs"):
        load_fixed_camera_rig(result)


def test_live_gate_requires_current_scene_and_explicit_acceptance(bound_pair, tmp_path):
    path, _, ref, _ = bound_pair
    rig = build_fixed_camera_rig(ref, [path], scene_id="scene")
    state = tmp_path / "state.json"
    with pytest.raises(ValueError, match="No scene"):
        assert_rig_valid_for_live_capture(rig, state)
    data = {
        "scene_id": "scene",
        "status": "candidate",
        "approved_rig_fingerprint": rig_geometry_fingerprint(rig),
    }
    state.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not accepted"):
        assert_rig_valid_for_live_capture(rig, state)
    data["status"] = "validated_for_capture"
    state.write_text(json.dumps(data))
    assert_rig_valid_for_live_capture(rig, state)
    data["scene_id"] = "camera_moved"
    state.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not accepted"):
        assert_rig_valid_for_live_capture(rig, state)


def test_calibration_files_are_never_overwritten(tmp_path):
    path = write_new_json(tmp_path / "calibration.json", {"a": 1})
    with pytest.raises(FileExistsError):
        write_new_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}


def test_ambiguous_chessboards_rejected_and_low_contrast_not_oriented():
    with pytest.raises(ValueError, match="mixed-parity"):
        CalibrationTarget(
            "chessboard",
            {"inner_corners_cols": 8, "inner_corners_rows": 8, "square_size_mm": 20},
        )
    points = chessboard_object_points(11, 8, 20)[:, :2].reshape(-1, 1, 2) + 20
    assert (
        orient_chessboard_corners(np.full((200, 260), 128, np.uint8), points, (11, 8))
        is None
    )


def test_frame_pair_skew_and_duplicate_frame_rejected():
    first = {"frames": [{"frame_number": 10}, {"frame_number": 11}], "skew_ms": 25}
    assert pair_frames_fresh(first, None)
    assert not pair_frames_fresh(first, first)
    assert not pair_frames_fresh({**first, "skew_ms": 200}, None)


def test_independent_geometry_checks_both_directions():
    matrix = np.array(intrinsic("a", "a")["camera_matrix"])
    objects = chessboard_object_points(11, 8, 0.02)
    transform = np.eye(4)
    transform[0, 3] = 0.2
    observations = []
    for tvec in ([0.0, 0.0, 1.0], [-0.2, 0.0, 1.0]):
        pixels = cv2.projectPoints(
            objects, np.zeros(3), np.array(tvec), matrix, np.zeros(5)
        )[0]
        observations.append(
            BoardObservation(pixels, np.arange(len(objects)), objects, 0)
        )
    report, _ = evaluate_observations(
        observations, [matrix] * 2, [np.zeros(5)] * 2, transform
    )
    assert all(v["rmse"] < 0.001 for v in report["cross_errors_px"])


def validation_reports(tmp_path, same_pose=False):
    paths = []
    for i in range(5):
        pose = np.eye(4)
        pose[0, 3] = 0 if same_pose else i * 0.03
        samples = []
        for j in range(3):
            images = []
            for role in ("ref", "side"):
                image = tmp_path / f"{i}-{j}-{role}.png"
                image.write_bytes(f"{i}-{j}-{role}".encode())
                images.append(image)
            samples.append(
                {
                    "sample": j,
                    "self_errors_px": [{"rmse": 0.2}] * 2,
                    "cross_errors_px": [{"rmse": 0.5, "p95": 0.8}] * 2,
                    "reference_from_board": pose.tolist(),
                    "image_paths": [p.name for p in images],
                    "image_sha256": [file_sha256(p) for p in images],
                }
            )
        report = {
            "type": "independent_board_check",
            "pair_geometry_fingerprint": "pair",
            "scene_id": "scene",
            "intrinsic_binding_verified": True,
            "complete": True,
            "stationarity_checked": True,
            "samples": samples,
        }
        paths.append(write_new_json(tmp_path / f"report_{i}.json", report))
    return paths


def test_independent_acceptance_requires_distinct_new_poses(tmp_path):
    paths = validation_reports(tmp_path)
    report = check_acceptance_reports(
        paths, {"pair": {"fitting-image"}}, "scene", 1.0, 2.0
    )
    assert not report["absolute_accuracy_validated"]
    assert report["distinct_poses"]["pair"] == 5
    with pytest.raises(ValueError, match="reuses"):
        check_acceptance_reports(
            paths, {"pair": {file_sha256(tmp_path / "0-0-ref.png")}}, "scene", 1.0, 2.0
        )
    with pytest.raises(ValueError, match="physical scene"):
        check_acceptance_reports(paths, {"pair": set()}, "other-scene", 1.0, 2.0)


def test_repeated_single_pose_is_not_five_pose_validation(tmp_path):
    with pytest.raises(ValueError, match="five distinct"):
        check_acceptance_reports(
            validation_reports(tmp_path, True), {"pair": set()}, "scene", 1.0, 2.0
        )


@pytest.mark.parametrize(
    "failure",
    [
        "error",
        "failed_observation",
        "unbound",
        "missing_metrics",
        "changed_image",
        "negative_error",
    ],
)
def test_acceptance_never_ignores_bad_evidence(tmp_path, failure):
    paths = validation_reports(tmp_path)
    report = json.loads(paths[0].read_text())
    if failure == "error":
        report["samples"][0]["cross_errors_px"][0]["rmse"] = 8.0
    elif failure == "failed_observation":
        report["samples"][0]["failure"] = "missing corner"
    elif failure == "missing_metrics":
        report["samples"][0]["cross_errors_px"] = []
    elif failure == "changed_image":
        (tmp_path / report["samples"][0]["image_paths"][0]).write_bytes(b"changed")
    elif failure == "negative_error":
        report["samples"][0]["cross_errors_px"][0]["rmse"] = -1.0
    else:
        report["intrinsic_binding_verified"] = False
    paths[0].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        check_acceptance_reports(paths, {"pair": set()}, "scene", 1.0, 2.0)


def test_static_replay_handles_more_than_one_board_pair(
    bound_pair, tmp_path, monkeypatch
):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/36_validate_charuco_extrinsics.py"
    )
    spec = importlib.util.spec_from_file_location("validate_pair_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path, pair, _, _ = bound_pair
    write_new_json(
        tmp_path / "validation.json",
        {"pair": str(path), "samples": [{"sample": 1}, {"sample": 2}]},
    )
    matrix = np.array(intrinsic("a", "a")["camera_matrix"])
    objects = chessboard_object_points(11, 8, 0.02)
    detections = []
    for tvec in ([0.0, 0.0, 1.0], [-0.2, 0.0, 1.0]):
        pixels = cv2.projectPoints(
            objects, np.zeros(3), np.array(tvec), matrix, np.zeros(5)
        )[0]
        detections.append(BoardObservation(pixels, np.arange(len(objects)), objects, 0))
    repeated = iter(detections * 2)
    monkeypatch.setattr(
        module.CalibrationTarget, "detect", lambda self, image: next(repeated)
    )
    monkeypatch.setattr(
        module.cv2, "imread", lambda path: np.zeros((480, 640, 3), np.uint8)
    )
    report = module.static_check(
        tmp_path,
        pair,
        [matrix] * 2,
        [np.zeros(5)] * 2,
        np.array(pair["transform_reference_from_target"]),
    )
    assert report["target_reprojection_px"]["count"] == 2 * len(objects)
    assert report["target_reprojection_px"]["rmse"] < 0.001
