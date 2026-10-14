"""Independent board checks and acceptance policy, separate from fitting."""

from pathlib import Path

import cv2
import numpy as np

from xarm_sdk_control.calibration_data import file_sha256, read_json
from xarm_sdk_control.multiview_calibration import (
    transform_from_rvec_tvec,
    transform_rotation_angle_deg,
)


def error_summary(values):
    values = np.asarray(values, float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Validation residuals must be nonempty and finite")
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p95": float(np.percentile(np.abs(values), 95)),
        "maximum": float(np.max(np.abs(values))),
        "count": len(values),
    }


def evaluate_observations(observations, matrices, distortions, transform):
    if any(len(items) != 2 for items in (observations, matrices, distortions)):
        raise ValueError("Independent check needs exactly two views")
    poses, self_errors = [], []
    for observation, matrix, distortion in zip(observations, matrices, distortions):
        if observation is None or len(observation.corner_ids) < 8:
            raise ValueError(
                "At least eight identified corners are needed in each view"
            )
        if not np.all(np.isfinite(observation.corners_px)):
            raise ValueError("Nonfinite detected corners")
        if (
            np.linalg.matrix_rank(
                observation.object_points_m - observation.object_points_m.mean(axis=0),
                tol=1e-8,
            )
            < 2
        ):
            raise ValueError("Collinear board corners")
        ok, rvec, tvec = cv2.solvePnP(
            observation.object_points_m, observation.corners_px, matrix, distortion
        )
        if not ok:
            raise ValueError("Board pose estimation failed")
        pose = transform_from_rvec_tvec(rvec, tvec)
        if (
            not np.all(np.isfinite(pose))
            or np.min(
                (pose[:3, :3] @ observation.object_points_m.T).T[:, 2] + pose[2, 3]
            )
            <= 0
        ):
            raise ValueError("Board pose is nonfinite or behind camera")
        projected = cv2.projectPoints(
            observation.object_points_m, rvec, tvec, matrix, distortion
        )[0].reshape(-1, 2)
        self_errors.append(
            error_summary(
                np.linalg.norm(
                    projected - observation.corners_px.reshape(-1, 2), axis=1
                )
            )
        )
        poses.append(pose)
    cross, predictions = [], []
    for destination, source, relative in (
        (0, 1, transform),
        (1, 0, np.linalg.inv(transform)),
    ):
        pose = relative @ poses[source]
        obs = observations[destination]
        predicted = cv2.projectPoints(
            obs.object_points_m,
            cv2.Rodrigues(pose[:3, :3])[0],
            pose[:3, 3],
            matrices[destination],
            distortions[destination],
        )[0].reshape(-1, 2)
        predictions.append(predicted)
        cross.append(
            error_summary(
                np.linalg.norm(predicted - obs.corners_px.reshape(-1, 2), axis=1)
            )
        )
    return {
        "self_errors_px": self_errors,
        "cross_errors_px": cross,
        "reference_from_board": poses[0].tolist(),
    }, predictions


def check_acceptance_reports(paths, expected_pairs, scene_id, max_rmse_px, max_p95_px):
    """Require five distinct new board poses per pair; no claim of millimetre accuracy."""
    if (
        not all(np.isfinite(v) and v > 0 for v in (max_rmse_px, max_p95_px))
        or max_p95_px < max_rmse_px
    ):
        raise ValueError("Acceptance thresholds must be explicit, positive and ordered")
    poses = {key: [] for key in expected_pairs}
    if not poses:
        raise ValueError("No camera pairs to validate")
    receipts, seen_hashes = [], set()
    for path in paths:
        report = read_json(path)
        key = report.get("pair_geometry_fingerprint")
        if (
            report.get("type") != "independent_board_check"
            or key not in poses
            or report.get("scene_id") != scene_id
        ):
            raise ValueError("Validation is not for this pair and physical scene")
        if (
            report.get("intrinsic_binding_verified") is not True
            or not report.get("complete")
            or not report.get("stationarity_checked")
        ):
            raise ValueError("Incomplete, unbound or nonstationary validation")
        samples = report.get("samples", [])
        if len(samples) < 3:
            raise ValueError(
                "Need at least three repeat frames at each independent pose"
            )
        for sample in samples:
            if "failure" in sample:
                raise ValueError("Validation contains failed observations")
            if (
                len(sample.get("self_errors_px", [])) != 2
                or len(sample.get("cross_errors_px", [])) != 2
            ):
                raise ValueError("Missing metrics for one or both cameras")
            for metric in sample["self_errors_px"]:
                if not np.isfinite(metric["rmse"]) or not 0 <= metric["rmse"] <= 1.0:
                    raise ValueError("Independent sample has poor local PnP fit")
            for metric in sample["cross_errors_px"]:
                if (
                    not np.isfinite(metric["rmse"])
                    or not np.isfinite(metric["p95"])
                    or not 0 <= metric["rmse"] <= max_rmse_px
                    or not 0 <= metric["p95"] <= max_p95_px
                ):
                    raise ValueError(
                        "Independent cross-view residual exceeds acceptance policy"
                    )
            hashes = sample.get("image_sha256", [])
            if (
                len(hashes) != 2
                or len(set(hashes)) != 2
                or any(h in seen_hashes or h in expected_pairs[key] for h in hashes)
            ):
                raise ValueError(
                    "Validation reuses fitting images or repeated evidence"
                )
            image_paths = sample.get("image_paths", [])
            if len(image_paths) != 2 or any(
                file_sha256(Path(path).parent / image) != expected
                for image, expected in zip(image_paths, hashes)
            ):
                raise ValueError("Independent raw images are missing or changed")
            seen_hashes.update(hashes)
        pose = np.array(samples[0]["reference_from_board"], float)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("Invalid independent board pose")
        if all(
            np.linalg.norm(pose[:3, 3] - old[:3, 3]) >= 0.02
            or transform_rotation_angle_deg(pose, old) >= 5
            for old in poses[key]
        ):
            poses[key].append(pose)
        receipts.append({"path": str(path), "sha256": file_sha256(path)})
    if any(len(items) < 5 for items in poses.values()):
        raise ValueError(
            "Each pair needs at least five distinct independent poses (20 mm translation or 5 deg rotation)"
        )
    return {
        "reports": receipts,
        "distinct_poses": {key: len(items) for key, items in poses.items()},
        "max_rmse_px": max_rmse_px,
        "max_p95_px": max_p95_px,
        "absolute_accuracy_validated": False,
        "scope": "static_rgbd_capture_only_not_robot_motion",
    }
