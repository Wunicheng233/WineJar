#!/usr/bin/env python3
"""Leave out paired board poses and check cross-view reprojection and metric scale."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from xarm_sdk_control.calibration_data import (
    file_sha256,
    intrinsic_health,
    load_pair_intrinsics,
)
from xarm_sdk_control.calibration_target import CalibrationTarget
from xarm_sdk_control.multiview_calibration import (
    calibrate_relative_camera_pose_from_observations,
    reprojection_error_px,
    transform_from_rvec_tvec,
)
from xarm_sdk_control.run_output import new_timestamped_run_dir


def summary(values):
    values = np.asarray(values, float)
    if not len(values):
        return None
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p95_abs": float(np.percentile(np.abs(values), 95)),
        "rmse": float(np.sqrt(np.mean(values**2))),
    }


def static_check(directory, payload, matrices, distortions, transform):
    metadata = json.loads((directory / "validation.json").read_text())
    source = json.loads(Path(metadata["pair"]).read_text())
    for role in ["reference_camera", "target_camera"]:
        if (
            source[role]["serial"] != payload[role]["serial"]
            or source[role]["camera_id"] != payload[role]["camera_id"]
        ):
            raise ValueError(
                "Stationary validation camera identity differs from the pair"
            )
    calibration_target = CalibrationTarget.from_pair(payload)
    if source.get(calibration_target.kind) != payload[calibration_target.kind]:
        raise ValueError("Stationary validation uses another board specification")
    errors = []
    for sample in metadata["samples"]:
        if "failure" in sample:
            continue
        detections = []
        for role in ["reference_camera", "target_camera"]:
            image = cv2.imread(
                str(
                    directory
                    / f"{sample['sample']:02d}_{payload[role]['camera_id']}.png"
                )
            )
            if image is None or image.shape[:2] != (
                payload["capture"]["height"],
                payload["capture"]["width"],
            ):
                raise ValueError("Missing or mismatched stationary image")
            detection = calibration_target.detect(image)
            if detection is None:
                raise ValueError("Stationary image cannot be redetected")
            detections.append(detection)
        reference, target = detections
        ok, rvec, tvec = cv2.solvePnP(
            reference.object_points_m, reference.corners_px, matrices[0], distortions[0]
        )
        if not ok:
            raise ValueError("Stationary board pose failed")
        pose = np.linalg.inv(transform) @ transform_from_rvec_tvec(rvec, tvec)
        predicted, _ = cv2.projectPoints(
            target.object_points_m,
            cv2.Rodrigues(pose[:3, :3])[0],
            pose[:3, 3],
            matrices[1],
            distortions[1],
        )
        errors.extend(
            np.linalg.norm(
                predicted.reshape(-1, 2) - target.corners_px.reshape(-1, 2), axis=1
            )
        )
    if not errors:
        raise ValueError("No valid stationary observations")
    return {
        "capture": str(directory.resolve()),
        "target_reprojection_px": summary(errors),
        "limitation": "Full calibration fit tested on separate images of one pose; does not establish performance at other poses.",
    }


def validate_pair(
    path, method="pose_average", fold=4, stationary_capture=None, allow_legacy=False
):
    payload = json.loads(path.read_text())
    ordinary = payload.get("type") == "fixed_camera_pair_chessboard_extrinsics"
    if not ordinary and payload.get("type") != "fixed_camera_pair_charuco_extrinsics":
        raise ValueError("This validator requires saved paired board observations")
    matrices, distortions, _ = load_pair_intrinsics(
        payload, path, allow_legacy=allow_legacy
    )
    calibration_target = CalibrationTarget.from_pair(payload)
    for sample in payload["samples"]:
        for prefix in ("reference", "target"):
            observation = calibration_target.observation(sample, prefix)
            sample[prefix + "_corner_ids"] = observation.corner_ids.tolist()
    spec = calibration_target.spec
    objects = calibration_target.objects
    samples = payload["samples"]
    held_out = [i for i in range(len(samples)) if i % 5 == fold]
    training = [i for i in range(len(samples)) if i not in held_out]
    if len(held_out) < 2 or len(training) < 8:
        raise ValueError(
            "At least ten paired samples are required for leave-out validation"
        )

    def objects_at(index, prefix):
        return objects[np.array(samples[index][prefix + "_corner_ids"], int)]

    def pixels_at(index, prefix):
        return np.array(samples[index][prefix + "_corner_points_px"], float).reshape(
            -1, 2
        )

    def fit_indices(indices):
        return calibrate_relative_camera_pose_from_observations(
            [objects_at(i, "reference") for i in indices],
            [pixels_at(i, "reference") for i in indices],
            [objects_at(i, "target") for i in indices],
            [pixels_at(i, "target") for i in indices],
            matrices[0],
            distortions[0],
            matrices[1],
            distortions[1],
            method=method,
            image_size_px=(payload["capture"]["width"], payload["capture"]["height"]),
        )

    fit = fit_indices(training)
    target_from_reference = np.linalg.inv(fit.transform_reference_from_target)
    errors, scale_errors, sample_reports = [], [], []
    for index in held_out:
        ok, rvec, tvec = cv2.solvePnP(
            objects_at(index, "reference"),
            pixels_at(index, "reference"),
            matrices[0],
            distortions[0],
        )
        if not ok:
            raise ValueError(f"Held-out reference PnP failed for sample {index + 1}")
        reference_from_board = transform_from_rvec_tvec(rvec, tvec)
        self_errors = {
            "reference": reprojection_error_px(
                objects_at(index, "reference"),
                pixels_at(index, "reference"),
                matrices[0],
                distortions[0],
                rvec,
                tvec,
            )
        }
        target_ok, target_rvec, target_tvec = cv2.solvePnP(
            objects_at(index, "target"),
            pixels_at(index, "target"),
            matrices[1],
            distortions[1],
        )
        self_errors["target"] = (
            reprojection_error_px(
                objects_at(index, "target"),
                pixels_at(index, "target"),
                matrices[1],
                distortions[1],
                target_rvec,
                target_tvec,
            )
            if target_ok
            else None
        )
        target_from_board = target_from_reference @ reference_from_board
        predicted, _ = cv2.projectPoints(
            objects_at(index, "target"),
            cv2.Rodrigues(target_from_board[:3, :3])[0],
            target_from_board[:3, 3],
            matrices[1],
            distortions[1],
        )
        pixel_error = np.linalg.norm(
            predicted.reshape(-1, 2) - pixels_at(index, "target"), axis=1
        )
        errors.extend(pixel_error)
        sample = samples[index]
        common, ir, it = np.intersect1d(
            sample["reference_corner_ids"],
            sample["target_corner_ids"],
            return_indices=True,
        )
        if len(common) >= 2:
            left = cv2.undistortPoints(
                pixels_at(index, "reference")[ir, None], matrices[0], distortions[0]
            ).reshape(-1, 2)
            right = cv2.undistortPoints(
                pixels_at(index, "target")[it, None], matrices[1], distortions[1]
            ).reshape(-1, 2)
            homogeneous = cv2.triangulatePoints(
                np.eye(3, 4), target_from_reference[:3], left.T, right.T
            )
            reconstructed = (homogeneous[:3] / homogeneous[3]).T
            for a in range(len(common)):
                for b in range(a + 1, len(common)):
                    distance = np.linalg.norm(objects[common[a]] - objects[common[b]])
                    if np.isclose(distance, spec["square_size_mm"] / 1000, atol=1e-6):
                        scale_errors.append(
                            (
                                np.linalg.norm(reconstructed[a] - reconstructed[b])
                                - distance
                            )
                            * 1000
                        )
        sample_reports.append(
            {
                "sample": index + 1,
                "target_reprojection_px": summary(pixel_error),
                "shared_corner_count": len(common),
                "single_view_pnp_rmse_px": self_errors,
                "observation_quality_pass": all(
                    value is not None and np.isfinite(value) and value <= 1.0
                    for value in self_errors.values()
                ),
            }
        )
    static_result = None
    if stationary_capture is not None:
        full_fit = fit_indices(range(len(samples)))
        static_result = static_check(
            stationary_capture,
            payload,
            matrices,
            distortions,
            full_fit.transform_reference_from_target,
        )
    return {
        "source": str(path.resolve()),
        "source_sha256": file_sha256(path),
        "intrinsic_binding_verified": all(
            "intrinsics_fingerprint" in payload[k]
            for k in ("reference_camera", "target_camera")
        ),
        "intrinsic_health": [
            intrinsic_health(
                k, d, (payload["capture"]["width"], payload["capture"]["height"])
            )
            for k, d in zip(matrices, distortions)
        ],
        "method": method,
        "fold": fold,
        "retained_training_sample_indices": [
            training[i - 1] + 1 for i in fit.accepted_indices
        ],
        "training_stereo_rms_px": fit.stereo_rms_px,
        "transform_reference_from_target": fit.transform_reference_from_target.tolist(),
        "camera_id": payload["target_camera"]["camera_id"],
        "training_sample_count": len(training),
        "held_out_sample_indices": [i + 1 for i in held_out],
        "cross_view_reprojection_error_px": summary(errors),
        "adjacent_square_length_error_mm": summary(scale_errors),
        "samples": sample_reports,
        "full_fit_stationary_check": static_result,
        "limitation": "Poses are withheld from this extrinsic refit only. Intrinsics and physical printed-square dimensions are not independently validated. This does not bound jar localization error.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True, action="append")
    parser.add_argument(
        "--allow-legacy-input",
        action="store_true",
        help="Offline historical diagnostics only; missing provenance is not certified",
    )
    parser.add_argument(
        "--method",
        choices=["pose_average", "stereo", "compare"],
        default="pose_average",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Use all five interleaved held-out splits",
    )
    parser.add_argument(
        "--stationary-capture",
        type=Path,
        help="Optional saved 38 output for this camera pair; no live streams",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/multiview_neck/calibration_validation",
    )
    args = parser.parse_args()
    try:
        methods = (
            ["pose_average", "stereo"] if args.method == "compare" else [args.method]
        )
        results = [
            validate_pair(
                path.expanduser().resolve(),
                method,
                fold,
                args.stationary_capture if fold == 4 else None,
                args.allow_legacy_input,
            )
            for path in args.pair
            for fold in (range(5) if args.all_folds else [4])
            for method in methods
        ]
    except (ValueError, KeyError, OSError, cv2.error) as error:
        print(f"ERROR: {error}")
        return 2
    output = new_timestamped_run_dir(args.output_root)
    aggregate = []
    for source, method in sorted({(r["source"], r["method"]) for r in results}):
        group = [r for r in results if r["source"] == source and r["method"] == method]
        metrics = [r["cross_view_reprojection_error_px"] for r in group]
        count = sum(m["count"] for m in metrics)
        aggregate.append(
            {
                "source": source,
                "method": method,
                "folds": len(group),
                "pooled_cross_rmse_px": float(
                    np.sqrt(sum(m["count"] * m["rmse"] ** 2 for m in metrics) / count)
                ),
            }
        )
    (output / "validation.json").write_text(
        json.dumps(
            {
                "type": "paired_board_leave_out_validation",
                "pairs": results,
                "aggregate": aggregate,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(f"Saved validation: {output}")
    for result in results:
        print(
            result["camera_id"],
            result["method"],
            f"fold={result['fold']}",
            "pixel error",
            result["cross_view_reprojection_error_px"],
            "square error mm",
            result["adjacent_square_length_error_mm"],
        )
        if result["full_fit_stationary_check"] is not None:
            print(
                "Stationary",
                result["method"],
                result["full_fit_stationary_check"]["target_reprojection_px"],
            )
    for item in aggregate:
        print("Aggregate", item["method"], item["folds"], item["pooled_cross_rmse_px"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
