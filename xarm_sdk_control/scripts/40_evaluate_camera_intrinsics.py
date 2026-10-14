#!/usr/bin/env python3
"""Compare Brown4/Brown5 intrinsics offline; save candidates, never activate them."""

import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from xarm_sdk_control.calibration_data import (
    file_sha256,
    intrinsic_health,
    load_intrinsics,
    write_new_json,
)
from xarm_sdk_control.multiview_calibration import (
    calibrate_chessboard_intrinsics,
    chessboard_object_points,
    find_chessboard_corners,
)
from xarm_sdk_control.run_output import new_timestamped_run_dir


def evaluate(path, output, redetect):
    old_k, old_d, payload = load_intrinsics(path)
    size = tuple(payload["capture"][k] for k in ("width", "height"))
    spec = payload["chessboard"]
    pattern = (spec["inner_corners_cols"], spec["inner_corners_rows"])
    square = spec["square_size_mm"] / 1000
    objects = chessboard_object_points(*pattern, square)
    points = []
    for sample in payload["samples"]:
        if redetect:
            image_path = Path(payload["sample_run_dir"]) / sample["color_path"]
            image = cv2.imread(str(image_path))
            if image is None or image.shape[:2] != size[::-1]:
                raise ValueError(f"Missing/mismatched intrinsic image {image_path}")
            found, corners = find_chessboard_corners(image, pattern)
            if not found:
                raise ValueError(
                    f"Redetection failed; no silent sample dropping: {image_path}"
                )
            sample["corner_points_px"] = corners.reshape(-1, 2).tolist()
            sample["color_path"] = str(image_path.resolve())
            sample["image_sha256"] = file_sha256(image_path)
        points.append(
            np.array(sample["corner_points_px"], np.float32).reshape(-1, 1, 2)
        )
    if len(points) < 10:
        raise ValueError("Need at least 10 observations for five-fold evaluation")
    report = {
        "source": str(path),
        "source_sha256": file_sha256(path),
        "redetected": redetect,
        "existing_health": intrinsic_health(old_k, old_d, size),
        "models": [],
    }
    for model in ("brown4", "brown5"):
        errors, folds = [], []
        for fold in range(5):
            training = [p for i, p in enumerate(points) if i % 5 != fold]
            fit = calibrate_chessboard_intrinsics(
                training, size, *pattern, square, model=model
            )
            held_out = []
            for index, pixels in enumerate(points):
                if index % 5 != fold:
                    continue
                ok, rvec, tvec = cv2.solvePnP(
                    objects, pixels, fit.camera_matrix, fit.distortion_coefficients
                )
                if not ok:
                    raise ValueError("Held-out board pose failed")
                projected = cv2.projectPoints(
                    objects, rvec, tvec, fit.camera_matrix, fit.distortion_coefficients
                )[0]
                values = np.linalg.norm(projected - pixels, axis=2).ravel()
                held_out.extend(values)
                errors.extend(values)
            folds.append(
                {
                    "fold": fold,
                    "held_out_local_rmse_px": float(
                        np.sqrt(np.mean(np.square(held_out)))
                    ),
                    "health": intrinsic_health(
                        fit.camera_matrix, fit.distortion_coefficients, size
                    ),
                }
            )
        fit = calibrate_chessboard_intrinsics(
            points, size, *pattern, square, model=model
        )
        candidate = copy.deepcopy(payload)
        health = intrinsic_health(fit.camera_matrix, fit.distortion_coefficients, size)
        candidate.update(
            schema_version=2,
            calibration_status="candidate",
            intrinsic_model=model,
            created_at=datetime.now().astimezone().isoformat(),
            camera_matrix=fit.camera_matrix.tolist(),
            distortion_coefficients=fit.distortion_coefficients.tolist(),
        )
        candidate["quality"] = {
            "numerical_health": health,
            "absolute_accuracy_validated": False,
            "rms_reprojection_error_px": fit.rms_reprojection_error_px,
            "per_view_reprojection_error_px": fit.per_view_reprojection_error_px,
            "held_out_local_rmse_px": float(np.sqrt(np.mean(np.square(errors)))),
            "folds": folds,
        }
        pixels = np.concatenate(points).reshape(-1, 2)
        candidate["quality"]["sample_coverage"] = {
            "pixel_min": pixels.min(0).tolist(),
            "pixel_max": pixels.max(0).tolist(),
            "note": "Extrema are not proof of full-field coverage; independent edge and metric validation remain required",
        }
        candidate["provenance"] = {
            "source": str(path),
            "source_sha256": file_sha256(path),
            "redetected": redetect,
            "protocol": "five interleaved folds; intrinsics fitted on training only, held-out board pose fitted locally; no metric accuracy bound",
        }
        result_path = output / f"{payload['camera_id']}_{model}.json"
        write_new_json(result_path, candidate)
        report["models"].append(
            {
                "model": model,
                "candidate": str(result_path),
                "health": health,
                "held_out_local_rmse_px": candidate["quality"][
                    "held_out_local_rmse_px"
                ],
                "all_folds_healthy": all(f["health"]["passed"] for f in folds),
            }
        )
    healthy = [
        m for m in report["models"] if m["health"]["passed"] and m["all_folds_healthy"]
    ]
    report["candidate_for_independent_validation"] = (
        min(healthy, key=lambda m: m["held_out_local_rmse_px"])["candidate"]
        if healthy
        else None
    )
    report["live_configuration_changed"] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intrinsics", type=Path, required=True, action="append")
    parser.add_argument("--redetect", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data/multiview_neck/intrinsic_evaluation",
    )
    args = parser.parse_args()
    output = new_timestamped_run_dir(args.output_root.expanduser().resolve())
    reports = [
        evaluate(p.expanduser().resolve(), output, args.redetect)
        for p in args.intrinsics
    ]
    write_new_json(
        output / "evaluation.json",
        {"cameras": reports, "absolute_accuracy_validated": False},
    )
    for report in reports:
        print(report["source"])
        for model in report["models"]:
            print(
                model["model"],
                "heldout",
                model["held_out_local_rmse_px"],
                "health",
                model["health"]["passed"],
            )
    print(f"Saved offline evaluation: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
