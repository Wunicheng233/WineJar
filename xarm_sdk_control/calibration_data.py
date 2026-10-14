"""Calibration identity, immutable numerical provenance and intrinsic diagnostics."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

SCENE_STATE_PATH = (
    Path(__file__).resolve().parent / "calibration/multiview_neck/scene_status.json"
)


def current_scene_id():
    state = read_json(SCENE_STATE_PATH)
    if not isinstance(state.get("scene_id"), str) or not state["scene_id"]:
        raise ValueError(
            "Scene has no identity; initialize scene_status.json before calibration"
        )
    return state["scene_id"]


def fit_quality(fit, sample_count):
    def stats(values):
        values = np.asarray(values, float)
        return {"mean": float(values.mean()), "max": float(values.max())}

    return {
        "paired_sample_count": sample_count,
        "method": fit.method,
        "stereo_training_rms_px": fit.stereo_rms_px,
        "pnp_accepted_indices": fit.pnp_accepted_indices,
        "accepted_indices": fit.accepted_indices,
        "rejected_indices": fit.rejected_indices,
        "reference_pnp_reprojection_error_px": stats(
            fit.reference_reprojection_errors_px
        ),
        "target_pnp_reprojection_error_px": stats(fit.target_reprojection_errors_px),
        "transform_translation_residual_mm": stats(
            np.asarray(fit.translation_residuals_m) * 1000
        ),
        "transform_rotation_residual_deg": stats(fit.rotation_residuals_deg),
    }


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def write_new_json(path, payload):
    """Never silently overwrite a calibration or its evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return path


def intrinsic_snapshot(payload):
    if payload.get("type") != "fixed_camera_chessboard_intrinsics":
        raise ValueError("Unsupported intrinsic calibration type")
    serial = payload.get("camera", {}).get("serial")
    size = [payload.get("capture", {}).get(k) for k in ("width", "height")]
    matrix = np.asarray(payload.get("camera_matrix"), float)
    distortion = np.asarray(payload.get("distortion_coefficients"), float).reshape(-1)
    if (
        not isinstance(serial, str)
        or not serial
        or not all(type(v) is int and v > 0 for v in size)
    ):
        raise ValueError("Missing camera serial or invalid image dimensions")
    if (
        matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.allclose(matrix[2], [0, 0, 1])
        or not np.allclose([matrix[0, 1], matrix[1, 0]], 0)
    ):
        raise ValueError("Invalid OpenCV camera matrix")
    if distortion.size not in (4, 5) or not np.all(np.isfinite(distortion)):
        raise ValueError("Expected finite Brown-Conrady k1,k2,p1,p2[,k3] coefficients")
    return {
        "serial": serial,
        "image_size_px": size,
        "model": "opencv_brown_conrady",
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.tolist(),
    }


def load_intrinsics(path, expected_serial=None, image_size=None):
    payload = read_json(path)
    snapshot = intrinsic_snapshot(payload)
    if expected_serial is not None and snapshot["serial"] != expected_serial:
        raise ValueError("Intrinsic camera serial mismatch")
    if image_size is not None and snapshot["image_size_px"] != list(image_size):
        raise ValueError("Intrinsic image size mismatch")
    return (
        np.array(snapshot["camera_matrix"]),
        np.array(snapshot["distortion_coefficients"]),
        payload,
    )


def bind_intrinsics(entry, payload):
    snapshot = intrinsic_snapshot(payload)
    entry["intrinsics_snapshot"] = snapshot
    entry["intrinsics_fingerprint"] = fingerprint(snapshot)


def load_pair_intrinsics(pair, owner, *, allow_legacy=False):
    matrices, distortions, payloads = [], [], []
    size = [pair["capture"][k] for k in ("width", "height")]
    for role in ("reference_camera", "target_camera"):
        entry = pair[role]
        path = (Path(owner).parent / entry["intrinsics_path"]).resolve()
        matrix, distortion, payload = load_intrinsics(path, entry["serial"], size)
        snapshot = intrinsic_snapshot(payload)
        recorded = entry.get("intrinsics_fingerprint")
        if recorded is None:
            if not allow_legacy:
                raise ValueError(
                    "Legacy pair has no intrinsic binding; refit into a new candidate first"
                )
        elif (
            recorded != fingerprint(snapshot)
            or entry.get("intrinsics_snapshot") != snapshot
        ):
            raise ValueError("Pair intrinsic values changed since fitting")
        matrices.append(matrix)
        distortions.append(distortion)
        payloads.append(payload)
    if pair["reference_camera"]["serial"] == pair["target_camera"]["serial"]:
        raise ValueError("A pair needs two distinct cameras")
    return matrices, distortions, payloads


def pair_geometry_fingerprint(pair):
    return fingerprint(
        {
            "cameras": [
                {
                    "id": pair[role]["camera_id"],
                    "intrinsics": pair[role]["intrinsics_fingerprint"],
                }
                for role in ("reference_camera", "target_camera")
            ],
            "transform": pair["transform_reference_from_target"],
        }
    )


def intrinsic_health(matrix, distortion, image_size):
    """Check model invertibility on a fixed full-image grid, not metrology accuracy."""
    matrix, distortion = (
        np.asarray(matrix, float),
        np.asarray(distortion, float).reshape(-1),
    )
    # Reuse validation without requiring an on-disk calibration.
    intrinsic_snapshot(
        {
            "type": "fixed_camera_chessboard_intrinsics",
            "camera": {"serial": "diagnostic"},
            "capture": dict(zip(("width", "height"), image_size)),
            "camera_matrix": matrix,
            "distortion_coefficients": distortion,
        }
    )
    width, height = image_size
    x, y = np.meshgrid(np.linspace(0, width - 1, 33), np.linspace(0, height - 1, 25))
    pixels = np.column_stack((x.ravel(), y.ravel()))
    rays = cv2.undistortPoints(pixels[:, None], matrix, distortion).reshape(-1, 2)
    projected = cv2.projectPoints(
        np.column_stack((rays, np.ones(len(rays)))),
        np.zeros(3),
        np.zeros(3),
        matrix,
        distortion,
    )[0].reshape(-1, 2)
    errors = np.linalg.norm(projected - pixels, axis=1)
    radius = np.linalg.norm(
        (pixels - matrix[:2, 2]) / np.diag(matrix)[:2], axis=1
    ).max()
    r = np.linspace(0, max(radius, float(np.linalg.norm(rays, axis=1).max())), 2049)
    k1, k2 = distortion[:2]
    k3 = distortion[4] if len(distortion) == 5 else 0.0
    derivative = 1 + 3 * k1 * r**2 + 5 * k2 * r**4 + 7 * k3 * r**6
    finite = np.all(np.isfinite(errors)) and np.all(np.isfinite(derivative))
    return {
        "passed": bool(finite and errors.max() <= 0.25 and derivative.min() > 0),
        "max_roundtrip_px": float(errors.max()) if finite else None,
        "p95_roundtrip_px": float(np.percentile(errors, 95)) if finite else None,
        "minimum_radial_derivative": float(derivative.min()) if finite else None,
        "roundtrip_limit_px": 0.25,
        "tested_grid": [33, 25],
        "scope": "full_image_numerical_health_not_absolute_accuracy",
    }
