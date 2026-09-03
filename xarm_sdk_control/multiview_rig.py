"""Validated calibration bundle for the fixed three-camera neck-localization rig."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


RIG_TYPE = "fixed_multiview_camera_rig"
INTRINSIC_TYPE = "fixed_camera_chessboard_intrinsics"
EXTRINSIC_TYPE = "fixed_camera_pair_chessboard_extrinsics"


@dataclass(frozen=True)
class FixedCameraModel:
    """One calibrated camera expressed in the rig reference-camera frame."""

    camera_id: str
    serial: str
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_size_px: tuple[int, int]
    transform_reference_from_camera: np.ndarray
    intrinsics_path: Path

    @property
    def transform_camera_from_reference(self) -> np.ndarray:
        return np.linalg.inv(self.transform_reference_from_camera)


@dataclass(frozen=True)
class FixedCameraRig:
    """A collection of fixed cameras sharing the reference camera coordinate frame."""

    reference_camera_id: str
    cameras: dict[str, FixedCameraModel]
    source_extrinsics_paths: tuple[Path, ...]

    def camera(self, camera_id: str) -> FixedCameraModel:
        try:
            return self.cameras[camera_id]
        except KeyError as exc:
            raise KeyError(f"Unknown rig camera {camera_id!r}; available: {sorted(self.cameras)}") from exc

    def transform_points_to_reference(self, camera_id: str, points_camera_m: np.ndarray) -> np.ndarray:
        """Transform N camera-frame points into the reference camera frame."""
        points = np.asarray(points_camera_m, dtype=np.float64).reshape(-1, 3)
        transform = self.camera(camera_id).transform_reference_from_camera
        return (transform[:3, :3] @ points.T).T + transform[:3, 3]

    def project_reference_points(self, camera_id: str, points_reference_m: np.ndarray) -> np.ndarray:
        """Project reference-frame 3D points into one calibrated camera image."""
        camera = self.camera(camera_id)
        rotation = camera.transform_camera_from_reference[:3, :3]
        translation = camera.transform_camera_from_reference[:3, 3]
        rvec, _ = cv2.Rodrigues(rotation)
        projected, _ = cv2.projectPoints(
            np.asarray(points_reference_m, dtype=np.float64).reshape(-1, 3),
            rvec,
            translation,
            camera.camera_matrix,
            camera.distortion_coefficients,
        )
        return projected.reshape(-1, 2)


def _read_json(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    return json.loads(resolved.read_text(encoding="utf-8"))


def _resolve_reference(value: str | Path, owner_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner_path.parent / path
    return path.resolve()


def _relative_reference(path: Path, owner_path: Path) -> str:
    return os.path.relpath(path.expanduser().resolve(), owner_path.expanduser().resolve().parent)


def _validated_transform(value: object, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{label} must use homogeneous final row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError(f"{label} rotation is not a proper orthonormal matrix")
    return transform


def _camera_from_intrinsics(path: Path, expected_camera_id: str | None = None, expected_serial: str | None = None) -> FixedCameraModel:
    resolved = path.expanduser().resolve()
    payload = _read_json(resolved)
    if payload.get("type") != INTRINSIC_TYPE:
        raise ValueError(f"{resolved} is not a fixed-camera chessboard intrinsic calibration")
    camera_id = payload.get("camera_id")
    serial = payload.get("camera", {}).get("serial")
    if not isinstance(camera_id, str) or not isinstance(serial, str):
        raise ValueError(f"{resolved} is missing camera_id or camera serial")
    if expected_camera_id is not None and camera_id != expected_camera_id:
        raise ValueError(f"{resolved} belongs to camera_id {camera_id!r}, expected {expected_camera_id!r}")
    if expected_serial is not None and serial != expected_serial:
        raise ValueError(f"{resolved} belongs to serial {serial!r}, expected {expected_serial!r}")
    capture = payload.get("capture", {})
    image_size = (capture.get("width"), capture.get("height"))
    if not all(isinstance(value, int) and value > 0 for value in image_size):
        raise ValueError(f"{resolved} has invalid capture dimensions")
    matrix = np.asarray(payload.get("camera_matrix"), dtype=np.float64)
    distortion = np.asarray(payload.get("distortion_coefficients"), dtype=np.float64).reshape(-1)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError(f"{resolved} has invalid camera matrix")
    if distortion.size < 4 or not np.all(np.isfinite(distortion)):
        raise ValueError(f"{resolved} has invalid distortion coefficients")
    return FixedCameraModel(
        camera_id=camera_id,
        serial=serial,
        camera_matrix=matrix,
        distortion_coefficients=distortion,
        image_size_px=(int(image_size[0]), int(image_size[1])),
        transform_reference_from_camera=np.eye(4, dtype=np.float64),
        intrinsics_path=resolved,
    )


def build_fixed_camera_rig(reference_intrinsics_path: Path, pair_extrinsics_paths: Iterable[Path]) -> FixedCameraRig:
    """Build a rig from one reference intrinsic file and reference-to-target pairs."""
    reference = _camera_from_intrinsics(reference_intrinsics_path)
    cameras: dict[str, FixedCameraModel] = {reference.camera_id: reference}
    sources: list[Path] = []
    for extrinsics_path in pair_extrinsics_paths:
        resolved = Path(extrinsics_path).expanduser().resolve()
        payload = _read_json(resolved)
        if payload.get("type") != EXTRINSIC_TYPE:
            raise ValueError(f"{resolved} is not a fixed-camera pair extrinsic calibration")
        pair_reference = payload.get("reference_camera", {})
        pair_target = payload.get("target_camera", {})
        if (
            pair_reference.get("camera_id") != reference.camera_id
            or pair_reference.get("serial") != reference.serial
        ):
            raise ValueError(f"{resolved} does not use the selected reference camera")
        target_id = pair_target.get("camera_id")
        target_serial = pair_target.get("serial")
        target_intrinsics_path = pair_target.get("intrinsics_path")
        if not isinstance(target_id, str) or not isinstance(target_serial, str) or not isinstance(target_intrinsics_path, str):
            raise ValueError(f"{resolved} is missing target-camera calibration metadata")
        if target_id in cameras:
            raise ValueError(f"Camera {target_id!r} appears in more than one pair extrinsic calibration")
        target = _camera_from_intrinsics(
            _resolve_reference(target_intrinsics_path, resolved),
            target_id,
            target_serial,
        )
        transform = _validated_transform(payload.get("transform_reference_from_target"), resolved.name)
        cameras[target_id] = FixedCameraModel(
            camera_id=target.camera_id,
            serial=target.serial,
            camera_matrix=target.camera_matrix,
            distortion_coefficients=target.distortion_coefficients,
            image_size_px=target.image_size_px,
            transform_reference_from_camera=transform,
            intrinsics_path=target.intrinsics_path,
        )
        sources.append(resolved)
    if len(cameras) < 2:
        raise ValueError("A multiview rig requires a reference camera and at least one target camera")
    return FixedCameraRig(reference.camera_id, cameras, tuple(sources))


def rig_to_dict(rig: FixedCameraRig, owner_path: Path | None = None) -> dict:
    """Serialize a validated rig without duplicating the intrinsic matrices' provenance."""
    return {
        "schema_version": 1,
        "type": RIG_TYPE,
        "reference_camera_id": rig.reference_camera_id,
        "coordinate_convention": {
            "units": "metres",
            "reference_frame": "reference camera optical frame",
            "camera_axes": "+x right, +y down, +z forward",
            "transform": "p_reference = T_reference_from_camera @ p_camera_homogeneous",
        },
        "cameras": [
            {
                "camera_id": camera.camera_id,
                "serial": camera.serial,
                "intrinsics_path": (
                    _relative_reference(camera.intrinsics_path, owner_path)
                    if owner_path is not None
                    else str(camera.intrinsics_path)
                ),
                "image_size_px": list(camera.image_size_px),
                "transform_reference_from_camera": camera.transform_reference_from_camera.tolist(),
            }
            for camera in rig.cameras.values()
        ],
        "source_extrinsics_paths": [
            _relative_reference(path, owner_path) if owner_path is not None else str(path)
            for path in rig.source_extrinsics_paths
        ],
    }


def write_fixed_camera_rig(rig: FixedCameraRig, path: Path) -> Path:
    """Write a validated rig manifest for capture and offline localization scripts."""
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(rig_to_dict(rig, resolved), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved


def load_fixed_camera_rig(path: Path) -> FixedCameraRig:
    """Load a rig manifest and revalidate referenced intrinsic files and transforms."""
    resolved = path.expanduser().resolve()
    payload = _read_json(resolved)
    if payload.get("type") != RIG_TYPE:
        raise ValueError(f"{resolved} is not a fixed multiview camera rig")
    reference_id = payload.get("reference_camera_id")
    entries = payload.get("cameras")
    if not isinstance(reference_id, str) or not isinstance(entries, list):
        raise ValueError(f"{resolved} is missing reference camera or camera list")
    cameras: dict[str, FixedCameraModel] = {}
    for entry in entries:
        camera_id = entry.get("camera_id")
        serial = entry.get("serial")
        intrinsics_path = entry.get("intrinsics_path")
        if not isinstance(camera_id, str) or not isinstance(serial, str) or not isinstance(intrinsics_path, str):
            raise ValueError(f"{resolved} has invalid camera entry")
        if camera_id in cameras:
            raise ValueError(f"{resolved} contains duplicate camera_id {camera_id!r}")
        camera = _camera_from_intrinsics(
            _resolve_reference(intrinsics_path, resolved),
            camera_id,
            serial,
        )
        transform = _validated_transform(entry.get("transform_reference_from_camera"), f"{resolved.name}:{camera_id}")
        cameras[camera_id] = FixedCameraModel(
            camera_id=camera.camera_id,
            serial=camera.serial,
            camera_matrix=camera.camera_matrix,
            distortion_coefficients=camera.distortion_coefficients,
            image_size_px=camera.image_size_px,
            transform_reference_from_camera=transform,
            intrinsics_path=camera.intrinsics_path,
        )
    if reference_id not in cameras:
        raise ValueError(f"{resolved} reference camera {reference_id!r} is absent")
    if not np.allclose(cameras[reference_id].transform_reference_from_camera, np.eye(4), atol=1e-8):
        raise ValueError(f"{resolved} reference camera transform must be identity")
    source_paths = tuple(
        _resolve_reference(path, resolved) for path in payload.get("source_extrinsics_paths", [])
    )
    return FixedCameraRig(reference_id, cameras, source_paths)
