"""Pure geometry helpers for the fixed multi-camera neck-localization rig."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ChessboardIntrinsicCalibration:
    """Intrinsic result plus per-image reprojection errors in pixels."""

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rms_reprojection_error_px: float
    per_view_reprojection_error_px: list[float]
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]


@dataclass(frozen=True)
class RelativeCameraPoseCalibration:
    """Rigid transform from one fixed camera frame into another."""

    transform_reference_from_target: np.ndarray
    pnp_accepted_indices: list[int]
    accepted_indices: list[int]
    rejected_indices: list[int]
    reference_reprojection_errors_px: list[float]
    target_reprojection_errors_px: list[float]
    translation_residuals_m: list[float]
    rotation_residuals_deg: list[float]
    method: str = "pose_average"
    stereo_rms_px: float | None = None


@dataclass(frozen=True)
class BoardObservation:
    """Identified corners and board-frame coordinates for either calibration target."""

    corners_px: np.ndarray
    corner_ids: np.ndarray
    object_points_m: np.ndarray
    marker_count: int


CharucoDetection = BoardObservation  # Compatibility for existing callers.


def create_charuco_board(
    squares_x: int,
    squares_y: int,
    square_size_m: float,
    marker_size_m: float,
    dictionary_name: str,
) -> cv2.aruco.CharucoBoard:
    """Construct a validated OpenCV ChArUco board from physical dimensions."""
    if min(squares_x, squares_y) < 3:
        raise ValueError("ChArUco board must contain at least 3 by 3 squares")
    if not 0 < marker_size_m < square_size_m:
        raise ValueError(
            "marker_size_m must be positive and smaller than square_size_m"
        )
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None or not dictionary_name.startswith("DICT_"):
        raise ValueError(f"Unknown OpenCV ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.CharucoBoard(
        (int(squares_x), int(squares_y)),
        float(square_size_m),
        float(marker_size_m),
        dictionary,
    )


def detect_charuco_corners(
    color_bgr: np.ndarray,
    board: cv2.aruco.CharucoBoard,
) -> CharucoDetection | None:
    """Detect identified ChArUco corners; partial board visibility is supported."""
    gray = cv2.cvtColor(np.asarray(color_bgr), cv2.COLOR_BGR2GRAY)
    corners, corner_ids, _marker_corners, marker_ids = cv2.aruco.CharucoDetector(
        board
    ).detectBoard(gray)
    if corners is None or corner_ids is None or len(corners) < 4:
        return None
    ids = np.asarray(corner_ids, dtype=np.int32).reshape(-1)
    board_points = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    if np.any(ids < 0) or np.any(ids >= len(board_points)):
        raise ValueError("Detected ChArUco corner ID is outside the configured board")
    return CharucoDetection(
        corners_px=np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
        corner_ids=ids,
        object_points_m=board_points[ids],
        marker_count=0 if marker_ids is None else len(marker_ids),
    )


def orient_chessboard_corners(gray, corners, pattern_size):
    """Resolve 180-degree ordering using local square colour, never camera geometry.

    Only mixed-parity inner-corner counts distinguish the two ends by colour.
    Return None for ambiguous size, low contrast, or inconsistent square colours.
    """
    if pattern_size[0] % 2 == pattern_size[1] % 2:
        return None
    grid = corners.reshape(pattern_size[1], pattern_size[0], 2)
    centers = (grid[:-1, :-1] + grid[1:, :-1] + grid[:-1, 1:] + grid[1:, 1:]) / 4
    if (
        not np.all(np.isfinite(centers))
        or np.any(centers < 1)
        or np.any(centers > np.array(gray.shape[::-1]) - 2)
    ):
        return None
    cells = np.array(
        [
            float(np.mean(cv2.getRectSubPix(gray, (3, 3), tuple(map(float, p)))))
            for p in centers.reshape(-1, 2)
        ]
    ).reshape(centers.shape[:2])
    local = cells[:-1, :-1] + cells[1:, 1:] - cells[:-1, 1:] - cells[1:, :-1]
    rows, cols = np.indices(local.shape)
    votes = (local * (-1.0) ** (rows + cols)).ravel()
    if len(votes) < 4:
        return None
    median = float(np.median(votes))
    if abs(median) < 40 or np.mean(votes * np.sign(median) >= 20) < 0.8:
        return None
    return corners[::-1].copy() if median > 0 else corners


def find_chessboard_corners(
    color_bgr: np.ndarray,
    pattern_size: tuple[int, int],
    *,
    require_orientation: bool = False,
) -> tuple[bool, np.ndarray | None]:
    """Find every inner corner, preferring OpenCV's robust SB detector."""
    gray = cv2.cvtColor(np.asarray(color_bgr), cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
            | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            corners = corners.astype(np.float32)
            oriented = orient_chessboard_corners(gray, corners, pattern_size)
            if require_orientation and oriented is None:
                return False, None
            return True, corners if oriented is None else oriented
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return False, None
    # A refinement window must not reach neighbouring corners on small/tilted boards.
    grid = corners.reshape(pattern_size[1], pattern_size[0], 2)
    spacings = np.concatenate(
        (
            np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
            np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
        )
    )
    window = max(1, min(11, int(np.floor(float(np.min(spacings)) * 0.4))))
    refined = cv2.cornerSubPix(
        gray,
        corners,
        (window, window),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    refined = refined.astype(np.float32)
    oriented = orient_chessboard_corners(gray, refined, pattern_size)
    if require_orientation and oriented is None:
        return False, None
    return True, refined if oriented is None else oriented


def observe_chessboard(color, pattern, square_size_m, *, require_orientation=True):
    found, corners = find_chessboard_corners(
        color, pattern, require_orientation=require_orientation
    )
    if not found:
        return None
    return BoardObservation(
        corners,
        np.arange(len(corners)),
        chessboard_object_points(*pattern, square_size_m),
        0,
    )


def chessboard_object_points(
    pattern_cols: int,
    pattern_rows: int,
    square_size_m: float,
) -> np.ndarray:
    """Return chessboard inner-corner coordinates in its planar board frame."""
    if pattern_cols < 2 or pattern_rows < 2:
        raise ValueError(
            "Chessboard pattern must contain at least 2 by 2 inner corners"
        )
    if not np.isfinite(square_size_m) or square_size_m <= 0:
        raise ValueError("square_size_m must be finite and positive")
    points = np.zeros((pattern_cols * pattern_rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    points *= float(square_size_m)
    return points


def normalized_board_area_ratio(
    corners_px: np.ndarray, image_size_px: tuple[int, int]
) -> float:
    """Return convex-hull board coverage as a fraction of the image area."""
    width, height = image_size_px
    if width <= 0 or height <= 0:
        raise ValueError("image_size_px must contain positive width and height")
    corners = np.asarray(corners_px, dtype=np.float32).reshape(-1, 2)
    if len(corners) < 3:
        return 0.0
    hull = cv2.convexHull(corners)
    return float(cv2.contourArea(hull) / float(width * height))


def calibrate_chessboard_intrinsics(
    corner_sets_px: Sequence[np.ndarray],
    image_size_px: tuple[int, int],
    pattern_cols: int,
    pattern_rows: int,
    square_size_m: float,
    *,
    model: str = "brown5",
) -> ChessboardIntrinsicCalibration:
    """Fit one camera's intrinsic matrix from complete chessboard observations."""
    width, height = image_size_px
    if model not in ("brown5", "brown4"):
        raise ValueError("Unknown intrinsic model")
    if width <= 0 or height <= 0:
        raise ValueError("image_size_px must contain positive width and height")
    if len(corner_sets_px) < 4:
        raise ValueError("At least four chessboard observations are required")

    object_template = chessboard_object_points(
        pattern_cols, pattern_rows, square_size_m
    )
    expected_count = len(object_template)
    image_points: list[np.ndarray] = []
    for index, corner_set in enumerate(corner_sets_px, start=1):
        points = np.asarray(corner_set, dtype=np.float32).reshape(-1, 1, 2)
        if len(points) != expected_count:
            raise ValueError(
                f"Observation {index} contains {len(points)} corners; expected {expected_count}"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("Nonfinite image corners")
        image_points.append(points)

    object_points = [object_template.copy() for _ in image_points]
    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        (int(width), int(height)),
        None,
        None,
        flags=cv2.CALIB_FIX_K3 if model == "brown4" else 0,
    )
    errors: list[float] = []
    for object_points_view, image_points_view, rvec, tvec in zip(
        object_points,
        image_points,
        rvecs,
        tvecs,
    ):
        projected, _ = cv2.projectPoints(
            object_points_view,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        residual = projected.reshape(-1, 2) - image_points_view.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(np.square(residual), axis=1)))))

    return ChessboardIntrinsicCalibration(
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        distortion_coefficients=np.asarray(distortion, dtype=np.float64).reshape(-1),
        rms_reprojection_error_px=float(rms),
        per_view_reprojection_error_px=errors,
        rvecs=[np.asarray(value, dtype=np.float64) for value in rvecs],
        tvecs=[np.asarray(value, dtype=np.float64) for value in tvecs],
    )


def transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Return a homogeneous transform that maps object coordinates into camera coordinates."""
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def transform_rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return the absolute rotational distance between two homogeneous transforms."""
    relative_rotation = (
        np.asarray(first, dtype=np.float64)[:3, :3]
        @ np.asarray(second, dtype=np.float64)[:3, :3].T
    )
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def average_rigid_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Average rigid transforms using chordal rotation averaging and mean translation."""
    if not transforms:
        raise ValueError("At least one transform is required")
    matrices = [np.asarray(transform, dtype=np.float64) for transform in transforms]
    if any(matrix.shape != (4, 4) for matrix in matrices):
        raise ValueError("Every transform must have shape (4, 4)")
    rotation_sum = sum(matrix[:3, :3] for matrix in matrices)
    left, _, right_transpose = np.linalg.svd(rotation_sum)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.mean([matrix[:3, 3] for matrix in matrices], axis=0)
    return transform


def reprojection_error_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    """Return RMS corner reprojection error for one solvePnP pose."""
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        distortion_coefficients,
    )
    residual = projected.reshape(-1, 2) - np.asarray(
        image_points, dtype=np.float64
    ).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(np.square(residual), axis=1))))


def calibrate_relative_camera_pose(
    reference_corner_sets_px: Sequence[np.ndarray],
    target_corner_sets_px: Sequence[np.ndarray],
    object_points: np.ndarray,
    reference_camera_matrix: np.ndarray,
    reference_distortion_coefficients: np.ndarray,
    target_camera_matrix: np.ndarray,
    target_distortion_coefficients: np.ndarray,
    max_pnp_reprojection_error_px: float = 1.0,
) -> RelativeCameraPoseCalibration:
    """Estimate ``T_reference_from_target`` from paired complete chessboard views.

    Each paired observation must show the same static chessboard pose.  Candidate
    camera-to-camera transforms are formed via solvePnP, then filtered by both
    corner reprojection error and robust pose consistency before averaging.
    """
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    reference_objects = [object_points for _ in reference_corner_sets_px]
    target_objects = [object_points for _ in target_corner_sets_px]
    return calibrate_relative_camera_pose_from_observations(
        reference_objects,
        reference_corner_sets_px,
        target_objects,
        target_corner_sets_px,
        reference_camera_matrix,
        reference_distortion_coefficients,
        target_camera_matrix,
        target_distortion_coefficients,
        max_pnp_reprojection_error_px,
    )


def calibrate_relative_camera_pose_from_observations(
    reference_object_sets: Sequence[np.ndarray],
    reference_corner_sets_px: Sequence[np.ndarray],
    target_object_sets: Sequence[np.ndarray],
    target_corner_sets_px: Sequence[np.ndarray],
    reference_camera_matrix: np.ndarray,
    reference_distortion_coefficients: np.ndarray,
    target_camera_matrix: np.ndarray,
    target_distortion_coefficients: np.ndarray,
    max_pnp_reprojection_error_px: float = 1.0,
    *,
    method: str = "pose_average",
    image_size_px: tuple[int, int] | None = None,
) -> RelativeCameraPoseCalibration:
    """Estimate a camera transform from paired, independently visible target points."""
    lengths = {
        len(reference_object_sets),
        len(reference_corner_sets_px),
        len(target_object_sets),
        len(target_corner_sets_px),
    }
    if len(lengths) != 1:
        raise ValueError("Reference and target observations must have equal lengths")
    if len(reference_corner_sets_px) < 4:
        raise ValueError(
            "At least four paired calibration-target observations are required"
        )
    if (
        not np.isfinite(max_pnp_reprojection_error_px)
        or max_pnp_reprojection_error_px <= 0
    ):
        raise ValueError("max_pnp_reprojection_error_px must be positive")
    if method not in {"pose_average", "stereo"}:
        raise ValueError("Unknown pair calibration method")
    if method == "stereo" and (image_size_px is None or min(image_size_px) <= 0):
        raise ValueError("Stereo calibration requires the actual image size")

    candidate_indices: list[int] = []
    candidates: list[np.ndarray] = []
    reference_errors: list[float] = []
    target_errors: list[float] = []
    rejected_indices: list[int] = []
    for index, (
        reference_objects,
        reference_corners,
        target_objects,
        target_corners,
    ) in enumerate(
        zip(
            reference_object_sets,
            reference_corner_sets_px,
            target_object_sets,
            target_corner_sets_px,
        ),
        start=1,
    ):
        reference_objects = np.asarray(reference_objects, dtype=np.float64).reshape(
            -1, 3
        )
        target_objects = np.asarray(target_objects, dtype=np.float64).reshape(-1, 3)
        reference_corners = np.asarray(reference_corners, dtype=np.float64).reshape(
            -1, 1, 2
        )
        target_corners = np.asarray(target_corners, dtype=np.float64).reshape(-1, 1, 2)
        if len(reference_corners) != len(reference_objects) or len(
            target_corners
        ) != len(target_objects):
            raise ValueError(
                f"Paired observation {index} has mismatched object/image point counts"
            )
        if min(len(reference_objects), len(target_objects)) < 4:
            raise ValueError(
                f"Paired observation {index} contains fewer than four points"
            )
        if not all(
            np.all(np.isfinite(v))
            for v in (
                reference_objects,
                target_objects,
                reference_corners,
                target_corners,
            )
        ):
            raise ValueError(f"Paired observation {index} has nonfinite coordinates")
        if any(
            np.linalg.matrix_rank(v - v.mean(axis=0), tol=1e-8) < 2
            for v in (reference_objects, target_objects)
        ):
            raise ValueError(f"Paired observation {index} contains collinear points")
        reference_ok, reference_rvec, reference_tvec = cv2.solvePnP(
            reference_objects,
            reference_corners,
            reference_camera_matrix,
            reference_distortion_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        target_ok, target_rvec, target_tvec = cv2.solvePnP(
            target_objects,
            target_corners,
            target_camera_matrix,
            target_distortion_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not reference_ok or not target_ok:
            rejected_indices.append(index)
            continue
        reference_error = reprojection_error_px(
            reference_objects,
            reference_corners,
            reference_camera_matrix,
            reference_distortion_coefficients,
            reference_rvec,
            reference_tvec,
        )
        target_error = reprojection_error_px(
            target_objects,
            target_corners,
            target_camera_matrix,
            target_distortion_coefficients,
            target_rvec,
            target_tvec,
        )
        if (
            not np.all(np.isfinite([reference_error, target_error]))
            or max(reference_error, target_error) > max_pnp_reprojection_error_px
        ):
            rejected_indices.append(index)
            continue
        reference_from_board = transform_from_rvec_tvec(reference_rvec, reference_tvec)
        target_from_board = transform_from_rvec_tvec(target_rvec, target_tvec)
        if any(
            np.min((pose[:3, :3] @ points.T).T[:, 2] + pose[2, 3]) <= 0
            for pose, points in (
                (reference_from_board, reference_objects),
                (target_from_board, target_objects),
            )
        ):
            rejected_indices.append(index)
            continue
        candidate_indices.append(index)
        candidates.append(reference_from_board @ np.linalg.inv(target_from_board))
        reference_errors.append(reference_error)
        target_errors.append(target_error)

    if len(candidates) < 4:
        raise ValueError(
            "Fewer than four paired observations survived the PnP error filter"
        )

    # Start from the candidate closest to all others, then reject transforms that
    # disagree with that central pose by a robust median/MAD threshold.
    pairwise_costs = []
    for candidate in candidates:
        distances = [
            np.linalg.norm(candidate[:3, 3] - other[:3, 3]) / 0.01
            + transform_rotation_angle_deg(candidate, other) / 2.0
            for other in candidates
        ]
        pairwise_costs.append(float(np.sum(distances)))
    centre = candidates[int(np.argmin(pairwise_costs))]
    translation_to_centre = np.array(
        [np.linalg.norm(candidate[:3, 3] - centre[:3, 3]) for candidate in candidates],
        dtype=np.float64,
    )
    rotation_to_centre = np.array(
        [transform_rotation_angle_deg(candidate, centre) for candidate in candidates],
        dtype=np.float64,
    )
    translation_median = float(np.median(translation_to_centre))
    rotation_median = float(np.median(rotation_to_centre))
    translation_mad = float(
        np.median(np.abs(translation_to_centre - translation_median))
    )
    rotation_mad = float(np.median(np.abs(rotation_to_centre - rotation_median)))
    translation_limit = max(
        0.003, translation_median + 3.0 * max(translation_mad, 1e-6)
    )
    rotation_limit = max(0.5, rotation_median + 3.0 * max(rotation_mad, 1e-4))
    keep = (translation_to_centre <= translation_limit) & (
        rotation_to_centre <= rotation_limit
    )
    if int(np.count_nonzero(keep)) < 4:
        raise ValueError(
            "Fewer than four paired observations survived pose-consistency filtering"
        )
    accepted_indices = [
        index for index, accepted in zip(candidate_indices, keep) if accepted
    ]
    rejected_indices.extend(
        index for index, accepted in zip(candidate_indices, keep) if not accepted
    )
    accepted_transforms = [
        candidate for candidate, accepted in zip(candidates, keep) if accepted
    ]
    mean_transform = average_rigid_transforms(accepted_transforms)
    stereo_rms = None
    if method == "stereo":
        objects, reference_pixels, target_pixels = [], [], []
        for index in accepted_indices:
            ro = np.asarray(reference_object_sets[index - 1]).reshape(-1, 3)
            to = np.asarray(target_object_sets[index - 1]).reshape(-1, 3)
            # Object coordinates come from the same board definition, not estimated 3D points.
            target_lookup = {tuple(point): i for i, point in enumerate(to)}
            matches = [
                (i, target_lookup[tuple(point)])
                for i, point in enumerate(ro)
                if tuple(point) in target_lookup
            ]
            if len(matches) < 8:
                raise ValueError(
                    f"Sample {index}: stereo refinement needs at least eight shared corners"
                )
            ir, it = np.asarray(matches).T
            common = ro[ir].astype(np.float32)
            singular = np.linalg.svd(common - common.mean(axis=0), compute_uv=False)
            if singular[1] < 1e-6:
                raise ValueError(f"Sample {index}: shared corners are collinear")
            objects.append(common)
            reference_pixels.append(
                np.asarray(reference_corner_sets_px[index - 1], np.float32).reshape(
                    -1, 2
                )[ir]
            )
            target_pixels.append(
                np.asarray(target_corner_sets_px[index - 1], np.float32).reshape(-1, 2)[
                    it
                ]
            )
        initial = np.linalg.inv(mean_transform)
        result = cv2.stereoCalibrateExtended(
            objects,
            reference_pixels,
            target_pixels,
            np.asarray(reference_camera_matrix, float).copy(),
            np.asarray(reference_distortion_coefficients, float).copy(),
            np.asarray(target_camera_matrix, float).copy(),
            np.asarray(target_distortion_coefficients, float).copy(),
            image_size_px,
            initial[:3, :3].copy(),
            initial[:3, 3].copy(),
            flags=cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_USE_EXTRINSIC_GUESS,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-9),
        )
        stereo_rms = float(result[0])
        refined = np.eye(4)
        refined[:3, :3], refined[:3, 3] = result[5], result[6].reshape(3)
        if not np.isfinite(stereo_rms) or not np.all(np.isfinite(refined)):
            raise ValueError("Stereo refinement did not produce finite geometry")
        mean_transform = np.linalg.inv(refined)
    translation_residuals = [
        float(np.linalg.norm(candidate[:3, 3] - mean_transform[:3, 3]))
        for candidate in accepted_transforms
    ]
    rotation_residuals = [
        transform_rotation_angle_deg(candidate, mean_transform)
        for candidate in accepted_transforms
    ]
    return RelativeCameraPoseCalibration(
        transform_reference_from_target=mean_transform,
        pnp_accepted_indices=candidate_indices,
        accepted_indices=accepted_indices,
        rejected_indices=sorted(set(rejected_indices)),
        reference_reprojection_errors_px=reference_errors,
        target_reprojection_errors_px=target_errors,
        translation_residuals_m=translation_residuals,
        rotation_residuals_deg=rotation_residuals,
        method=method,
        stereo_rms_px=stereo_rms,
    )
