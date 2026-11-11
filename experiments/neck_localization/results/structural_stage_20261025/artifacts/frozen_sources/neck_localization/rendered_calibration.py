"""Image-detected checkerboard calibration; metres and OpenCV integer pixels.

Renderer inputs and evaluation truth stay outside the intrinsic solver. A known
surveyed board pose is explicitly required to establish the world extrinsic.
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


CV_FROM_SAPIEN = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])


@dataclass(frozen=True)
class BoardSettings:
    columns: int = 9
    rows: int = 6
    square_m: float = .025
    border_m: float = .020
    marker_half_size_m: float = .0045

    def object_points(self):
        x, y = np.meshgrid(np.arange(self.columns), np.arange(self.rows))
        return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float32)*self.square_m

    def marker_points(self):
        # Both lie outside the pattern, on the upper white margin.
        offset = self.square_m+.0085
        return np.array([[-offset, -offset], [(self.columns-1)*self.square_m+offset, -offset]])


def transform_from_rt(rvec, tvec):
    out = np.eye(4)
    out[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float))[0]
    out[:3, 3] = np.asarray(tvec).reshape(3)
    return out


def world_camera_from_known_board(T_world_from_board, rvec, tvec):
    return np.asarray(T_world_from_board) @ np.linalg.inv(transform_from_rt(rvec, tvec))


def pose_error(T_estimated, T_true):
    a, b = np.asarray(T_estimated), np.asarray(T_true)
    return dict(translation_mm=float(np.linalg.norm(a[:3, 3]-b[:3, 3])*1000),
                rotation_deg=float(np.rad2deg(Rotation.from_matrix(a[:3, :3].T@b[:3, :3]).magnitude())))


def _marker_centres(rgb):
    image = np.asarray(rgb, float)
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    masks = [(r > 110) & (r > 1.7*g) & (r > 1.7*b),
             (g > 110) & (g > 1.7*r) & (g > 1.7*b)]
    centres = []
    for mask in masks:
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        if n <= 1:
            raise ValueError('Printed red/green direction marker not detected')
        index = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[index, cv2.CC_STAT_AREA] < 3:
            raise ValueError('Printed direction marker too small')
        centres.append(centroids[index])
    return np.asarray(centres)


def order_detected_corners(corners, marker_centres, settings=BoardSettings()):
    """Resolve flips using PRINTED marker locations and observed marker pixels.

    No camera matrix, true pose or ideal corner projection enters this function.
    """
    grid = np.asarray(corners, np.float32).reshape(settings.rows, settings.columns, 2)
    object_xy = settings.object_points()[:, :2]
    candidates = [grid, grid[:, ::-1], grid[::-1], grid[::-1, ::-1]]
    errors = []
    for candidate in candidates:
        H, _ = cv2.findHomography(object_xy, candidate.reshape(-1, 2), method=0)
        projected_markers = cv2.perspectiveTransform(settings.marker_points()[None].astype(np.float32), H)[0]
        errors.append(float(np.sqrt(np.mean(np.sum((projected_markers-marker_centres)**2, axis=1)))))
    selected = int(np.argmin(errors))
    if errors[selected] > 8.:
        raise ValueError(f'Direction marker/corner correspondence inconsistent: {errors}')
    return candidates[selected].reshape(-1, 1, 2).copy(), dict(
        selected_flip=selected, marker_reprojection_rms_px=errors[selected], all_flip_errors_px=errors)


def detect_checkerboard(rgb, settings=BoardSettings()):
    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    # These raster images already have high contrast. Histogram equalization
    # (NORMALIZE_IMAGE) made SB fail on this near-binary render; do not apply it.
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, (settings.columns, settings.rows), flags=flags)
    if not found:
        raise ValueError('findChessboardCornersSB did not detect the complete checkerboard')
    # SB returns subpixel corners directly. The OpenCV documentation explicitly
    # distinguishes these from cornerSubPix; a second gradient-based refinement
    # degraded this raster calibration and is retained only in the v2 record.
    markers = _marker_centres(rgb)
    corners, ordering = order_detected_corners(corners, markers, settings)
    return corners, dict(detector='OpenCV findChessboardCornersSB direct subpixel output',
                         observed_red_green_marker_centres_px=markers.tolist(), **ordering)


def calibrate_intrinsics(detected_corners, image_size, settings=BoardSettings()):
    """No true K, board poses, extrinsic initialization or ideal pixels accepted."""
    if len(detected_corners) < 10:
        raise ValueError('At least 10 different board views required')
    points = [settings.object_points() for _ in detected_corners]
    images = [np.asarray(c, np.float32).reshape(-1, 1, 2) for c in detected_corners]
    # Pinhole raster images have zero lens distortion. Fixing it avoids fitting
    # fictitious high-order terms to raster sampling. No intrinsic guess flag.
    flags = (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    rms, K, distortion, rvecs, tvecs = cv2.calibrateCamera(points, images, tuple(image_size), None, None,
        flags=flags, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 150, 1e-12))
    return dict(K=K, distortion=distortion, rms_px=float(rms), rvecs=rvecs, tvecs=tvecs,
                flags=int(flags), distortion_model='fixed_zero_matching_ideal_pinhole_renderer')


def estimate_camera_pose(detected_corners, K, distortion, T_world_from_known_board, settings=BoardSettings()):
    """One observed board + independent ideal fixture pose anchors world frame."""
    ok, rvec, tvec = cv2.solvePnP(settings.object_points(), np.asarray(detected_corners, np.float32),
                                np.asarray(K), np.asarray(distortion), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError('solvePnP failed')
    predicted = cv2.projectPoints(settings.object_points(), rvec, tvec, K, distortion)[0]
    error = predicted.reshape(-1, 2)-np.asarray(detected_corners).reshape(-1, 2)
    return world_camera_from_known_board(T_world_from_known_board, rvec, tvec), dict(
        rvec_camera_from_board=rvec.ravel().tolist(), tvec_camera_from_board_m=tvec.ravel().tolist(),
        board_reprojection_rms_px=float(np.sqrt(np.mean(np.sum(error*error, axis=1)))))


def board_pose_plan(seed=2026102551):
    """Predeclared 15 train, five new validation poses, one fixed world anchor."""
    rng = np.random.default_rng(seed)
    result = []
    for split, count in [('train', 15), ('validation', 5)]:
        for index in range(count):
            # Centre stays inside view, while depth/tilt/roll and image position vary.
            result.append(dict(frame_id=f'{split}_{index:02d}', split=split,
                distance_m=float(rng.uniform(.52, 1.02)),
                center_xy_m=rng.uniform([-.105, -.055], [.105, .055]).tolist(),
                euler_xyz_deg=rng.uniform([-28, -30, -23], [28, 30, 23]).tolist()))
    result.append(dict(frame_id='fixed_world_fixture', split='world_anchor', distance_m=.76,
                       center_xy_m=[.025, -.015], euler_xyz_deg=[17., -13., 8.]))
    return result


def T_camera_from_board_plan(item, settings=BoardSettings()):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', item['euler_xyz_deg'], degrees=True).as_matrix()
    centre = settings.object_points().mean(axis=0)
    T[:3, 3] = np.r_[item['center_xy_m'], item['distance_m']]-T[:3, :3]@centre
    return T


class RasterCheckerboard:
    """Actual SAPIEN raster chessboard capture, no projected-corner substitute."""
    def __init__(self, camera, renderer_K, settings=BoardSettings()):
        import sapien
        from pathlib import Path
        self.camera, self.settings = camera, settings
        raster_dir = str(Path(sapien.__file__).parent/'vulkan_shader/default')
        self.scene = sapien.Scene()
        self.scene.set_ambient_light([1., 1., 1.])
        builder = self.scene.create_actor_builder()
        s = settings.square_m
        for row in range(settings.rows+1):
            for column in range(settings.columns+1):
                value = .015 if (row+column) % 2 == 0 else .95
                builder.add_box_visual(pose=sapien.Pose([(column-.5)*s, (row-.5)*s, .0004]),
                    half_size=[s/2, s/2, .0004], material=[value, value, value, 1])
        centre = settings.object_points().mean(axis=0)
        builder.add_box_visual(pose=sapien.Pose([centre[0], centre[1], .002]),
            half_size=[(settings.columns+1)*s/2+settings.border_m,
                       (settings.rows+1)*s/2+settings.border_m, .001], material=[.95, .95, .95, 1])
        for point, color in zip(settings.marker_points(), [[.95, .01, .01, 1], [.01, .95, .01, 1]]):
            builder.add_box_visual(pose=sapien.Pose([*point, .0003]),
                half_size=[settings.marker_half_size_m, settings.marker_half_size_m, .0003], material=color)
        self.board = builder.build_kinematic(name='real_rendered_checkerboard')
        mount = self.scene.create_actor_builder().build_kinematic(name='calibration_camera')
        pose = camera.T_world_from_camera.copy()
        pose[:3, :3] = pose[:3, :3]@CV_FROM_SAPIEN
        mount.set_pose(sapien.Pose(pose))
        self.render_camera = sapien.render.RenderCameraComponent(camera.width, camera.height, raster_dir)
        self.render_camera.set_perspective_parameters(.05, 10., renderer_K[0, 0], renderer_K[1, 1],
                                                      renderer_K[0, 2], renderer_K[1, 2], 0)
        mount.add_component(self.render_camera)

    def capture(self, T_world_from_board):
        import sapien
        self.board.set_pose(sapien.Pose(T_world_from_board))
        self.scene.update_render()
        self.render_camera.take_picture()
        expected = self.camera.T_world_from_camera.copy()
        expected[:3, :3] = expected[:3, :3]@np.diag([1., -1., -1.])
        np.testing.assert_allclose(self.render_camera.get_model_matrix(), expected, atol=2e-6)
        return np.clip(self.render_camera.get_picture('Color')[..., :3]*255, 0, 255).astype(np.uint8)
