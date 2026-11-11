"""MuJoCo inspection window: camera rig, thin result annotations and saved inputs.

All camera props live in MjvScene only. RGB/depth panels read the saved inputs;
neither the measurement model nor estimator is changed. Result curves are
explicit through-surface projections, not newly detected image boundaries.
"""
from dataclasses import dataclass
import gzip
from hashlib import sha256
import json
from pathlib import Path
import time

import cv2
import glfw
import mujoco
import numpy as np
from OpenGL import GL
from PIL import Image, ImageDraw, ImageFont

from .geometry import Camera, circle_points
from .point_cloud_display import DisplayCloud, observation_cloud, display_samples
from .viewer_navigation import CONDITION_LABELS, NavButton, Rect, framebuffer_point, hit_button, viewer_layout


CAMERA_COLORS = [(82, 176, 248), (116, 218, 150), (238, 184, 92)]
ESTIMATE_COLOR = (255, 104, 91)
TRUTH_COLOR = (94, 224, 246)
BAND_COLOR = (255, 198, 80)
RANGE_COLOR = (177, 239, 146)
REJECTED_COLOR = (255, 181, 86)
METHOD_LABELS = dict(baseline='原定位器', stereo_robust='原几何定位器', portfolio_plain='多候选·普通', portfolio_robust='多候选·鲁棒',
    portfolio='多候选·区域反馈', portfolio_single='单候选对照', portfolio_anisotropic='方差传播加权',
    portfolio_fixed='口沿轴对照', visible_profile='可见区母线拟合', standard_sor='标准旋转曲面')
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def present_image(image, context):
    """Present a complete 2D composite regardless of the previous 3D GL state.

    mjr_drawPixels does not turn off the depth test left by mjr_render. Without
    this scope, text/result lines disappear behind the previous render even
    though a screenshot of the CPU composite looks correct.
    """
    height, width = image.shape[:2]
    GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    try:
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glColorMask(True, True, True, True)
        mujoco.mjr_drawPixels(np.ascontiguousarray(image[::-1]).ravel(), None,
                             mujoco.MjrRect(0, 0, width, height), context)
    finally:
        GL.glPopAttrib()


def read_records(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def project_scene(scene, points, width, height):
    """Project the MONO average camera to top-left image pixel centres.

    Left/right eye poses have an IPD offset, even in a mono scene. Also,
    frustum_width is not used by MuJoCo's renderer; derive it from aspect.
    """
    camera = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
    forward = np.array(camera.forward, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, camera.up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = np.asarray(points)-camera.pos
    z, x, y = delta @ forward, delta @ right, delta @ up
    sy = float(camera.frustum_top-camera.frustum_bottom)
    sx = sy*width/height
    left = camera.frustum_center-sx/2
    with np.errstate(divide="ignore", invalid="ignore"):
        u = width*(camera.frustum_near*x/z-left)/sx-.5
        v = height*(1-(camera.frustum_near*y/z-camera.frustum_bottom)/sy)-.5
    pixels = np.c_[u, v]
    pixels[z <= camera.frustum_near] = np.nan
    return pixels


def draw_cloud(scene, points, colors, width, height):
    """Use MuJoCo's installed projection and depth convention, with mono pose.

    MuJoCo 3.8 uses reversed Z with a zero-to-one clip range. Replacing its
    projection by glFrustum/GL_LEQUAL silently hides points behind clear depth.
    """
    if not len(points):
        return
    camera = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
    forward = np.asarray(camera.forward, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, camera.up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([right, up, -forward])
    matrix[:3, 3] = -matrix[:3, :3] @ camera.pos
    old_mode = GL.glGetIntegerv(GL.GL_MATRIX_MODE)
    GL.glPushAttrib(GL.GL_ALL_ATTRIB_BITS)
    GL.glPushClientAttrib(GL.GL_CLIENT_VERTEX_ARRAY_BIT)
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glPushMatrix()
    GL.glLoadMatrixd(matrix.T)
    try:
        GL.glViewport(0, 0, width, height)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(True)
        for flag in (GL.GL_LIGHTING, GL.GL_TEXTURE_2D, GL.GL_BLEND, GL.GL_FOG, GL.GL_POINT_SMOOTH):
            GL.glDisable(flag)
        GL.glPointSize(2.)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, points)
        GL.glColorPointer(3, GL.GL_UNSIGNED_BYTE, 0, colors)
        GL.glDrawArrays(GL.GL_POINTS, 0, len(points))
    finally:
        GL.glPopMatrix()
        GL.glMatrixMode(old_mode)
        GL.glPopClientAttrib()
        GL.glPopAttrib()


def draw_result(image, project, result, color=ESTIMATE_COLOR, dashed=False, arrow=True):
    """One thin circle + one thin normal arrow. Never invent a rejected result."""
    if not result:
        return
    center = np.asarray(result["center_m"], float)
    normal = np.asarray(result["normal"], float)
    ring = circle_points(center, normal, result["radius_m"], 240)
    pixels = project(np.r_[ring, ring[:1]])
    for i, (a, b) in enumerate(zip(pixels[:-1], pixels[1:])):
        if dashed and (i//5) % 2:
            continue
        if np.all(np.isfinite([a, b])) and np.max(np.abs([a, b])) < 1e7:
            cv2.line(image, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), color, 1, cv2.LINE_AA)
    tip = project(np.array([center, center+.075*normal]))
    if arrow and np.all(np.isfinite(tip)) and np.max(np.abs(tip)) < 1e7:
        cv2.arrowedLine(image, tuple(np.rint(tip[0]).astype(int)), tuple(np.rint(tip[1]).astype(int)),
                       color, 1, cv2.LINE_AA, tipLength=.12)


def draw_band_boundaries(image, project, band):
    """Axial-width edges only: no invented thickness or material contact."""
    center, normal = np.array(band['center_m']), np.array(band['normal'])
    for sign in (-1, 1):
        boundary = dict(center_m=center+sign*band['width_m']/2*normal,
                        normal=normal, radius_m=band['radius_m'])
        draw_result(image, project, boundary, BAND_COLOR, arrow=False)


def draw_truth_range(image, project, truth):
    """Evaluation-only near-min range, approximate radius for line display."""
    center, normal = np.array(truth['center_m']), np.array(truth['normal'])
    for bound in truth['near_min_interval_s_m']:
        boundary = dict(center_m=center+(bound-truth['s_m'])*normal,
                        normal=normal, radius_m=truth['radius_m'])
        draw_result(image, project, boundary, RANGE_COLOR, dashed=True, arrow=False)


def _add_geom(scene, kind, size, pos, mat, rgba):
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("Inspection geometry capacity exhausted")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, kind, np.asarray(size, float), np.asarray(pos, float),
                       np.asarray(mat, float).reshape(9), np.asarray(rgba, np.float32))
    scene.ngeom += 1
    return geom


def _line(scene, a, b, color, width=1):
    geom = _add_geom(scene, mujoco.mjtGeom.mjGEOM_LINE, [0, 0, 0], [0, 0, 0], np.eye(3), color)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a), np.asarray(b))


def add_camera_markers(scene, cameras, used):
    for index, camera in enumerate(cameras):
        rgb = np.array(CAMERA_COLORS[index % 3], float)/255
        if camera.name not in used:
            rgb *= .55
        rgba = np.r_[rgb, 1.]
        pose = camera.T_world_from_camera
        position, rotation = pose[:3, 3], pose[:3, :3]
        # Schematic housing behind the optical centre. It is not a D435 CAD.
        _add_geom(scene, mujoco.mjtGeom.mjGEOM_BOX, [.038, .019, .018],
                  position-.022*rotation[:, 2], rotation, rgba)
        _add_geom(scene, mujoco.mjtGeom.mjGEOM_CYLINDER, [.012, .004, 0],
                  position, rotation, [.15, .18, .22, 1])
        # Truncated FOV marker, derived from K; 0.18 m is only a drawing length.
        corners = camera.unproject(np.array([[0, 0], [camera.width-1, 0],
                                            [camera.width-1, camera.height-1], [0, camera.height-1]]),
                                   np.full(4, .18))
        for i in range(4):
            _line(scene, position, corners[i], rgba)
            _line(scene, corners[i], corners[(i+1) % 4], rgba)
        _line(scene, position, position+.20*rotation[:, 2], rgba, 1.5)


@dataclass
class CameraInput:
    camera: Camera
    rgb: np.ndarray
    depth: np.ndarray
    rgb_path: Path


class InspectionViewer:
    def __init__(self, scenes, records, case, camera_set="all", evidence="rgb", method="rim_guided_profile",
                 width=1480, height=960, visible=True, overlay=False, ground_truth=False, large_neck=False,
                 point_cloud=False, band=False, datasets=None):
        self.root, self.records_path = Path(scenes), Path(records)
        self.rows = read_records(records)
        settings_path = self.records_path.parent/"settings.json"
        self.record_settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
        self.evidence, self.method = evidence, method
        self.methods = sorted({r["method"] for r in self.rows if r["evidence"] == evidence})
        self.cases = sorted({r["case"] for r in self.rows if r["evidence"] == evidence and r["method"] == method
                             and (self.root/r["case"]/"scene.xml").exists()})
        if case not in self.cases:
            raise ValueError("Unknown case: "+case)
        self.case_index, self.camera_set = self.cases.index(case), camera_set
        self.overlay, self.show_truth, self.depth_mode = overlay, ground_truth, False
        self.large_neck = large_neck
        self.show_band = band
        self.point_cloud, self.cloud_crop, self.cloud_source_colors = point_cloud, False, False
        self.cloud_local = False
        self.cloud_weights = False
        self.show_estimate = self.show_neck = True
        self.show_rejected_candidate = False
        self.show_fit_pixels = False
        self.datasets = datasets or []
        self.nav_scroll = 0
        self._nav_buttons = []
        self._nav_content_height = 0
        self._cloud_cache_key = None
        self.dragging = None
        self.cursor = None
        self.context = None
        self.window = None
        self._fonts = {}
        if not glfw.init():
            raise RuntimeError("GLFW could not open the desktop display")
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
        glfw.window_hint(glfw.SAMPLES, 0)
        self.window = glfw.create_window(width, height, "WineJar | MuJoCo scene + actual camera inputs", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create a GLFW window")
        glfw.set_window_size_limits(self.window, 1000, 680, glfw.DONT_CARE, glfw.DONT_CARE)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        try:
            self._load_case()
        except Exception:
            self.close()
            raise
        glfw.set_mouse_button_callback(self.window, self._mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._move)
        glfw.set_scroll_callback(self.window, self._scroll)
        glfw.set_key_callback(self.window, self._key)

    @property
    def row(self):
        return next(r for r in self.rows if r["case"] == self.cases[self.case_index]
                    and r["camera_set"] == self.camera_set and r["evidence"] == self.evidence
                    and r["method"] == self.method)

    @property
    def used(self):
        return {f.camera.name for f in self.frames} if self.camera_set == "all" else set(self.camera_set.split(","))

    def _load_case(self):
        glfw.make_context_current(self.window)
        if self.context is not None:
            self.context.free()
            self.context = None
        self.folder = self.root/self.cases[self.case_index]
        self.model = mujoco.MjModel.from_xml_path(str(self.folder/"scene.xml"))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        metadata = json.loads((self.folder/"observations/camera_true_parameters.json").read_text())
        self.frames = []
        for item in metadata["cameras"]:
            camera = Camera(item["name"], np.array(item["K"]), np.array(item["T_world_from_camera"]), item["width"], item["height"])
            rgb_path = self.folder/"observations"/(camera.name+"_rgb.png")
            rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
            with np.load(self.folder/"observations"/(camera.name+"_depth_z_m.npz")) as source:
                depth = source["depth_z_m"]
            rgb.setflags(write=False)
            depth.setflags(write=False)
            self.frames.append(CameraInput(camera, rgb, depth, rgb_path))
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera.name)
            np.testing.assert_allclose(self.data.cam_xpos[camera_id], camera.T_world_from_camera[:3, 3], atol=1e-8)
            rotation = self.data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
            np.testing.assert_allclose(rotation, camera.T_world_from_camera[:3, :3], atol=1e-8)
        self.clouds = []
        self.foreground_mask_semantics = 'legacy_rgb_threshold'
        for f in self.frames:
            mask_path = self.folder/'foreground_masks'/(f.camera.name+'_foreground.png')
            mask = None
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
                if mask.shape != f.depth.shape:
                    raise ValueError('Saved foreground mask does not match observation dimensions')
                self.foreground_mask_semantics = 'saved_image_estimated_foreground_not_GT'
            self.clouds.append(observation_cloud(f.camera, f.rgb, f.depth, foreground_mask=mask))
        self._cloud_cache_key = None
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        self.scene = mujoco.MjvScene(self.model, maxgeom=512)
        self.neck_scene = mujoco.MjvScene(self.model, maxgeom=512)
        self.option = mujoco.MjvOption()
        self.camera = mujoco.MjvCamera()
        self.neck_camera = mujoco.MjvCamera()
        self._reset_camera()

    def _reset_camera(self):
        for cam in (self.camera, self.neck_camera):
            mujoco.mjv_defaultCamera(cam)
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0, 0, .37]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 2.35, 65, -27
        self.neck_camera.distance, self.neck_camera.azimuth, self.neck_camera.elevation = .30, 130, -15
        self._focus_neck()

    def _focus_neck(self):
        result = self.row.get("estimate")
        d = self.row.get('diagnostics', {})
        points = d.get('fit_point_positions_m')
        self.neck_camera.lookat[:] = (result['center_m'] if result else np.median(points, axis=0)
            if points else d.get('initial_mouth_center_m', [0, 0, .59]))

    def _layout(self, width, height):
        return viewer_layout(width, height, self.large_neck)

    def _pointer_view(self, x, y):
        """Map logical GLFW coordinates to the actual framebuffer layout."""
        size = glfw.get_framebuffer_size(self.window)
        point = framebuffer_point(x, y, glfw.get_window_size(self.window), size)
        if point is None:
            return None
        x, y = point
        layout = self._layout(*size)
        if self.show_neck and layout.neck.contains(x, y):
            return self.neck_camera, self.neck_scene, layout.neck.height
        if layout.main.contains(x, y):
            return self.camera, self.scene, layout.main.height
        return None

    def _truth(self):
        return json.loads((self.folder/"evaluation_only/target.json").read_text()) if self.show_truth else None

    def _draw_annotations(self, image, scene=None, camera=None):
        project = camera.project if camera is not None else lambda points: project_scene(scene, points, image.shape[1], image.shape[0])
        if self.show_estimate and self.row["accepted"]:
            draw_result(image, project, self.row.get("estimate"))
            if self.show_band and self.row.get('band'):
                draw_band_boundaries(image, project, self.row['band'])
        elif self.show_rejected_candidate and self.row.get('estimate'):
            draw_result(image, project, self.row['estimate'], REJECTED_COLOR, dashed=True)
        if self.show_truth:
            truth = self._truth()
            draw_result(image, project, truth, TRUTH_COLOR, dashed=True)
            if self.show_band and self.row.get('band'):
                draw_truth_range(image, project, truth)

    def _local_fit_clouds(self):
        """Recreate the saved method's observation selection, never read GT."""
        d, settings = self.row.get("diagnostics", {}), self.record_settings
        if 'fit_point_positions_m' in d:
            # New pipeline stores its observation-derived training subset;
            # validation points are not presented as fitted measurements.
            points=np.asarray(d['fit_point_positions_m'],float)
            ids=np.asarray(d['fit_camera_ids'])
            weights=np.asarray(d['fit_point_weights'])
            names=d['fit_camera_names']
            result=[]
            for frame in self.frames:
                name=frame.camera.name
                mask=ids==names.index(name) if name in names else np.zeros(len(ids),bool)
                local=points[mask]
                if len(local):
                    uv=np.rint(frame.camera.project(local)).astype(int)
                    if np.any(uv<0) or np.any(uv[:,0]>=frame.camera.width) or np.any(uv[:,1]>=frame.camera.height):
                        raise ValueError('Stored observation subset projects outside input image')
                    colors=frame.rgb[uv[:,1],uv[:,0]].copy()
                    if self.cloud_weights:
                        fraction=np.clip(weights[mask]/max(weights.max(),1e-12),0,1)
                        colors=(np.array([55,65,80])+(np.array([255,213,70])-np.array([55,65,80]))*fraction[:,None]).astype(np.uint8)
                else:
                    colors=np.empty((0,3),np.uint8)
                result.append(DisplayCloud(name,local,colors,np.ones(len(local),bool)))
            return result
        if not all(k in d for k in ("initial_mouth_center_m", "initial_mouth_normal",
                                   "local_selection_interval_from_initial_axis_m")):
            return None
        if not all(k in settings for k in ("fit_depth_mm", "exclude_top_mm", "facing_cosine_min", "max_fit_points_per_camera")):
            return None
        origin, normal = np.array(d["initial_mouth_center_m"]), np.array(d["initial_mouth_normal"])
        low, high = d["local_selection_interval_from_initial_axis_m"]
        result = []
        for cloud, frame in zip(self.clouds, self.frames):
            delta = cloud.points-origin
            s = delta @ normal
            radial = delta-s[:, None]*normal
            view = frame.camera.T_world_from_camera[:3, 3]-cloud.points
            view -= (view @ normal)[:, None]*normal
            cosine = np.sum(radial*view, axis=1)/np.maximum(np.linalg.norm(radial, axis=1)*np.linalg.norm(view, axis=1), 1e-12)
            keep = (cloud.station_mask & (s >= -settings["fit_depth_mm"]/1000)
                    & (s <= -settings["exclude_top_mm"]/1000) & (s >= low) & (s <= high)
                    & (cosine > settings["facing_cosine_min"]))
            indices = np.flatnonzero(keep)
            if len(indices):
                indices = indices[np.linspace(0, len(indices)-1,
                                  min(len(indices), settings["max_fit_points_per_camera"])).astype(int)]
            colors = cloud.colors[indices]
            if self.cloud_weights and 'fit_point_weights' in d and cloud.camera_name in self.used:
                names = d.get('fit_camera_names', [f.camera.name for f in self.frames if f.camera.name in self.used])
                weights = np.asarray(d['fit_point_weights'])
                camera_ids = np.asarray(d['fit_camera_ids'])
                local_weights = weights[camera_ids == names.index(cloud.camera_name)]
                if len(local_weights) != len(indices):
                    raise ValueError('Saved weights do not match reconstructed observation points')
                fraction = np.clip(local_weights/max(weights.max(), 1e-12), 0, 1)
                colors = (np.array([55,65,80])+(np.array([255,213,70])-np.array([55,65,80]))*fraction[:,None]).astype(np.uint8)
            result.append(DisplayCloud(cloud.camera_name, cloud.points[indices], colors, np.ones(len(indices), bool)))
        return result

    def _render_view(self, camera, scene, width, height, camera_props=False):
        mujoco.mjv_updateScene(self.model, self.data, self.option, None, camera,
                              mujoco.mjtCatBit.mjCAT_ALL, scene)
        scene.stereo = mujoco.mjtStereo.mjSTEREO_NONE
        if self.point_cloud:
            scene.ngeom = 0  # Hide surfaces; never show invisible mesh samples as measurements.
        if camera_props:
            add_camera_markers(scene, [f.camera for f in self.frames], self.used)
        rect = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(rect, scene, self.context)
        if self.point_cloud:
            key = (self.method, self.camera_set, self.cloud_crop, self.cloud_source_colors, self.cloud_local, self.cloud_weights)
            if key != self._cloud_cache_key:
                source_colors = {f.camera.name: CAMERA_COLORS[i % 3] for i, f in enumerate(self.frames)}
                local = self._local_fit_clouds() if self.cloud_local else None
                self._local_cloud_available = local is not None
                clouds = (local or []) if self.cloud_local else self.clouds
                self._cloud_data = display_samples(clouds,
                                                   self.used, self.cloud_crop or self.cloud_local,
                                                   source_colors if self.cloud_source_colors and not self.cloud_weights else None)
                self._cloud_cache_key = key
            draw_cloud(scene, *self._cloud_data[:2], width, height)
        rgb = np.empty((height, width, 3), np.uint8)
        mujoco.mjr_readPixels(rgb, None, rect, self.context)
        image = np.ascontiguousarray(rgb[::-1])
        self._draw_annotations(image, scene=scene)
        return image

    def _text(self, image, items):
        canvas = Image.fromarray(image)
        draw = ImageDraw.Draw(canvas)
        for x, y, text, size, color in items:
            if size not in self._fonts:
                self._fonts[size] = ImageFont.truetype(str(FONT), size) if FONT.exists() else ImageFont.load_default()
            draw.text((x, y), text, font=self._fonts[size], fill=color)
        return np.asarray(canvas)

    def _font(self, size):
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(str(FONT), size) if FONT.exists() else ImageFont.load_default()
        return self._fonts[size]

    def _draw_sidebar(self, image, layout):
        width, height = layout.sidebar.width, layout.sidebar.height
        content_bottom = height-164
        sections, buttons = [], []
        y = 8

        def section(label):
            nonlocal y
            sections.append((12, y, label))
            y += 24

        def row(items):
            nonlocal y
            gap = 5
            w = (width-24-gap*(len(items)-1))//len(items)
            for index, (action, label, active, enabled) in enumerate(items):
                buttons.append(NavButton(action, label, Rect(12+index*(w+gap), y, w, 30), active, enabled))
            y += 36

        if self.datasets:
            section('观测条件 · 同一坛型对照')
            for start in range(0, len(self.datasets), 2):
                row([(f"dataset:{d['condition']}", d['label'], Path(d['records']) == self.records_path, True)
                     for d in self.datasets[start:start+2]])
        # Keep the comparison's main decision visible at the default size.
        # Less frequently used drawing controls remain accessible by scrolling.
        if len(self.methods) > 1:
            section('估计算法')
            for start in range(0, len(self.methods), 2):
                row([(f'method:{m}', METHOD_LABELS.get(m,m)[:12], self.method == m, True)
                     for m in self.methods[start:start+2]])
        section(f'酒坛  {self.case_index+1} / {len(self.cases)}')
        row([('previous', '上一只', False, True), ('next', '下一只', False, True)])
        section('三维视图')
        row([('view:model', '实体', not self.point_cloud, True),
             ('view:cloud', '全坛点云', self.point_cloud and not self.cloud_local, True),
             ('view:fit', '拟合点', self.point_cloud and self.cloud_local, True)])
        row([('color:rgb', 'RGB色', not self.cloud_source_colors and not self.cloud_weights, True),
             ('color:source', '来源色', self.cloud_source_colors and not self.cloud_weights, True),
             ('color:weights', '权重', self.cloud_weights, 'fit_point_weights' in self.row.get('diagnostics', {}))])
        row([('toggle_crop', '工位裁剪', self.cloud_crop and not self.cloud_local, not self.cloud_local)])
        section('相机画面')
        row([('input:rgb', 'RGB', not self.depth_mode, True), ('input:depth', '深度', self.depth_mode, True)])
        has_fit = bool(self.row.get('diagnostics', {}).get('fit_point_positions_m'))
        row([('toggle_overlay', '画面叠线', self.overlay, True),
             ('toggle_fit_pixels', '拟合像素', self.show_fit_pixels and has_fit, has_fit)])
        section('几何标注')
        has_estimate = self.row['accepted'] and bool(self.row.get('estimate'))
        has_band = bool(self.row.get('band'))
        row([('toggle_estimate', '估计圆与法向', self.show_estimate and has_estimate, has_estimate)])
        row([('toggle_rejected', '被拒绝的候选（橙虚线）', self.show_rejected_candidate,
              not self.row['accepted'] and bool(self.row.get('estimate')))])
        row([('toggle_truth', '评估真值', self.show_truth, True),
             ('toggle_band', '带宽边界', self.show_band and has_band, has_band)])
        section('观察方式')
        row([('toggle_neck', '颈部近景', self.show_neck, True), ('toggle_large', '放大近景', self.large_neck, self.show_neck)])
        row([('reset', '复位视角', False, True)])
        choices = [('all', '三机'), ('cam_a', 'A'), ('cam_a,cam_b', 'A+B')]
        section('参与估计的视图')
        available = {r['camera_set'] for r in self.rows if r['case'] == self.cases[self.case_index]
                     and r['method'] == self.method and r['evidence'] == self.evidence}
        row([(f'camera:{value}', label, self.camera_set == value, value in available) for value, label in choices])
        self._nav_content_height = y+8
        viewport_height = content_bottom-64
        self.nav_scroll = min(max(0, self.nav_scroll), max(0, self._nav_content_height-viewport_height))
        content = Image.new('RGB', (width, max(self._nav_content_height, viewport_height)), (20, 29, 42))
        draw = ImageDraw.Draw(content)
        for x, yy, label in sections:
            draw.text((x, yy), label, font=self._font(12), fill=(141, 161, 183))
        for button in buttons:
            r = button.rect
            fill = (37, 74, 93) if button.active else (32, 44, 60) if button.enabled else (23, 32, 44)
            edge = (85, 183, 207) if button.active else (56, 73, 93) if button.enabled else (35, 45, 58)
            draw.rounded_rectangle((r.x, r.y, r.x+r.width-1, r.y+r.height-1), radius=5, fill=fill, outline=edge)
            font = self._font(13)
            tw = draw.textlength(button.label, font=font)
            draw.text((r.x+(r.width-tw)/2, r.y+5), button.label, font=font,
                      fill=(231, 243, 249) if button.enabled else (86, 103, 123))
        canvas = Image.fromarray(image)
        painter = ImageDraw.Draw(canvas)
        painter.rectangle((0, 0, width-1, height-1), fill=(20, 29, 42))
        painter.text((14, 10), 'WineJar', font=self._font(23), fill=(237, 243, 249))
        painter.text((15, 40), '离线观测与定位检查', font=self._font(12), fill=(142, 164, 186))
        canvas.paste(content.crop((0, int(self.nav_scroll), width, int(self.nav_scroll)+viewport_height)), (0, 64))
        self._nav_buttons = [NavButton(b.action, b.label,
            Rect(b.rect.x, int(b.rect.y+64-self.nav_scroll), b.rect.width, b.rect.height), b.active, b.enabled) for b in buttons]
        self._nav_clip = (64, content_bottom)
        painter = ImageDraw.Draw(canvas)
        if self._nav_content_height > viewport_height:
            track = content_bottom-64
            thumb = max(24, int(track*viewport_height/self._nav_content_height))
            thumb_y = 64+int((track-thumb)*self.nav_scroll/max(1, self._nav_content_height-viewport_height))
            painter.rounded_rectangle((width-5, thumb_y, width-3, thumb_y+thumb), radius=1, fill=(89, 115, 139))
        painter.line((0, content_bottom+3, width, content_bottom+3), fill=(54, 71, 91))
        d = self.row.get('diagnostics', {})
        status = '算法已接受输出' if self.row['accepted'] else '算法未接受输出'
        if self.show_truth and self.row['accepted'] and not self.row.get('truth_pass',True):
            status = '评估发现：错误接受'
        status_color = ((241, 118, 104) if self.show_truth and self.row['accepted'] and not self.row.get('truth_pass',True)
                        else (115, 217, 174) if self.row['accepted'] else (246, 177, 118))
        painter.text((12, content_bottom+12), status, font=self._font(15), fill=status_color)
        reason_labels = {
            'continuous_core_insufficient_extent': '连续核心范围不足',
            'continuous_core_insufficient_coverage': '核心圆周覆盖不足',
            'continuous_core_touches_candidate_boundary': '核心碰到候选边界',
            'near_minimum_interval_touches_observed_search_boundary': '最细范围碰到搜索边界',
            'no_supported_near_upward_mouth_plane': '口沿初始化支持不足',
            'unsupported_target_interval': '目标区间缺少观测支持',
            'core_model_mismatch': '核心模型与观测不一致',
            'target_readout_mismatch': '目标截面与观测不一致',
        }
        for index, reason in enumerate(self.row.get('reasons', [])[:2]):
            painter.text((12, content_bottom+39+index*18), reason_labels.get(reason, reason)[:26],
                         font=self._font(12), fill=(173, 188, 205))
        local = 'fit_point_positions_m' in d
        note = (f"拟合训练点：{len(d['fit_point_positions_m'])}" if local else '此例未形成颈部拟合点集')
        painter.text((12, height-79), note, font=self._font(12), fill=(176, 196, 216))
        painter.text((12, height-59), '失败点集是尝试范围，非已确认坛颈' if not self.row['accepted'] and local else '红：估计  青：评估真值',
                     font=self._font(11), fill=(142, 164, 186))
        if self.show_truth and self.row.get('center_mm') is not None:
            painter.text((12, height-36), f"评估中心误差 {self.row['center_mm']:.3f} mm", font=self._font(12), fill=TRUTH_COLOR)
        else:
            painter.text((12, height-36), '仅查看保存输入；不运行估计器', font=self._font(11), fill=(142, 164, 186))
        return np.asarray(canvas)

    def draw(self, capture_presented=False):
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width < 640 or height < 480:
            return None
        layout = self._layout(width, height)
        image = np.empty((height, width, 3), np.uint8)
        image[:] = [20, 28, 40]
        main = layout.main
        image[main.y:main.y+main.height, main.x:main.x+main.width] = self._render_view(
            self.camera, self.scene, main.width, main.height, camera_props=True)
        text = []
        condition = self.record_settings.get('preprocessing', {}).get('condition', self.row.get('condition_name', ''))
        condition = CONDITION_LABELS.get(condition, condition)
        case = self.cases[self.case_index]
        text.append((main.x+18, 12, case, 18, (233, 241, 247)))
        method_label = METHOD_LABELS.get(self.method, self.method)
        text.append((main.x+18, 38, f'{condition or "保存观测"} · {method_label}', 12, (147, 172, 194)))
        if self.show_rejected_candidate and not self.row['accepted'] and self.row.get('estimate'):
            text.append((main.x+18, 61, '橙色虚线：被拒绝候选，仅用于失败诊断', 12, REJECTED_COLOR))
        if self.show_truth and self.row.get('center_mm') is not None:
            yy = 80 if self.show_rejected_candidate and not self.row['accepted'] else 61
            text.append((main.x+18, yy, f"评估：中心 {self.row['center_mm']:.2f} mm · 法向 {self.row['normal_deg']:.2f}° · 直径 {self.row['diameter_mm']:.2f} mm", 12, TRUTH_COLOR))
        for index, frame in enumerate(self.frames):
            point = project_scene(self.scene, frame.camera.T_world_from_camera[:3, 3][None], main.width, main.height)[0]
            if np.all(np.isfinite(point)):
                x, y = point+[main.x, main.y]
                blocked = Rect(layout.neck.x, layout.neck.y-48, layout.neck.width, layout.neck.height+48)
                if main.contains(x, y) and not (self.show_neck and blocked.contains(x, y)):
                    text.append((min(int(x)+6, main.x+main.width-80), int(y), frame.camera.name, 14, CAMERA_COLORS[index % 3]))
        if self.show_neck:
            r = layout.neck
            image[r.y:r.y+r.height, r.x:r.x+r.width] = self._render_view(self.neck_camera, self.neck_scene, r.width, r.height)
            cv2.rectangle(image, (r.x-1, r.y-26), (r.x+r.width, r.y+r.height), (74, 96, 116), 1)
            text.append((r.x+9, r.y-23, '颈部近景', 14, (213, 229, 241)))
        region = '实体模型'
        if self.point_cloud:
            if self.cloud_local:
                region = ('拟合训练点' if self.row['accepted'] else '失败尝试的拟合训练点') if self._local_cloud_available else '尚无拟合点：初始化已失败'
            else:
                region = '工位裁剪点云' if self.cloud_crop else '全坛观测点云'
            region += f' · {len(self._cloud_data[0])}点'
        text.append((main.x+16, height-24, region, 12, (168, 192, 211)))
        native_mapping = bool(self.row.get('diagnostics', {}).get('native_color_mapping'))
        inputs = layout.inputs
        slot_height = (height-34)//len(self.frames)
        for index, frame in enumerate(self.frames):
            top = 12+index*slot_height
            color = CAMERA_COLORS[index % 3] if frame.camera.name in self.used else (128, 143, 160)
            mode = ('原生IR深度' if self.depth_mode else 'RGB→IR取色') if native_mapping else ('光轴深度' if self.depth_mode else '原始RGB')
            if self.show_fit_pixels:
                mode += ' · 拟合像素'
            text.append((inputs.x+12, top, f'{frame.camera.name}  ·  {mode}', 14, color))
            panel_top = top+29
            if native_mapping and not self.depth_mode:
                text.append((inputs.x+12, top+22, '按测得3D点取色，非原始RGB', 11, (144, 168, 188)))
                panel_top += 18
            if self.depth_mode:
                depth_low, depth_high = self.record_settings.get('depth_display_range_m', [.45, 1.35])
                scaled = np.clip((frame.depth-depth_low)/(depth_high-depth_low), 0, 1)
                panel = cv2.cvtColor(cv2.applyColorMap((scaled*255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
                panel[~np.isfinite(frame.depth) | (frame.depth <= 0)] = 0
            else:
                panel = frame.rgb.copy()
            if self.overlay:
                self._draw_annotations(panel, camera=frame.camera)
            if self.show_fit_pixels:
                d = self.row.get('diagnostics', {})
                names = d.get('fit_camera_names', [])
                if 'fit_point_positions_m' in d and frame.camera.name in names:
                    points = np.asarray(d['fit_point_positions_m'])
                    points = points[np.asarray(d['fit_camera_ids']) == names.index(frame.camera.name)]
                    uv = np.rint(frame.camera.project(points)).astype(int)
                    for x, y in uv:
                        if 0 <= x < frame.camera.width and 0 <= y < frame.camera.height:
                            cv2.circle(panel, (int(x), int(y)), 1, (255, 185, 90), -1)
            available_w, available_h = inputs.width-24, slot_height-(panel_top-top)-8
            scale = min(available_w/frame.camera.width, available_h/frame.camera.height)
            pw, ph = max(1, int(frame.camera.width*scale)), max(1, int(frame.camera.height*scale))
            panel = cv2.resize(panel, (pw, ph), interpolation=cv2.INTER_AREA)
            x = inputs.x+(inputs.width-pw)//2
            image[panel_top:panel_top+ph, x:x+pw] = panel
            cv2.rectangle(image, (x-1, panel_top-1), (x+pw, panel_top+ph), tuple(color), 1)
        if self.depth_mode:
            depth_low, depth_high = self.record_settings.get('depth_display_range_m', [.45, 1.35])
            text.append((inputs.x+12, height-24, f'{depth_low:.2f}–{depth_high:.2f} m；黑色为无效深度', 11, (162, 185, 206)))
        image = np.ascontiguousarray(self._text(image, text))
        image = np.ascontiguousarray(self._draw_sidebar(image, layout))
        present_image(image, self.context)
        if capture_presented:
            presented = np.empty_like(image)
            mujoco.mjr_readPixels(presented, None, mujoco.MjrRect(0, 0, width, height), self.context)
            image = np.ascontiguousarray(presented[::-1])
        glfw.swap_buffers(self.window)
        return image

    def _activate(self, action):
        self.dragging = None
        toggles = dict(toggle_overlay='overlay', toggle_truth='show_truth', toggle_band='show_band',
                       toggle_estimate='show_estimate', toggle_neck='show_neck', toggle_large='large_neck',
                       toggle_fit_pixels='show_fit_pixels', toggle_rejected='show_rejected_candidate')
        if action in toggles:
            name = toggles[action]
            setattr(self, name, not getattr(self, name))
        elif action.startswith('view:'):
            mode = action.split(':', 1)[1]
            self.point_cloud = mode != 'model'
            self.cloud_local = mode == 'fit'
            if not self.cloud_local:
                self.cloud_weights = False
        elif action.startswith('color:'):
            mode = action.split(':', 1)[1]
            self.cloud_weights = mode == 'weights'
            self.cloud_source_colors = mode == 'source'
            if self.cloud_weights:
                self.cloud_local = self.point_cloud = True
        elif action.startswith('input:'):
            self.depth_mode = action.endswith('depth')
        elif action == 'toggle_crop':
            self.cloud_crop = not self.cloud_crop
            self.cloud_local = self.cloud_weights = False
            self.point_cloud = True
        elif action in ('next', 'previous'):
            self.case_index = (self.case_index+(1 if action == 'next' else -1)) % len(self.cases)
            self._load_case()
        elif action == 'reset':
            self._reset_camera()
        elif action == 'method' or action.startswith('method:'):
            previous = self.method
            self.method = (action.split(':',1)[1] if action.startswith('method:') else
                           self.methods[(self.methods.index(previous)+1) % len(self.methods)])
            try:
                self.row
            except StopIteration:
                self.method = previous
            self._focus_neck()
        elif action.startswith('camera:'):
            previous = self.camera_set
            self.camera_set = action.split(':', 1)[1]
            try:
                self.row
            except StopIteration:
                self.camera_set = previous
        elif action.startswith('dataset:'):
            self._switch_dataset(action.split(':', 1)[1])
        self._cloud_cache_key = None

    def _switch_dataset(self, condition):
        dataset = next(d for d in self.datasets if d['condition'] == condition)
        if Path(dataset['records']) == self.records_path:
            return
        old_case = self.cases[self.case_index]
        poses = [{k:np.array(getattr(cam, k)).copy() for k in ('lookat', 'distance', 'azimuth', 'elevation')}
                 for cam in (self.camera, self.neck_camera)]
        self.root, self.records_path = Path(dataset['scenes']), Path(dataset['records'])
        self.rows = read_records(self.records_path)
        settings_path = self.records_path.parent/'settings.json'
        self.record_settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
        self.methods = sorted({r['method'] for r in self.rows if r['evidence'] == self.evidence})
        if self.method not in self.methods:
            self.method = dataset.get('method', self.methods[0])
        self.cases = sorted({r['case'] for r in self.rows if r['evidence'] == self.evidence and r['method'] == self.method
                             and (self.root/r['case']/'scene.xml').exists()})
        self.case_index = self.cases.index(old_case) if old_case in self.cases else 0
        if not any(r['camera_set'] == self.camera_set for r in self.rows):
            self.camera_set = 'all'
        self._load_case()
        if self.cases[self.case_index] == old_case:
            for cam, pose in zip((self.camera, self.neck_camera), poses):
                for key, value in pose.items():
                    if key == 'lookat':
                        cam.lookat[:] = value
                    else:
                        setattr(cam, key, float(value))

    def _mouse_button(self, window, button, action, mods):
        self.cursor = glfw.get_cursor_pos(window)
        point = framebuffer_point(*self.cursor, glfw.get_window_size(window), glfw.get_framebuffer_size(window))
        if point is not None and button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            selected = hit_button(self._nav_buttons, *point, *getattr(self, '_nav_clip', (0, 0)))
            if selected:
                self._activate(selected)
                return
        view = self._pointer_view(*self.cursor)
        self.dragging = (button, view) if action == glfw.PRESS and view is not None else None

    def _move(self, window, x, y):
        if self.dragging is None or self.cursor is None:
            self.cursor = (x, y)
            return
        ww, hh = glfw.get_window_size(window)
        fw, fh = glfw.get_framebuffer_size(window)
        button, (camera, scene, view_height) = self.dragging
        dx = (x-self.cursor[0])*fw/ww/view_height
        dy = (y-self.cursor[1])*fh/hh/view_height
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_V if button == glfw.MOUSE_BUTTON_LEFT else mujoco.mjtMouse.mjMOUSE_MOVE_V
        mujoco.mjv_moveCamera(self.model, action, dx, dy, scene, camera)
        self.cursor = (x, y)

    def _scroll(self, window, x, y):
        size = glfw.get_framebuffer_size(window)
        point = framebuffer_point(*glfw.get_cursor_pos(window), glfw.get_window_size(window), size)
        if point is not None and self._layout(*size).sidebar.contains(*point):
            self.nav_scroll -= y*42
            self.nav_scroll = max(0, self.nav_scroll)
            return
        view = self._pointer_view(*glfw.get_cursor_pos(window))
        if view is not None:
            camera, scene, _ = view
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -.05*y, scene, camera)

    def _key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
            return
        actions = {
            glfw.KEY_O: 'toggle_overlay', glfw.KEY_G: 'toggle_truth', glfw.KEY_B: 'toggle_band',
            glfw.KEY_N: 'toggle_large', glfw.KEY_R: 'reset', glfw.KEY_V: 'toggle_crop',
            glfw.KEY_M: 'method', glfw.KEY_LEFT: 'previous', glfw.KEY_RIGHT: 'next',
            glfw.KEY_D: 'input:rgb' if self.depth_mode else 'input:depth',
            glfw.KEY_P: 'view:model' if self.point_cloud else 'view:cloud',
            glfw.KEY_L: 'view:cloud' if self.cloud_local and self.point_cloud else 'view:fit',
            glfw.KEY_W: 'color:rgb' if self.cloud_weights else 'color:weights',
            glfw.KEY_C: 'color:rgb' if self.cloud_source_colors else 'color:source',
            glfw.KEY_1: 'camera:cam_a', glfw.KEY_2: 'camera:cam_a,cam_b', glfw.KEY_3: 'camera:all',
        }
        if key in actions:
            self._activate(actions[key])

    def snapshot(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        image = self.draw(capture_presented=True)
        if image is None:
            raise RuntimeError("Window is too small")
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        metadata = {"kind": "inspection_only_no_estimation", "case": self.cases[self.case_index],
                    "capture": "actual_window_back_buffer_after_presentation",
                    "camera_set": self.camera_set, "rgb_result_overlay": self.overlay,
                    "method": self.method,
                    "large_neck_view": self.large_neck,
                    "finite_band_overlay": self.show_band,
                    "sensor_perturbation": self.record_settings.get('sensor_perturbation'),
                    "depth_preprocessing": self.record_settings.get('preprocessing'),
                    "point_cloud_view": self.point_cloud, "point_cloud_station_crop": self.cloud_crop,
                    "point_cloud_local_fit": self.cloud_local,
                    "point_cloud_weights": self.cloud_weights,
                    "point_cloud_source_colors": self.cloud_source_colors,
                    "point_cloud_origin": "RGB foreground and valid saved axial depth; no mesh/GT/completion; deterministic display thinning",
                    "mouse_navigation": True, "navigation_scroll": self.nav_scroll,
                    "show_neck_view": self.show_neck, "show_estimate": self.show_estimate,
                    "show_rejected_candidate": self.show_rejected_candidate,
                    "fitted_pixel_display_overlay": self.show_fit_pixels,
                    "fit_subset_semantics": self.row.get('diagnostics', {}).get('fit_visualization_semantics',
                        'saved training subset, not guaranteed to be the physical neck'),
                    "ground_truth_overlay": self.show_truth, "depth_display": self.depth_mode,
                    "camera_props": "schematic; true poses and K-derived truncated FOV markers",
                    "result_lines": "through-surface projections, not observed image edges",
                    "accepted": self.row["accepted"], "reasons": self.row["reasons"],
                    "records_sha256": sha256(self.records_path.read_bytes()).hexdigest(),
                    "raw_rgb_sha256": {f.camera.name: sha256(f.rgb_path.read_bytes()).hexdigest() for f in self.frames}}
        path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")

    def run(self):
        while not glfw.window_should_close(self.window):
            self.draw()
            glfw.poll_events()
            time.sleep(.015)

    def close(self):
        if self.window is not None:
            glfw.make_context_current(self.window)
            if self.context is not None:
                self.context.free()
                self.context = None
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()
