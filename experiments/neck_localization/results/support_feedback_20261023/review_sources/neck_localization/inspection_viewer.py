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


CAMERA_COLORS = [(82, 176, 248), (116, 218, 150), (238, 184, 92)]
ESTIMATE_COLOR = (255, 104, 91)
TRUTH_COLOR = (94, 224, 246)
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


def draw_result(image, project, result, color=ESTIMATE_COLOR, dashed=False):
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
    arrow = project(np.array([center, center+.075*normal]))
    if np.all(np.isfinite(arrow)) and np.max(np.abs(arrow)) < 1e7:
        cv2.arrowedLine(image, tuple(np.rint(arrow[0]).astype(int)), tuple(np.rint(arrow[1]).astype(int)),
                       color, 1, cv2.LINE_AA, tipLength=.12)


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
                 point_cloud=False):
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
        self.point_cloud, self.cloud_crop, self.cloud_source_colors = point_cloud, False, False
        self.cloud_local = False
        self.cloud_weights = False
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
        self.clouds = [observation_cloud(f.camera, f.rgb, f.depth) for f in self.frames]
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
        self.neck_camera.lookat[:] = result["center_m"] if result else [0, 0, .59]

    def _layout(self, width, height):
        right_width = min(420, max(285, int(height*.42)))
        left_width = width-right_width
        if self.large_neck:
            inset_w, inset_h = min(720, left_width-32), min(480, int(height*.55))
        else:
            inset_w, inset_h = min(420, left_width//2), min(285, height//3)
        return left_width, right_width, inset_w, inset_h, height-inset_h-94

    def _pointer_view(self, x, y):
        """Map logical GLFW coordinates to the actual framebuffer layout."""
        ww, hh = glfw.get_window_size(self.window)
        fw, fh = glfw.get_framebuffer_size(self.window)
        if min(ww, hh, fw, fh) <= 0:
            return None
        x, y = x*fw/ww, y*fh/hh
        left_width, _, inset_w, inset_h, y0 = self._layout(fw, fh)
        if not 0 <= x < left_width or not 0 <= y < fh:
            return None
        if 16 <= x < 16+inset_w and y0 <= y < y0+inset_h:
            return self.neck_camera, self.neck_scene, inset_h
        return self.camera, self.scene, fh

    def _truth(self):
        return json.loads((self.folder/"evaluation_only/target.json").read_text()) if self.show_truth else None

    def _draw_annotations(self, image, scene=None, camera=None):
        project = camera.project if camera is not None else lambda points: project_scene(scene, points, image.shape[1], image.shape[0])
        if self.row["accepted"]:
            draw_result(image, project, self.row.get("estimate"))
        if self.show_truth:
            draw_result(image, project, self._truth(), TRUTH_COLOR, dashed=True)

    def _local_fit_clouds(self):
        """Recreate the saved method's observation selection, never read GT."""
        d, settings = self.row.get("diagnostics", {}), self.record_settings
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
                self._cloud_data = display_samples(local if local is not None else self.clouds,
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

    def draw(self, capture_presented=False):
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width < 640 or height < 480:
            return None
        left_width, right_width, inset_w, inset_h, y0 = self._layout(width, height)
        image = np.zeros((height, width, 3), np.uint8)
        image[:] = [22, 29, 40]
        image[:, :left_width] = self._render_view(self.camera, self.scene, left_width, height, camera_props=True)
        text = []
        for index, frame in enumerate(self.frames):
            point = project_scene(self.scene, frame.camera.T_world_from_camera[:3, 3][None], left_width, height)[0]
            covered = (point[0]+80 >= 15 and point[0] < 18+inset_w
                       and y0-48 <= point[1] < y0+inset_h+1)
            if (np.all(np.isfinite(point)) and 0 < point[0] < left_width
                    and 75 < point[1] < height-85 and not covered):
                text.append((int(point[0])+8, int(point[1]), frame.camera.name, 18, CAMERA_COLORS[index % 3]))
        # Neck inset: a genuine additional MuJoCo review camera, not estimator input.
        neck = self._render_view(self.neck_camera, self.neck_scene, inset_w, inset_h)
        image[y0:y0+inset_h, 16:16+inset_w] = neck
        cv2.rectangle(image, (15, y0-27), (17+inset_w, y0+inset_h+1), (90, 108, 125), 1)
        text.append((23, y0-25, "颈部近景：圆与法向 · 可独立旋转 · N放大/还原", 16, (231, 236, 242)))
        # Three panels show the actual saved inputs, with an optional display-only overlay.
        slot_h = (height-38)//len(self.frames)
        for index, frame in enumerate(self.frames):
            y = 12+index*slot_h
            active = frame.camera.name in self.used
            color = CAMERA_COLORS[index % 3] if active else (132, 142, 153)
            mode = "Z深度伪彩" if self.depth_mode else "原始RGB"
            if self.overlay:
                mode += " + 结果叠加"
            text.append((left_width+12, y, f"{frame.camera.name} · {'参与估计' if active else '仅预览'} · {mode}", 15, color))
            if self.depth_mode:
                scaled = np.clip((frame.depth-.45)/.90, 0, 1)
                panel = cv2.cvtColor(cv2.applyColorMap((scaled*255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
            else:
                panel = frame.rgb.copy()
            if self.overlay:
                self._draw_annotations(panel, camera=frame.camera)
            available_w, available_h = right_width-24, slot_h-34
            scale = min(available_w/frame.camera.width, available_h/frame.camera.height)
            pw, ph = int(frame.camera.width*scale), int(frame.camera.height*scale)
            panel = cv2.resize(panel, (pw, ph), interpolation=cv2.INTER_AREA)
            x = left_width+(right_width-pw)//2
            image[y+28:y+28+ph, x:x+pw] = panel
            cv2.rectangle(image, (x-1, y+27), (x+pw, y+28+ph), color, 1)
        reason_labels = {"unsupported_gap_in_radius_profile": "半径剖面存在未观测空档"}
        status = "算法接受" if self.row["accepted"] else "算法拒绝："+"; ".join(
            reason_labels.get(reason, reason) for reason in self.row["reasons"])
        input_note = "相机外形为示意；位姿和视场来自实际参数。右侧画面直接读取保存的输入。"
        if self.point_cloud:
            shown, total = len(self._cloud_data[0]), self._cloud_data[2]
            region = ('新方法实际拟合点' if self._local_cloud_available else '局部记录不可用，显示旧工位点') if self.cloud_local else ('旧工位裁剪' if self.cloud_crop else '全坛RGB前景')
            input_note = (f"点云：{region} · "
                          f"{'相机来源色' if self.cloud_source_colors else '原始RGB色'} · 显示{shown}/{total}点 · 仅参与估计相机")
            if self.cloud_weights:
                available = self.cloud_local and 'fit_point_weights' in self.row.get('diagnostics', {})
                input_note = (f"实际权重：灰色低→黄色高（按当前最大值归一显示） · {shown}点" if available
                              else "当前记录无点权重；显示原始点颜色")
        text += [(18, 12, f"{self.cases[self.case_index]} | {self.camera_set} | {self.method} | {status}", 17, (243, 244, 246)),
                 (18, 41, "红色：估计圆与法向；线条为结果投影，可穿过表面。G：显示/隐藏真值", 16, ESTIMATE_COLOR),
                 (18, 66, input_note, 15, (205, 216, 228)),
                 (18, height-75, "左键拖动旋转 · 右键拖动平移 · 滚轮缩放 · R复位", 16, (225, 233, 242)),
                 (18, height-49, "1/2/3：切换估计视图数   ←/→：切换酒坛   O：画面叠加   D：RGB/深度   Esc：退出", 15, (225, 233, 242)),
                 (18, height-23, "P点云  C来源色  V全坛/旧裁剪  L拟合点  W权重  N近景放大  M方法", 15, (225, 233, 242))]
        if self.show_truth:
            text.append((18, 92, "青色虚线：真值（仅用于评估比较）", 16, TRUTH_COLOR))
        if self.row["center_mm"] is not None:
            over_limit = "（超过4 mm）" if self.row["center_mm"] > 4 else ""
            text.append((30 if self.large_neck else left_width//2, height-116,
                         f"离线评估：圆心误差 {self.row['center_mm']:.2f} mm{over_limit}", 16, (226, 233, 241)))
        if self.depth_mode:
            text.append((left_width+12, height-24, "深度色标：0.45 m → 1.35 m（范围外截色）", 13, (208, 219, 230)))
        image = np.ascontiguousarray(self._text(image, text))
        present_image(image, self.context)
        if capture_presented:
            # Capture the actual window back buffer, not just the intended image.
            presented = np.empty_like(image)
            mujoco.mjr_readPixels(presented, None, mujoco.MjrRect(0, 0, width, height), self.context)
            image = np.ascontiguousarray(presented[::-1])
        glfw.swap_buffers(self.window)
        return image

    def _mouse_button(self, window, button, action, mods):
        self.cursor = glfw.get_cursor_pos(window)
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
        view = self._pointer_view(*glfw.get_cursor_pos(window))
        if view is not None:
            camera, scene, _ = view
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -.05*y, scene, camera)

    def _key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_O:
            self.overlay = not self.overlay
        elif key == glfw.KEY_D:
            self.depth_mode = not self.depth_mode
        elif key == glfw.KEY_G:
            self.show_truth = not self.show_truth
        elif key == glfw.KEY_R:
            self._reset_camera()
        elif key == glfw.KEY_N:
            self.dragging = None
            self.large_neck = not self.large_neck
        elif key == glfw.KEY_P:
            self.point_cloud = not self.point_cloud
        elif key == glfw.KEY_C:
            self.cloud_weights = False
            self.cloud_source_colors = not self.cloud_source_colors
        elif key == glfw.KEY_V:
            self.cloud_local = False
            self.cloud_crop = not self.cloud_crop
        elif key == glfw.KEY_L:
            self.cloud_local = not self.cloud_local
            self.point_cloud = True
        elif key == glfw.KEY_W:
            self.cloud_weights = not self.cloud_weights
            self.cloud_local = True
            self.point_cloud = True
        elif key == glfw.KEY_M:
            previous = self.method
            self.method = self.methods[(self.methods.index(previous)+1) % len(self.methods)]
            try:
                self.row
            except StopIteration:
                self.method = previous
            self._focus_neck()
        elif key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3):
            choice = {glfw.KEY_1: "cam_a", glfw.KEY_2: "cam_a,cam_b", glfw.KEY_3: "all"}[key]
            previous, self.camera_set = self.camera_set, choice
            try:
                self.row
            except StopIteration:
                self.camera_set = previous
            self._focus_neck()
        elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
            self.dragging = None
            self.case_index = (self.case_index+(1 if key == glfw.KEY_RIGHT else -1)) % len(self.cases)
            self._load_case()

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
                    "point_cloud_view": self.point_cloud, "point_cloud_station_crop": self.cloud_crop,
                    "point_cloud_local_fit": self.cloud_local,
                    "point_cloud_weights": self.cloud_weights,
                    "point_cloud_source_colors": self.cloud_source_colors,
                    "point_cloud_origin": "RGB foreground and valid saved axial depth; no mesh/GT/completion; deterministic display thinning",
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
