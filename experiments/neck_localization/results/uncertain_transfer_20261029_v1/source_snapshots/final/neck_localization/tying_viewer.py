"""Three-station playback in the original InspectionViewer and GLFW UI."""
import json
import time

import cv2
import glfw
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .inspection_viewer import (InspectionViewer, draw_result, draw_band_boundaries,
    draw_truth_range, project_scene, TRUTH_COLOR, REJECTED_COLOR, add_camera_markers,
    draw_cloud, display_samples, CAMERA_COLORS, _line)
from .station_tying import timeline, move_circle, GunSpec
from .viewer_navigation import NavButton, Rect, framebuffer_point, hit_button


PHASE_LABELS = dict(recognize='识别停拍', to_loading='输送至上料', cover='上料遮盖',
    to_tying='输送至扎口', above='枪头在口部上方', descend='沿估计法向下降',
    close='两圆弧闭合', tighten='收带指令演示', open='张开枪头', retract='退出')


class TyingViewer(InspectionViewer):
    def __init__(self, *args, station_config, **kwargs):
        self.station_config = station_config
        self.clock_s, self.playing, self.last_wall = 0., False, time.monotonic()
        self.stage_state = None
        self.stage_buttons = []
        super().__init__(*args, **kwargs)

    def _load_case(self):
        super()._load_case()
        # Original scenes were framed around one jar at <2 m; the long line
        # needs a farther viewing clip plane (camera inputs are unchanged).
        self.model.vis.map.zfar = 20.
        self.plan = json.loads((self.folder/'plan.json').read_text())
        self.stage_eval = json.loads((self.folder/'evaluation.json').read_text())
        self.ik = json.loads((self.folder/'ik_path.json').read_text())
        self.scene_meta = json.loads((self.folder/'scene_metadata.json').read_text())
        self.joint_addresses = [self.model.jnt_qposadr[self.model.joint(f'right_joint{i}').id] for i in range(1, 7)]
        self.jaw_addresses = [self.model.jnt_qposadr[self.model.joint(f'jaw_joint_{s}').id] for s in (1, -1)]
        self.clock_s = 0.
        self._apply_stage()

    def _reset_camera(self):
        super()._reset_camera()
        self.camera.lookat[:] = [1.2, 0, .38]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 4.4, 90, -24

    def _focus_neck(self):
        super()._focus_neck()
        if self.stage_state is not None:
            self.neck_camera.lookat[:] += self.stage_state['displacement_m']

    def _apply_stage(self):
        self.stage_state = timeline(self.station_config, self.plan, self.clock_s)
        state = self.stage_state
        delta = state['displacement_m']
        for name, origin in [('jar', np.zeros(3)), ('loading_cover', np.array(self.scene_meta['cover_position_m']))]:
            body = self.model.body(name).id
            self.data.mocap_pos[self.model.body_mocapid[body]] = origin+delta
        self.model.geom_rgba[self.model.geom('cover_visual').id, 3] = 1. if state['covered'] else 0.
        samples = self.ik.get('samples', [])
        if samples:
            fraction = 0.
            if state['phase'] == 'descend':
                above, end = np.array(self.plan['W_T_above']), np.array(self.plan['W_T_G_t2'])
                fraction = np.clip(np.linalg.norm(state['W_T_G'][:3, 3]-above[:3, 3])/self.plan['descent_distance_m'], 0, 1)
            elif state['phase'] in ('close', 'tighten', 'open'):
                fraction = 1.
            elif state['phase'] == 'retract':
                fraction = np.clip(np.linalg.norm(state['W_T_G'][:3, 3]-np.array(self.plan['W_T_above'])[:3, 3])/self.plan['descent_distance_m'], 0, 1)
            index = fraction*(len(samples)-1)
            low, high = int(np.floor(index)), min(int(np.floor(index))+1, len(samples)-1)
            q = (1-(index-low))*np.array(samples[low]['q_rad'])+(index-low)*np.array(samples[high]['q_rad'])
            self.data.qpos[self.joint_addresses] = q
        self.data.qpos[self.jaw_addresses] = np.deg2rad(state.get('opening_deg', 60.)/2)
        mujoco.mj_forward(self.model, self.data)
        self._focus_neck()

    def _draw_annotations(self, image, scene=None, camera=None):
        if camera is not None:
            # These panels are the original t0 camera inputs, never post-cover observations.
            return super()._draw_annotations(image, camera=camera)
        project = lambda p: project_scene(scene, p, image.shape[1], image.shape[0])
        transform = np.eye(4)
        transform[:3, 3] = self.stage_state['displacement_m']
        if self.row.get('estimate') and (self.show_estimate and self.row['accepted'] or self.show_rejected_candidate):
            circle = move_circle(self.row['estimate'], transform)
            draw_result(image, project, circle, dashed=not self.row['accepted'])
            if self.show_band and self.row.get('band'):
                band = self.row['band'] | move_circle(self.row['band'], transform)
                draw_band_boundaries(image, project, band)
        if self.show_truth:
            truth = self._truth()
            moved = truth | move_circle(truth, transform)
            draw_result(image, project, moved, TRUTH_COLOR, dashed=True)
            if self.show_band:
                draw_truth_range(image, project, moved)
                # Factory criterion applies to the complete physical neck,
                # distinct from the narrower near-minimum proxy.
                for bound in moved['search_interval_s_m']:
                    edge = moved | dict(center_m=np.array(moved['center_m'])+(bound-moved['s_m'])*np.array(moved['normal']))
                    draw_result(image, project, edge, (122, 224, 141), dashed=True, arrow=False)

    def _render_view(self, camera, scene, width, height, camera_props=False):
        if not self.point_cloud:
            mujoco.mjv_updateScene(self.model, self.data, self.option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            if camera_props:
                add_camera_markers(scene, [f.camera for f in self.frames], self.used)
            if self.stage_state.get('band_visible') and self.plan.get('waypoints'):
                T = np.array(self.plan['W_T_G_t2'])
                radius = self.stage_state['band_radius_m']
                theta = np.linspace(0, 2*np.pi, 121)
                for z in (-.0015, .0015):
                    points = np.c_[radius*np.cos(theta), radius*np.sin(theta), np.full(len(theta), z)] @ T[:3, :3].T+T[:3, 3]
                    for a, b in zip(points, points[1:]):
                        _line(scene, a, b, [1., .8, .12, 1.], 1.)
            rect = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjr_render(rect, scene, self.context)
            rgb = np.empty((height, width, 3), np.uint8)
            mujoco.mjr_readPixels(rgb, None, rect, self.context)
            image = np.ascontiguousarray(rgb[::-1])
            self._draw_annotations(image, scene=scene)
            return image
        # Preserve original point selection/colours; move t0 measurements rigidly.
        # The label always discloses that these are recorded, NOT re-observed points.
        mujoco.mjv_updateScene(self.model, self.data, self.option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        scene.ngeom = 0
        if camera_props:
            add_camera_markers(scene, [f.camera for f in self.frames], self.used)
        local = self._local_fit_clouds() if self.cloud_local else None
        self._local_cloud_available = local is not None
        clouds = (local or []) if self.cloud_local else self.clouds
        colors = {f.camera.name: CAMERA_COLORS[i % 3] for i, f in enumerate(self.frames)}
        self._cloud_data = display_samples(clouds, self.used, self.cloud_crop or self.cloud_local,
                                          colors if self.cloud_source_colors and not self.cloud_weights else None)
        pts, rgb = self._cloud_data[:2]
        rect = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(rect, scene, self.context)
        draw_cloud(scene, pts+self.stage_state['displacement_m'], rgb, width, height)
        pixels = np.empty((height, width, 3), np.uint8)
        mujoco.mjr_readPixels(pixels, None, rect, self.context)
        image = np.ascontiguousarray(pixels[::-1])
        self._draw_annotations(image, scene=scene)
        return image

    def _text(self, image, items):
        layout = self._layout(image.shape[1], image.shape[0])
        x = layout.main.x+18
        state = self.stage_state
        items = items+[(x, 91, f"{self.clock_s:.1f}s · {PHASE_LABELS[state['phase']]}", 15, (245, 211, 116)),
            (layout.inputs.x+12, image.shape[0]-18, '识别工位t0保存输入；遮盖后不再观测颈部', 11, (241, 208, 114)),
            (x, 115, '收带仅为运动演示；锁紧/剪断和臂身碰撞尚未验证', 11, (177, 192, 211))]
        if self.plan.get('waypoints'):
            items.append((x, 137, '红：估计  青：真值  绿：物理颈段边界  黄：3mm带', 11, (177, 192, 211)))
        if self.show_truth and self.stage_eval.get('head_clearance'):
            clearance = self.stage_eval['head_clearance']
            if not clearance['passed']:
                items.append((x, 180, '枪头净空尚无法确认；不等同于已证实碰撞', 12, (246, 150, 110)))
        for label, pos in [('识别工位', [0, -.33, .03]), ('上料工位', [1.2, -.33, .03]), ('扎口工位', [2.4, -.33, .03])]:
            uv = project_scene(self.scene, np.array([pos]), layout.main.width, layout.main.height)[0]
            if np.isfinite(uv).all():
                items.append((int(uv[0])+layout.main.x-25, int(uv[1])+layout.main.y, label, 12, (170, 220, 210)))
        if self.point_cloud:
            items.append((x, 202, '点云：t0保存测量随刚性输送变换', 12, (241, 208, 114)))
        if not self.ik.get('passed'):
            items.append((x, 158, '离线IK未通过：所示臂姿为失败诊断，不是可执行轨迹', 13, (245, 130, 100)))
        if self.show_truth and self.stage_eval.get('evaluated'):
            margin = self.stage_eval['physical_band']['minimum_margin_mm']
            label = '颈段内' if margin >= 0 else '越出颈段'
            items.append((x, 158, f'3mm带：{label} · 最小余量 {margin:+.3f} mm', 13,
                          (115, 217, 174) if margin >= 0 else (246, 150, 110)))
        return super()._text(image, items)

    def _draw_sidebar(self, image, layout):
        image = super()._draw_sidebar(image, layout)
        canvas = Image.fromarray(image)
        draw = ImageDraw.Draw(canvas)
        width, height = layout.sidebar.width, layout.sidebar.height
        y = height-163
        draw.rectangle((0, y, width-1, height-1), fill=(20, 29, 42))
        labels = [('stage:play', '暂停' if self.playing else '播放全流程'), ('stage:recognize', '识别工位'),
                  ('stage:cover', '上料遮盖'), ('stage:above', '扎口上方'), ('stage:descend', '下降到位'),
                  ('stage:close', '闭合枪头'), ('stage:tighten', '收带演示'), ('stage:retract', '退出')]
        self.stage_buttons = []
        for i, (action, label) in enumerate(labels):
            bx, by, bw, bh = 10+(i % 2)*(width//2), y+6+(i//2)*31, width//2-17, 26
            draw.rounded_rectangle((bx, by, bx+bw, by+bh), 4, fill=(37, 65, 80), outline=(73, 119, 137))
            draw.text((bx+5, by+4), label, font=self._font(12), fill=(231, 243, 249))
            self.stage_buttons.append(NavButton(action, label, Rect(bx, by, bw, bh), False, True))
        draw.text((12, height-25), '内净径150 mm · 带宽3 mm', font=self._font(12), fill=(231, 209, 145))
        return np.asarray(canvas)

    def _activate(self, action):
        if action.startswith('stage:'):
            selected = action.split(':')[1]
            if selected == 'play':
                self.playing = not self.playing
                self.last_wall = time.monotonic()
                return
            phases = list(self.station_config['durations_s'])
            begin = sum(self.station_config['durations_s'][p] for p in phases[:phases.index(selected)])
            self.clock_s = begin+self.station_config['durations_s'][selected]*(.96 if selected in ('descend', 'close', 'tighten', 'retract') else .3)
            self.playing = False
            self._apply_stage()
        else:
            super()._activate(action)

    def _mouse_button(self, window, button, action, mods):
        point = framebuffer_point(*glfw.get_cursor_pos(window), glfw.get_window_size(window), glfw.get_framebuffer_size(window))
        if point is not None and button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            selected = hit_button(self.stage_buttons, *point, 0, glfw.get_framebuffer_size(window)[1])
            if selected:
                self._activate(selected)
                return
        super()._mouse_button(window, button, action, mods)

    def draw(self, capture_presented=False):
        now = time.monotonic()
        if self.playing:
            self.clock_s += min(now-self.last_wall, .2)
            total = sum(self.station_config['durations_s'].values())
            if self.clock_s >= total:
                self.clock_s, self.playing = total, False
        self.last_wall = now
        self._apply_stage()
        return super().draw(capture_presented)
