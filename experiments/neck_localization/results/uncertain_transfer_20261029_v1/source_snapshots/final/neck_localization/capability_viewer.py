"""Read-only replay of the saved capability audit; never rerun estimators.

Truth is used only for explicit display and common viewport selection. L-NDF
is shown as a center/axis, with no invented radius or predicted circle.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/neck_localization'
OPTIONS = ('SimSense / noise', 'FoundationStereo / noise',
           'SimSense / clean', 'FoundationStereo / clean',
           'P2P input / raw', 'P2P restored / raw',
           'P2P input / ideal', 'P2P restored / ideal',
           'L-NDF / raw', 'L-NDF / ideal', 'Ideal IR / control')


@lru_cache(maxsize=30)
def read_json(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=16)
def read_rows(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines()]
    return read_json(str(path))


def circle(center, normal, radius):
    normal = np.asarray(normal, float)
    normal /= np.linalg.norm(normal)
    u = np.cross(normal, [1., 0., 0.])
    if np.linalg.norm(u) < .1:
        u = np.cross(normal, [0., 1., 0.])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    angle = np.linspace(0, 2*np.pi, 200)
    return np.asarray(center) + radius*(np.cos(angle)[:, None]*u + np.sin(angle)[:, None]*v)


@lru_cache(maxsize=24)
def reference_mesh_points(path):
    # Saved CAD is evaluation/display-only, never supplied to an estimator.
    with Path(path).open() as stream:
        points = np.array([[float(x) for x in line.split()[1:4]]
                           for line in stream if line.startswith('v ')])
    return points[np.linspace(0,len(points)-1,min(9000,len(points)),dtype=int)]


def cloud_from_frames(frames, maximum=6000):
    from .stereo_robust import blue_screen_foreground
    clouds, ids = [], []
    for i, frame in enumerate(frames):
        valid = blue_screen_foreground(frame.rgb) & np.isfinite(frame.depth_z_m) & (frame.depth_z_m > 0)
        yy, xx = np.nonzero(valid)
        q = frame.camera.unproject(np.c_[xx, yy], frame.depth_z_m[yy, xx])
        keep = (q[:, 2] >= 0) & (q[:, 2] <= 1.4) & (np.linalg.norm(q[:, :2], axis=1) <= 1.5)
        q = q[keep]
        clouds.append(q); ids.append(np.full(len(q), i))
    points, ids = np.concatenate(clouds), np.concatenate(ids)
    if len(points) > maximum:
        take = np.linspace(0, len(points)-1, maximum, dtype=int)
        points, ids = points[take], ids[take]
    return points, ids


def load_case(layout, case, option, backend='portfolio'):
    """Load precisely the saved branch, retaining prediction/input provenance."""
    from scripts.evaluate_capability_depth import load_prediction_index, read_prediction, prepare_frame
    from scripts.evaluate_capability_cloud import load_rgb_frames, source_camera_zbuffers
    from .point_restoration_baseline import load_observation_frames
    source = BASE/'runs/structural_stage_20261025_v1'/layout
    rendered = source/read_json(str(source/'input_source.json'))['render_directory']
    truth = read_json(str(rendered/'geometry'/case/'evaluation_only/target.json'))
    reference_frames = load_observation_frames(rendered/'ideal_rgb/scenes'/case/'observations')
    reference = reference_mesh_points(str(rendered/'ideal_rgb/scenes'/case/'assets/jar.obj'))
    title = option
    if option.startswith(('SimSense', 'FoundationStereo', 'Ideal IR')):
        condition = 'shared_noise' if option.endswith('noise') else 'clean'
        if option.startswith('FoundationStereo'):
            predictions = BASE/'runs/foundation_stereo_capability_20261027_v1'
            evaluation = BASE/'results/foundation_stereo_capability_eval_20261027_v1'
        elif option.startswith('SimSense'):
            predictions = BASE/'runs/simsense_capability_20261027_v1'
            evaluation = BASE/'results/simsense_capability_20261027_v1'
        else:
            predictions = BASE/'runs/ideal_native_reference_capability_20261027_v1'
            evaluation = BASE/'results/ideal_native_capability_eval_20261027_v1'
        pairs = [p for p in read_json(str(BASE/'runs/capability_inputs_20261027_v1/manifest.json'))['pairs']
                 if (p['layout'], p['case'], p['condition']) == (layout, case, condition)]
        index = load_prediction_index(predictions)
        frames = []
        for pair in sorted(pairs, key=lambda p:p['camera']):
            depth, _ = read_prediction(pair, predictions, index, {})
            frame, _, _ = prepare_frame(pair, depth, {})
            frames.append(frame)
        row = next(r for r in read_rows(str(evaluation/'records.jsonl'))
                   if (r['layout'],r['case'],r['condition'],r['camera_set'],r['backend']) ==
                   (layout,case,condition,'all',backend))
        points, ids = cloud_from_frames(frames)
        note = '三视RGB-D前景展示；并非全部都是拟合点。原生IR深度按审查流程量化/注册。'
        radius = row.get('estimate', {}).get('radius_m')
        estimate = row.get('estimate')
    elif option.startswith('P2P'):
        condition = 'ideal_rgb' if option.endswith('ideal') else 'stereo_raw'
        variant = 'restored_reprojected' if option.startswith('P2P restored') else 'sampled_input_reprojected'
        predictions = BASE/f'runs/p2p_capability_{layout}_20261027_v2'
        evaluation = BASE/f'results/p2p_capability_{layout}_eval_20261027_v2'
        item = next(r for r in read_json(str(predictions/'manifest.json')) if (r['case'],r['condition']) == (case,condition))
        with np.load(item['pointcloud']) as data:
            points = data['points_world_m' if variant.startswith('restored') else 'input_points_world_m'].copy()
            ids = data['source_camera_ids'].copy()
        frames = load_rgb_frames(item['observations_directory'], item['camera_names'], {})
        frames, _ = source_camera_zbuffers(points, ids, frames)
        # Show the actual quantized/reprojected readout input, not a denser cloud.
        points, ids = cloud_from_frames(frames)
        row = next(r for r in read_rows(str(evaluation/'records.jsonl'))
                   if (r['case'],r['condition'],r['variant'],r['backend']) == (case,condition,variant,backend))
        note = '三视联合恢复；展示同来源相机回投、1mm量化后的实际读出输入。未补洞。'
        estimate = row.get('estimate'); radius = (estimate or {}).get('radius_m')
    else:
        condition = 'ideal_rgb' if option.endswith('ideal') else 'stereo_raw'
        predictions = BASE/'runs/capability_lndf_frozen_cos_full_20261027_v1'
        evaluation = BASE/'results/capability_lndf_frozen_cos_full_20261027_v1'
        row = next(r for r in read_rows(str(evaluation/'records.json'))
                   if (r['layout'],r['case'],r['condition']) == (layout,case,condition))
        path = predictions/'predictions'/f'{layout}/{condition}/{case}.npz'
        with np.load(path) as data:
            points = data['input_points_world_m'].copy()
        if len(points) > 6000:
            points = points[np.linspace(0,len(points)-1,6000,dtype=int)]
        ids = np.zeros(len(points),int)
        frames = load_observation_frames(row['observations'])[:1]
        estimate = dict(center_m=row['center_m'], normal=row['normal'])
        radius = None
        note = 'cam_a单视实际输入；仅预测中心/轴，紫线没有半径含义，不绘制预测圆。'
    return dict(title=title, layout=layout, case=case, truth=truth, reference=reference,
                frames=frames, station_cameras=[f.camera for f in reference_frames],
                points=points, ids=ids, row=row, estimate=estimate,
                radius=radius, note=note, prediction_root=str(predictions), evaluation_root=str(evaluation))


class CapabilityViewer:
    def __init__(self, layout='close', case='structural_test_nominal_000', option=OPTIONS[1]):
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        from matplotlib.widgets import RadioButtons, Button, CheckButtons
        font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
        if font.exists():
            fm.fontManager.addfont(str(font))
            plt.rcParams['font.family'] = [fm.FontProperties(fname=str(font)).get_name(), 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        self.plt, self.layout, self.option, self.backend = plt, layout, option, 'portfolio'
        config = read_json(str(BASE/'runs/structural_stage_20261025_v1/close/render_inputs/effective_config.json'))
        self.cases = [r['name'] for r in config['cases']]
        self.index = self.cases.index(case)
        self.truth_on = True
        self.fig = plt.figure(figsize=(17,9), facecolor='#f5f7fa')
        self.full = self.fig.add_axes([.19,.44,.37,.48], projection='3d')
        self.neck = self.fig.add_axes([.58,.44,.37,.48], projection='3d')
        self.full.view_init(elev=18,azim=-65); self.neck.view_init(elev=18,azim=-65)
        self.images = [self.fig.add_axes([.20+i*.255,.17,.24,.24]) for i in range(3)]
        self.info = self.fig.add_axes([.20,.015,.76,.14]); self.info.axis('off')
        self.methods = RadioButtons(self.fig.add_axes([.012,.34,.17,.56]), OPTIONS, active=OPTIONS.index(option))
        self.methods.on_clicked(self.select_method)
        self.inputs = Button(self.fig.add_axes([.012,.91,.17,.045]), '查看实际输入图像')
        self.inputs.on_clicked(lambda event:self.show_inputs())
        self.layouts = RadioButtons(self.fig.add_axes([.012,.23,.17,.09]), ['近距 close','远距 far'], active=0 if layout=='close' else 1)
        self.layouts.on_clicked(self.select_layout)
        self.backends = RadioButtons(self.fig.add_axes([.012,.11,.17,.10]), ['portfolio','stereo_robust'])
        self.backends.on_clicked(self.select_backend)
        self.truth_widget = CheckButtons(self.fig.add_axes([.012,.055,.17,.045]), ['显示理想参考/真值'], [True])
        self.truth_widget.on_clicked(self.toggle_truth)
        self.prev = Button(self.fig.add_axes([.012,.005,.08,.04]), '上一坛 ←')
        self.next = Button(self.fig.add_axes([.10,.005,.08,.04]), '下一坛 →')
        self.prev.on_clicked(lambda event:self.move(-1)); self.next.on_clicked(lambda event:self.move(1))
        self.fig.canvas.mpl_connect('key_press_event', self.key)
        self.draw()

    def select_method(self, value):
        self.option = value; self.draw()

    def select_layout(self, value):
        self.layout = 'close' if 'close' in value else 'far'; self.draw()

    def select_backend(self, value):
        self.backend = value; self.draw()

    def toggle_truth(self, _):
        self.truth_on = not self.truth_on; self.draw()

    def move(self, step):
        self.index = (self.index+step) % len(self.cases); self.draw()

    def key(self, event):
        if event.key in ('right','left'):
            self.move(1 if event.key=='right' else -1)

    def show_inputs(self, snapshot=None):
        import cv2
        stereo = self.option.startswith(('SimSense','FoundationStereo','Ideal IR'))
        if stereo:
            condition='shared_noise' if self.option.endswith('noise') else 'clean'
            pairs=[p for p in read_json(str(BASE/'runs/capability_inputs_20261027_v1/manifest.json'))['pairs']
                   if (p['layout'],p['case'],p['condition']) == (self.layout,self.cases[self.index],condition)]
            fig,axes=self.plt.subplots(3,2,figsize=(12,9))
            for i,pair in enumerate(sorted(pairs,key=lambda p:p['camera'])):
                for j,key in enumerate(('left_path','right_path')):
                    axes[i,j].imshow(cv2.imread(pair[key],cv2.IMREAD_GRAYSCALE),cmap='gray',vmin=0,vmax=255)
                    axes[i,j].set_title(pair['camera']+(' 左IR' if j==0 else '右IR'));axes[i,j].axis('off')
            fig.suptitle(f'实际保存并输入模型的IR图像 · {self.cases[self.index]} · {self.layout} · {condition}')
        else:
            fig,axes=self.plt.subplots(1,len(self.data['frames']),figsize=(12,5),squeeze=False)
            for axis,frame in zip(axes[0],self.data['frames']):
                axis.imshow(frame.rgb);axis.set_title(frame.camera.name+' 原始RGB');axis.axis('off')
            fig.suptitle('当前方法使用RGB-D；RGB如下，深度反投影点云在主窗口显示')
        fig.tight_layout()
        if snapshot:
            fig.savefig(snapshot,dpi=130);self.plt.close(fig)
        else:
            fig.show()

    def draw(self):
        d = load_case(self.layout, self.cases[self.index], self.option, self.backend)
        self.data = d
        truth, estimate = d['truth'], d['estimate']
        true_ring = circle(truth['center_m'], truth['normal'], truth['radius_m'])
        pred_ring = circle(estimate['center_m'], estimate['normal'], d['radius']) if estimate and d['radius'] else None
        is_pose = self.option.startswith('L-NDF')
        color = '#b326b2' if is_pose else '#df7900'
        for axis in (self.full,self.neck):
            angles = axis.elev, axis.azim
            axis.clear()
            if self.truth_on:
                q = d['reference']; axis.scatter(*q.T,c='#929ba4',s=.4,alpha=.13,depthshade=False)
                axis.plot(*true_ring.T,c='#168c3d',lw=2,label='真值截面（仅评价）')
            for i in np.unique(d['ids']):
                q = d['points'][d['ids']==i]
                axis.scatter(*q.T,s=1.0,c=['#4279ba','#c46d32','#945ec0'][int(i)],alpha=.58,depthshade=False)
            if estimate:
                c = np.array(estimate['center_m']); n = np.array(estimate['normal'])
                axis.scatter(*c,c=color,s=32)
                axis.plot(*np.array([c,c+n*.045]).T,c=color,lw=2)
                if pred_ring is not None:
                    axis.plot(*pred_ring.T,c=color,lw=2,ls='--',label='保存的候选圆（可能被拒绝）')
            axis.set_xlabel('X / m'); axis.set_ylabel('Y / m'); axis.set_zlabel('Z / m')
            axis.view_init(elev=angles[0],azim=angles[1])
        self.full.set_title('工位 / 输入前景点云（可鼠标旋转）',fontsize=12)
        camera_positions=[]
        for camera in d['station_cameras']:
            c=camera.T_world_from_camera[:3,3]
            camera_positions.append(c)
            self.full.scatter(*c,marker='^',c='#555555',s=28)
            self.full.text(*c,camera.name,fontsize=8)
            self.full.plot(*np.array([c,c+camera.T_world_from_camera[:3,2]*.07]).T,c='#555555',lw=1)
        extent_points = np.vstack([d['reference'],camera_positions])
        if estimate:
            extent_points = np.vstack([extent_points,estimate['center_m']])
        lo,hi = np.percentile(extent_points,[0,100],axis=0)
        span = max(float(np.max(hi-lo)),.65)
        mid = (lo+hi)/2
        self.full.set(xlim=(mid[0]-span*.55,mid[0]+span*.55),ylim=(mid[1]-span*.55,mid[1]+span*.55),zlim=(mid[2]-span*.55,mid[2]+span*.55))
        self.full.set_box_aspect((1,1,1))
        c = np.array(truth['center_m'])
        self.neck.set(xlim=(c[0]-.075,c[0]+.075),ylim=(c[1]-.075,c[1]+.075),zlim=(c[2]-.055,c[2]+.055))
        self.neck.set_box_aspect((1,1,.73)); self.neck.set_title('颈部放大：绿色真值 / 橙色候选 / 紫色功能帧',fontsize=12)
        if estimate:
            delta=np.abs(np.asarray(estimate['center_m'])-c)
            if np.any(delta > [.075,.075,.055]):
                self.neck.text2D(.04,.94,'预测中心超出放大区域，请看左侧全局图',transform=self.neck.transAxes,color=color,fontsize=10)
        # Shared evaluation-only viewport; never used for model input cropping.
        for i,axis in enumerate(self.images):
            axis.clear(); axis.axis('off')
            if i >= len(d['frames']):
                axis.text(.5,.5,'此方法未使用该相机',ha='center'); continue
            frame = d['frames'][i]
            axis.imshow(frame.rgb)
            uv = frame.camera.project(np.array([truth['center_m']]))[0]
            axis.set_xlim(uv[0]-95,uv[0]+95); axis.set_ylim(uv[1]+65,uv[1]-65)
            if self.truth_on:
                xy=frame.camera.project(true_ring);axis.plot(*xy.T,color='#14d354',lw=1.2)
            if pred_ring is not None:
                xy=frame.camera.project(pred_ring);axis.plot(*xy.T,color=color,lw=1.2,ls='--')
            elif estimate:
                world=np.array([estimate['center_m'],np.array(estimate['center_m'])+np.array(estimate['normal'])*.045])
                local=(world-frame.camera.T_world_from_camera[:3,3])@frame.camera.T_world_from_camera[:3,:3]
                if np.all(local[:,2]>0):
                    xy=frame.camera.project(world)
                    axis.plot(*xy.T,c=color,lw=1.5,marker='o',ms=3)
                    if abs(xy[0,0]-uv[0])>95 or abs(xy[0,1]-uv[1])>65:
                        axis.text(.03,.06,'预测中心超出当前放大图',transform=axis.transAxes,c=color,fontsize=9)
                else:
                    axis.text(.03,.06,'预测坐标系位于相机后方',transform=axis.transAxes,c=color,fontsize=9)
            axis.set_title(frame.camera.name+' RGB（仅展示时放大）',fontsize=11)
        row=d['row']; self.info.clear(); self.info.axis('off')
        def fmt(key):
            value=row.get(key); return '未输出' if value is None else f'{value:.3f}'
        line=f"圆心误差 {fmt('center_mm')} mm    轴/法向误差 {fmt('axis_deg' if is_pose else 'normal_deg')} °    直径误差 {fmt('diameter_mm')} mm"
        if is_pose:
            line+='    仅功能帧，无半径输出'
        else:
            line+=f"    候选几何达标={row.get('truth_pass')}    实际接受={row.get('accepted')}"
        reasons=' / '.join(row.get('reasons',[])) or ('L-NDF不包含几何后端拒绝门槛' if is_pose else '无拒绝原因')
        self.info.text(0,.94,line,fontsize=11,va='top')
        self.info.text(0,.66,d['note'],fontsize=10,va='top')
        self.info.text(0,.40,'拒绝/说明：'+reasons,fontsize=8.5,va='top',wrap=True)
        self.info.text(0,.13,'保存结果回放，不重新训练/推理。灰色理想表面、绿色真值和统一放大区域只用于展示；候选达标不等于工业成功。',fontsize=9,va='top')
        self.fig.suptitle(f"能力审查回放 · {self.index+1}/12 {self.cases[self.index]} · {self.layout} · {self.option}",fontsize=15)
        self.fig.canvas.draw_idle()

    def save(self, destination):
        destination = Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
        self.fig.savefig(destination,dpi=130)
        (destination.with_suffix('.json')).write_text(json.dumps(dict(case=self.data['case'],layout=self.layout,
            option=self.option,backend=self.backend,prediction_root=self.data['prediction_root'],
            evaluation_root=self.data['evaluation_root'],display_only=True,no_inference=True,
            truth_only_for_display=True,saved_result=self.data['row']),ensure_ascii=False,indent=2)+'\n')
