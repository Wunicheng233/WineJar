"""Read saved estimates and render the existing MuJoCo viewer for briefing.

No estimator, geometry, camera input, calibration or source result is changed.
The freely positioned review camera is for presentation, never a new input.
"""
import os
from pathlib import Path
import sys
import json
import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.pop('MUJOCO_GL', None)
import cv2
import mujoco
import numpy as np
from PIL import Image
from neck_localization.tying_viewer import TyingViewer
from neck_localization.inspection_viewer import draw_result

HERE = Path(__file__).resolve().parent
OUT = HERE/'assets'
BASE = ROOT/'experiments/neck_localization/results/cross_station_20261028_v3'


def save(name, pixels):
    Image.fromarray(pixels).save(OUT/(name+'.png'))


def main():
    OUT.mkdir(exist_ok=True)
    datasets = json.loads((BASE/'viewer/datasets.json').read_text())
    d = next(x for x in datasets if x['condition'] == 'new8/fusion/clean')
    v = TyingViewer(d['scenes'], d['records'], 'confirmation_nominal_003',
        method=d['method'], datasets=datasets, station_config=json.loads((BASE/'viewer/config.json').read_text()),
        visible=False, overlay=True, ground_truth=False, band=False, width=1920, height=1080)
    sources = {}
    try:
        def setup_buffer():
            mujoco.mjr_resizeOffscreen(1920, 1440, v.context)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, v.context)

        def camera(look, distance, azimuth=90, elevation=-20):
            v.camera.lookat[:] = look
            v.camera.distance = distance
            v.camera.azimuth = azimuth
            v.camera.elevation = elevation

        def render(name, size=(1600, 1000), markers=False):
            save(name, v._render_view(v.camera, v.scene, *size, camera_props=markers))

        setup_buffer()
        v.show_estimate = False
        v._activate('stage:recognize')
        camera([1.15, -.05, .30], 3.85, 90, -23)
        render('line_overview', (1920, 800), True)
        camera([0, 0, .40], 1.35, 78, -22)
        render('station_1_recognition', markers=True)
        v._activate('stage:cover')
        camera([1.2, 0, .35], 1.30, 90, -20)
        render('station_2_covered')
        for phase in ('above', 'descend', 'close'):
            v._activate('stage:'+phase)
            camera([2.4, -.12, .47], 1.45, 66, -24)
            render('station_3_'+phase)
            camera(np.asarray(v.row['estimate']['center_m'])+np.array([2.4, 0, -.015]), .34, 68, -23)
            render('gun_closeup_'+phase, (1400, 1000))
        v._activate('stage:recognize')
        v.show_estimate = v.show_truth = True
        camera(v.row['estimate']['center_m'], .29, 130, -20)
        render('neck_estimate_vs_truth', (1600, 1000))
        v.point_cloud = True
        v.cloud_crop = True
        camera([0, 0, .55], .50, 130, -18)
        render('stereo_point_cloud', (1200, 1000))
        v.point_cloud = False
        for frame in v.frames:
            save('input_'+frame.camera.name+'_rgb', frame.rgb)
            overlay = frame.rgb.copy()
            draw_result(overlay, frame.camera.project, v.row['estimate'])
            save('input_'+frame.camera.name+'_overlay', overlay)
            valid = np.isfinite(frame.depth) & (frame.depth > 0)
            scalar = np.clip((frame.depth-.15)/.75, 0, 1)
            grey = np.uint8(np.nan_to_num(1-scalar)*255)
            color = cv2.cvtColor(cv2.applyColorMap(grey, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
            color[~valid] = 0
            save('input_'+frame.camera.name+'_depth', color)
        controls = json.loads((BASE/'band_controls.json').read_text())['rows']
        rows = [r for r in controls if r['method'] == 'pose_fs_radius']
        cases = sorted({(r['cohort'], r['case']) for r in rows})
        table = []
        for index, (cohort, case) in enumerate(cases, 1):
            d = next(x for x in datasets if x['condition'] == f'{cohort}/fusion/clean')
            v.root = Path(d['scenes'])
            v.records_path = Path(d['records'])
            from neck_localization.inspection_viewer import read_records
            v.rows = read_records(v.records_path)
            v.cases = sorted({r['case'] for r in v.rows})
            v.case_index = v.cases.index(case)
            v.method = d['method']
            v._load_case()
            setup_buffer()
            v._activate('stage:recognize')
            v.show_truth = False
            v.show_estimate = True
            v.show_band = False
            truth = json.loads((v.folder/'evaluation_only/target.json').read_text())
            camera([0, 0, truth['center_m'][2]*.48], 1.35, 90, -10)
            render(f'jar_{index:02d}', (800, 1100))
            camera(v.row['estimate']['center_m'], .28, 130, -17)
            v.show_truth = True
            render(f'neck_{index:02d}', (1200, 800))
            pairs = [r for r in rows if (r['cohort'], r['case']) == (cohort, case)]
            table.append(dict(id=index, cohort=cohort, case=case,
                center_max_mm=max(r['center_mm'] for r in pairs),
                normal_max_deg=max(r['normal_deg'] for r in pairs),
                diameter_max_mm=max(r['diameter_mm'] for r in pairs),
                band_contained_all=all(r['band_contained'] for r in pairs),
                minimum_band_margin_mm=min(r['band_margin_mm'] for r in pairs)))
            for p in (v.folder/'scene.xml', v.records_path, v.folder/'evaluation_only/target.json'):
                sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        (HERE/'metrics.json').write_text(json.dumps(dict(rows=table,
            n_models=len(table), n_input_conditions=2, n_results=len(rows),
            center_max_mm=max(r['center_mm'] for r in rows),
            normal_max_deg=max(r['normal_deg'] for r in rows),
            diameter_max_mm=max(r['diameter_mm'] for r in rows),
            band_contained_per_condition={c:sum(r['band_contained'] for r in rows if r['condition']==c) for c in ('clean','shared_noise')},
            band_width_mm=3, physical_neck_mm=[6,10], maximum_tilt_deg=15,
            interpretation='Already-seen synthetic models, optional candidate composition; not real fastening success.'), indent=2))
        for p in (BASE/'band_controls.json', Path(__file__)):
            sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        (HERE/'asset_provenance.json').write_text(json.dumps(dict(source_sha256=sources,
            renderer='Existing TyingViewer / MuJoCo; free review cameras only',
            depth_preview_range_m=[.15,.90],depth_preview='Turbo, black pixels are missing depth',
            no_new_estimation_training_or_hardware=True), indent=2))
        print('Captured existing simulation evidence:', len(table), 'models')
    finally:
        v.close()


if __name__ == '__main__':
    main()
