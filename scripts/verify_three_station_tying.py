#!/usr/bin/env python3
"""Saved-plan FK audit and actual GLFW mouse-callback checks. No devices."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get('MUJOCO_GL') == 'egl':
    del os.environ['MUJOCO_GL']
import glfw
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from neck_localization.tying_viewer import TyingViewer
from neck_localization.tying_scene import OfflineArm
from neck_localization.station_tying import GunSpec
from scripts.verify_viewer_navigation import click


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_click(viewer, phase):
    viewer.draw()
    button = next(b for b in viewer.stage_buttons if b.action == 'stage:'+phase)
    fw, fh = glfw.get_framebuffer_size(viewer.window)
    ww, hh = glfw.get_window_size(viewer.window)
    pos = ((button.rect.x+button.rect.width/2)*ww/fw, (button.rect.y+button.rect.height/2)*hh/fh)
    with patch('glfw.get_cursor_pos', return_value=pos):
        viewer._mouse_button(viewer.window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
    viewer.draw()


def run(bundle, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    config = json.loads((bundle/'config.json').read_text())
    datasets = json.loads((bundle/'datasets.json').read_text())
    protected = json.loads((bundle/'provenance.json').read_text())['input_and_source_sha256']
    assert all(digest(p) == h for p, h in protected.items())
    paths, assets = [], {}
    for dataset in datasets:
        for folder in sorted(Path(dataset['scenes']).iterdir()):
            model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
            arm = OfflineArm(model)
            plan = json.loads((folder/'plan.json').read_text())
            ik = json.loads((folder/'ik_path.json').read_text())
            errors, angles = [], []
            samples = ik['samples']
            for index in range(2*len(samples)-1):
                u = index/(2*len(samples)-2)
                f = u*(len(samples)-1)
                lo, hi = int(f), min(int(f)+1, len(samples)-1)
                q = (1-(f-lo))*np.array(samples[lo]['q_rad'])+(f-lo)*np.array(samples[hi]['q_rad'])
                actual = arm.forward(q)
                desired = np.array(plan['W_T_above'])
                desired[:3, 3] = (1-u)*desired[:3, 3]+u*np.array(plan['W_T_G_t2'])[:3, 3]
                errors.append(np.linalg.norm(actual[:3, 3]-desired[:3, 3]))
                angles.append(np.rad2deg(Rotation.from_matrix(desired[:3, :3] @ actual[:3, :3].T).magnitude()))
            assert max(errors) < .0001 and max(angles) < .05
            # A separate FK composition verifies the saved base/flange/guide contract.
            gun = GunSpec(**config['gun'])
            chained = np.linalg.inv(np.array(config['B_T_W'])) @ np.array(plan['B_T_F']) @ gun.T_F_G
            np.testing.assert_allclose(chained, plan['W_T_G_t2'], atol=1e-12)
            paths.append(dict(condition=dataset['condition'], case=folder.name,
                              max_interpolated_position_error_mm=float(max(errors)*1000),
                              max_interpolated_angle_error_deg=float(max(angles)), checked_poses=len(errors)))
            for asset in ET.parse(folder/'scene.xml').getroot().find('asset'):
                if asset.get('file'):
                    p = Path(asset.get('file')).resolve()
                    assets[str(p)] = digest(p)
    d = next(x for x in datasets if x['condition'] == 'new8/fs/clean')
    viewer = TyingViewer(d['scenes'], d['records'], 'confirmation_nominal_001', method=d['method'],
        datasets=datasets, station_config=config, ground_truth=True, overlay=True, band=True, visible=False)
    interactions = []
    try:
        original_rgb = [f.rgb.copy() for f in viewer.frames]
        original_depth = [f.depth.copy() for f in viewer.frames]
        for phase in ('recognize', 'cover', 'above', 'descend', 'close', 'tighten', 'retract'):
            stage_click(viewer, phase)
            assert viewer.stage_state['phase'] == phase
            for f, rgb, depth in zip(viewer.frames, original_rgb, original_depth):
                np.testing.assert_array_equal(f.rgb, rgb)
                np.testing.assert_array_equal(f.depth, depth)
            viewer.snapshot(output/(phase+'.png'))
            interactions.append('mouse:stage:'+phase)
        click(viewer, 'view:fit')
        stage_click(viewer, 'close')
        viewer.snapshot(output/'recorded_fit_at_tying.png')
        click(viewer, 'input:depth')
        viewer.snapshot(output/'recorded_depth_at_tying.png')
        click(viewer, 'view:model')
        routes = []
        for dataset in datasets:
            click(viewer, 'dataset:'+dataset['condition'])
            for i, case in enumerate(viewer.cases):
                viewer.case_index = i
                viewer._load_case()
                stage_click(viewer, 'close')
                assert viewer.stage_state['covered']
                assert np.isclose(viewer.data.mocap_pos[viewer.model.body_mocapid[viewer.model.body('jar').id], 0], 2.4)
                routes.append(dict(condition=dataset['condition'], case=case))
        # Finite playback finishes, not an endless loop or fake full fastening flag.
        viewer.clock_s = sum(config['durations_s'].values())-.01
        viewer.playing = True
        viewer.last_wall -= .1
        viewer.draw()
        assert not viewer.playing
        glfw.set_window_size(viewer.window, 1000, 680)
        glfw.poll_events()
        stage_click(viewer, 'recognize')
        viewer.snapshot(output/'small_window.png')
    finally:
        viewer.close()
    assert all(digest(p) == h for p, h in protected.items())
    assets[str(ROOT/'mujoco_xarm6/scene/xarm6_right_bare.xml')] = digest(ROOT/'mujoco_xarm6/scene/xarm6_right_bare.xml')
    (output/'validation.json').write_text(json.dumps(dict(passed=True, paths=paths, routes=routes,
        interactions=interactions, protected_files_unchanged=len(protected), input_pixels_unchanged_across_stages=True,
        scope='actual_GLFW_callback_and_framebuffer_FK_audit_not_robot_execution',
        asset_sha256=assets, viewer_source_sha256={str(ROOT/p): digest(ROOT/p) for p in
          ['neck_localization/tying_viewer.py', 'scripts/view_tying_stage.py', 'scripts/verify_three_station_tying.py']}),
        ensure_ascii=False, indent=2)+'\n')
    print('PASS', len(paths), 'paths;', len(routes), 'scene routes;', len(protected), 'unchanged protected files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.viewer.resolve(), args.output.resolve())
