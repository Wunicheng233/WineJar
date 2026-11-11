#!/usr/bin/env python3
"""Exercise mouse callback navigation on saved studies and capture the window.

Pointer positions are injected into the real callback; this is an integration
check, not a claim that a human operated the UI. No estimator or hardware runs.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.pop('MUJOCO_GL', None)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import glfw
import numpy as np
from neck_localization.inspection_viewer import InspectionViewer
from neck_localization.viewer_navigation import active_stereo_datasets


def click(viewer, action):
    viewer.draw()
    button = next(b for b in viewer._nav_buttons if b.action == action and b.enabled)
    top, bottom = viewer._nav_clip
    if button.rect.y < top or button.rect.y+button.rect.height > bottom:
        viewer.nav_scroll += button.rect.y-top-8
        viewer.draw()
        button = next(b for b in viewer._nav_buttons if b.action == action and b.enabled)
    fw, fh = glfw.get_framebuffer_size(viewer.window)
    ww, hh = glfw.get_window_size(viewer.window)
    point = ((button.rect.x+button.rect.width/2)*ww/fw, (button.rect.y+button.rect.height/2)*hh/fh)
    with patch('glfw.get_cursor_pos', return_value=point):
        viewer._mouse_button(viewer.window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
    viewer.draw()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    runs = ROOT/'experiments/neck_localization/runs'
    datasets = active_stereo_datasets(runs)
    raw = next(d for d in datasets if d['condition'] == 'stereo_raw')
    protected = {Path(d['records']) for d in datasets}
    for dataset in datasets:
        for case in ['test_nominal_000', 'test_nominal_001']:
            for path in (Path(dataset['scenes'])/case/'observations').glob('*'):
                if path.is_file():
                    protected.add(path)
    hashes = {str(p):sha256(p.read_bytes()).hexdigest() for p in protected}
    actions = []
    viewer = InspectionViewer(raw['scenes'], raw['records'], 'test_nominal_000', method=raw['method'],
        visible=False, point_cloud=True, large_neck=True, overlay=True, datasets=datasets)
    try:
        before = np.array(viewer.camera.lookat).copy()
        for action in ['view:fit', 'input:depth', 'dataset:stereo_spatial', 'toggle_truth', 'toggle_fit_pixels']:
            click(viewer, action); actions.append(action)
        assert viewer.cloud_local and viewer.depth_mode and viewer.show_truth
        assert viewer.show_fit_pixels
        assert '/stereo_spatial/' in str(viewer.records_path)
        assert viewer.cases[viewer.case_index] == 'test_nominal_000'
        np.testing.assert_array_equal(before, viewer.camera.lookat)
        viewer.snapshot(args.output/'filtered_fit_depth_sidebar.png')
        original_rgb = [frame.rgb.copy() for frame in viewer.frames]
        for condition in ['stereo_gp', 'stereo_gp_spatial', 'ideal_ir_registered', 'ideal_rgb']:
            click(viewer, f'dataset:{condition}'); actions.append(f'dataset:{condition}')
            assert viewer.cases[viewer.case_index] == 'test_nominal_000'
            for frame, rgb in zip(viewer.frames, original_rgb):
                np.testing.assert_array_equal(frame.rgb, rgb)
            np.testing.assert_array_equal(before, viewer.camera.lookat)
        assert viewer.row['accepted']
        for action in ['input:rgb', 'color:weights', 'next', 'previous']:
            click(viewer, action); actions.append(action)
        assert viewer.cases[viewer.case_index] == 'test_nominal_000'
        viewer.snapshot(args.output/'ideal_fit_weights_sidebar.png')
        glfw.set_window_size(viewer.window, 1000, 680); glfw.poll_events(); viewer.draw()
        click(viewer, 'reset'); actions.append('small_window:reset')
        viewer.snapshot(args.output/'small_window_sidebar.png')
        assert viewer.nav_scroll > 0
    finally:
        viewer.close()
    native = active_stereo_datasets(runs, 'native')
    source = next(d for d in native if d['condition'] == 'native_raw')
    viewer = InspectionViewer(source['scenes'], source['records'], 'development_stress_000', method=source['method'],
        visible=False, point_cloud=True, large_neck=True, datasets=native)
    try:
        click(viewer, 'view:fit')
        assert not viewer._local_cloud_available
        assert len(viewer._cloud_data[0]) == 0, 'Missing fitted points must not silently fall back to the full jar'
        viewer.snapshot(args.output/'native_initialization_no_fit.png')
        actions.append('native_initialization_failure:no_fabricated_fit_cloud')
    finally:
        viewer.close()
    assert all(sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    report = dict(scope='display-only integration; mouse callback receives injected pointer coordinates',
        actions=actions, condition_count=len(datasets), native_count=len(native),
        same_case_and_view_pose_preserved=True, RGB_unchanged_across_conditions=True,
        protected_input_file_count=len(hashes), protected_inputs_unchanged=True,
        no_fallback_when_fit_absent=True, rendered_snapshots='actual window back-buffer readback',
        sources={n:sha256((ROOT/n).read_bytes()).hexdigest() for n in
            ['neck_localization/inspection_viewer.py', 'neck_localization/viewer_navigation.py']})
    (args.output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
