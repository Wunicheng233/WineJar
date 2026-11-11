#!/usr/bin/env python3
"""Check saved capability routes with the original MuJoCo GL viewer.

Only replay and injected mouse callbacks run. Model/estimator inference and
hardware access are intentionally absent. Screenshots read the presented GL
back buffer; they are not CPU-only composites.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

os.environ.pop('MUJOCO_GL', None)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from neck_localization.inspection_viewer import InspectionViewer
from scripts.verify_viewer_navigation import click
from scripts.view_structural_study import select_dataset, DEFAULT, CAPABILITY_DEFAULT


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def run(bundle, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest, datasets, selected = select_dataset(bundle)
    validation = json.loads((bundle/'validation.json').read_text())
    assert validation['passed'] and validation['no_estimator_calls']
    protected = dict(validation['input_sha256'])
    protected.update({str(Path(d['records'])): digest(d['records']) for d in datasets})
    protected[str(manifest)] = digest(manifest)
    assert all(digest(p) == h for p, h in protected.items()), 'Saved inputs changed before replay'
    assert selected['condition'] == 'close/foundation_stereo/clean'
    assert selected['method'] == 'visible_profile'
    assert select_dataset()[0] == CAPABILITY_DEFAULT/'datasets.json'
    checks, actions, selected_examples = [], [], []
    viewer = InspectionViewer(selected['scenes'], selected['records'], 'structural_test_nominal_000',
        method=selected['method'], visible=False, point_cloud=False, large_neck=True,
        overlay=True, ground_truth=True, datasets=datasets)

    def act(action):
        click(viewer, action)
        actions.append(action)

    def case(name):
        for _ in range(len(viewer.cases)):
            if viewer.cases[viewer.case_index] == name:
                return
            act('next')
        raise AssertionError('Case missing: '+name)

    def save(filename):
        viewer.nav_scroll = 0
        viewer.snapshot(output/filename)
        row = viewer.row
        selected_examples.append(dict(snapshot=filename, condition=next(
            d['condition'] for d in datasets if Path(d['records']) == viewer.records_path),
            case=row['case'], method=row['method'], accepted=row['accepted'],
            center_mm=row.get('center_mm'), normal_deg=row.get('normal_deg'),
            diameter_mm=row.get('diameter_mm'), reasons=row['reasons']))

    try:
        # Match --rejected-candidate in the original CLI; the button is disabled
        # for an accepted row, but this preference persists across cases.
        viewer.show_rejected_candidate = True
        viewer.draw()
        top, bottom = viewer._nav_clip
        method_buttons = [b for b in viewer._nav_buttons if b.action.startswith('method:')]
        assert len(method_buttons) == 4
        assert all(top <= b.rect.y and b.rect.y+b.rect.height <= bottom for b in method_buttons)
        assert not viewer.point_cloud
        save('mujoco_default_entity_methods_visible.png')
        act('view:fit')
        act('color:weights')
        act('toggle_fit_pixels')
        save('mujoco_foundation_corrected_near.png')
        for dataset in datasets:
            old_case = viewer.cases[viewer.case_index]
            old_pose = np.array(viewer.camera.lookat).copy()
            act('dataset:'+dataset['condition'])
            if old_case == viewer.cases[viewer.case_index]:
                np.testing.assert_array_equal(viewer.camera.lookat, old_pose)
            assert viewer.row['camera_set'] == 'all'
            for method in list(viewer.methods):
                act('method:'+method)
                assert viewer.method == method
                for mode in ('input:depth', 'input:rgb', 'view:model', 'view:cloud', 'view:fit'):
                    act(mode)
                count = len(viewer.row.get('diagnostics', {}).get('fit_point_positions_m', []))
                if count:
                    assert viewer._local_cloud_available
                    assert len(viewer._cloud_data[0]) > 0
                else:
                    assert not viewer._local_cloud_available
                    assert len(viewer._cloud_data[0]) == 0
                checks.append(dict(condition=dataset['condition'], method=method,
                    case=viewer.row['case'], saved_fit_points=count,
                    RGB_depth_model_cloud_and_fit=True,
                    no_fabricated_fit_when_absent=True))
        act('dataset:close/foundation_stereo_confirm/shared_noise')
        act('method:visible_profile')
        case('confirmation_nominal_004')
        assert not viewer.row['accepted'] and viewer.row['center_mm'] < 1
        act('toggle_rejected')
        act('toggle_rejected')
        save('mujoco_new8_004_shared_rejected.png')
        case('confirmation_nominal_005')
        assert viewer.row['accepted'] and viewer.row['normal_deg'] > 2
        save('mujoco_new8_005_shared_axis.png')
        act('dataset:close/foundation_stereo/shared_noise')
        act('method:standard_sor')
        case('structural_test_stress_001')
        assert viewer.row['accepted'] and viewer.row['center_mm'] > 15
        act('input:depth')
        save('mujoco_standard_sor_shared_center_error.png')
        act('dataset:far/foundation_stereo/clean')
        act('method:visible_profile')
        save('mujoco_foundation_far.png')
    finally:
        viewer.close()

    old_manifest, old_datasets, old = select_dataset(DEFAULT)
    legacy = InspectionViewer(old['scenes'], old['records'], 'structural_test_nominal_000',
        method=old.get('method', 'portfolio'), visible=False, point_cloud=True,
        large_neck=True, ground_truth=True, datasets=old_datasets)
    try:
        legacy.show_rejected_candidate = True
        click(legacy, 'view:fit')
        legacy.snapshot(output/'mujoco_explicit_legacy.png')
    finally:
        legacy.close()
    assert all(digest(p) == h for p, h in protected.items()), 'Saved inputs changed during replay'
    report = dict(passed=True, bundle=str(bundle), default_condition=selected['condition'],
        default_method=selected['method'], route_count=len(datasets), checks=checks,
        selected_examples=selected_examples, actions=actions,
        source_files_checked=len(protected), sources_unchanged=True,
        original_entry_point='scripts/view_structural_study.py --ground-truth',
        explicit_legacy_manifest=str(old_manifest), explicit_legacy_bundle_passed=True,
        default_entity_visible=True, default_all_four_method_buttons_visible=True,
        all_GL_windows_closed=True, no_inference=True, no_hardware=True,
        interaction='Actual GLFW mouse callbacks with injected pointer coordinates',
        capture='Actual presented GLFW back buffer readback',
        scope='Post-hoc visualization of saved results; selected examples are not a new evaluation')
    (output/'interaction_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(passed=True, route_count=len(datasets), method_route_checks=len(checks),
        examples=selected_examples, output=str(output)), ensure_ascii=False, indent=2))


def run_sam2(bundle, output):
    """Check the separate paired SAM2 bundle, preserving the main default."""
    output.mkdir(parents=True, exist_ok=False)
    manifest, datasets, selected = select_dataset(bundle)
    assert selected['condition'] == 'close/foundation_stereo_sam/clean'
    assert select_dataset()[0] == CAPABILITY_DEFAULT/'datasets.json'
    validation = json.loads((bundle/'validation.json').read_text())
    assert validation['passed'] and validation['no_estimator_calls']
    protected = dict(validation['input_sha256'])
    protected.update({str(Path(d['records'])): digest(d['records']) for d in datasets})
    protected[str(manifest)] = digest(manifest)
    assert all(digest(p) == h for p, h in protected.items())
    checks, actions = [], []
    viewer = InspectionViewer(selected['scenes'], selected['records'], 'structural_test_nominal_000',
        method=selected['method'], visible=False, point_cloud=False, large_neck=True,
        overlay=True, ground_truth=True, datasets=datasets)
    try:
        viewer.show_rejected_candidate = True
        viewer.draw()
        top, bottom = viewer._nav_clip
        buttons = [b for b in viewer._nav_buttons if b.action.startswith('method:')]
        assert len(buttons) == 2 and all(top <= b.rect.y and b.rect.y+b.rect.height <= bottom for b in buttons)
        viewer.snapshot(output/'mujoco_foundation_sam2_entity.png')
        for dataset in datasets:
            assert dataset['condition'].startswith('close/') and '_sam/' in dataset['condition']
            click(viewer, 'dataset:'+dataset['condition'])
            for method in list(viewer.methods):
                for action in ('method:'+method, 'input:depth', 'view:fit', 'input:rgb', 'view:model'):
                    click(viewer, action)
                    actions.append(action)
                    if action == 'view:fit':
                        assert viewer._local_cloud_available and len(viewer._cloud_data[0]) > 0
                foreground = viewer.row['diagnostics']['foreground']
                assert 'sam' in foreground['method'].lower()
                checks.append(dict(condition=dataset['condition'], case=viewer.row['case'], method=method,
                    saved_SAM2_masks=True, RGB_depth_model_and_saved_fit_points=True))
        for condition, target, filename in [
            ('close/foundation_stereo_confirm_sam/shared_noise', 'confirmation_nominal_005', 'mujoco_sam2_new8_005_band.png'),
            ('close/foundation_stereo_confirm_sam/clean', 'confirmation_nominal_006', 'mujoco_sam2_new8_006_rejected.png'),
        ]:
            click(viewer, 'dataset:'+condition)
            click(viewer, 'method:visible_profile')
            for _ in range(len(viewer.cases)):
                if viewer.row['case'] == target:
                    break
                click(viewer, 'next')
            viewer.show_band = True
            viewer.nav_scroll = 0
            viewer.snapshot(output/filename)
        click(viewer, 'dataset:close/foundation_stereo_sam/clean')
        click(viewer, 'method:visible_profile')
        for _ in range(len(viewer.cases)):
            if viewer.row['case'] == 'structural_test_nominal_000':
                break
            click(viewer, 'next')
        click(viewer, 'color:weights')
        viewer.show_band = False
        viewer.nav_scroll = 0
        viewer.snapshot(output/'mujoco_foundation_sam2_fit_weights.png')
    finally:
        viewer.close()
    assert all(digest(p) == h for p, h in protected.items())
    report = dict(passed=True, bundle=str(bundle), route_count=len(datasets), checks=checks,
        actions=actions, sources_unchanged=True, protected_source_files=len(protected),
        main_v1_default_unchanged=True, SAM2_bundle_explicit_only=True, all_GL_windows_closed=True,
        capture='Actual presented GLFW back buffer readback',
        interaction='Actual GLFW mouse callbacks with injected pointer coordinates',
        no_inference=True, no_hardware=True,
        scope='Replay of paired RGB segmentation adaptation; does not constitute unseen validation')
    (output/'interaction_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(passed=True, routes=len(datasets), method_routes=len(checks), output=str(output)), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=CAPABILITY_DEFAULT)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sam2', action='store_true', help='Check a separate two-method paired SAM2 replay bundle')
    a = p.parse_args()
    (run_sam2 if a.sam2 else run)(a.bundle.resolve(), a.output.resolve())
