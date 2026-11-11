#!/usr/bin/env python3
"""Paired frozen-estimator axial-depth-bias sensitivity study (offline only)."""
import argparse
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neck_localization.band_geometry import plan_band, evaluate_band
from neck_localization.depth_perturbation import axial_depth_bias
from neck_localization.evaluation import score
from neck_localization.geometry import Estimate
from neck_localization.joint_profile import JointSettings
from neck_localization.task_validated_output import estimate_task_validated_output
from scripts.run_depth_profile_baseline import load_frames


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def export_observations(source, destination, frames, bias):
    """Persist EXACT estimator inputs; immutable original assets are linked."""
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('assets', 'scene.xml', 'evaluation_only', 'model_manifest.json'):
        (destination/name).symlink_to((source/name).resolve(), target_is_directory=(source/name).is_dir())
    obs = destination/'observations'
    obs.mkdir()
    (obs/'camera_true_parameters.json').symlink_to((source/'observations/camera_true_parameters.json').resolve())
    for frame in frames:
        name = frame.camera.name
        (obs/(name+'_rgb.png')).symlink_to((source/'observations'/(name+'_rgb.png')).resolve())
        np.savez_compressed(obs/(name+'_depth_z_m.npz'), depth_z_m=frame.depth_z_m)
    write_json(obs/'sensor_perturbation.json', dict(model='additive_axial_Z_bias', bias_by_camera_m=bias,
               rgb_and_camera_parameters_unchanged=True, assumed_not_measured=True))


def summarize(rows, conditions):
    summaries = []
    for condition in conditions:
        for group in ('all', 'nominal', 'stress'):
            selected = [r for r in rows if r['condition_name'] == condition['name'] and (group == 'all' or r['group'] == group)]
            def success(row, policy):
                return bool(row['accepted'] and row['truth_pass'] and row.get('band')
                            and row['band']['nominal_compatible'] and row['band_evaluation'][policy]['contained'])
            summary = dict(condition_name=condition['name'], bias_by_camera_mm=condition['bias_by_camera_mm'],
                           group=group, n=len(selected), accepted=sum(r['accepted'] for r in selected),
                           rejected=sum(not r['accepted'] for r in selected),
                           correct_circle=sum(r['accepted'] and r['truth_pass'] for r in selected),
                           wrong_accepted_circle=sum(r['accepted'] and not r['truth_pass'] for r in selected),
                           solved_but_rejected_correct_circle=sum(not r['accepted'] and r['truth_pass'] for r in selected))
            for policy in ('near_min_proxy', 'full_annotated_neck'):
                summary[policy+'_task_success'] = sum(success(r, policy) for r in selected)
                summary[policy+'_false_compatible'] = sum(bool(r['accepted'] and r.get('band') and r['band']['nominal_compatible']
                    and not r['band_evaluation'][policy]['contained']) for r in selected)
                vals = [r['band_evaluation'][policy]['minimum_margin_mm'] for r in selected if r.get('band_evaluation')]
                summary[policy+'_minimum_margin_mm'] = float(min(vals)) if vals else None
            for metric in ('center_mm', 'normal_deg', 'diameter_mm', 'normal_component_mm', 'tangent_mm', 'runtime_ms', 'residual'):
                # Include solved rejected cases; NEVER success-only statistics.
                vals = [abs(r[metric]) for r in selected if r.get(metric) is not None]
                for suffix, percentile in [('median', 50), ('p95', 95), ('max', 100)]:
                    summary[metric+'_'+suffix] = float(np.percentile(vals, percentile)) if vals else None
            summary['reasons'] = {reason:sum(reason in r['reasons'] for r in selected)
                                  for reason in sorted({reason for r in selected for reason in r['reasons']})}
            summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'experiments/neck_localization/configs/depth_bias_stage1_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    source_root = ROOT/plan['scenes']
    frozen = json.loads((ROOT/'experiments/neck_localization/results/reliable_output_20261023/v2_freeze.json').read_text())
    for name, digest in frozen['sources'].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'plan.json', plan)
    observed_hashes, rows, input_hashes = {}, [], {}
    stage_sources = ['scripts/run_depth_bias_stage1.py', 'neck_localization/depth_perturbation.py',
                     'neck_localization/band_geometry.py', 'tests/neck_localization/test_depth_perturbation.py']
    stage_hashes = {p:sha256((ROOT/p).read_bytes()).hexdigest() for p in stage_sources}
    for condition in plan['conditions']:
        folder = args.output/condition['name']
        write_json(folder/'settings.json', dict(**asdict(JointSettings()), band_width_mm=plan['band_width_mm'],
                   sensor_perturbation=condition, assumption='fixed additive axial Z, no random error'))
    with (args.output/'records.jsonl').open('w') as journal:
        for case in plan['cases']:
            source = source_root/case['name']
            spec = json.loads((source/'model_manifest.json').read_text())['jar_spec']
            assert spec['neck_length_m'] >= plan['minimum_model_neck_length_m']
            frames = load_frames(source/'observations', 'rgb')
            for path in sorted((source/'observations').glob('*')):
                if path.name.endswith(('_rgb.png', '_depth_z_m.npz')) or path.name == 'camera_true_parameters.json':
                    input_hashes[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
            for condition in plan['conditions']:
                bias = {name:value/1000 for name,value in condition['bias_by_camera_mm'].items()}
                perturbed = axial_depth_bias(frames, bias)
                export_observations(source, args.output/condition['name']/'scenes'/case['name'], perturbed, bias)
                # No model/target data passed to estimator. Plan selection is
                # synthetic experimental design, never a per-object fit prior.
                try:
                    estimate = estimate_task_validated_output(perturbed)
                except Exception as error:
                    estimate = Estimate(reasons=['estimator_exception'], diagnostics=dict(exception_type=type(error).__name__,
                                        exception_message=str(error), traceback=traceback.format_exc()))
                # Propose using observations BEFORE accessing evaluation truth.
                diagnostics = estimate.diagnostics
                required = ('axis_origin_m', 'near_min_interval_from_axis_origin_m')
                band = (plan_band(dict(center_m=estimate.center, normal=estimate.normal, radius_m=estimate.radius), diagnostics,
                                  plan['band_width_mm']/1000)
                        if estimate.center is not None and all(k in diagnostics for k in required) else None)
                truth_path = source/'evaluation_only/target.json'
                truth = json.loads(truth_path.read_text())
                observed_hashes[case['name']] = sha256(truth_path.read_bytes()).hexdigest()
                row = dict(case=case['name'], group=case['group'], split='paired_sensitivity', evidence='rgb',
                           geometry='photo_short_neck', camera_set='all', factor=condition['name'],
                           condition_name=condition['name'], method='task_validated_output', **score(estimate, truth))
                row.setdefault('diagnostics', diagnostics)
                row['band'] = band
                row['band_evaluation'] = ({'near_min_proxy':evaluate_band(band, truth, truth['near_min_interval_s_m']),
                    'full_annotated_neck':evaluate_band(band, truth, truth['search_interval_s_m'])} if band else None)
                if row.get('estimate'):
                    row['signed_diameter_error_mm'] = 2000*(row['estimate']['radius_m']-truth['radius_m'])
                rows.append(row)
                journal.write(json.dumps(row, ensure_ascii=False)+'\n')
                journal.flush()
                print(case['name'], condition['name'], 'accepted='+str(row['accepted']), 'center='+str(row['center_mm']),
                      'diameter='+str(row['diameter_mm']), 'reasons='+str(row['reasons']), flush=True)
    summaries = summarize(rows, plan['conditions'])
    write_json(args.output/'summary.json', summaries)
    write_json(args.output/'provenance.json', dict(plan_sha256=sha256(args.plan.read_bytes()).hexdigest(),
               frozen_estimator_sources=frozen['sources'], stage_sources=stage_hashes, input_sources_sha256=input_hashes,
               evaluation_only_truth_sha256=observed_hashes, no_new_scenes=True, no_parameter_tuning=True,
               scope='12 previously viewed models >=6 mm, paired depth-bias sensitivity; not unseen or real hardware validation'))
    for condition in plan['conditions']:
        selected = [r for r in rows if r['condition_name'] == condition['name']]
        (args.output/condition['name']/'records.json.gz').write_bytes(gzip.compress(json.dumps(selected).encode(), mtime=0))
    (args.output/'records.json.gz').write_bytes(gzip.compress(json.dumps(rows).encode(), mtime=0))
    for name, digest in frozen['sources'].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    assert all(sha256((ROOT/p).read_bytes()).hexdigest() == h for p,h in stage_hashes.items())
    assert all(sha256((ROOT/p).read_bytes()).hexdigest() == h for p,h in input_hashes.items())
    for name in stage_sources:
        target = args.output/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    print('Completed', len(rows), 'paired estimates; input/source hashes verified unchanged', flush=True)


if __name__ == '__main__':
    main()
