#!/usr/bin/env python3
"""Filter saved jars by physical factory constraints and run offline station study."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from neck_localization.jar_model import JarSpec
from neck_localization.station_tying import GunSpec, jaw_metrics, plan_tying
from neck_localization.tying_scene import build_scene, OfflineArm
from neck_localization.tying_evaluation import factory_filter, HeadClearance, evaluate_transfer


def read(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes())


def write(path, value):
    def encode(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=encode)+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


COHORTS = {
    'old12': ('development_fs12', 'development_ss12', 'foundation_stereo_sam', 'simsense_sam'),
    'old8': ('development_fs8', 'development_ss8', 'foundation_stereo_confirm_sam', 'simsense_confirm_sam'),
    'new8': ('locked_fs8', 'locked_ss8', 'fs_test8', 'ss_test8'),
}


def run(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    config = read(config_path)
    # The first timeline supports straight transport; never silently discard rotation.
    M = np.asarray(config['M_W_t2_t0'])
    if not np.allclose(M[:3, :3], np.eye(3)) or not np.allclose(M[:3, 3][1:], 0):
        raise ValueError('v1 scene/timeline requires straight world-X conveyor')
    output.mkdir(parents=True)
    write(output/'config.json', config)
    protected = {str(config_path): digest(config_path)}
    def track(path):
        path = Path(path).resolve()
        protected[str(path)] = digest(path)
        return path
    results = ROOT/'experiments/neck_localization/results'
    census, records, datasets = [], [], []
    for cohort, (fs, ss, fs_view, ss_view) in COHORTS.items():
        dirs = [results/f'task_surface_{name}_20261028_v1' for name in (fs, ss)]
        provenance = read(track(dirs[0]/'provenance.json'))
        manifest = read(track(provenance['manifest']))
        pairs = {p['case']: p for p in manifest['pairs'] if p['layout'] == 'close'}
        retained = {}
        for case, pair in sorted(pairs.items()):
            model_path = track(Path(pair['render_root'])/'geometry'/case/'model_manifest.json')
            spec = JarSpec(**read(model_path)['jar_spec'])
            domain = config['factory_domain']
            decision = factory_filter(spec, domain['physical_neck_min_m'], domain['physical_neck_max_m'], domain['mouth_tilt_max_deg'])
            census.append(dict(jar_id=f'{cohort}/{case}', cohort=cohort, case=case,
                               source_model=str(model_path), **decision))
            if decision['retained']:
                retained[case] = (spec, pair)
        colliders = {}
        for frontend, result_dir, view_id, backend in [('fs', dirs[0], fs_view, 'visible_profile'),
                                                      ('ss', dirs[1], ss_view, 'standard_sor')]:
            source_records = track(result_dir/'records.json.gz')
            rows = [r for r in read(source_records) if r['layout'] == 'close' and r['backend'] == backend and r['case'] in retained]
            for condition in ('clean', 'shared_noise'):
                route = f'{cohort}/{frontend}/{condition}'
                folder = output/route
                folder.mkdir(parents=True)
                view_bundle = 'task_surface_viewer_20261028_v1' if cohort == 'new8' else 'capability_structural_viewer_sam2_20261027_v1'
                viewer_root = results/view_bundle/'close'/view_id/condition
                original_view_rows = read(track(viewer_root/'records.json.gz'))
                selected = []
                for original in rows:
                    if original['condition'] != condition:
                        continue
                    row = copy.deepcopy(original)
                    case = row['case']
                    reference = next(r for r in original_view_rows if r['case'] == case and r['method'] == backend)
                    if (row['accepted'] != reference['accepted'] or any(not np.allclose(row['estimate'][k], reference['estimate'][k],
                            atol=1e-9, rtol=0) for k in ('center_m', 'normal', 'radius_m'))):
                        raise ValueError('saved viewer prediction differs from frozen baseline')
                    source = viewer_root/'scenes'/case
                    destination = folder/'scenes'/case
                    destination.mkdir(parents=True)
                    for child in ('assets', 'evaluation_only', 'observations', 'foreground_masks', 'model_manifest.json'):
                        p = source/child
                        (destination/child).symlink_to(p.resolve(), target_is_directory=p.is_dir())
                        for f in sorted(p.rglob('*')) if p.is_dir() else [p]:
                            if f.is_file():
                                track(f)
                    spec, pair = retained[case]
                    observation = dict(jar_id=f'{cohort}/{case}', time_s=0., encoder_position_m=0.,
                        calibration_id='synthetic_exact_K_T_frozen_near_layout', accepted=row['accepted'],
                        reasons=row['reasons'], estimate=row['estimate'],
                        mouth_seed_m=row.get('diagnostics', {}).get('initial_mouth_center_m'))
                    write(destination/'observation_t0.json', observation)
                    started = time.perf_counter()
                    plan = plan_tying(observation, config)
                    plan['planning_runtime_ms'] = (time.perf_counter()-started)*1000
                    write(destination/'plan.json', plan)
                    scene = build_scene(track(source/'scene.xml'), destination/'scene.xml', spec, config, ROOT)
                    write(destination/'scene_metadata.json', scene)
                    truth = read(track(source/'evaluation_only/target.json'))
                    evaluation = evaluate_transfer(plan, truth, truth['search_interval_s_m'], row, config)
                    # Explicit oracle branch is ONLY a transform/geometry positive control.
                    oracle_obs = observation | dict(accepted=True, estimate={k: truth[k] for k in ('center_m', 'normal', 'radius_m')},
                        mouth_seed_m=(np.array(truth['center_m'])+.015*np.array(truth['normal'])).tolist())
                    oracle_plan = plan_tying(oracle_obs, config)
                    oracle_row = row | dict(normal_deg=0.)
                    oracle = evaluate_transfer(oracle_plan, truth, truth['search_interval_s_m'], oracle_row, config)
                    evaluation['oracle_geometry_diagnostic'] = {k: oracle[k] for k in ('center_mm', 'normal_deg', 'diameter_mm', 'physical_band')}
                    if case not in colliders:
                        colliders[case] = HeadClearance(source/'assets/jar.obj', spec, config)
                    clearance = colliders[case].evaluate(plan)
                    model = mujoco.MjModel.from_xml_path(str(destination/'scene.xml'))
                    ik_started = time.perf_counter()
                    ik = OfflineArm(model).descent_path(plan)
                    ik['runtime_ms'] = (time.perf_counter()-ik_started)*1000
                    write(destination/'ik_path.json', ik)
                    evaluation.update(head_clearance=clearance, ik_passed=ik['passed'],
                        ik_runtime_ms=ik['runtime_ms'],
                        joint_path_max_step_rad=ik.get('max_joint_step_rad'),
                        sampled_kinematic_usable=bool(evaluation.get('physical_band', {}).get('contained') and clearance['passed'] and ik['passed']))
                    write(destination/'evaluation.json', evaluation)
                    row.update(method=backend, evidence='rgb', condition_name=route, station_plan=plan,
                               station_evaluation=evaluation)
                    selected.append(row)
                    records.append(dict(cohort=cohort, frontend=frontend, condition=condition, case=case,
                        jar_id=observation['jar_id'], accepted=row['accepted'],
                        original_center_mm=row['center_mm'], original_normal_deg=row['normal_deg'],
                        original_diameter_mm=row['diameter_mm'], source_records=str(source_records),
                        original_near_min_band=row['band_evaluation'], plan_status=plan['status'],
                        planning_runtime_ms=plan['planning_runtime_ms'], **evaluation))
                    print(route, case, 'IK', ik['passed'], 'head', clearance['passed'],
                          'band', evaluation.get('physical_band', {}).get('contained'), flush=True)
                (folder/'records.json.gz').write_bytes(gzip.compress(json.dumps(selected, allow_nan=False).encode(), mtime=0))
                write(folder/'settings.json', dict(preprocessing=dict(condition=route), depth_display_range_m=[.2, .9]))
                datasets.append(dict(condition=route, label=f'{cohort}·{frontend}·'+('原始' if condition == 'clean' else '扰动'),
                    scenes=str(folder/'scenes'), records=str(folder/'records.json.gz'), method=backend))
    write(output/'filter_manifest.json', dict(total=len(census), retained=sum(c['retained'] for c in census),
        excluded=sum(not c['retained'] for c in census), entries=census, rule=config['factory_domain'],
        no_source_case_deleted_or_rescaled=True))
    write(output/'datasets.json', datasets)
    write(output/'records.json', records)
    summaries = []
    for frontend in ('fs', 'ss'):
        for condition in ('clean', 'shared_noise'):
            group = [r for r in records if r['frontend'] == frontend and r['condition'] == condition]
            valid = [r for r in group if r['evaluated']]
            summary = dict(frontend=frontend, condition=condition, total=len(group), accepted=sum(r['accepted'] for r in group),
                center_false_rejects=sum(not r['accepted'] and r['original_center_mm'] <= 4 for r in group),
                center_false_accepts=sum(r['accepted'] and r['original_center_mm'] > 4 for r in group),
                physical_band_contained=sum(r['physical_band']['contained'] for r in valid),
                near_min_proxy_contained=sum(r['near_min_proxy_band']['contained'] for r in valid),
                head_clearance_passed=sum(r['head_clearance']['passed'] for r in group),
                ik_passed=sum(r['ik_passed'] for r in group),
                sampled_kinematic_usable=sum(r['sampled_kinematic_usable'] for r in group),
                final_fastening_success=None, failures=[r['jar_id'] for r in group if not r['sampled_kinematic_usable']])
            for key in ('original_center_mm', 'original_normal_deg', 'original_diameter_mm', 'planning_runtime_ms'):
                summary[key] = dict(zip(('median', 'p95', 'max'), np.percentile([r[key] for r in group], [50, 95, 100]).tolist()))
            summary['max_center_error_invariance_m'] = max(r['center_error_invariance_m'] for r in valid)
            summaries.append(summary)
    write(output/'summary.json', dict(groups=summaries, gun_geometry=jaw_metrics(GunSpec(**config['gun'])),
        physical_interval_assumption='entire_model_neck_is_usable_per_user_factory_domain_not_measured_contact_groove',
        final_fastening_not_validated=True))
    for source in ('neck_localization/station_tying.py', 'neck_localization/tying_scene.py',
                   'neck_localization/tying_evaluation.py', 'neck_localization/tying_cover.py', 'scripts/run_three_station_tying.py'):
        track(ROOT/source)
    changed = [p for p, h in protected.items() if digest(p) != h]
    write(output/'provenance.json', dict(config_path=str(config_path), input_and_source_sha256=protected,
        changed_protected_files=changed, zero_added_motion_or_calibration_noise=True,
        no_new_rendering_inference_or_training=True, no_devices=True))
    if changed:
        raise RuntimeError('protected input changed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'experiments/neck_localization/configs/three_station_tying_20261028_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
