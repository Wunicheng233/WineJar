#!/usr/bin/env python3
"""Paired band controls, hierarchical virtual calibration, encoder and tool study.

All numerical uncertainty priors in the supplied protocol are assumptions.
No target truth enters a calibration fit, channel composition, or motion plan.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from neck_localization.station_tying import GunSpec, frame, pose, plan_tying
from neck_localization.jar_model import JarSpec
from neck_localization.task_circle_fusion import pose_radius_control
from neck_localization.cross_station_uncertainty import (
    perturb, reference_points, tool_points, fit_rigid_points, fit_encoder_scale,
    encoder_reading, evaluate_arrays)
from neck_localization.tool_clearance import head_bore_certificate
from neck_localization.tying_evaluation import evaluate_transfer
from neck_localization.tying_evaluation import _obj_triangles
from neck_localization.tying_cover import cover_mesh


def encode(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode, allow_nan=False)+'\n')


def read(path):
    p = Path(path)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    return dict(zip(('median', 'p95', 'max'), np.percentile(values, [50, 95, 100]).tolist()))


def band_controls(source, output, tracked):
    """Freeze observation-only predictions first, then evaluate target truth."""
    config, cases, all_rows, results = read(source/'config.json'), {}, [], []
    for cohort in ('old12', 'old8', 'new8'):
        for condition in ('clean', 'shared_noise'):
            fs_path, ss_path = [source/cohort/f/condition/'records.json.gz' for f in ('fs', 'ss')]
            for p in (fs_path, ss_path):
                tracked[str(p)] = digest(p)
            fs, ss = read(fs_path), read(ss_path)
            by_ss = {r['case']: r for r in ss}
            for f in fs:
                s = by_ss[f['case']]
                folder = source/cohort/'ss'/condition/'scenes'/s['case']
                fs_folder = source/cohort/'fs'/condition/'scenes'/s['case']
                # Equal RGB exposures/segmentation provenance, different depths.
                for name in ('cam_a', 'cam_b', 'cam_c'):
                    assert digest(folder/'observations'/f'{name}_rgb.png') == digest(fs_folder/'observations'/f'{name}_rgb.png')
                composed = [pose_radius_control(s, f), pose_radius_control(s, f, refit=True)]
                candidates = [('fs_visible', f), ('ss_sor', s)]+[(r['method'], r) for r in composed]
                # All four predictions exist before evaluator reads truth.
                predictions = [(name, deepcopy(row)) for name, row in candidates]
                truth = read(folder/'evaluation_only/target.json')
                spec = JarSpec(**read(folder/'model_manifest.json')['jar_spec'])
                cases[(cohort, s['case'])] = dict(folder=folder, spec=spec, truth=truth)
                for name, row in predictions:
                    c = row['estimate']
                    value = evaluate_arrays(np.array(c['center_m'])[None], np.array(c['normal'])[None],
                        np.array([c['radius_m']]), np.array(truth['center_m'])[None], np.array(truth['normal'])[None],
                        np.array([spec.neck_length_m]))
                    evaluation = {k: v[0] for k, v in value.items()}
                    evaluation['diameter_mm'] = abs(c['radius_m']-truth['radius_m'])*2000
                    result = dict(cohort=cohort, condition=condition, case=s['case'], method=name,
                        accepted=row['accepted'], reasons=row['reasons'], estimate=c, **evaluation)
                    if 'channel_composition' in row['diagnostics']:
                        result['composition'] = row['diagnostics']['channel_composition']
                    results.append(result)
                    all_rows.append(dict(cohort=cohort, condition=condition, row=row, method=name,
                        truth=truth, spec=spec, source_folder=folder))
    summary = []
    for method in ('fs_visible', 'ss_sor', 'pose_fs_radius', 'pose_fs_radius_refit'):
        for condition in ('clean', 'shared_noise'):
            group = [r for r in results if r['method'] == method and r['condition'] == condition]
            summary.append(dict(method=method, condition=condition, total=len(group),
                accepted=sum(r['accepted'] for r in group), center_false_rejects=sum(not r['accepted'] and r['center_mm'] <= 4 for r in group),
                center_false_accepts=sum(r['accepted'] and r['center_mm'] > 4 for r in group),
                band_contained=sum(r['band_contained'] for r in group),
                max_normal_deg=max(r['normal_deg'] for r in group), center_mm=quantiles([r['center_mm'] for r in group]),
                diameter_mm=quantiles([r['diameter_mm'] for r in group]),
                diameter_pass_2mm=sum(r['diameter_mm'] <= 2 for r in group),
                joint_pose_band_diameter_pass=sum(r['band_contained'] and r['diameter_mm'] <= 2 for r in group),
                radius_fallbacks=sum(r.get('composition', {}).get('radius_fallback', False) for r in group)))
    write(output/'band_controls.json', dict(rows=results, summary=summary,
        split='posthoc_same_11_saved_jars_not_new_independent_test', diameter_2mm_is_research_not_factory_requirement=True))
    return config, all_rows, cases


def calibration_session(rng, scenario, protocol, station):
    b = protocol['assumed_bounds']
    combined = scenario.startswith('combined')
    active = lambda name: combined or scenario == name
    T = np.array(station['B_T_W'])
    F = GunSpec(**station['gun']).T_F_G
    bounded = lambda name, enabled: rng.uniform(-1, 1, len(b[name]))*np.array(b[name]) if enabled else np.zeros(len(b[name]))
    dt = bounded('nominal_registration_translation_m', active('registration'))
    dr = np.deg2rad(bounded('nominal_registration_rotation_deg', active('registration')))
    Braw = perturb(dt, dr) @ T
    Ft = bounded('tool_nominal_translation_m', active('tcp'))
    Fr = np.deg2rad(bounded('tool_nominal_rotation_deg', active('tcp')))
    Fraw = F @ perturb(Ft, Fr)
    common = bounded('survey_common_translation_m', active('survey_bias'))
    scale = rng.uniform(-1, 1)*b['artifact_scale_ppm']*1e-6 if active('survey_bias') else 0.
    sigma = np.array(b['survey_point_std_m']) if (combined or scenario in ('registration', 'survey_bias')) else np.full(3, 1e-10)
    tool_sigma = np.array(b['tool_point_std_m']) if active('tcp') else np.full(3, 1e-10)
    common_tool = bounded('tool_survey_common_translation_m', combined or scenario == 'survey_bias')
    surveys = {}
    for layout in ('wide', 'narrow'):
        p = reference_points(layout)
        q = (p*(1+scale)) @ T[:3, :3].T+T[:3, 3]+common+rng.normal(size=p.shape)*sigma
        if scenario == 'combined_survey_outlier':
            q[-1] += b['survey_outlier_m']
        surveys[layout] = q
    p = tool_points()
    qtool = p*(1+scale) @ F[:3, :3].T+F[:3, 3]+common_tool+rng.normal(size=p.shape)*tool_sigma
    # Line direction survey shares its instrument scale/common translation
    # with registration. Reuse this same measured line for every strategy.
    line = np.array([[0, 0, .35], [2.4, 0, .35]])
    qline = line*(1+scale) @ T[:3, :3].T+T[:3, 3]+common+rng.normal(size=line.shape)*sigma
    scale_encoder = rng.uniform(-1, 1)*b['encoder_scale_ppm']*1e-6 if active('encoder_scale') else 0.
    amp = rng.uniform(0, b['encoder_cyclic_amplitude_m']) if active('encoder_cyclic') else 0.
    phase = rng.uniform(0, 2*np.pi)
    tick = b['encoder_tick_m'] if active('encoder_quantization') else 1e-12
    encoder = dict(scale_error=scale_encoder, cyclic_amplitude_m=amp, phase_rad=phase,
                   cyclic_period_m=b['encoder_cyclic_period_m'], tick_m=tick)
    cal_s = np.linspace(0, 2.4, 13)
    cal_read = encoder_reading(cal_s, **encoder)-encoder_reading(0., **encoder)
    cal_survey = cal_s*(1+scale)+rng.normal(size=cal_s.shape)*sigma[0]
    # Known difference 0 is exactly zero, regardless of common instrument zero.
    cal_survey -= cal_survey[0]
    enc_fit = fit_encoder_scale(cal_read, cal_survey)
    learned = {}
    for layout, robust in [('wide', False), ('wide', True), ('narrow', False)]:
        key = layout+('_huber' if robust else '_ls')
        base = fit_rigid_points(reference_points(layout), surveys[layout], sigma, robust=robust)
        tool = fit_rigid_points(tool_points(), qtool, tool_sigma, robust=robust)
        direction_B = qline[1]-qline[0]
        direction_B /= np.linalg.norm(direction_B)
        direction_W = base['transform'][:3, :3].T @ direction_B
        learned[key] = dict(B=base['transform'], F=tool['transform'], direction=direction_W,
                            encoder_scale=enc_fit, base_fit=base, tool_fit=tool)
    return dict(Braw=Braw, Fraw=Fraw, learned=learned, encoder=encoder,
                calibration_measurements=dict(wide=surveys['wide'], narrow=surveys['narrow'], tool=qtool,
                    line=qline, encoder_readout=cal_read, encoder_survey=cal_survey),
                truth_latents_evaluation_only=dict(registration_translation=dt, registration_rotation_rad=dr,
                    survey_bias=common, artifact_scale=scale, encoder=encoder, tool_translation=Ft, tool_rotation_rad=Fr))


def monte_carlo(protocol, station, rows, output):
    # Same noise draws across counterfactual cases/methods; do not count these
    # as independent physical observations when reporting sample sizes.
    count = len(rows)
    C0 = np.array([x['row']['estimate']['center_m'] for x in rows])
    N0 = np.array([x['row']['estimate']['normal'] for x in rows])
    radii = np.array([x['row']['estimate']['radius_m'] for x in rows])
    rotations = np.array([frame(n, station['preferred_guide_x_W']) for n in N0])
    true_C = np.array([x['truth']['center_m'] for x in rows])
    true_N = np.array([x['truth']['normal'] for x in rows])
    lengths = np.array([x['spec'].neck_length_m for x in rows])
    truth_B, truth_F = np.array(station['B_T_W']), GunSpec(**station['gun']).T_F_G
    world_B = np.linalg.inv(truth_B)
    S, K = protocol['calibration_sessions'], protocol['cycles_per_session']
    conditions, scenarios, strategies = protocol['band_controls'], protocol['scenarios'], protocol['strategies']
    shape = (len(scenarios), len(strategies), S, K, count)
    arrays = {key: np.empty(shape, np.float32) for key in ('center_mm', 'normal_deg', 'axial_mm', 'tangent_mm', 'band_margin_mm')}
    calibration_rows, measurement_arrays, latents = [], {}, []
    holdout = np.array([[.1, -.25, .4], [.9, .15, .6], [2.2, -.15, .58], [2.5, .25, .2]])
    started = perf_counter()
    for scenario_id, scenario in enumerate(scenarios):
        for session in range(S):
            # Scenario-independent seed gives common random-number streams,
            # but enabling an error does not masquerade as new measured data.
            paired_id = protocol.get('scenario_seed_ids', {}).get(scenario, scenario_id)
            rng = np.random.default_rng(np.random.SeedSequence([protocol['seed'], paired_id, session]))
            cal = calibration_session(rng, scenario, protocol, station)
            latents.append(dict(scenario=scenario, session=session, **cal['truth_latents_evaluation_only']))
            for key, value in cal['calibration_measurements'].items():
                measurement_arrays.setdefault(key, []).append(value)
            cycle_rng = np.random.default_rng(np.random.SeedSequence([protocol['seed'], 888, session]))
            actual_D = 2.4+cycle_rng.uniform(-1, 1, K)*protocol['assumed_bounds']['stop_repeatability_half_range_m']*(scenario in ('stop_variation', 'combined', 'combined_survey_outlier'))
            start_encoder = cycle_rng.uniform(0, cal['encoder']['cyclic_period_m'], K)
            measured_D = encoder_reading(start_encoder+actual_D, **cal['encoder'])-encoder_reading(start_encoder, **cal['encoder'])
            for strategy_id, strategy in enumerate(strategies):
                if strategy.startswith('nominal_'):
                    Bhat, Fhat, e = cal['Braw'], cal['Fraw'], np.array([1., 0, 0])
                    Dhat = np.full(K, 2.4) if strategy.endswith('preset_distance') else measured_D
                else:
                    key = 'narrow_ls' if strategy.startswith('narrow') else 'wide_huber' if strategy.endswith('huber') else 'wide_ls'
                    fitted = cal['learned'][key]
                    Bhat, Fhat, e = fitted['B'], fitted['F'], fitted['direction']
                    Dhat = measured_D*fitted['encoder_scale']
                    predicted = holdout @ Bhat[:3, :3].T+Bhat[:3, 3]
                    exact = holdout @ truth_B[:3, :3].T+truth_B[:3, 3]
                    calibration_rows.append(dict(scenario=scenario, session=session, strategy=strategy,
                        base_fit_rms_normalized=fitted['base_fit']['normalized_rms'],
                        base_holdout_max_mm=float(np.linalg.norm(predicted-exact, axis=1).max()*1000),
                        information_condition=fitted['base_fit']['information_condition'],
                        noise_only_covariance=fitted['base_fit']['local_noise_only_covariance']))
                E = world_B @ Bhat
                Q = np.linalg.inv(Fhat) @ truth_F
                # Exact homogeneous chain, vectorized over saved jar predictions.
                local_c = C0+np.einsum('nij,j->ni', rotations, Q[:3, 3])
                local_n = np.einsum('nij,j->ni', rotations, Q[:3, 2])
                actual_n = local_n @ E[:3, :3].T
                for cycle in range(K):
                    actual_c = (local_c+Dhat[cycle]*e) @ E[:3, :3].T+E[:3, 3]
                    target_c = true_C+[actual_D[cycle], 0, 0]
                    result = evaluate_arrays(actual_c, actual_n, radii, target_c, true_N, lengths)
                    for key in arrays:
                        arrays[key][scenario_id, strategy_id, session, cycle] = result[key]
        print('uncertainty', scenario, 'completed', S, 'calibration sessions', flush=True)
    np.savez_compressed(output/'conditional_samples.npz', **arrays)
    np.savez_compressed(output/'calibration_measurements.npz', **{k: np.array(v) for k, v in measurement_arrays.items()})
    write(output/'evaluation_only_session_latents.json', latents)
    write(output/'calibration_validation.json', calibration_rows)
    write(output/'sample_axes.json', dict(scenarios=scenarios, strategies=strategies,
        sessions=S, cycles=K, predictions=[dict(cohort=x['cohort'], case=x['row']['case'], condition=x['condition'], method=x['method']) for x in rows],
        calibration_measurement_order='scenario_major_then_session', independent_units='24 hypothetical calibration sessions; paired counterfactual jar/cycle comparisons',
        runtime_s=perf_counter()-started, status=protocol['uncertainty_status']))
    summary = []
    for i, scenario in enumerate(scenarios):
        for j, strategy in enumerate(strategies):
            for method in conditions:
                ids = [k for k, row in enumerate(rows) if row['method'] == method]
                metrics = {key: array[i, j][:, :, ids] for key, array in arrays.items()}
                summary.append(dict(scenario=scenario, strategy=strategy, method=method,
                    predictions=len(ids), paired_conditional_draws=metrics['center_mm'].size,
                    band_contained_fraction=float(np.mean(metrics['band_margin_mm'] >= -1e-6)),
                    accepted_pose_center_over_4mm_fraction=float(np.mean(metrics['center_mm'] > 4)),
                    center_mm=quantiles(metrics['center_mm']), normal_deg=quantiles(metrics['normal_deg']),
                    minimum_band_margin_mm=float(metrics['band_margin_mm'].min()),
                    between_session_axial_std_mm=float(metrics['axial_mm'].mean(axis=1).std(axis=0).mean()),
                    within_session_axial_std_mm=float(metrics['axial_mm'].std(axis=1).mean()),
                    not_real_world_success_probability=True, final_fastening_success=None))
    write(output/'uncertainty_summary.json', summary)
    return summary


def tool_scan(protocol, station, cases, rows, output):
    results = []
    for (cohort, case), values in cases.items():
        jar_triangles = _obj_triangles(values['folder']/'assets/jar.obj')
        covers = {}
        for amount in protocol['tool_scans']['cover_radial_allowance_m']:
            cover_config = deepcopy(station)
            cover_config['cover_radial_allowance_m'] = amount
            cm = cover_mesh(values['spec'], cover_config)
            covers[amount] = cm['vertices'][cm['faces']] @ cm['rotation'].T+cm['origin']
        for tube in protocol['tool_scans']['head_tube_radius_m']:
            for cover in protocol['tool_scans']['cover_radial_allowance_m']:
                config = deepcopy(station)
                config['gun']['head_tube_radius_m'], config['cover_radial_allowance_m'] = tube, cover
                for method in ('fs_visible', 'ss_sor'):
                    selected = next(x for x in rows if x['cohort'] == cohort and x['row']['case'] == case and x['condition'] == 'clean' and x['method'] == method)
                    row = selected['row']
                    obs = dict(jar_id=cohort+'/'+case, time_s=0, calibration_id='frozen_exact', accepted=row['accepted'],
                               estimate=row['estimate'], reasons=row['reasons'], mouth_seed_m=row['diagnostics']['initial_mouth_center_m'])
                    plan = plan_tying(obs, config)
                    certificate = head_bore_certificate(values['folder']/'assets/jar.obj', values['spec'], config, plan,
                        jar_triangles=jar_triangles, cover_triangles=covers[cover])
                    results.append(dict(cohort=cohort, case=case, method=method, head_tube_radius_mm=tube*1000,
                                        cover_allowance_mm=cover*1000, **certificate))
        print('tool bounds', cohort, case, flush=True)
    write(output/'tool_scan.json', dict(rows=results, assumption_scan_not_measured_material=True,
        status='sufficient_two_jaw_mesh_clearance_only_not_actual_contact_or_locking'))


def export_fusion_viewer(source, destination, station, rows):
    destination.mkdir()
    datasets = read(source/'datasets.json')
    write(destination/'config.json', station)
    for cohort in ('old12', 'old8', 'new8'):
        for condition in ('clean', 'shared_noise'):
            folder = destination/cohort/'fusion'/condition
            folder.mkdir(parents=True)
            records = []
            for item in rows:
                if (item['cohort'], item['condition'], item['method']) != (cohort, condition, 'pose_fs_radius'):
                    continue
                row, src = deepcopy(item['row']), item['source_folder']
                dst = folder/'scenes'/row['case']
                dst.mkdir(parents=True)
                for child in src.iterdir():
                    if child.name not in ('plan.json', 'evaluation.json', 'observation_t0.json'):
                        (dst/child.name).symlink_to(child.resolve(), target_is_directory=child.is_dir())
                obs = read(src/'observation_t0.json')
                obs['estimate'] = row['estimate']
                plan = plan_tying(obs, station)
                original = read(src/'plan.json')
                # Only radius changed: saved SS FK and head-path audit remain
                # the same. Do not silently reuse IK for a different pose.
                np.testing.assert_allclose(plan['W_T_G_t2'], original['W_T_G_t2'], atol=1e-12)
                np.testing.assert_allclose(plan['W_T_above'], original['W_T_above'], atol=1e-12)
                write(dst/'plan.json', plan)
                write(dst/'observation_t0.json', obs)
                e = evaluate_arrays(np.array(row['estimate']['center_m'])[None], np.array(row['estimate']['normal'])[None],
                    np.array([row['estimate']['radius_m']]), np.array(item['truth']['center_m'])[None],
                    np.array(item['truth']['normal'])[None], np.array([item['spec'].neck_length_m]))
                original_evaluation = read(src/'evaluation.json')
                evaluation = evaluate_transfer(plan, item['truth'], item['truth']['search_interval_s_m'], row, station)
                for key in ('head_clearance', 'ik_passed', 'ik_runtime_ms', 'joint_path_max_step_rad'):
                    evaluation[key] = original_evaluation[key]
                evaluation['sampled_kinematic_usable'] = bool(evaluation['physical_band']['contained'] and
                    evaluation['head_clearance']['passed'] and evaluation['ik_passed'])
                evaluation['channel_composition'] = row['diagnostics']['channel_composition']
                evaluation['viewer_pose_cloud_source'] = 'SS pose fit; FS supplies radius only; both channels saved in control audit'
                write(dst/'evaluation.json', evaluation)
                row.update(method='pose_fs_radius', condition_name=f'{cohort}/fusion/{condition}', station_plan=plan,
                           station_evaluation=evaluation, diameter_mm=abs(row['estimate']['radius_m']-item['truth']['radius_m'])*2000)
                row['band_evaluation'] = evaluation['near_min_proxy_band']
                row['truth_pass'] = bool(row['center_mm'] <= 4 and row['normal_deg'] <= 2 and row['diameter_mm'] <= 2)
                row['viewer_truth_pass_criterion'] = 'center_4mm_normal_2deg_diameter_2mm_research_only'
                records.append(row)
            (folder/'records.json.gz').write_bytes(gzip.compress(json.dumps(records, default=encode, allow_nan=False).encode(), mtime=0))
            write(folder/'settings.json', read(source/cohort/'ss'/condition/'settings.json'))
            datasets.append(dict(condition=f'{cohort}/fusion/{condition}', label=cohort+'·姿态/半径组合·'+('原始' if condition == 'clean' else '扰动'),
                scenes=str((folder/'scenes').resolve()), records=str((folder/'records.json.gz').resolve()), method='pose_fs_radius'))
    write(destination/'datasets.json', datasets)


def run(config, output):
    if output.exists():
        raise FileExistsError(output)
    protocol = read(config)
    source = ROOT/protocol['source_bundle']
    original_provenance = read(source/'provenance.json')['input_and_source_sha256']
    assert all(digest(p) == h for p, h in original_provenance.items())
    sources = [config, ROOT/'scripts/run_cross_station_study.py', ROOT/'neck_localization/task_circle_fusion.py',
               ROOT/'neck_localization/cross_station_uncertainty.py', ROOT/'neck_localization/tool_clearance.py']
    tracked = {str(p.resolve()): digest(p) for p in sources}
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    snapshot.mkdir()
    for path in sources:
        relative = path.resolve().relative_to(ROOT)
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write(output/'protocol.json', protocol)
    station, rows, cases = band_controls(source, output, tracked)
    write(output/'observation_predictions.json', [dict(cohort=x['cohort'], condition=x['condition'], method=x['method'],
        case=x['row']['case'], estimate=x['row']['estimate'], accepted=x['row']['accepted']) for x in rows])
    export_fusion_viewer(source, output/'viewer', station, rows)
    monte_carlo(protocol, station, rows, output)
    for amplitude in protocol.get('combined_amplitude_sweep', []):
        if amplitude == 1:
            continue
        scaled = deepcopy(protocol)
        scaled['scenarios'] = ['combined', 'combined_survey_outlier']
        for key, value in scaled['assumed_bounds'].items():
            if key != 'encoder_cyclic_period_m':
                scaled['assumed_bounds'][key] = (np.asarray(value)*amplitude).tolist()
        folder = output/f'combined_amplitude_{amplitude:g}'
        folder.mkdir()
        write(folder/'protocol.json', scaled)
        monte_carlo(scaled, station, rows, folder)
    tool_scan(protocol, station, cases, rows, output)
    assert all(digest(p) == h for p, h in tracked.items())
    assert all(digest(p) == h for p, h in original_provenance.items())
    write(output/'provenance.json', dict(source_sha256=tracked, previous_465_protected_inputs_unchanged=True,
        no_training_or_devices=True, runtime_body_or_carrier_observation=False,
        independent_jar_evaluation=False, all_uncertainty_priors_assumed=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'experiments/neck_localization/configs/cross_station_20261028_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve())
