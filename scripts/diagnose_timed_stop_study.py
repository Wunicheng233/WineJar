#!/usr/bin/env python3
"""Independent homogeneous-chain replay and axis/normal error decomposition."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from neck_localization.station_tying import GunSpec, frame, pose
from neck_localization.cross_station_uncertainty import executed_circle, evaluate_arrays
from scripts.run_cross_station_study import read, write, digest, calibration_session, quantiles


def run(bundle):
    protocol, axes = read(bundle/'protocol.json'), read(bundle/'axes.json')
    provenance = read(bundle/'provenance.json')
    for path, sha in provenance['input_and_source_sha256'].items():
        assert digest(path) == sha, path
    source = ROOT/protocol['source_bundle']
    station = read(source/'config.json')
    rows = read(bundle/'band_controls.json')['rows']
    tc, tn, lengths = [], [], []
    for row in rows:
        folder = source/row['cohort']/'ss'/row['condition']/'scenes'/row['case']
        truth = read(folder/'evaluation_only/target.json')
        spec = read(folder/'model_manifest.json')['jar_spec']
        tc.append(truth['center_m'])
        tn.append(truth['normal'])
        lengths.append(spec['neck_length_m'])
    tc, tn, lengths = np.array(tc), np.array(tn), np.array(lengths)
    radii = np.array([r['estimate']['radius_m'] for r in rows])
    cal_path = ROOT/protocol['calibration_protocol']
    cal_protocol = deepcopy(read(cal_path))
    for key in ('encoder_scale_ppm', 'encoder_cyclic_amplitude_m', 'stop_repeatability_half_range_m'):
        cal_protocol['assumed_bounds'][key] = 0.
    cal_protocol['assumed_bounds']['encoder_tick_m'] = 1e-12
    B, F = np.array(station['B_T_W']), GunSpec(**station['gun']).T_F_G
    main = dict(np.load(bundle/'conditional_samples.npz'))
    shape = main['center_mm'].shape
    extra = {key: np.empty(shape, np.float32) for key in ('axial_mm', 'tangent_mm', 'normal_deg')}
    extra['xyz_mm'] = np.empty(shape+(3,), np.float32)
    S, K = protocol['calibration_sessions'], protocol['cycles_per_session']
    direction = np.array(protocol['position_error_direction_W'], float)
    direction /= np.linalg.norm(direction)
    np.testing.assert_array_equal(direction, [1., 0., 0.])
    G0 = [pose(frame(r['estimate']['normal'], station['preferred_guide_x_W']), r['estimate']['center_m']) for r in rows]
    maximum_replay_difference = 0.
    for session in range(S):
        cal = calibration_session(np.random.default_rng(np.random.SeedSequence([protocol['seed'], session])), 'combined', cal_protocol, station)
        rng = np.random.default_rng(np.random.SeedSequence([protocol['seed'], session, 99]))
        stops = [rng.uniform(-1, 1, K), rng.normal(size=K)/np.sqrt(3)]
        for j, strategy in enumerate(axes['strategies']):
            if strategy == 'exact_reference_tcp_diagnostic':
                bhat, fhat = B, F
            elif strategy == 'nominal_reference_tcp_timer':
                bhat, fhat = cal['Braw'], cal['Fraw']
            else:
                bhat, fhat = cal['learned']['wide_huber']['B'], cal['learned']['wide_huber']['F']
            # Matrix product per actual guide frame, independently of the main
            # runner's vectorized E/Q translation and normal derivation.
            actual = np.array([executed_circle(g, B, bhat, F, fhat, 2.4, direction) for g in G0])
            for h, stop in enumerate(stops):
                for k, bound in enumerate(axes['ranges_m']):
                    target = tc[None]+(2.4+bound*stop[:, None, None])*direction
                    result = evaluate_arrays(actual[None, :, :3, 3], actual[None, :, :3, 2], radii[None], target, tn[None], lengths[None])
                    for key in ('center_mm', 'band_margin_mm'):
                        expected = main[key][h, k, j, session]
                        maximum_replay_difference = max(maximum_replay_difference, float(np.max(abs(result[key]-expected))))
                        np.testing.assert_allclose(result[key], expected, atol=1e-5, rtol=0)
                    for key in extra:
                        extra[key][h, k, j, session] = result[key]
    # No target rotation was introduced by a stop-position perturbation.
    n = extra['normal_deg']
    np.testing.assert_array_equal(n, np.broadcast_to(n[:1, :1, :, :, :1], n.shape))
    np.savez_compressed(bundle/'error_components.npz', **extra)
    summary = []
    for h, distribution in enumerate(axes['distributions']):
        for k, bound in enumerate(axes['ranges_m']):
            for j, strategy in enumerate(axes['strategies']):
                for method in sorted({r['method'] for r in rows}):
                    ids = [i for i, r in enumerate(rows) if r['method'] == method]
                    group = {key: value[h, k, j][:, :, ids] for key, value in extra.items()}
                    summary.append(dict(distribution=distribution, uniform_equivalent_half_range_mm=bound*1000,
                        strategy=strategy, method=method,
                        absolute_axial_mm=quantiles(abs(group['axial_mm'])),
                        tangent_mm=quantiles(group['tangent_mm']), normal_deg=quantiles(group['normal_deg']),
                        signed_xyz_mean_mm=group['xyz_mm'].mean(axis=(0, 1, 2)),
                        absolute_xyz_p95_mm=np.percentile(abs(group['xyz_mm']), 95, axis=(0, 1, 2)),
                        normal_sign='G +Z towards mouth; oriented normals with angle in [0,180], not abs(dot)'))
    write(bundle/'component_summary.json', summary)
    write(bundle/'component_verification.json', dict(passed=True, original_inputs_unchanged=True,
        independent_homogeneous_chain_replay=True, normal_independent_of_stop_position=True,
        maximum_center_or_band_replay_difference_mm=maximum_replay_difference,
        diagnosis_script_sha256=digest(__file__), final_fastening_success=None))
    print('timed-stop independent matrix replay and component diagnosis passed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    run(p.parse_args().bundle.resolve())
