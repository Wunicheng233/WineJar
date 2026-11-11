#!/usr/bin/env python3
"""User-authorized timed-stop main case: small random position error, no encoder."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from neck_localization.station_tying import GunSpec, frame
from neck_localization.cross_station_uncertainty import evaluate_arrays
from neck_localization.jar_model import centerline
from scripts.run_cross_station_study import (read, write, digest, band_controls, calibration_session, quantiles)


def run(protocol_path, output):
    if output.exists():
        raise FileExistsError(output)
    protocol, tracked = read(protocol_path), {str(protocol_path): digest(protocol_path)}
    source = ROOT/protocol['source_bundle']
    cal_path = ROOT/protocol['calibration_protocol']
    tracked[str(cal_path)] = digest(cal_path)
    cal_protocol = deepcopy(read(cal_path))
    # Encoder mechanisms from the research comparison are disabled/unused.
    for key in ('encoder_scale_ppm', 'encoder_cyclic_amplitude_m', 'stop_repeatability_half_range_m'):
        cal_protocol['assumed_bounds'][key] = 0.
    cal_protocol['assumed_bounds']['encoder_tick_m'] = 1e-12
    output.mkdir(parents=True)
    write(output/'protocol.json', protocol)
    station, rows, _ = band_controls(source, output, tracked)
    C = np.array([r['row']['estimate']['center_m'] for r in rows])
    N = np.array([r['row']['estimate']['normal'] for r in rows])
    R = np.array([frame(n, station['preferred_guide_x_W']) for n in N])
    radii = np.array([r['row']['estimate']['radius_m'] for r in rows])
    tc = np.array([r['truth']['center_m'] for r in rows])
    tn = np.array([r['truth']['normal'] for r in rows])
    lengths = np.array([r['spec'].neck_length_m for r in rows])
    mouth = np.array([centerline(r['spec'], r['spec'].height_m) for r in rows])
    mouth_radius = np.array([r['spec'].lip_radius_m+station['cover_radial_allowance_m'] for r in rows])
    B, F = np.array(station['B_T_W']), GunSpec(**station['gun']).T_F_G
    gun = GunSpec(**station['gun'])
    sag = gun.arc_radius_m*(1-np.cos(np.pi/(2*gun.arc_segments_per_half)))
    S, K = protocol['calibration_sessions'], protocol['cycles_per_session']
    ranges, shapes, strategies = protocol['stop_half_ranges_m'], protocol['distribution_controls'], protocol['strategies']
    array_shape = (len(shapes), len(ranges), len(strategies), S, K, len(rows))
    arrays = {k: np.empty(array_shape, np.float32) for k in ('center_mm', 'band_margin_mm', 'mouth_bore_margin_mm')}
    direction = np.asarray(protocol['position_error_direction_W'], float)
    direction /= np.linalg.norm(direction)
    for session in range(S):
        cal = calibration_session(np.random.default_rng(np.random.SeedSequence([protocol['seed'], session])), 'combined', cal_protocol, station)
        rng = np.random.default_rng(np.random.SeedSequence([protocol['seed'], session, 99]))
        standardized = [rng.uniform(-1, 1, K), rng.normal(size=K)/np.sqrt(3)]
        for j, strategy in enumerate(strategies):
            if strategy == 'exact_reference_tcp_diagnostic':
                bhat, fhat = B, F
            elif strategy == 'nominal_reference_tcp_timer':
                bhat, fhat = cal['Braw'], cal['Fraw']
            else:
                bhat, fhat = cal['learned']['wide_huber']['B'], cal['learned']['wide_huber']['F']
            E, Q = np.linalg.inv(B) @ bhat, np.linalg.inv(fhat) @ F
            actual_c = (C+np.einsum('nij,j->ni', R, Q[:3, 3])+2.4*direction) @ E[:3, :3].T+E[:3, 3]
            actual_n = np.einsum('nij,j->ni', R, Q[:3, 2]) @ E[:3, :3].T
            # No strategy receives the actual stop error. All commands use
            # nominal 2.4m and the frozen recognition observation.
            for shape_id in range(len(shapes)):
                for range_id, half_range in enumerate(ranges):
                    d = 2.4+half_range*standardized[shape_id]
                    truth_centers = tc[None]+d[:, None, None]*direction
                    value = evaluate_arrays(actual_c[None], actual_n[None], radii[None], truth_centers, tn[None], lengths[None])
                    mouth_centers = mouth[None]+d[:, None, None]*direction
                    delta = mouth_centers-actual_c[None]
                    axial = np.sum(delta*actual_n[None], axis=-1)
                    transverse = np.linalg.norm(delta-axial[..., None]*actual_n[None], axis=-1)
                    mouth_margin = (gun.inner_diameter_m/2-sag-mouth_radius[None]-transverse)*1000
                    for key in ('center_mm', 'band_margin_mm'):
                        arrays[key][shape_id, range_id, j, session] = value[key]
                    arrays['mouth_bore_margin_mm'][shape_id, range_id, j, session] = mouth_margin
    np.savez_compressed(output/'conditional_samples.npz', **arrays)
    summary = []
    for h, shape in enumerate(shapes):
        for k, bound in enumerate(ranges):
            for j, strategy in enumerate(strategies):
                for method in ('fs_visible', 'ss_sor', 'pose_fs_radius', 'pose_fs_radius_refit'):
                    ids = [i for i, r in enumerate(rows) if r['method'] == method]
                    a = {key: v[h, k, j][:, :, ids] for key, v in arrays.items()}
                    summary.append(dict(distribution=shape, uniform_equivalent_half_range_mm=bound*1000,
                        strategy=strategy, method=method, center_mm=quantiles(a['center_mm']),
                        band_conditional_fraction=float(np.mean(a['band_margin_mm'] >= -1e-6)),
                        mouth_bore_sufficient_fraction=float(np.mean(a['mouth_bore_margin_mm'] > 0)),
                        mouth_scope='projected_mouth_plus_assumed_cover_only_not_complete_head_collision',
                        final_fastening_success=None, priors_not_measured=True))
    write(output/'summary.json', summary)
    write(output/'axes.json', dict(distributions=shapes, ranges_m=ranges, strategies=strategies, sessions=S, cycles=K,
        predictions=[dict(cohort=r['cohort'], case=r['row']['case'], method=r['method'], condition=r['condition']) for r in rows]))
    tracked[str(Path(__file__).resolve())] = digest(__file__)
    for p, h in tracked.items():
        assert digest(p) == h
    write(output/'provenance.json', dict(input_and_source_sha256=tracked, no_encoder_observation_used=True,
        no_runtime_reobservation=True, user_stop_model_override_applied=True))
    print('timed stop complete', len(summary), 'groups; no encoder or reobservation')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'experiments/neck_localization/configs/timed_stop_20261028_v1.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.config.resolve(), args.output.resolve())
