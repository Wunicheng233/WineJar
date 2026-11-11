#!/usr/bin/env python3
"""Read-only scoring/audit of the frozen calibration-transfer experiments.

Loads evaluation truth only after fits exist. Never returns data to a solver,
contacts hardware, or issues a process-release decision.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from neck_localization.experiment_io import read, write, digest, quantiles
from neck_localization.robot_error_model import VirtualRobot
from neck_localization.station_tying import pose


COMPARISONS = [
    'confirmation_baselines/records.json',
    'confirmation_poe/records.json',
    'confirmation_poe_block/records.json',
    'confirmation_evaluation_baselines/records.json',
    'confirmation_evaluation_poe/records.json',
    'confirmation_evaluation_poe_block/records.json',
    'partial_rendered_board/records.json',
    'partial_rendered_board/summary.json',
    'partial_rendered_evaluation/records.json',
    'bounded_transport/records.json',
]


def numerical_comparison(a, b):
    """Compare every field except elapsed time and relocated input paths."""
    if isinstance(a, dict):
        return (isinstance(b, dict) and a.keys() == b.keys() and
                all(numerical_comparison(a[k], b[k]) for k in a
                    if k not in ('elapsed_ms', 'source_observations')))
    if isinstance(a, list):
        return (isinstance(b, list) and len(a) == len(b) and
                all(numerical_comparison(x, y) for x, y in zip(a, b)))
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return isinstance(b, (int, float)) and bool(np.isclose(a, b, rtol=0, atol=1e-9))
    return a == b


def rendered_geometry_audit(bundle):
    """Measure mj_saveLastXML rounding against the actual rendered board body."""
    inputs = bundle/'confirmation_inputs'
    folder = bundle/'rendered_board/rendered'
    geometry = read(inputs/'geometry.json')
    C, Y, K = (np.array(geometry[k]) for k in ('W_T_C', 'nominal_F_T_P', 'K'))
    parameters = read(inputs/'evaluation_only/test/2026102981/kinematic_5/truth.json')['actual_robot_parameters']
    robot = VirtualRobot((inputs/'robot_nominal.xml').read_text(), parameters)
    model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
    data = mujoco.MjData(model)
    body = model.body('calibration_board').id
    addresses = model.jnt_qposadr[[model.joint(f'right_joint{i}').id for i in range(1, 7)]]
    points = np.array(read(bundle/'rendered_board/board_points.json')['points_P_m'])
    q = np.load(inputs/'measurements/test/2026102981/kinematic_5/observations.npz')['q_reported']
    def project(T):
        p = points @ T[:3, :3].T + T[:3, 3]
        return p[:, :2]/p[:, 2, None]*[K[0, 0], K[1, 1]] + K[:2, 2]
    distance, angle, pixels = [], [], []
    for values in q:
        data.qpos[addresses] = values*robot.scale + robot.zero
        mujoco.mj_forward(model, data)
        actual = pose(data.xmat[body].reshape(3, 3), data.xpos[body])
        original = robot.forward(values) @ Y
        distance.append(np.linalg.norm(actual[:3, 3] - original[:3, 3])*1000)
        angle.append(np.rad2deg(Rotation.from_matrix(actual[:3, :3] @ original[:3, :3].T).magnitude()))
        pixels.extend(np.linalg.norm(project(np.linalg.inv(C) @ actual) -
                                     project(np.linalg.inv(C) @ original), axis=1))
    return dict(frames=len(q), board_translation_rounding_mm=quantiles(distance),
                board_rotation_rounding_deg=quantiles(angle),
                projected_corner_rounding_px=quantiles(pixels),
                not_a_sensor_or_corner_extraction_accuracy_claim=True)


def run(bundle, output, replay=None):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    records = read(bundle/'analysis/confirmation_records.json')
    detailed = []
    for condition, method in sorted({(r['condition'], r['method']) for r in records}):
        g = [r for r in records if (r['condition'], r['method'], r['tool_policy']) ==
             (condition, method, 'measured_board_jig')]
        d = dict(condition=condition, method=method, total=len(g),
                 solver_rejections=sum(not r['accepted'] for r in g),
                 ik_failures=sum(not r['nominal_IK_passed'] for r in g),
                 actual_center_over_4mm=sum(r['center_mm'] > 4 for r in g),
                 geometric_band_failures=sum(not r['band_contained'] for r in g),
                 numerical_accept_with_bad_band=sum(r['accepted'] and not r['band_contained'] for r in g),
                 process_release_evaluated=False, false_process_rejection=None,
                 false_process_acceptance=None, final_fastening_success=None)
        d.update({k: quantiles([abs(r[k]) for r in g]) for k in
                  ('axial_mm', 'tangent_mm', 'normal_deg', 'center_mm',
                   'calibration_and_execution_only_mm')})
        d['abs_xyz_mm'] = {axis: quantiles([abs(r['xyz_mm'][i]) for r in g])
                           for i, axis in enumerate('xyz')}
        d['minimum_band_margin_mm'] = min(r['band_margin_mm'] for r in g)
        detailed.append(d)
    write(output/'task_metrics.json', detailed)
    simple=bundle/'simplicity_evaluation/records.json'
    if simple.exists():
        simple_rows=read(simple)
        summary=[]
        for condition, method in sorted({(r['condition'],r['method']) for r in simple_rows}):
            g=[r for r in simple_rows if (r['condition'],r['method'],r['tool_policy'])==
               (condition,method,'measured_board_jig')]
            summary.append(dict(condition=condition,method=method,total=len(g),
                                **{k:quantiles([r[k] for r in g]) for k in
                                   ('center_mm','normal_deg','calibration_and_execution_only_mm')},
                                band_contained=sum(r['band_contained'] for r in g),
                                supplementary_after_main_results=True,final_fastening_success=None))
        write(output/'simple_control_summary.json',summary)
    rounding = rendered_geometry_audit(bundle)
    assert rounding['projected_corner_rounding_px']['max'] < .01
    write(output/'rendered_geometry_audit.json', rounding)
    checks = {}
    for name in ('protected_before.json', 'confirmation_lock.json'):
        p = bundle/name
        if p.exists():
            saved = read(p)
            hashes = saved['source_sha256'] if name == 'confirmation_lock.json' else saved
            for path, expected in hashes.items():
                assert digest(path) == expected, path
            checks[name] = len(hashes)
    if replay:
        comparisons = []
        paths=COMPARISONS.copy()
        if simple.exists() and (replay/'simplicity_evaluation/records.json').exists():
            paths+=['simplicity_controls/records.json','simplicity_evaluation/records.json']
        for path in paths:
            same = numerical_comparison(read(bundle/path), read(replay/path))
            comparisons.append(dict(path=path, matches_at_1e_9=same))
            assert same, path
        pngs = list((bundle/'rendered_board/rendered').glob('[0-9][0-9].png'))
        assert len(pngs) == 30
        assert all(digest(p) == digest(replay/'rendered_board/rendered'/p.name) for p in pngs)
        checks.update(replay_tables=comparisons, replay_identical_RGB_frames=len(pngs))
    checks.update(passed=True, source_sha256={str(Path(__file__).resolve()): digest(__file__)},
                  truth_used_only_for_scoring=True)
    write(output/'verification.json', checks)
    print(json.dumps(dict(passed=True, groups=len(detailed),
                          max_rounding_px=rounding['projected_corner_rounding_px']['max'],
                          replay_verified=bool(replay)), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--replay', type=Path)
    a = p.parse_args()
    run(a.bundle.resolve(), a.output.resolve(), a.replay.resolve() if a.replay else None)
