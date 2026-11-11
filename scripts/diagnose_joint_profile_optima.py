#!/usr/bin/env python3
"""Post-hoc, explicitly oracle-assisted solver/model diagnosis; not evaluation.

Does not alter the frozen estimator. Truth is deliberately used for axis probes,
oracle ROI, and model-consistent synthetic points; none are production results.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neck_localization.joint_profile import (AxisSplineProblem, JointSettings,
    observed_initialization, supported_interval)
from neck_localization.jar_model import JarSpec, outer_radius
from scripts.run_depth_profile_baseline import load_frames
from scripts.run_joint_profile import source_hashes


def normal_error(n, truth):
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(n, truth)), n @ truth)))


def truth_parameters(problem, truth):
    n = np.array(truth['normal'])
    c = np.array(truth['center_m'])*1000
    a = c+(problem.origin[2]-c[2])/n[2]*n
    return np.r_[a[:2]-problem.origin[:2], problem.U.T @ n/(problem.n0 @ n)]


def describe(problem, x, truth):
    state = problem.evaluate(x)
    nd = len(problem.points)
    result = dict(x=np.asarray(x).tolist(), objective_mm2=state['objective'],
        weighted_data_cost_mm2=float(state['residual'][:nd] @ state['residual'][:nd]),
        regularizer_cost_mm2=float(state['residual'][nd:] @ state['residual'][nd:]),
        radial_rms_mm=float(np.sqrt(np.mean(state['radial_residual']**2))),
        normal_deg=normal_error(state['normal'], np.array(truth['normal'])),
        scaled_gradient_norm=float(np.linalg.norm(2*state['jacobian'].T @ state['residual']*
                                                  np.array([5, 5, .1, .1]))))
    try:
        mid, radius, _ = supported_interval(problem, state, problem.points, problem.settings)
        result['center_mm'] = float(np.linalg.norm(state['origin']+mid*state['normal']-
                                                    np.array(truth['center_m'])*1000))
        result['diameter_mm'] = float(abs(radius-truth['radius_m']*1000)*2)
    except Exception as error:
        result['target_rejection'] = str(error)
    return result


def solve(problem, x, truth):
    bounds = np.array([20., 20., .24, .24])
    fit = least_squares(problem.fun, x, jac=problem.jac, bounds=(-bounds, bounds),
        x_scale=[5, 5, .1, .1], max_nfev=300, ftol=1e-11, xtol=1e-11, gtol=1e-9)
    return dict(**describe(problem, fit.x, truth), start=np.asarray(x).tolist(),
                converged=bool(fit.success), status=int(fit.status),
                optimality=float(fit.optimality), nfev=int(fit.nfev))


def curvature_check(problem, x):
    """Numerical full reduced-objective Hessian, not Gauss-Newton J.T@J."""
    scale = np.array([5., 5., .1, .1])
    x = np.array(x)
    def grad(y):
        st = problem.evaluate(y)
        return 2*(st['jacobian'].T @ st['residual'])*scale
    steps = []
    for h in (1e-4, 3e-5):
        H = np.column_stack([(grad(x+np.eye(4)[k]*scale*h)-
                              grad(x-np.eye(4)[k]*scale*h))/(2*h) for k in range(4)])
        steps.append(dict(scaled_step=h, eigenvalues=np.linalg.eigvalsh((H+H.T)/2).tolist(),
                          relative_asymmetry=float(np.linalg.norm(H-H.T)/np.linalg.norm(H))))
    return dict(scaled_gradient_norm=float(np.linalg.norm(grad(x))), steps=steps)


def run_case(folder, camera_names, n_random, seed):
    settings = JointSettings()
    frames = load_frames(folder/'observations', 'rgb', camera_names)
    selected, ids, indices, center, normal, _, _, selection = observed_initialization(frames, settings, 4101)
    points, sources = selected[indices]*1000, ids[indices]
    problem = AxisSplineProblem(points, center*1000, normal, sources, settings)
    truth = json.loads((folder/'evaluation_only/target.json').read_text())
    xtrue = truth_parameters(problem, truth)
    assert normal_error(problem.axis(xtrue)[1], np.array(truth['normal'])) < 1e-10
    assert np.linalg.norm(np.cross(np.array(truth['center_m'])*1000-problem.axis(xtrue)[0],
                                   np.array(truth['normal']))) < 1e-9
    rng = np.random.default_rng(seed)
    starts = [('initial', np.zeros(4)), ('oracle_true_axis', xtrue)]
    for k in range(4):
        for sign in (-1, 1):
            x = np.zeros(4); x[k] = sign*([8., 8., .12, .12][k])
            starts.append((f'coordinate_{k}_{sign}', x))
    for j in range(n_random):
        starts.append((f'random_{j}', rng.uniform(-1, 1, 4)*[12, 12, .18, .18]))
    fits = []
    for name, x in starts:
        fits.append(dict(name=name, **solve(problem, x, truth)))
    best = min(fits, key=lambda f: f['objective_mm2'])
    worse = min((f for f in fits if f['objective_mm2'] > best['objective_mm2']+.001),
                key=lambda f: f['objective_mm2'], default=None)
    # Same point set, all four axis variables interpolated. Profile re-solved
    # for EVERY point. This is a cross-section, not a global minimum proof.
    line = [dict(t=float(t), **describe(problem, (1-t)*xtrue+t*np.array(best['x']), truth))
            for t in np.linspace(-.5, 1.5, 81)]
    # Truth-only ROI: exclude both transition and lip, with 0.3 mm boundary margin.
    nt = np.array(truth['normal']); ct = np.array(truth['center_m'])*1000
    absolute_s = truth['s_m']*1000+(points-ct) @ nt
    lo, hi = np.array(truth['search_interval_s_m'])*1000
    keep = (absolute_s > lo+.3) & (absolute_s < hi-.3)
    cropped = AxisSplineProblem(points[keep], center*1000, normal, sources[keep], settings)
    cropped_truth = describe(cropped, truth_parameters(cropped, truth), truth)
    cropped_fits = [solve(cropped, x, truth) for _, x in starts[:10]]
    regional = []
    for label, mask in [('remove_lower_transition', absolute_s > lo+.3),
                        ('remove_upper_lip', absolute_s < hi-.3)]:
        sub = AxisSplineProblem(points[mask], center*1000, normal, sources[mask], settings)
        subset_fits = [solve(sub, x, truth) for _, x in starts[:2]]
        regional.append(dict(name=label, points=int(mask.sum()),
            true_axis=describe(sub, truth_parameters(sub, truth), truth), fits=subset_fits,
            best=min(subset_fits, key=lambda f: f['objective_mm2'])))
    # Model-consistent analytic quadratic, identical observed axial samples,
    # azimuths and camera IDs. This discards rendered radial geometry entirely.
    atrue, _, _ = problem.axis(xtrue)
    s = (points-atrue) @ nt
    radial = points-atrue-s[:, None]*nt
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    ideal_r = truth['radius_m']*1000+.04*(s-np.mean(s))**2
    ideal_points = atrue+s[:, None]*nt+ideal_r[:, None]*radial
    analytic = AxisSplineProblem(ideal_points, center*1000, normal, sources, settings)
    analytic_fits = [solve(analytic, x, truth) for _, x in starts[:10]]
    spec = JarSpec(**json.loads((folder/'model_manifest.json').read_text())['jar_spec'])
    nominal_r = outer_radius(spec, absolute_s/1000)*1000
    nominal_points = atrue+s[:, None]*nt+nominal_r[:, None]*radial
    nominal = AxisSplineProblem(nominal_points, center*1000, normal, sources, settings)
    nominal_fits = [solve(nominal, x, truth) for _, x in starts[:2]]
    # On REAL rendered input, check derivative and an independent numerical-J solver.
    steps = [1e-4, 1e-4, 1e-6, 1e-6]
    numerical = np.column_stack([(problem.fun(xtrue+np.eye(4)[k]*h)-
                                  problem.fun(xtrue-np.eye(4)[k]*h))/(2*h)
                                 for k, h in enumerate(steps)])
    analytic_jac = problem.jac(xtrue)
    numeric_fit = least_squares(problem.fun, xtrue, jac='3-point',
        bounds=([-20, -20, -.24, -.24], [20, 20, .24, .24]),
        x_scale=[5, 5, .1, .1], max_nfev=300, ftol=1e-11, xtol=1e-11, gtol=1e-9)
    regions = []
    for label, mask in [('below_neck', absolute_s < lo),
                        ('straight_neck', (absolute_s >= lo) & (absolute_s <= hi)),
                        ('above_neck', absolute_s > hi)]:
        costs = []
        for x in [xtrue, np.array(best['x'])]:
            st = problem.evaluate(x)
            costs.append(float(st['residual'][:len(points)][mask] @ st['residual'][:len(points)][mask]))
        regions.append(dict(name=label, count=int(mask.sum()), true_axis_data_cost_mm2=costs[0],
                            best_axis_data_cost_mm2=costs[1]))
    # Data-only local singular values, with profile nuisance directions removed.
    state = problem.evaluate(np.array(best['x']))
    q, sv, _ = np.linalg.svd(state['data_profile_basis'], full_matrices=False)
    q = q[:, sv > sv[0]*1e-8]
    j = state['data_axis_jacobian']; j = (j-q @ (q.T @ j))*[5, 5, .1, .1]
    singular = np.linalg.svd(j, compute_uv=False)
    return dict(case=folder.name, camera_names=camera_names or 'all', fit_points=len(points),
        scene=str(folder.relative_to(ROOT)), input_sha256={str(p.relative_to(folder)): sha256(p.read_bytes()).hexdigest()
            for p in sorted((folder/'observations').iterdir()) if p.suffix in ('.json', '.png', '.npz')},
        selection=selection, oracle_true_axis=describe(problem, xtrue, truth),
        initial_axis=describe(problem, np.zeros(4), truth), fits=fits, best=best,
        truth_to_best_line=line, effective_axis_singular_values=singular.tolist(),
        regional_costs=regions, oracle_one_sided_cuts=regional,
        numerical_jacobian_check=dict(relative_frobenius_error=float(
            np.linalg.norm(numerical-analytic_jac)/np.linalg.norm(numerical)),
            max_absolute_error=float(np.max(np.abs(numerical-analytic_jac))),
            numerical_jacobian_fit=describe(problem, numeric_fit.x, truth)),
        curvature_diagnostics=dict(best=curvature_check(problem, best['x']),
            representative_worse_solution=(dict(fit=worse, curvature=curvature_check(problem, worse['x']))
                                             if worse else None)),
        analytic_nominal_profile=dict(fits=nominal_fits,
            best=min(nominal_fits, key=lambda f: f['objective_mm2']),
            note='Oracle common-axis surface with nominal r(s) at observed s/azimuth; removes mesh and collar departure, keeps sharp profile and sampling'),
        oracle_straight_neck_only=dict(points=int(keep.sum()), true_axis=cropped_truth,
            fits=cropped_fits, best=min(cropped_fits, key=lambda f: f['objective_mm2']),
            note='Oracle ROI; no observable target endpoints; axis diagnosis only'),
        analytic_same_sampling=dict(fits=analytic_fits,
            best=min(analytic_fits, key=lambda f: f['objective_mm2']),
            note='Oracle synthetic quadratic, same s/azimuth/camera; target errors not meaningful'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--random-starts', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20261023)
    args = parser.parse_args()
    frozen = json.loads((ROOT/'experiments/neck_localization/results/joint_profile_20261023/freeze.json').read_text())
    assert frozen['sources'] == source_hashes(), 'Frozen estimator/runner was changed'
    assert frozen['settings'] == asdict(JointSettings()), 'Frozen estimator settings changed'
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [
        ('joint_profile_test_scenes_20261023_v1', 'test_oblique_flat', None),
        ('preparation_20261022_v5', 'reference_shallow', None),
        ('preparation_20261022_v5', 'tilted_neck', None),
        ('short_neck_heldout_20261022_v1', 'broad_reverse_tilt', ['cam_a'])]
    result = dict(kind='post_hoc_ORACLE_DIAGNOSTIC_NOT_TEST', seed=args.seed,
        random_starts=args.random_starts, settings=asdict(JointSettings()),
        source_sha256={name: sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['neck_localization/joint_profile.py', 'neck_localization/depth_profile.py',
             'scripts/diagnose_joint_profile_optima.py']},
        note='All scenes previously inspected. Truth used explicitly for probes/ROI/synthetic points. No tuning of frozen estimator.',
        cases=[])
    started = time.perf_counter()
    for root, name, cameras in cases:
        row = run_case(ROOT/'experiments/neck_localization/runs'/root/name,
                       cameras, args.random_starts, args.seed)
        result['cases'].append(row)
        result['elapsed_s'] = time.perf_counter()-started
        (args.output/'diagnostics.json').write_text(json.dumps(result, indent=2)+'\n')
        print(name, 'truth', row['oracle_true_axis']['objective_mm2'],
              'best', row['best']['objective_mm2'], 'normal', row['best']['normal_deg'],
              'oracle ROI normal', row['oracle_straight_neck_only']['best']['normal_deg'],
              'analytic normal', row['analytic_same_sampling']['best']['normal_deg'], flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, case in zip(axes.flat, result['cases']):
        line = case['truth_to_best_line']; v = np.array([r['objective_mm2'] for r in line])
        ax.plot([r['t'] for r in line], v-v.min(), lw=1.5)
        ax.axvline(0, color='teal', ls='--', label='True axis (oracle probe)')
        ax.axvline(1, color='orangered', ls=':', label='Best fitted axis')
        ax.set(title=case['case'], xlabel='Axis interpolation t; profile refitted at every t',
               ylabel='Objective above line minimum (mm^2)')
        ax.legend(fontsize=8)
    fig.savefig(args.output/'objective_sections.png', dpi=160)


if __name__ == '__main__':
    main()
