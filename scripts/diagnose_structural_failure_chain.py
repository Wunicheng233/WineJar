#!/usr/bin/env python3
"""Post-hoc failure attribution on saved inputs, NOT a new estimator benchmark.

The frozen estimator is replayed unchanged. A Python return hook captures its
chosen seed's local state. Explicitly labelled counterfactuals freeze its point
pool and weights; oracle axes/depths/regions are diagnostic inputs only.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import least_squares
from neck_localization import structure_portfolio as portfolio
from neck_localization.evaluation import score
from neck_localization.geometry import Estimate
from neck_localization.support_feedback import WeightedAxisSplineProblem
from scripts.report_structural_stage import render_root
from scripts.run_depth_profile_baseline import load_frames


def replay_capture(frames):
    """Capture without wrapping/replacing any estimator operation."""
    captures = []
    def hook(frame, event, arg):
        if event == 'return' and frame.f_code is portfolio.fit_seed.__code__ and isinstance(arg, Estimate):
            captures.append((arg, frame.f_locals.copy()))
    previous = sys.getprofile()
    try:
        sys.setprofile(hook)
        result = portfolio.estimate_portfolio(frames)
    finally:
        sys.setprofile(previous)
    return result, next((v for r, v in captures if r is result), None)


def read_curve(points, ids, origin, normal, joint, weights, interval=None):
    """One fixed-weight fit; no admission and no additional tuning."""
    problem = WeightedAxisSplineProblem(points, origin, normal, ids, joint)
    problem.set_weights(weights)
    state = problem.evaluate(np.zeros(4))
    axial = (points-origin)@normal
    lo, hi = (np.percentile(axial, 2)+.5, np.percentile(axial, 98)-.5) if interval is None else interval
    grid = np.arange(lo, hi, .1)
    if len(grid) < 3:
        raise ValueError('diagnostic_insufficient_range')
    values = problem.basis(grid)@state['beta']
    index = int(np.argmin(values)); left = right = index
    while left > 0 and values[left-1] <= values[index]+1: left -= 1
    while right+1 < len(grid) and values[right+1] <= values[index]+1: right += 1
    midpoint = (grid[left]+grid[right])/2
    radius = float(problem.basis(midpoint)@state['beta'])
    estimate = Estimate(center=(origin+midpoint*normal)/1000, normal=normal,
        radius=radius/1000, accepted=False, reasons=['diagnostic_counterfactual_not_admitted'])
    return estimate, dict(boundary=bool(left == 0 or right == len(grid)-1),
        span_mm=float(grid[right]-grid[left]), points=len(points), objective=float(state['objective']),
        grid_mm=grid.tolist(), radius_mm=values.tolist())


def axis_at_z(truth, z):
    n = np.array(truth['normal']); c = np.array(truth['center_m'])*1000
    return c+(z-c[2])/n[2]*n, n


def ideal_correspondences(points, ids, ideal, truth, ss, rr):
    """Same RGB pixels, ideal rendered Z; not original measured XYZ."""
    clean = np.zeros_like(points); valid = np.zeros(len(points), bool)
    for i, f in enumerate(ideal):
        use = ids == i; uv = np.rint(f.camera.project(points[use]/1000)).astype(int)
        z = f.depth_z_m[uv[:, 1], uv[:, 0]]
        clean[use] = f.camera.unproject(uv, z)*1000
        valid[use] = np.isfinite(z) & (z > 0)
    c = np.array(truth['center_m'])*1000; n = np.array(truth['normal'])
    axial = (clean-c)@n; s = axial/1000+truth['s_m']
    rho = np.linalg.norm(clean-c-axial[:, None]*n, axis=1)/1000
    lo, hi = truth['search_interval_s_m']
    # In the straight-neck range profile s is a true axial coordinate.
    exterior = valid & (s >= lo) & (s <= hi) & (abs(rho-np.interp(s, ss, rr)) < .001)
    return clean, valid, exterior, axial


def metric_stats(x):
    if not len(x): return dict(n=0)
    return dict(n=len(x), mean=float(np.mean(x)), median=float(np.median(x)),
        p95_abs=float(np.percentile(abs(x), 95)), rms=float(np.sqrt(np.mean(x*x))))


def solver_probe(v, truth, seed):
    """Keep final axis objective/weights fixed; compare starts and budget."""
    problem = v['problem']; final = v['x'].copy(); bounds = np.array([15.,15.,.22,.22])
    a, n = axis_at_z(truth, problem.origin[2])
    true_x = np.r_[a[:2]-problem.origin[:2], problem.U.T@n/(problem.n0@n)]
    starts = [('final_restart', final), ('original_zero', np.zeros(4))]
    if np.all(abs(true_x) < bounds): starts.append(('oracle_true_axis', true_x))
    rng = np.random.default_rng(seed)
    for i in range(3):
        starts.append((f'random_{i}', rng.uniform(-.8,.8,4)*bounds))
    original_state = problem.evaluate(final)
    analytic = problem.jac(final)
    numerical = np.empty_like(analytic)
    for j,h in enumerate([1e-4,1e-4,1e-6,1e-6]):
        step = np.eye(4)[j]*h
        numerical[:,j] = (problem.fun(final+step)-problem.fun(final-step))/(2*h)
    rows=[]
    for label,x in starts:
        fit=least_squares(problem.fun,x,jac=problem.jac,bounds=(-bounds,bounds),
            x_scale=[5,5,.1,.1],max_nfev=400,ftol=1e-7,xtol=1e-7,gtol=1e-6)
        st=problem.evaluate(fit.x)
        rows.append(dict(start=label,success=bool(fit.success),nfev=fit.nfev,objective=float(st['objective']),
            normal_deg=float(np.degrees(np.arccos(np.clip(st['normal']@n,-1,1)))),x=fit.x.tolist()))
    return dict(original_objective=float(original_state['objective']),
        oracle_axis_objective=float(problem.evaluate(true_x)['objective']),
        oracle_axis_within_bounds=bool(np.all(abs(true_x)<bounds)),
        jacobian_relative_error=float(np.linalg.norm(analytic-numerical)/max(np.linalg.norm(numerical),1e-12)),
        original_nfev=v['fit'].nfev,original_success=bool(v['fit'].success),restarts=rows)


def run(source, output):
    output.mkdir(parents=True,exist_ok=False)
    hashed={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT/'neck_localization').glob('*.py'))}
    rows=[]; cases=[]; probes=[]
    for layout in ['far','close']:
        folder=render_root(source,layout)
        saved=json.loads(gzip.decompress((source/layout/'main/records.json.gz').read_bytes()))
        selected=[r for r in saved if r['condition_name'] in ['ideal_rgb','stereo_raw'] and r['method']=='portfolio']
        for old in selected:
            case=old['case'];mode=old['condition_name']
            scene=folder/mode/'scenes'/case
            frames=load_frames(scene/'observations','rgb')
            result,v=replay_capture(frames)
            if result.center is None or v is None: raise ValueError(f'Cannot capture saved solved case: {layout}/{mode}/{case}')
            np.testing.assert_allclose(result.center,old['estimate']['center_m'],atol=1e-10,rtol=0)
            np.testing.assert_allclose(result.normal,old['estimate']['normal'],atol=1e-10,rtol=0)
            np.testing.assert_allclose(result.radius,old['estimate']['radius_m'],atol=1e-10,rtol=0)
            assert result.reasons==old['reasons']
            truth=json.loads((scene/'evaluation_only/target.json').read_text())
            with np.load(scene/'evaluation_only/profile.npz') as p:ss,rr=p['s_m'].copy(),p['radius_m'].copy()
            ideal=load_frames(folder/'ideal_rgb/scenes'/case/'observations','rgb')
            points=v['points'][v['training']]; ids=v['ids'][v['training']]
            a,n=v['a'],v['n']; weights=v['read_used_weights']; joint=v['joint']
            clean,valid,exterior,_=ideal_correspondences(points,ids,ideal,truth,ss,rr)
            true_a,true_n=axis_at_z(truth,a[2]);interval=(np.array(truth['search_interval_s_m'])-truth['s_m'])*1000+(np.array(truth['center_m'])*1000-true_a)@true_n
            # Frozen readout grid was determined from train + held-out points.
            # Do not silently replace it by training-only percentiles.
            observed_interval=(float(v['grid'][0]),float(v['grid'][-1]+.05))
            true_axial=(v['points']-true_a)@true_n
            true_axis_interval=(float(np.percentile(true_axial,2)+.5),float(np.percentile(true_axial,98)-.5))
            key=dict(layout=layout,case=case,condition_name=mode,diagnostic_only=True)
            native=score(result,truth)
            rows.append(dict(**key,intervention='frozen_baseline',**native))
            variants=[('readout_replay',points,ids,a,n,weights,observed_interval),
                ('readout_axis_weights',points,ids,a,n,v['axis_used_weights'],observed_interval),
                ('oracle_axis',points,ids,true_a,true_n,weights,true_axis_interval),
                ('oracle_depth_same_pixels',clean[valid],ids[valid],a,n,weights[valid],observed_interval),
                ('oracle_axis_and_depth',clean[valid],ids[valid],true_a,true_n,weights[valid],true_axis_interval),
                ('oracle_axis_outer_support',points[exterior],ids[exterior],true_a,true_n,weights[exterior],true_axis_interval),
                ('oracle_axis_outer_support_search',points[exterior],ids[exterior],true_a,true_n,weights[exterior],interval),
                ('oracle_axis_depth_outer_support_search',clean[exterior],ids[exterior],true_a,true_n,weights[exterior],interval)]
            for name,p,i,origin,normal,w,limits in variants:
                try:
                    if len(p)<30:raise ValueError('insufficient_oracle_selected_points')
                    estimate,details=read_curve(p,i,origin,normal,joint,w,limits)
                    scored=score(estimate,truth)
                    if name=='readout_replay':
                        np.testing.assert_allclose(estimate.center,result.center,atol=1e-10,rtol=0)
                        np.testing.assert_allclose(estimate.radius,result.radius,atol=1e-10,rtol=0)
                    row=dict(**key,intervention=name,**scored,curve=details)
                except (ValueError,np.linalg.LinAlgError) as error:
                    row=dict(**key,intervention=name,error=str(error),truth_pass=False,accepted=False)
                rows.append(row)
            # Diagnose same selected rays at known axis, not after curve fitting.
            dc=clean-true_a;sc=dc@true_n;rc=np.linalg.norm(dc-sc[:,None]*true_n,axis=1)
            dm=points-true_a;sm=dm@true_n;rm=np.linalg.norm(dm-sm[:,None]*true_n,axis=1)
            radial_bias=rm-rc
            near=exterior & (abs((clean-np.array(truth['center_m'])*1000)@true_n)<1.5)
            grid=np.array(result.diagnostics['profile_s_m'])*1000; low,high=result.diagnostics['near_min_interval_from_axis_origin_m']
            entry=dict(**key,selected_train=len(points),ideal_valid=int(valid.sum()),
                ideal_outer_neck_count=int(exterior.sum()),ideal_outer_neck_fraction=float(exterior.mean()),
                near_target_points=int(near.sum()),radial_measured_minus_ideal_mm=metric_stats(radial_bias[exterior]),
                near_target_radial_bias_mm=metric_stats(radial_bias[near]),
                by_camera=[dict(camera=f.camera.name,outer_neck_points=int((exterior&(ids==i)).sum()),
                    bias=metric_stats(radial_bias[exterior&(ids==i)])) for i,f in enumerate(frames)],
                chosen_seed=result.diagnostics['chosen_seed'],baseline_truth_pass=bool(native['truth_pass']),
                baseline_boundary=bool(abs(low*1000-grid[0])<1e-6 or abs(high*1000-grid[-1])<1e-6),
                baseline_span_mm=(high-low)*1000,original_solver_success=bool(v['fit'].success),
                original_solver_nfev=v['fit'].nfev)
            cases.append(entry)
            if mode=='stereo_raw' and ((layout=='far' and case in ['structural_test_stress_003','structural_test_nominal_005']) or (layout=='close' and case in ['structural_test_nominal_000','structural_test_stress_003'])):
                probes.append(dict(**key,**solver_probe(v,truth,2026102607)))
            print(layout,mode,case,'replayed; outer=',entry['ideal_outer_neck_count'],flush=True)
            (output/'progress.json').write_text(json.dumps(dict(completed_cases=len(cases),rows=len(rows)))+'\n')
    after={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in hashed}
    assert after==hashed,'Frozen source changed during diagnosis'
    (output/'records.json.gz').write_bytes(gzip.compress(json.dumps(rows).encode(),mtime=0))
    (output/'case_diagnostics.json').write_text(json.dumps(cases,indent=2)+'\n')
    (output/'solver_probes.json').write_text(json.dumps(probes,indent=2)+'\n')
    summaries=[]
    for layout in ['far','close']:
        for mode in ['ideal_rgb','stereo_raw']:
            for name in sorted(set(r['intervention'] for r in rows)):
                group=[r for r in rows if r['layout']==layout and r['condition_name']==mode and r['intervention']==name]
                row=dict(layout=layout,condition=mode,intervention=name,n=len(group),solved=sum('center_mm' in r for r in group),geometric_pass=sum(r['truth_pass'] for r in group))
                for metric in ['center_mm','normal_deg','diameter_mm']:
                    vals=[r[metric] for r in group if r.get(metric) is not None]
                    row[metric]={k:float(np.percentile(vals,v)) if vals else None for k,v in [('median',50),('p95',95),('max',100)]}
                summaries.append(row)
    (output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (output/'provenance.json').write_text(json.dumps(dict(source=str(source.resolve()),sources=hashed,
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),replayed_cases=len(cases),
        previous_test_now_diagnostic=True,estimator_unchanged=True,renderer_unchanged=True,
        oracle_inputs='Axis, ideal rendered Z at SAME measured RGB pixels, true straight-neck exterior labels and search range; diagnostic only',
        weights='Frozen final readout weights unless axis-weights intervention; no reweighting after oracle changes',
        no_admission_claim=True,settings=asdict(portfolio.PortfolioSettings())),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.run,args.output)
