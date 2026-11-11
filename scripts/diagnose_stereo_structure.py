#!/usr/bin/env python3
"""Explicit ORACLE diagnosis: keep measured depths, change seed/support only.

NOT an estimator result. Synthetic geometry is used in the two labelled oracle
interventions. Neither substitutes true coordinates for a measured 3D point.
"""
import argparse
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from neck_localization import stereo_robust as old
from neck_localization.depth_profile import ProfileSettings, ProfileRejection
from neck_localization.evaluation import score
from neck_localization.joint_profile import JointSettings
from neck_localization.support_feedback import WeightedAxisSplineProblem
from scripts.run_depth_profile_baseline import load_frames


def cloud_and_ids(frames, masks):
    points, ids = [], []
    for i, (f, mask) in enumerate(zip(frames, masks)):
        yy, xx = np.nonzero(mask & (f.depth_z_m > 0) & np.isfinite(f.depth_z_m))
        q = f.camera.unproject(np.c_[xx, yy], f.depth_z_m[yy, xx])
        keep = (q[:, 2] > .4) & (q[:, 2] < .76) & (np.linalg.norm(q[:, :2], axis=1) < .28)
        points.append(q[keep]); ids.append(np.full(keep.sum(), i))
    return np.concatenate(points), np.concatenate(ids)


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((source/'effective_config.json').read_text())
    original_prepare = old._prepare
    rows = []
    for case in config['cases']:
        name = case['name']
        geometry = source/'geometry'/name
        truth = json.loads((geometry/'evaluation_only/target.json').read_text())
        with np.load(geometry/'evaluation_only/profile.npz') as p:
            ss, rr = p['s_m'].copy(), p['radius_m'].copy()
            mouth = p['centerline_m'][-1].copy()
        n = np.array(truth['normal'])
        ideal = load_frames(source/'ideal_rgb/scenes'/name/'observations', 'rgb')
        region_masks = []
        for f in ideal:
            yy, xx = np.indices(f.depth_z_m.shape)
            q = f.camera.unproject(np.c_[xx.ravel(), yy.ravel()], f.depth_z_m.ravel()).reshape(*xx.shape, 3)
            d = q-truth['center_m']; s = d@n+truth['s_m']
            rho = np.linalg.norm(d-(d@n)[..., None]*n, axis=-1)
            lo, hi = truth['search_interval_s_m']
            region_masks.append((f.depth_z_m > 0) & (s > lo-.005) & (s < hi+.006) &
                                (abs(rho-np.interp(s, ss, rr)) < .001))
        for mode in ['ideal_rgb', 'stereo_raw']:
            frames = load_frames(source/mode/'scenes'/name/'observations', 'rgb')
            for intervention in ['observed', 'oracle_seed', 'oracle_region', 'oracle_both']:
                seed_oracle = intervention in ['oracle_seed', 'oracle_both']
                region_oracle = intervention in ['oracle_region', 'oracle_both']

                def prepare(f, masks, depth):
                    if not region_oracle:
                        if not seed_oracle:
                            return original_prepare(f, masks, depth)
                        with patch.object(old, '_mouth_plane', return_value=(mouth, n, np.zeros((100, 3)))), \
                             patch.object(old, '_mouth_center', return_value=(mouth, .09)):
                            return original_prepare(f, masks, depth)
                    cloud, _ = cloud_and_ids(f, masks)
                    if seed_oracle:
                        origin, normal, lip = mouth, n, .09
                    else:
                        o, normal, inliers = old._mouth_plane(cloud, ProfileSettings(), np.random.default_rng(4101))
                        origin, lip = old._mouth_center(o, normal, inliers)
                    selected, ids = cloud_and_ids(f, [a & b for a, b in zip(masks, region_masks)])
                    if len(selected) < 200:
                        raise ProfileRejection('oracle_region_insufficient_observed_depth')
                    indices = np.concatenate([idx[np.linspace(0, len(idx)-1, min(len(idx), 2500)).astype(int)]
                        for i in range(len(f)) if len(idx := np.flatnonzero(ids == i))])
                    held, train = indices[::4], indices[np.arange(len(indices)) % 4 != 0]
                    joint = replace(JointSettings(), fit_depth_mm=depth)
                    problem = WeightedAxisSplineProblem(selected[train]*1000, origin*1000, normal, ids[train], joint)
                    problem.set_weights(np.ones(len(train)))
                    return dict(problem=problem, joint=joint, selected=selected, ids=ids, train=train, held=held,
                        origin=origin, normal=normal, lip=lip, counts={}, selection=dict(
                            diagnostic_oracle_support=True, support='ideal ray identifies exterior; depth remains measured'))

                with patch.object(old, '_prepare', side_effect=prepare):
                    estimate = old.estimate_stereo_robust(frames)
                row = dict(case=name, condition_name=mode, intervention=intervention,
                           diagnostic_only=True, **score(estimate, truth))
                row.setdefault('diagnostics', estimate.diagnostics)
                rows.append(row)
                print(name, mode, intervention, row['accepted'], row.get('center_mm'),
                      row.get('normal_deg'), row.get('diameter_mm'), row['reasons'], flush=True)
                with (output/'records.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
    (output/'records.json.gz').write_bytes(gzip.compress(json.dumps(rows).encode(), mtime=0))
    (output/'provenance.json').write_text(json.dumps(dict(source=str(source.resolve()),
        diagnostic_only=True, oracle_lip_radius_m=.09,
        oracle_radius_semantics='broad permissive bound to isolate seed/region; not measured or claimed deployable',
        measured_points_never_replaced=True, core_solver_and_gates_unchanged=True), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.run, a.output)
