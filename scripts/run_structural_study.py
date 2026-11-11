#!/usr/bin/env python3
"""Observation-only structural alternatives, evaluated AFTER inference."""
import argparse
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from neck_localization.band_geometry import plan_band, evaluate_band
from neck_localization.evaluation import score, summarize
from neck_localization.geometry import Estimate
from neck_localization.stereo_robust import estimate_stereo_robust
from neck_localization.structural_localization import StructuralSettings, image_rim_seed, estimate_structural
from neck_localization.ray_profile import estimate_ray_profile
from neck_localization.structure_portfolio import estimate_portfolio
from scripts.run_depth_profile_baseline import load_frames


def sources():
    paths = [ROOT/'scripts/run_structural_study.py', *sorted((ROOT/'neck_localization').glob('*.py'))]
    return {str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in paths
            if p.name not in ['inspection_viewer.py', 'viewer_navigation.py']}


def run(source, output, modes, methods, freeze=None):
    config = json.loads((source/'effective_config.json').read_text())
    locked = sources()
    if freeze:
        frozen = json.loads(freeze.read_text())
        assert frozen['sources'] == locked, 'Source changed after pre-test freeze'
        assert frozen['config_sha256'] == sha256((source/'effective_config.json').read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    (output/'effective_config.json').write_text(json.dumps(config, indent=2)+'\n')
    (output/'provenance.json').write_text(json.dumps(dict(source=str(source.resolve()), sources=locked,
        settings=asdict(StructuralSettings()), modes=modes, methods=methods,
        no_hardware=True, true_K_T=True, truth='evaluation only; rim-only mode is NOT a tying-circle prediction'), indent=2)+'\n')
    rows = []
    for mode in modes:
        (output/mode/'scenes').mkdir(parents=True)
        (output/mode/'settings.json').write_text(json.dumps(dict(preprocessing=dict(condition=mode,
            kind='frozen_active_stereo_observations')), indent=2)+'\n')
        for case in config['cases']:
            name = case['name']; scene = source/mode/'scenes'/name
            (output/mode/'scenes'/name).symlink_to(scene.resolve(), target_is_directory=True)
            frames = load_frames(scene/'observations', 'rgb')
            for method in methods:
                if method == 'baseline':
                    estimate = estimate_stereo_robust(frames)
                elif method == 'rim_seed':
                    estimate = estimate_structural(frames)
                elif method in ['ray_rgb', 'ray_fused']:
                    estimate = estimate_ray_profile(frames, depth_enabled=method == 'ray_fused')
                elif method.startswith('portfolio'):
                    estimate = estimate_portfolio(frames, robust=method != 'portfolio_plain',
                        feedback=method not in ['portfolio_plain', 'portfolio_robust'], multi=method != 'portfolio_single',
                        anisotropic=method in ['portfolio_anisotropic','portfolio_fixed'], fixed_axis=method == 'portfolio_fixed')
                elif method == 'rim_only':
                    try:
                        center, normal, radius, detail = image_rim_seed(frames)
                        estimate = Estimate(center=center, normal=normal, radius=radius, accepted=True,
                                            diagnostics=detail)
                    except (ValueError, np.linalg.LinAlgError) as error:
                        estimate = Estimate(reasons=[str(error)])
                else:
                    raise ValueError(method)
                # No target, mesh or true dimensions have been opened until here.
                truth = json.loads((scene/'evaluation_only/target.json').read_text())
                if method == 'rim_only':
                    with np.load(scene/'evaluation_only/profile.npz') as profile:
                        truth = dict(truth, center_m=profile['centerline_m'][-1].tolist(),
                                     radius_m=float(profile['radius_m'][-1]))
                row = dict(case=name, group=case['group'], condition_name=mode,
                    split=config['dataset_split'], geometry='photo_short_neck', evidence='rgb', camera_set='all',
                    factor=mode, method=method, diagnostic_rim_not_tying_circle=method == 'rim_only', **score(estimate, truth))
                row.setdefault('diagnostics', estimate.diagnostics)
                band = (plan_band(row['estimate'], estimate.diagnostics) if estimate.center is not None and
                        'near_min_interval_from_axis_origin_m' in estimate.diagnostics else None)
                row['band'] = band
                row['band_evaluation'] = ({label:evaluate_band(band, truth, truth[field]) for label, field in
                    [('near_min_proxy','near_min_interval_s_m'), ('full_annotated_neck','search_interval_s_m')]} if band else None)
                rows.append(row)
                with (output/'records.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
                print(name, mode, method, row['accepted'], row['center_mm'], row['normal_deg'], row['diameter_mm'], row['reasons'], flush=True)
        selected = [r for r in rows if r['condition_name'] == mode]
        (output/mode/'records.json.gz').write_bytes(gzip.compress(json.dumps(selected).encode(), mtime=0))
    (output/'records.json.gz').write_bytes(gzip.compress(json.dumps(rows).encode(), mtime=0))
    (output/'summary.json').write_text(json.dumps(summarize(rows), indent=2)+'\n')
    assert locked == sources(), 'Sources changed during evaluation'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--modes', nargs='+', default=['stereo_raw']); p.add_argument('--methods', nargs='+', default=['rim_seed'])
    p.add_argument('--freeze', type=Path)
    a = p.parse_args(); run(a.run, a.output, a.modes, a.methods, a.freeze)
