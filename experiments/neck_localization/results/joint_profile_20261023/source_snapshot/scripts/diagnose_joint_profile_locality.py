#!/usr/bin/env python3
"""Development-only ablation: local neck vs broad collar data under same fitter."""
from dataclasses import replace
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neck_localization.joint_profile import JointSettings, estimate_joint_profile
from neck_localization.evaluation import score
from scripts.run_depth_profile_baseline import load_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    rows = []
    for case in ['reference_shallow', 'tilted_neck']:
        folder = args.scenes/case
        frames = load_frames(folder/'observations', 'rgb')
        for name, settings in [('local', JointSettings()),
                               ('broad_28mm', replace(JointSettings(), local_half_span_mm=1000.))]:
            estimate = estimate_joint_profile(frames, settings=settings)
            truth = json.loads((folder/'evaluation_only/target.json').read_text())
            result = score(estimate, truth)
            result.setdefault('diagnostics', estimate.diagnostics)
            rows.append(dict(case=case, ablation=name, split='development_diagnostic_not_test', **result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(kind='development_locality_ablation; does not tune or replace frozen test estimator',
        estimator_sha256=sha256((ROOT/'neck_localization/joint_profile.py').read_bytes()).hexdigest(),
        note='Broad mode uses all candidates in original observed 28mm band. Different point sets; objectives are not directly comparable.',
        records=rows), indent=2)+'\n')
    for row in rows:
        print(row['case'], row['ablation'], row['accepted'], row['center_mm'], row['normal_deg'], row['residual'])


if __name__ == '__main__':
    main()
