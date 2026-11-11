#!/usr/bin/env python3
"""Development-only mechanism ablation: one region update vs converged feedback.

One-update candidates are deliberately not called converged successes. No new
test scenes are used here, and the frozen estimator is not modified.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from neck_localization.support_feedback import FeedbackSettings,estimate_support_feedback
from neck_localization.evaluation import score
from scripts.run_depth_profile_baseline import load_frames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    records=json.loads((args.development/'records.json').read_text())
    provenance=json.loads((args.development/'provenance.json').read_text())
    assert provenance['split']=='development'
    rows=[]
    for source in provenance['scene_roots']:
        for camera_file in sorted(Path(source).glob('*/observations/camera_true_parameters.json')):
            folder=camera_file.parent.parent
            for cs in ['cam_a','cam_a,cam_b','all']:
                frames=load_frames(folder/'observations','rgb',None if cs=='all' else cs.split(','))
                est=estimate_support_feedback(frames,mode='region',settings=replace(FeedbackSettings(),maximum_outer_iterations=1))
                truth=json.loads((folder/'evaluation_only/target.json').read_text())
                once=score(est,truth)
                full=next(r for r in records if r['case']==folder.name and r['camera_set']==cs and r['method']=='region_equal_trust')
                rows.append(dict(case=folder.name,camera_set=cs,
                    single_update_normal_deg=once['normal_deg'],converged_normal_deg=full['normal_deg'],
                    single_update_center_mm=once['center_mm'],converged_center_mm=full['center_mm'],
                    full_iterations=len(full['diagnostics'].get('feedback_history',[])),
                    single_update_reasons=once['reasons'],full_reasons=full['reasons']))
    args.output.write_text(json.dumps(dict(kind='development_mechanism_diagnostic_not_test',
        note='Single-update variant intentionally fails feedback convergence; compare candidate errors only, not success rates.',records=rows),indent=2)+'\n')
    for r in rows:
        if r['case'] in ['test_oblique_flat','tilted_neck','broad_reverse_tilt'] and r['camera_set']=='all':print(r)


if __name__=='__main__':main()
