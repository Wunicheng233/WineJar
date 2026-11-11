#!/usr/bin/env python3
"""Tables and figures for the paired axial-depth-bias audit, no estimation."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.run/'summary.json').read_text())
    rows = json.load(gzip.open(args.run/'records.json.gz', 'rt'))
    with (args.output/'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v, dict) else v for k,v in row.items()} for row in summary)
    metrics = ['case', 'group', 'condition_name', 'accepted', 'truth_pass', 'center_mm', 'normal_component_mm',
               'tangent_mm', 'normal_deg', 'diameter_mm', 'signed_diameter_error_mm', 'runtime_ms', 'residual', 'reasons']
    case_rows = []
    for row in rows:
        item = {k:row.get(k) for k in metrics}
        for policy in ('near_min_proxy', 'full_annotated_neck'):
            value = (row.get('band_evaluation') or {}).get(policy, {})
            item[policy+'_minimum_margin_mm'] = value.get('minimum_margin_mm')
            item[policy+'_contained'] = value.get('contained')
        case_rows.append(item)
    with (args.output/'cases.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    common = sorted([s for s in summary if s['group']=='all' and (s['condition_name']=='zero' or s['condition_name'].startswith('common_'))],
                    key=lambda s:s['bias_by_camera_mm'].get('cam_a', 0))
    x = [s['bias_by_camera_mm'].get('cam_a', 0) for s in common]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0,0].plot(x, [s['correct_circle']/s['n'] for s in common], 'o-', label='Correct output')
    axes[0,0].plot(x, [s['rejected']/s['n'] for s in common], 'o-', label='Rejected (task failure)')
    axes[0,0].plot(x, [s['wrong_accepted_circle']/s['n'] for s in common], 'o-', label='Wrong accepted')
    axes[0,0].set(ylabel='Fraction / 12', ylim=(-.05,1.05))
    axes[0,0].legend(fontsize=8)
    axes[0,1].plot(x, [s['center_mm_p95'] for s in common], 'o-', label='Centre P95 [mm]')
    axes[0,1].plot(x, [s['diameter_mm_p95'] for s in common], 'o-', label='Diameter P95 [mm]')
    axes[0,1].set(ylabel='Solved candidates including rejected')
    axes[0,1].legend(fontsize=8)
    zero = {r['case']:r for r in rows if r['condition_name']=='zero'}
    slopes = []
    for case in zero:
        local = sorted([r for r in rows if r['case']==case and (r['condition_name']=='zero' or r['condition_name'].startswith('common_'))
                        and r.get('signed_diameter_error_mm') is not None], key=lambda r: next(s['bias_by_camera_mm'].get('cam_a',0) for s in common if s['condition_name']==r['condition_name']))
        bx = [next(s['bias_by_camera_mm'].get('cam_a',0) for s in common if s['condition_name']==r['condition_name']) for r in local]
        axes[1,0].plot(bx, [r['signed_diameter_error_mm'] for r in local], '.-', alpha=.5, lw=.8)
        minus = next(r for r in local if r['condition_name']=='common_m1')
        plus = next((r for r in local if r['condition_name']=='common_p1'), None)
        if plus:
            slopes.append((plus['signed_diameter_error_mm']-minus['signed_diameter_error_mm'])/2)
    axes[1,0].set(ylabel='Signed diameter error [mm] (12 cases)')
    shown = [s for s in summary if s['group']=='all' and s['condition_name'] in ('zero','only_a_m2','only_a_p2')]
    labels = [s['condition_name'] for s in shown]
    positions = np.arange(len(shown))
    axes[1,1].bar(positions-.2, [s['correct_circle'] for s in shown], width=.2, label='Correct output')
    axes[1,1].bar(positions, [s['solved_but_rejected_correct_circle'] for s in shown], width=.2, label='Geometrically correct, rejected')
    axes[1,1].bar(positions+.2, [s['wrong_accepted_circle'] for s in shown], width=.2, label='Wrong accepted')
    axes[1,1].set(xticks=positions, xticklabels=labels, ylabel='Cases / 12', ylim=(0,13))
    axes[1,1].legend(fontsize=7)
    for ax in (axes[0,0], axes[0,1], axes[1,0]):
        ax.set(xlabel='Assumed common additive axial depth bias [mm]')
        ax.grid(alpha=.2)
    fig.suptitle('Fixed estimator: depth-bias sensitivity only; NOT measured camera performance', fontsize=11)
    fig.savefig(args.output/'depth_bias_sensitivity.png', dpi=150)
    plt.close(fig)
    (args.output/'paired_diameter_slope.json').write_text(json.dumps(dict(
        units='mm diameter / mm axial depth bias', n=len(slopes), finite_difference='common +1 vs -1 mm; solved both sides only',
        median=float(np.median(slopes)), minimum=float(min(slopes)), maximum=float(max(slopes))), indent=2)+'\n')
    print('Tables, figure and paired diameter response saved to', args.output)


if __name__ == '__main__':
    main()
