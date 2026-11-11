#!/usr/bin/env python3
"""Summarize saved sensor experiments; never rerun or alter an estimator."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.evaluate_active_stereo import summaries

RUNS = ROOT/'experiments/neck_localization/runs'
SOURCES = {
    'true': 'active_stereo_adapted_test_20261025_v1',
    'gp': 'active_stereo_gp_adapted_test_20261025_v1',
    'board': 'active_stereo_board_calibrated_test_20261025_v1',
}
ARTIFACTS = ROOT/'experiments/neck_localization/results/active_stereo_20261025/artifacts'
RECORD_FILES = {'true':'true_parameters_records.json.gz', 'gp':'gp_records.json.gz',
                'board':'board_parameters_records.json.gz'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    grouped, rows = [], []
    for mode, run in SOURCES.items():
        # Original inference records are retained in runs; identical compact
        # copies keep this report reproducible without migrating all images.
        saved = ARTIFACTS/RECORD_FILES[mode]
        records = json.loads(gzip.decompress(saved.read_bytes()))
        for row in records:
            rows.append(dict(parameter_comparison=mode, **row))
        for summary in summaries(records):
            selected = [r for r in records if r['condition_name'] == summary['condition']]
            grouped.append(dict(parameter_comparison=mode, **summary,
                candidates_with_circle=sum(r.get('center_mm') is not None for r in selected),
                metric_population='all solved circles, including rejected; absent values are not zeros'))
    (args.output/'summary.json').write_text(json.dumps(grouped, indent=2)+'\n')
    keys = [k for k in grouped[0] if k != 'reasons']
    with (args.output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in keys} for r in grouped)
    fields = ['parameter_comparison','condition_name','case','group','accepted','truth_pass',
        'center_mm','normal_deg','diameter_mm','normal_component_mm','tangent_mm','runtime_ms','reasons']
    with (args.output/'cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k]) if k == 'reasons' else row.get(k) for k in fields})
    labels = [f"{r['parameter_comparison']}: {r['condition']}" for r in grouped]
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    y = np.arange(len(grouped))
    correct = np.array([r['correct_circle'] for r in grouped])
    wrong = np.array([r['wrong_accepted'] for r in grouped])
    reject = np.array([r['rejected'] for r in grouped])
    ax.barh(y, correct, color='#338769', label='Correct accepted')
    ax.barh(y, wrong, left=correct, color='#c33c38', label='Wrong accepted')
    ax.barh(y, reject, left=correct+wrong, color='#a6b0bf', label='Rejected = task failure')
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 8.6)
    ax.set_xticks(range(9))
    ax.set_xlabel('Eight locked jar shapes per condition')
    ax.set_title('Sensor-model transfer: frozen geometry and unchanged quality gates')
    ax.legend(loc='lower center', bbox_to_anchor=(.5, -0.27), ncol=3, frameon=False)
    for i, r in enumerate(grouped):
        ax.text(8.12, i, f"{r['correct_circle']}/8", va='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output/'output_rates.png', dpi=160)
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output), cases=len(rows), comparisons=len(grouped))))


if __name__ == '__main__':
    main()
