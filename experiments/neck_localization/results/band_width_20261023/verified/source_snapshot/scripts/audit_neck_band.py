#!/usr/bin/env python3
"""Audit a finite-width strap using frozen circle predictions; no new fitting.

GT is loaded only AFTER the observation-only band proposal. Existing scenes and
records are never modified. Both interval policies are synthetic assumptions.
"""
import argparse
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neck_localization.band_geometry import plan_band, evaluate_band


def read_json(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith('.gz') else data)


def statistics(rows, width_mm, policy, group):
    selected = [r for r in rows if group == 'all' or r['group'] == group]
    scores = [r['band_evaluation'][policy] for r in selected if r.get('band_evaluation')]
    proposals = [r['band'] for r in selected if r.get('band')]
    good = lambda r: bool(r['accepted'] and r['truth_pass'] and r.get('band')
                          and r['band']['nominal_compatible'] and r['band_evaluation'][policy]['contained'])
    result = dict(width_mm=width_mm, policy=policy, group=group, n=len(selected),
                  circle_correct_n=sum(r['accepted'] and r['truth_pass'] for r in selected),
                  proposed_compatible_n=sum(p['nominal_compatible'] for p in proposals),
                  geometrically_contained_n=sum(s['contained'] for s in scores),
                  proxy_task_success_n=sum(good(r) for r in selected),
                  false_compatible_n=sum(s['false_compatible'] for s in scores),
                  false_incompatible_n=sum(s['false_incompatible'] for s in scores),
                  ideal_true_axis_center_can_fit_n=sum(s['ideal_true_axis_center_can_fit'] for s in scores))
    for name in ('minimum_margin_mm', 'max_axial_overhang_mm', 'tilt_swing_mm'):
        values = [s[name] for s in scores]
        for label, percentile in [('median', 50), ('p95', 95), ('min', 0), ('max', 100)]:
            result[name+'_'+label] = float(np.percentile(values, percentile)) if values else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=Path, default=ROOT/'experiments/neck_localization/runs/reliable_output_v2_test_scenes')
    parser.add_argument('--records', type=Path, default=ROOT/'experiments/neck_localization/runs/reliable_output_v2_test/records.json.gz')
    parser.add_argument('--method', default='task_validated_output')
    parser.add_argument('--width-mm', type=float, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.width_mm) or args.width_mm < 0:
        parser.error('width must be nonnegative and finite')
    rows = [r for r in read_json(args.records) if r['method'] == args.method and r['camera_set'] == 'all']
    if not rows or len({r['case'] for r in rows}) != len(rows):
        raise ValueError('Need exactly one all-camera row per case')
    args.output.mkdir(parents=True, exist_ok=False)
    truth_hashes, truths, times, all_summary, primary = {}, {}, [], [], None
    for width in sorted(set([0., 3., args.width_mm, 7., 10.])):
        evaluated = []
        for source in rows:
            row = dict(source)
            # Propose BEFORE loading GT; keep every source row in denominator.
            start = time.perf_counter()
            band = plan_band(row['estimate'], row['diagnostics'], width/1000) if row.get('estimate') else None
            elapsed_ms = (time.perf_counter()-start)*1000
            row['band'] = band
            truth_path = args.scenes/row['case']/'evaluation_only/target.json'
            truth = read_json(truth_path)
            truths[row['case']] = truth
            truth_hashes[row['case']] = sha256(truth_path.read_bytes()).hexdigest()
            policies = dict(near_min_proxy=truth['near_min_interval_s_m'],
                            full_annotated_neck=truth['search_interval_s_m'])
            row['band_evaluation'] = {name: evaluate_band(band, truth, bounds) for name, bounds in policies.items()} if band else None
            evaluated.append(row)
            if width == args.width_mm:
                times.append(elapsed_ms)
        for policy in ('near_min_proxy', 'full_annotated_neck'):
            for group in ('all', 'nominal', 'stress'):
                all_summary.append(statistics(evaluated, width, policy, group))
        if width == args.width_mm:
            primary = evaluated
    with gzip.open(args.output/'records.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(primary, stream, ensure_ascii=False)
    settings = read_json(args.records.parent/'settings.json')
    settings.update(band_width_mm=args.width_mm, band_uncertainty='not_estimated',
                    band_estimator_source='frozen_predictions_no_refitting',
                    band_planner_criterion='observed_near_min_range_proxy',
                    band_scope='constant_radius_axial_strip; no contact, thickness, collision, transport or robot')
    (args.output/'settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(all_summary, indent=2)+'\n')
    with (args.output/'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_summary[0]))
        writer.writeheader()
        writer.writerows(all_summary)
    case_rows = []
    for row in primary:
        for policy, score in (row.get('band_evaluation') or {}).items():
            case_rows.append(dict(case=row['case'], group=row['group'], policy=policy,
                                  nominal_compatible=row['band']['nominal_compatible'],
                                  nominal_margin_mm=row['band']['nominal_margin_m']*1000,
                                  **{k: v for k, v in score.items() if not isinstance(v, (dict, list))}))
    with (args.output/'cases.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    files = [ROOT/'neck_localization/band_geometry.py', Path(__file__), ROOT/'tests/neck_localization/test_band_geometry.py']
    provenance = dict(kind='posthoc_geometry_audit_not_new_unseen_estimator_test', scenes=str(args.scenes.resolve()),
                      records=str(args.records.resolve()), input_records_sha256=sha256(args.records.read_bytes()).hexdigest(),
                      evaluation_only_truth_sha256=truth_hashes,
                      sources={str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in files},
                      planner_runtime_ms={label: float(np.percentile(times, p)) for label, p in [('median', 50), ('p95', 95), ('max', 100)]},
                      original_estimator_runtime_excluded=True)
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for policy, color in [('near_min_proxy', 'tab:blue'), ('full_annotated_neck', 'tab:orange')]:
        margins = [r['band_evaluation'][policy]['minimum_margin_mm'] for r in primary]
        axes[0].plot(range(len(primary)), margins, '.-', label=policy, color=color, lw=.7)
        curve = [r for r in all_summary if r['policy'] == policy and r['group'] == 'all']
        axes[1].plot([r['width_mm'] for r in curve], [r['geometrically_contained_n'] for r in curve], 'o-', color=color, label=policy+' contained')
        axes[1].plot([r['width_mm'] for r in curve], [r['ideal_true_axis_center_can_fit_n'] for r in curve], '--', color=color, label=policy+' ideal geometric capacity')
    axes[0].axhline(0, color='black', lw=.7)
    axes[0].set(xlabel='Frozen case index (source record order)', ylabel='Minimum axial margin [mm]', title=f'{args.width_mm:g} mm band: axial containment only; synthetic interval policies')
    axes[1].set(xlabel='Band axial width [mm]', ylabel=f'Cases / {len(primary)}')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(args.output/'band_margins.png', dpi=150)
    plt.close(fig)
    # Diagnostic only: align all intervals to true centre/normal AFTER scoring.
    failures = [r for r in primary if not r['band_evaluation']['near_min_proxy']['contained']]
    examples = [primary[0]]+[r for r in failures if r['case'] != primary[0]['case']]
    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 2*len(examples)), squeeze=False, constrained_layout=True)
    for ax, row in zip(axes[:, 0], examples):
        truth, band = truths[row['case']], row['band']
        score = row['band_evaluation']['near_min_proxy']
        nt, n = np.array(truth['normal']), np.array(band['normal'])
        offset = score['center_axial_offset_mm']
        intervals = [np.array(truth['search_interval_s_m'])*1000-truth['s_m']*1000,
                     np.array(truth['near_min_interval_s_m'])*1000-truth['s_m']*1000,
                     np.array(band['observed_usable_interval_relative_center_m'])*1000*(n @ nt)+offset,
                     score['actual_strip_axial_envelope_relative_truth_center_mm']]
        for y, bounds, color, label in zip([3, 2, 1, 0], intervals, ['.6', 'tab:green', 'tab:blue', 'tab:orange'],
                                           ['GT complete neck', 'GT near-min proxy', 'Observed proxy', 'Entire band envelope']):
            ax.plot(bounds, [y, y], color=color, lw=5, solid_capstyle='butt')
        ax.axvline(0, color='black', lw=.5, ls=':')
        ax.set(yticks=[3, 2, 1, 0], yticklabels=['GT complete neck', 'GT near-min proxy', 'Observed proxy', 'Entire band envelope'],
               ylim=(-.7, 3.7), xlabel='True-axis displacement from true circle centre [mm]',
               title=f"{row['case']}: nominal compatible={band['nominal_compatible']}, actual minimum margin={score['minimum_margin_mm']:.3f} mm")
        ax.grid(axis='x', alpha=.2)
    fig.savefig(args.output/'interval_diagnostic.png', dpi=150)
    plt.close(fig)
    print(json.dumps([s for s in all_summary if s['width_mm'] == args.width_mm and s['group'] == 'all'], indent=2))


if __name__ == '__main__':
    main()
