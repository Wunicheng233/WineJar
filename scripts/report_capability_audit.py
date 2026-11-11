#!/usr/bin/env python3
"""Collect saved capability results without rerunning inference or selection."""
import argparse
import csv
import json
from pathlib import Path


def dump_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def collect(arguments):
    arguments.output.mkdir(parents=True, exist_ok=False)
    geometry, measured, common = [], [], []
    inputs = {}
    overrides = dict(item.split('=', 1) for item in arguments.measurement_overrides)
    for specification in arguments.evaluations:
        name, directory = specification.split('=', 1)
        directory = Path(directory)
        inputs[name] = str(directory.resolve())
        if (directory / 'summary.json').exists():
            for row in json.loads((directory / 'summary.json').read_text()):
                geometry.append(dict(method=name, **row))
        groups = {}
        measurement_directory = Path(overrides.get(name, directory))
        if name in overrides:
            inputs[name + '_measurement_override'] = str(measurement_directory.resolve())
        for row in json.loads((measurement_directory / 'measurement_metrics.json').read_text()):
            if row['status'] != 'ok':
                continue
            for region, metric in row['regions'].items():
                key = row['layout'], row['condition'], region
                groups.setdefault(key, []).append(metric)
        for (layout, condition, region), metrics in sorted(groups.items()):
            count = sum(m['reference_pixels'] for m in metrics)
            valid = sum(m['valid_prediction_pixels'] for m in metrics)
            pooled_mse = sum(m['rmse_mm']**2 * m['valid_prediction_pixels']
                             for m in metrics if m['rmse_mm'] is not None)
            signed = sum(m['signed_mean_mm'] * m['valid_prediction_pixels']
                         for m in metrics if m['signed_mean_mm'] is not None)
            measured.append(dict(method=name, layout=layout, condition=condition,
                region=region, views=len(metrics), reference_pixels=count,
                valid_pixels=valid, coverage=valid/count if count else None,
                pooled_rmse_mm=(pooled_mse/valid)**.5 if valid else None,
                pooled_signed_mean_mm=signed/valid if valid else None))
            comparisons = [m.get('common_valid_comparison') for m in metrics]
            comparisons = [m for m in comparisons if m and m.get('status') == 'ok']
            if comparisons:
                cv = sum(m['valid_prediction_pixels'] for m in comparisons)
                mse = sum(m['rmse_mm']**2 * m['valid_prediction_pixels']
                          for m in comparisons if m['rmse_mm'] is not None)
                common.append(dict(method=name, layout=layout, condition=condition,
                    region=region, views=len(comparisons), common_pixels=cv,
                    pooled_rmse_mm=(mse/cv)**.5 if cv else None,
                    comparison_members=comparisons[0].get('members')))
    result = dict(inputs=inputs, geometry=geometry, measurements=measured,
                  common_support=common,
                  scope='Previously viewed synthetic capability cases; paired layouts are the same jars',
                  pooling='Measurement RMSE weighted by valid reference rays; coverage uses all fixed reference rays')
    (arguments.output / 'combined.json').write_text(json.dumps(result, indent=2) + '\n')
    dump_csv(arguments.output / 'geometry.csv', geometry)
    dump_csv(arguments.output / 'measurements.csv', measured)
    dump_csv(arguments.output / 'common_support.csv', common)
    if arguments.plot:
        plot(arguments.output, geometry, measured)


def plot(output, geometry, measurements):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    methods = list(dict.fromkeys(row['method'] for row in measurements))
    figure, axes = plt.subplots(1, 3, figsize=(13.3, 4), constrained_layout=True)
    labels = ['far / clean', 'far / noise', 'close / clean', 'close / noise']
    keys = [('far', 'clean'), ('far', 'shared_noise'),
            ('close', 'clean'), ('close', 'shared_noise')]
    width = .8 / max(len(methods), 1)
    for i, method in enumerate(methods):
        points = { (r['layout'], r['condition']): r for r in measurements
                   if r['method'] == method and r['region'] == 'GT_visible_outer_neck_evaluation_only'}
        geometry_points = {(r['layout'], r['condition']): r for r in geometry
                           if r['method'] == method and r['backend'] == 'portfolio'
                           and r['camera_set'] == 'all'}
        x = np.arange(4) + (i-(len(methods)-1)/2)*width
        axes[0].bar(x, [points.get(k, {}).get('pooled_rmse_mm', np.nan) for k in keys], width, label=method)
        axes[1].bar(x, [100*points[k]['coverage'] if k in points else np.nan for k in keys], width)
        axes[2].bar(x, [geometry_points[k]['candidate_geometry_pass'] if k in geometry_points else np.nan
                        for k in keys], width)
    for axis in axes:
        axis.set_xticks(range(4), labels, rotation=25)
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel('Native neck Z RMSE (mm), valid rays')
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Fixed neck reference rays covered (%)')
    axes[1].set_ylim(0, 105)
    axes[2].set_ylabel('Candidate circles within all thresholds / 12')
    axes[2].set_ylim(0, 12.7)
    figure.suptitle('Fixed synthetic capability audit: depth and downstream geometry')
    figure.savefig(output / 'depth_and_geometry.png', dpi=170)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations', nargs='+', required=True, metavar='NAME=EVAL_DIRECTORY')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--measurement-overrides', nargs='*', default=[], metavar='NAME=EVAL_DIRECTORY')
    parser.add_argument('--plot', action='store_true')
    collect(parser.parse_args())
