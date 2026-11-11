#!/usr/bin/env python3
"""Check timed-stop invariants and expose conditional errors, not factory rates."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.run_cross_station_study import read, write, digest, quantiles


def run(bundle):
    axes = read(bundle/'axes.json')
    provenance = read(bundle/'provenance.json')
    for path, sha in provenance['input_and_source_sha256'].items():
        assert digest(path) == sha, path
    assert provenance['no_encoder_observation_used'] and provenance['no_runtime_reobservation']
    baseline = read(bundle/'band_controls.json')['rows']
    data = dict(np.load(bundle/'conditional_samples.npz'))
    # At zero stopping error, cycle sampling must not change a command or truth.
    for key, a in data.items():
        assert np.isfinite(a).all()
        np.testing.assert_array_equal(a[0, 0], a[1, 0])
        np.testing.assert_array_equal(a[:, 0], np.broadcast_to(a[:, 0, :, :, :1], a[:, 0].shape))
    # Exact-reference diagnostic retains visual estimation error: it is NOT a
    # true target circle supplied to an estimator or motion planner.
    for key in ('center_mm', 'band_margin_mm'):
        expected = np.array([r[key] for r in baseline], np.float32)
        np.testing.assert_allclose(data[key][0, 0, 0], np.broadcast_to(expected, data[key][0, 0, 0].shape), atol=1e-5, rtol=0)
    # Commanded guide pose is fixed; changing a bounded physical stop position
    # cannot change any distance or clearance margin by more than that bound.
    for k, bound in enumerate(axes['ranges_m']):
        for key in data:
            assert np.max(np.abs(data[key][0, k]-data[key][0, 0])) <= bound*1000+2e-5
    # Summaries distinguish an accepted vision result from a usable transported
    # command. The vision gate has not seen these newly injected errors.
    assert all(r['accepted'] for r in baseline)
    table = []
    for h, shape in enumerate(axes['distributions']):
        for k, half_range in enumerate(axes['ranges_m']):
            for j, strategy in enumerate(axes['strategies']):
                for method in sorted({r['method'] for r in axes['predictions']}):
                    ids = [i for i, row in enumerate(axes['predictions']) if row['method'] == method]
                    a = {key: value[h, k, j][:, :, ids] for key, value in data.items()}
                    table.append(dict(distribution=shape, uniform_equivalent_half_range_mm=half_range*1000,
                        strategy=strategy, method=method, conditional_draws=a['center_mm'].size,
                        center_median_mm=quantiles(a['center_mm'])['median'],
                        center_p95_mm=quantiles(a['center_mm'])['p95'],
                        center_max_mm=quantiles(a['center_mm'])['max'],
                        vision_false_rejects=0,
                        unchanged_vision_gate_accepted_center_over_4mm_fraction=float(np.mean(a['center_mm'] > 4)),
                        band_contained_fraction=float(np.mean(a['band_margin_mm'] >= -1e-6)),
                        mouth_only_bore_certificate_fraction=float(np.mean(a['mouth_bore_margin_mm'] > 0)),
                        minimum_band_margin_mm=float(a['band_margin_mm'].min()),
                        minimum_mouth_only_bore_margin_mm=float(a['mouth_bore_margin_mm'].min()),
                        final_fastening_success=None))
    write(bundle/'checked_metrics.json', table)
    with (bundle/'checked_metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    # Retain each worst band case for the uniform calibrated main strategy,
    # including cases with good center error and inadequate process margin.
    j = axes['strategies'].index('calibrated_reference_tcp_timer')
    examples = []
    for k, half_range in enumerate(axes['ranges_m']):
        ids = [i for i, r in enumerate(axes['predictions']) if r['method'] == 'pose_fs_radius']
        a = data['band_margin_mm'][0, k, j][:, :, ids]
        s, c, i = np.unravel_index(np.argmin(a), a.shape)
        row_id = ids[i]
        examples.append(dict(half_range_mm=half_range*1000, calibration_session=int(s), cycle=int(c),
            prediction=axes['predictions'][row_id],
            **{key: float(v[0, k, j, s, c, row_id]) for key, v in data.items()}))
    write(bundle/'retained_failure_examples.json', examples)
    plot(table, bundle)
    write(bundle/'verification.json', dict(passed=True, zero_stop_retains_visual_error=True,
        zero_stop_has_no_cycle_dependence=True, bounded_stop_lipschitz_checks=True,
        previous_inputs_unchanged=True, no_encoder_or_runtime_observation=True,
        conditional_only_not_measured_probability=True, groups=len(table),
        validator_sha256=digest(__file__), samples_sha256=digest(bundle/'conditional_samples.npz')))
    print('timed-stop validation passed:', len(table), 'groups')


def plot(table, bundle):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
    strategies = [('exact_reference_tcp_diagnostic', '准确变换（诊断）', '#777777'),
                  ('nominal_reference_tcp_timer', '名义变换＋预设启停', '#d17c22'),
                  ('calibrated_reference_tcp_timer', '独立标定＋预设启停', '#2275ad')]
    for strategy, label, color in strategies:
        rows = [r for r in table if r['distribution'] == 'uniform_bound' and r['method'] == 'pose_fs_radius' and r['strategy'] == strategy]
        x = [r['uniform_equivalent_half_range_mm'] for r in rows]
        for ax, key, scale in zip(axs, ['center_p95_mm', 'band_contained_fraction', 'mouth_only_bore_certificate_fraction'], [1, 100, 100]):
            ax.plot(x, [r[key]*scale for r in rows], 'o-', color=color, label=label)
    # Distribution control has the same standard deviation; dashed curve does
    # not inherit a finite uniform support bound.
    rows = [r for r in table if r['distribution'] == 'normal_same_standard_deviation' and r['method'] == 'pose_fs_radius' and r['strategy'] == 'calibrated_reference_tcp_timer']
    for ax, key, scale in zip(axs, ['center_p95_mm', 'band_contained_fraction', 'mouth_only_bore_certificate_fraction'], [1, 100, 100]):
        ax.plot([r['uniform_equivalent_half_range_mm'] for r in rows], [r[key]*scale for r in rows], 's--', color='#6d4398', label='独立标定＋正态对照（同标准差）')
        ax.set_xlabel('假设到位误差半幅 a（mm）\n均匀 ±a；正态 σ=a/√3，无界')
        ax.grid(alpha=.22)
    for ax, title in zip(axs, ['圆心误差 P95（mm）', '3 mm 带在物理颈段内（%）', '仅坛口/包覆净空充分条件（%）']):
        ax.set_title(title)
    axs[0].legend(fontsize=8, loc='upper left')
    axs[1].set_ylim(80, 92)
    axs[2].set_ylim(98, 100.15)
    fig.suptitle('预设启停主场景｜11 个已看坛型、两保存输入条件｜姿态/半径组合对照', fontsize=13)
    fig.text(.5, .01, '全部幅度与分布为假设扫描；24 次虚拟标定。百分比为条件仿真比例，不是工厂成功率。净空不含枪体/机械臂，未模拟实际扎紧。', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.savefig(bundle/'timed_stop_comparison.png', dpi=160)
    fig.savefig(bundle/'timed_stop_comparison.svg')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    run(parser.parse_args().bundle.resolve())
