#!/usr/bin/env python3
"""Reproduce engineering figures/CSV from saved conditional simulation results."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(p):
    return json.loads(Path(p).read_text())


def run(bundle):
    from matplotlib.font_manager import FontProperties
    font_path = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font_path.exists():
        matplotlib.font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    figure = bundle/'figures'
    figure.mkdir(exist_ok=True)
    control = read(bundle/'band_controls.json')['summary']
    methods = ['fs_visible', 'ss_sor', 'pose_fs_radius', 'pose_fs_radius_refit']
    labels = ['FS原方案', '传统双目/SOR', '姿态+FS半径', '姿态+局部半径重拟合']
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    x = np.arange(4)
    for shift, condition, label, color in [(-.18, 'clean', '原始保存输入', '#3977a8'), (.18, 'shared_noise', '扰动保存输入', '#d18a32')]:
        rows = [next(r for r in control if r['method'] == m and r['condition'] == condition) for m in methods]
        a = axes[0, 0].bar(x+shift, [r['band_contained'] for r in rows], .34, label=label, color=color)
        axes[0, 0].bar_label(a, fmt='%d/11', padding=3, fontsize=9)
        axes[0, 1].bar(x+shift, [r['diameter_mm']['max'] for r in rows], .34, color=color)
    for a in axes[0]:
        a.set_xticks(x, labels, rotation=12, ha='right')
        a.grid(axis='y', alpha=.2)
    axes[0, 0].set_ylim(0, 12.5)
    axes[0, 0].set_ylabel('整条3 mm带在物理颈段内：例数')
    axes[0, 0].set_title('同一11只坛的事后对照；不是实机扎口率')
    axes[0, 0].legend(fontsize=9, loc='lower right')
    axes[0, 1].set_ylabel('最大直径误差 / mm')
    axes[0, 1].set_title('尺寸与落位分别评价')
    axes[0, 1].axhline(2, color='.4', linestyle='--', linewidth=1, label='2 mm仅研究参考')
    axes[0, 1].legend(fontsize=9)
    strategies = ['nominal_preset_distance', 'nominal_encoder_endpoints', 'wide_calibration_ls', 'wide_calibration_huber', 'narrow_calibration_ls']
    names = ['名义工位距离', '直接读编码器', '跨工位标定LS', '跨工位标定Huber', '局部小范围标定LS']
    amplitude = [.5, 1, 2]
    summaries = [read(bundle/'combined_amplitude_0.5/uncertainty_summary.json'), read(bundle/'uncertainty_summary.json'), read(bundle/'combined_amplitude_2/uncertainty_summary.json')]
    for strategy, label in zip(strategies, names):
        values = [next(r for r in rows if r['scenario'] == 'combined' and r['strategy'] == strategy and r['method'] == 'pose_fs_radius')['center_mm']['p95'] for rows in summaries]
        axes[1, 0].plot(amplitude, values, marker='o', label=label, linewidth=1.4)
    axes[1, 0].set_xlabel('假设误差幅度倍数（不是设备规格）')
    axes[1, 0].set_ylabel('最终工具圆心条件样本P95 / mm')
    axes[1, 0].set_title('保留视觉误差；加入标定/输送/TCP误差')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=.2)
    calib = read(bundle/'calibration_validation.json')
    for shift, scenario, label, color in [(-.18, 'combined', '无异常测量', '#3977a8'), (.18, 'combined_survey_outlier', '仅一标定点异常', '#d18a32')]:
        values = [np.percentile([r['base_holdout_max_mm'] for r in calib if r['scenario'] == scenario and r['strategy'] == s], 95) for s in strategies[2:]]
        a = axes[1, 1].bar(np.arange(3)+shift, values, .34, label=label, color=color)
        axes[1, 1].bar_label(a, fmt='%.2f', padding=3, fontsize=9)
    axes[1, 1].set_xticks(np.arange(3), names[2:], rotation=12, ha='right')
    axes[1, 1].set_ylabel('独立校验点最大误差的条件P95 / mm')
    axes[1, 1].set_title('用独立基准验标定，防止偶然抵消视觉偏差')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(axis='y', alpha=.2)
    fig.suptitle('WineJar：保存观测对照 + 假设先验下的跨工位实验；不代表现场精度或最终扎口', fontsize=13)
    fig.savefig(figure/'engineering_comparison.png', dpi=170)
    fig.savefig(figure/'engineering_comparison.svg')
    plt.close(fig)
    with (figure/'band_controls.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['method', 'condition', 'n', 'accepted', 'band_contained', 'center_median_mm', 'center_p95_mm', 'center_max_mm', 'normal_max_deg', 'diameter_max_mm', 'joint_band_diameter_research_pass'])
        for r in control:
            w.writerow([r['method'], r['condition'], r['total'], r['accepted'], r['band_contained'], *r['center_mm'].values(), r['max_normal_deg'], r['diameter_mm']['max'], r['joint_pose_band_diameter_pass']])
    with (figure/'conditional_uncertainty.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['assumed_amplitude', 'scenario', 'strategy', 'method', 'center_p95_mm', 'band_conditional_fraction', 'center_over_4mm_fraction', 'between_session_axial_std_mm', 'within_session_axial_std_mm'])
        for amp, rows in zip(amplitude, summaries):
            for r in rows:
                w.writerow([amp, r['scenario'], r['strategy'], r['method'], r['center_mm']['p95'], r['band_contained_fraction'], r['accepted_pose_center_over_4mm_fraction'], r['between_session_axial_std_mm'], r['within_session_axial_std_mm']])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    args = p.parse_args()
    run(args.bundle.resolve())
