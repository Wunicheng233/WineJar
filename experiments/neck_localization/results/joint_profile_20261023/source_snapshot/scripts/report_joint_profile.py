#!/usr/bin/env python3
"""Small persistent report of the fixed-rig joint-profile experiment."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neck_localization.evaluation import summarize
from scripts.run_joint_profile import METHODS, CAMERA_SETS

LABELS = {"rim_guided_profile": "旧固定轴剖面", "spline_fixed_axis": "新筛选/样条，固定轴",
          "joint_axis_spline": "新筛选/样条，联合轴"}


def cell(summary, metric):
    values = [summary[f"{metric}_{s}"] for s in ("median", "p95", "max")]
    return " / ".join(f"{v:.3f}" if v is not None else "—" for v in values)


def tables(rows, title):
    lines = [f"## {title}", "", "误差和耗时单元格为中位数 / P95 / 最大值。误差包括已求出但被拒绝的候选；",
             "未求出圆的失败不填零。成功需算法接受且通过4 mm/2°/2 mm分析阈值，后两项不是工艺定案。", "",
             "| 视图 | 方法 | 成功/拒绝/错误接受/总数 | 圆心mm | 法向° | 直径mm | 耗时ms |",
             "|---|---|---|---|---|---|---|"]
    for s in summarize(rows):
        group = [r for r in rows if r['method'] == s['method'] and r['factor'] == s['factor']]
        success = sum(r['accepted'] and r['truth_pass'] for r in group)
        rejection = sum(not r['accepted'] for r in group)
        false = sum(r['accepted'] and not r['truth_pass'] for r in group)
        lines.append(f"| {group[0]['camera_set']} | {LABELS[s['method']]} | {success}/{rejection}/{false}/{len(group)} | "
                     + " | ".join(cell(s, m) for m in ['center_mm', 'normal_deg', 'diameter_mm', 'runtime_ms'])+" |")
    lines += ["", "| 视图 | 方法 | XYZ最大绝对误差mm | 沿法向/切向最大误差mm | 已接受圆心最大误差mm |",
              "|---|---|---|---|---|"]
    for camera in CAMERA_SETS:
        for method in METHODS:
            group = [r for r in rows if r['camera_set'] == camera and r['method'] == method]
            solved = [r for r in group if r['center_mm'] is not None]
            xyz = np.max(np.abs([r['xyz_mm'] for r in solved]), axis=0) if solved else [None]*3
            axial = max((abs(r['normal_component_mm']) for r in solved), default=None)
            tangent = max((r['tangent_mm'] for r in solved), default=None)
            accepted = max((r['center_mm'] for r in group if r['accepted'] and r['center_mm'] is not None), default=None)
            fmt = lambda v: f"{v:.3f}" if v is not None else "—"
            lines.append(f"| {camera} | {LABELS[method]} | {' / '.join(fmt(v) for v in xyz)} | {fmt(axial)} / {fmt(tangent)} | {fmt(accepted)} |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    development = json.loads((args.development/'records.json').read_text())
    test = json.loads((args.test/'records.json').read_text())
    for name, source in [('development', args.development), ('test', args.test)]:
        folder = args.output/name; folder.mkdir()
        (folder/'records.json.gz').write_bytes(gzip.compress((source/'records.json').read_bytes(), mtime=0))
        for file in ('summary.json', 'settings.json', 'provenance.json'):
            shutil.copy2(source/file, folder/file)
    shutil.copy2(args.freeze, args.output/'freeze.json')
    lines = ['# 固定相机布局下的局部联合估计首轮结果', '',
             '准确相机参数、理想保存深度、普通RGB前景；没有相机位置搜索、传感器噪声或学习训练。',
             '8个已检查案例只作开发，6个新坛型在开发前锁定规格、开发后冻结求解器再渲染/评价。',
             '同一合成几何家族的小测试，不是跨真实酒坛泛化或实机精度证据。', '',
             '旧基线保持冻结；固定轴样条与联合轴样条共享观测筛选、曲线、权重和正则化，隔离轴更新的贡献。',
             '目标范围定义与生成器颈段标注仍有约1 mm量级的边界差异，平颈误差不能全归于求解器。',
             '运行耗时含初始化/筛选/拟合/判断，不含渲染、输入文件读取、采集传输；不同机器会变化。',
             '本轮没有校准协方差或概率置信度。数据条件数/多初值分歧是启发式拒绝证据。', '']
    lines += tables(test, '新测试结果')+['']+tables(development, '开发结果（不可当未见测试）')
    lines += ['', '## 残差与定位的对照', '',
              '| 测试案例 | 视图 | 联合/固定轴目标函数比 | 固定轴→联合圆心mm | 固定轴→联合法向° |',
              '|---|---|---|---|---|']
    for r in test:
        if r['method'] != METHODS[2]: continue
        fixed = next(v for v in test if v['case'] == r['case'] and v['camera_set'] == r['camera_set'] and v['method'] == METHODS[1])
        d, f = r['diagnostics'], fixed['diagnostics']
        if 'final_objective_mm2' not in d or 'final_objective_mm2' not in f: continue
        assert abs(d['initial_objective_mm2']-f['final_objective_mm2']) < 1e-5*max(1, f['final_objective_mm2'])
        ratio = d['final_objective_mm2']/max(f['final_objective_mm2'], 1e-12)
        fmt = lambda v: f'{v:.3f}' if v is not None else '—'
        lines.append(f"| {r['case']} | {r['camera_set']} | {ratio:.4f} | {fmt(fixed['center_mm'])}→{fmt(r['center_mm'])} | {fmt(fixed['normal_deg'])}→{fmt(r['normal_deg'])} |")
    lines += ['', '## 拒绝记录', '']
    for r in test:
        if not r['accepted']:
            lines.append(f"- {r['case']} / {r['camera_set']} / {r['method']}: {', '.join(r['reasons'])}")
    if all(r['accepted'] for r in test): lines.append('本批次无拒绝；这不证明拒绝机制在真实异常下充分有效。')
    lines += ['', '逐次记录（包含失败、分轴误差、拟合/多初值诊断）见development/和test/；源码/参数散列见freeze.json。',
              '绝对精度目标4 mm来自历史要求，法向2°与直径2 mm仅用于本轮比较。']
    (args.output/'tables.md').write_text('\n'.join(lines)+'\n')
    cases = sorted({r['case'] for r in test})
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
    for col, camera in enumerate(CAMERA_SETS):
        for row, (metric, title, limit) in enumerate([('center_mm', 'Centre error (mm)', 4),
                ('normal_deg', 'Normal error (deg)', 2), ('diameter_mm', 'Diameter error (mm)', 2)]):
            ax = axes[row, col]
            for index, method in enumerate(METHODS):
                values = [next(r for r in test if r['case'] == c and r['camera_set'] == camera and r['method'] == method) for c in cases]
                y = [r[metric] if r[metric] is not None else np.nan for r in values]
                ax.plot(np.arange(len(cases)), y, marker='o', markersize=4, label=method)
                for x, r in enumerate(values):
                    if not r['accepted'] and r[metric] is not None: ax.plot(x, r[metric], 'kx', markersize=8)
            ax.axhline(limit, color='.5', linestyle='--', linewidth=.8)
            ax.set_ylabel(title); ax.grid(alpha=.2); ax.set_ylim(bottom=0)
            if row == 0: ax.set_title(camera)
            if row == 2: ax.set_xticks(np.arange(len(cases)), [str(i+1) for i in range(len(cases))]); ax.set_xlabel('Test case index')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(args.output/'test_errors.png', dpi=150); plt.close(fig)
    (args.output/'case_indices.json').write_text(json.dumps(dict(enumerate(cases, 1)), indent=2)+'\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
