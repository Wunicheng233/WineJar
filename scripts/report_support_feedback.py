#!/usr/bin/env python3
"""Report all attempted cases, rejected estimates, weights and factorial effects."""
import argparse
import gzip
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neck_localization.evaluation import summarize
from scripts.run_support_feedback import MODES, CAMERA_MODES, CAMERA_SETS


def tables(rows, title):
    lines = [f'## {title}', '',
        '全部尝试计入成功/拒绝/错误接受分母；误差统计含已求出但被拒绝的候选。',
        '误差/耗时为中位数 / P95 / 最大值。4 mm、2°、直径2 mm为分析阈值，后两项尚非工艺验收标准。', '',
        '|相机|方法|成功/拒绝/错误接受/总数|圆心mm|法向°|直径mm|耗时ms|',
        '|---|---|---|---|---|---|---|']
    for s in summarize(rows):
        group = [r for r in rows if r['method']==s['method'] and r['factor']==s['factor']]
        counts = [sum(r['accepted'] and r['truth_pass'] for r in group),
                  sum(not r['accepted'] for r in group),
                  sum(r['accepted'] and not r['truth_pass'] for r in group), len(group)]
        cells = []
        for metric in ['center_mm','normal_deg','diameter_mm','runtime_ms']:
            cells.append(' / '.join(f'{s[metric+"_"+q]:.3f}' if s[metric+'_'+q] is not None else '—'
                                   for q in ['median','p95','max']))
        lines.append('|'+group[0]['camera_set']+'|'+s['method']+'|'+'/'.join(map(str,counts))+'|'+'|'.join(cells)+'|')
    lines += ['', '|相机|方法|XYZ最大绝对误差mm|沿法向/切向最大mm|', '|---|---|---|---|']
    for camera in CAMERA_SETS:
        for mode in MODES:
            for cm in CAMERA_MODES:
                method = f'{mode}_{cm}'
                solved = [r for r in rows if r['camera_set']==camera and r['method']==method and r['center_mm'] is not None]
                if not solved:continue
                xyz = np.max(np.abs([r['xyz_mm'] for r in solved]),axis=0)
                lines.append(f'|{camera}|{method}|'+ ' / '.join(f'{v:.3f}' for v in xyz)+
                    f'|{max(abs(r["normal_component_mm"]) for r in solved):.3f} / {max(r["tangent_mm"] for r in solved):.3f}|')
    lines += ['', '拒绝案例（包含无法求出圆、因此没有填零误差的尝试）：', '']
    for r in rows:
        if not r['accepted']:
            lines.append(f'- {r["case"]} / {r["camera_set"]} / {r["method"]}: '+', '.join(r['reasons']))
    return lines


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development',type=Path,required=True)
    p.add_argument('--test',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    dev=json.loads((args.development/'records.json').read_text())
    test=json.loads((args.test/'records.json').read_text())
    for label, rows in [('development',dev),('test',test)]:
        (args.output/f'{label}_records.json.gz').write_bytes(gzip.compress(json.dumps(rows).encode(),mtime=0))
    lines=['# 支持区域反馈与相机权重对照', '',
        '固定640×480三相机、准确相机参数、理想渲染深度、普通RGB前景；同一候选点集。',
        '14个已查看开发例；6个事先锁定新规格，开发后冻结源码再评价。',
        'balanced为每相机总权重相同；equal_trust为每点基础权重1，不按相机点数折减。',
        'fixed为固定权重；huber为0.10 mm阈值IRLS；region为观测低半径连续核心的轴/区域反馈。',
        '所有方法共享完整上下文的固定轴样条读出；轴核心不直接决定作业范围端点。',
        '拒绝检查同时涉及轴拟合、目标范围上下文、数据条件和区域支持；无校准概率不确定性。', '',
        '原固定权重与Huber的点集不变；region也是同一超集，更新软权重，不使用真值裁剪。',
        '代价函数不同，不能用不同方法的原始目标数值直接排名。运行时间不含采集、加载与渲染。', '']
    lines += tables(test,'新测试')+['']+tables(dev,'开发（已看过，不作未见评价）')
    lines += ['', '## 相机规则的配对影响', '',
              '|相机|策略|equal_trust−balanced 法向误差，中位数/最大绝对变化°|圆心误差，中位数/最大绝对变化mm|',
              '|---|---|---|---|']
    for camera in CAMERA_SETS:
        for mode in MODES:
            diffs=[]
            for a in test:
                if a['camera_set']!=camera or a['method']!=f'{mode}_balanced' or a['center_mm'] is None: continue
                b=next(b for b in test if b['case']==a['case'] and b['camera_set']==camera and b['method']==f'{mode}_equal_trust')
                if b['center_mm'] is None:continue
                diffs.append([b['normal_deg']-a['normal_deg'], b['center_mm']-a['center_mm']])
            if diffs:
                v=np.array(diffs)
                lines.append(f'|{camera}|{mode}|{np.median(v[:,0]):.6f} / {np.max(np.abs(v[:,0])):.6f}|{np.median(v[:,1]):.6f} / {np.max(np.abs(v[:,1])):.6f}|')
    (args.output/'tables.md').write_text('\n'.join(lines)+'\n')
    cases=sorted({r['case'] for r in test})
    fig, axes=plt.subplots(3,3,figsize=(13,9),constrained_layout=True)
    for col,camera in enumerate(CAMERA_SETS):
        for row,(metric,label) in enumerate([('normal_deg','Normal error (deg)'),('center_mm','Centre error (mm)'),('diameter_mm','Diameter error (mm)')]):
            ax=axes[row,col]
            for mode in MODES:
                rr=[next(r for r in test if r['case']==c and r['camera_set']==camera and r['method']==f'{mode}_equal_trust') for c in cases]
                ax.plot(range(1,len(cases)+1),[r[metric] if r[metric] is not None else np.nan for r in rr],'-o',ms=4,label=mode)
                for index,r in enumerate(rr,1):
                    if not r['accepted'] and r[metric] is not None:ax.plot(index,r[metric],'kx',ms=8)
            ax.set(xlabel='Locked test case index',ylabel=label,title=camera if row==0 else '')
            ax.set_ylim(bottom=0); ax.grid(alpha=.2)
            if row==0:ax.legend()
    fig.savefig(args.output/'test_comparison.png',dpi=150)
    (args.output/'case_indices.json').write_text(json.dumps(dict(enumerate(cases,1)),indent=2)+'\n')
    # Point-level plot is reconstructed from OBSERVATIONS. Truth is not used.
    from scripts.run_depth_profile_baseline import load_frames
    from neck_localization.joint_profile import observed_initialization, JointSettings
    from neck_localization.support_feedback import WeightedAxisSplineProblem
    provenance=json.loads((args.development/'provenance.json').read_text())
    case='test_oblique_flat'
    folder=next(Path(s)/case for s in provenance['scene_roots'] if (Path(s)/case/'scene.xml').exists())
    frames=load_frames(folder/'observations','rgb')
    selected,ids,indices,center,normal,*_=observed_initialization(frames,JointSettings(),4101)
    problem=WeightedAxisSplineProblem(selected[indices]*1000,center*1000,normal,ids[indices])
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True,sharex=True,sharey=True)
    for ax,mode in zip(axes,MODES):
        r=next(r for r in dev if r['case']==case and r['camera_set']=='all' and r['method']==f'{mode}_equal_trust')
        d=r['diagnostics']; state=problem.evaluate(np.array(d['axis_parameters']))
        weights=np.array(d['fit_point_weights']); colors=weights/weights.max()
        sc=ax.scatter(state['s'],np.linalg.norm(problem.points-state['origin']-state['s'][:,None]*state['normal'],axis=1),
                      c=colors,s=3,cmap='cividis',vmin=0,vmax=1,rasterized=True)
        ax.plot(np.array(d['profile_s_m'])*1000,np.array(d['profile_radius_m'])*1000,'r-',lw=1)
        ax.set(title=f'{mode}: normal error {r["normal_deg"]:.3f} deg',xlabel='s along fitted axis (mm)')
    axes[0].set_ylabel('Distance to fitted axis (mm)')
    fig.colorbar(sc,ax=axes,label='Actual fit weight / maximum in this method')
    fig.savefig(args.output/'development_weight_profiles.png',dpi=170)
    print(args.output)


if __name__=='__main__':main()
