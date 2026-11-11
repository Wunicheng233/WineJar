#!/usr/bin/env python3
"""All-attempt success rates, errors, timings and recovery paths."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from neck_localization.evaluation import summarize


def table(rows,title):
    lines=[f'## {title}','',
        '成功=接受且圆心≤4 mm/法向≤2°/直径≤2 mm，后两项仍为分析阈值；拒绝和错误输出计失败。',
        '误差含已求出的拒绝候选，未求出不填零；误差/耗时为中位数 / P95 / 最大。','',
        '|组|视图|方法|成功/拒绝/错误接受/总数|圆心mm|法向°|直径mm|耗时ms|>1000ms|',
        '|---|---|---|---|---|---|---|---|---|']
    for s in summarize(rows):
        rr=[r for r in rows if r['split']==s['split'] and r['method']==s['method'] and r['factor']==s['factor']]
        counts=[sum(r['accepted'] and r['truth_pass'] for r in rr),sum(not r['accepted'] for r in rr),sum(r['accepted'] and not r['truth_pass'] for r in rr),len(rr)]
        cells=[]
        for metric in ['center_mm','normal_deg','diameter_mm','runtime_ms']:
            cells.append(' / '.join(f'{s[metric+"_"+q]:.3f}' if s[metric+'_'+q] is not None else '—' for q in ['median','p95','max']))
        lines.append('|'+rr[0]['group']+'|'+rr[0]['camera_set']+'|'+s['method']+'|'+'/'.join(map(str,counts))+'|'+'|'.join(cells)+'|'+str(sum(r['over_illustrative_1000ms'] for r in rr))+'|')
    lines+=['','1000 ms仅为离线说明线，真实工位预算未给定；不含采集、传输、机器人执行。','',
            '|组|视图|方法|XYZ最大绝对误差mm|沿法向/切向最大mm|','|---|---|---|---|---|']
    for s in summarize(rows):
        rr=[r for r in rows if r['split']==s['split'] and r['method']==s['method'] and r['factor']==s['factor'] and r['center_mm'] is not None]
        if not rr:continue
        xyz=np.max(np.abs([r['xyz_mm'] for r in rr]),axis=0)
        lines.append('|'+rr[0]['group']+'|'+rr[0]['camera_set']+'|'+s['method']+'|'+' / '.join(f'{v:.3f}' for v in xyz)+
                     f'|{max(abs(r["normal_component_mm"]) for r in rr):.3f} / {max(r["tangent_mm"] for r in rr):.3f}|')
    lines+=['','拒绝/错误接受：','']
    for r in rows:
        if not r['accepted'] or not r['truth_pass']:
            lines.append(f'- {r["case"]} / {r["camera_set"]} / {r["method"]}: '+('拒绝 '+','.join(r['reasons']) if not r['accepted'] else '错误接受'))
    return lines


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development',type=Path,required=True)
    p.add_argument('--test',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--primary-v2',action='store_true',help='Report the second independent three-camera test')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    dev=json.loads(gzip.decompress((args.development/'records.json.gz').read_bytes()))
    test=json.loads(gzip.decompress((args.test/'records.json.gz').read_bytes()))
    lines=['# 正确输出率改进：理想观测阶段','',
        '开发36个已看随机坛型；新测试48个在实现前锁定，开发后冻结；同一圆截面模型家族、固定相机、准确K/T。',
        '三方法：frozen_region（旧区域）、stable_local（连续核心+阻尼+留出验证）、adaptive_validated（再加范围扩张/验证备用解）。',
        '新两方法25%确定性留出点不参与轴/曲线目标，但初始ROI/口沿和支撑检查仍使用全部观测；不是独立相机或独立标定验证。',
        '更改了支持模型和验证结构，没有靠真值放行、删除所有质量检查或调高误差限。',
        '尚不包含5 mm带宽、相机误差、输送或机械臂端到端任务。','']
    if args.primary_v2:
        lines[:]=['# 目标范围验证备用解：第二次独立评价','',
            '首批新测试暴露核心跨度限制：v1三视图46/48、旧法47/48；原结果保持冻结。',
            '首批48例现仅作为v2开发，另锁新种子32常规+16压力；本轮只评价当前三相机主工况。',
            '对照frozen_region、adaptive_validated（冻结v1）、task_validated_output（v2）。',
            'v2先运行v1；必要时用训练点得到备用轴，以完整目标范围、留出点、分相机残差、信息和候选一致性验证。',
            '不要求备用解具有3mm内部轴核心；目标范围至少1.5mm，残差0.4mm/留出均值0.2mm限保持。',
            '留出点不能校验共同标定偏差；尚无5mm带宽/输送/机器人验证。','']
    lines+=table(test,'新测试')+['']+table(dev,'开发（非未见测试）')
    lines+=['','## 扩张与备用路径（新测试）','',
            '|组|视图|28mm接受|42mm接受|56mm接受|备用固定权重接受|无输出或拒绝|','|---|---|---|---|---|---|']
    for group in ['nominal','stress']:
        for cs in ['cam_a','cam_a,cam_b','all']:
            rr=[r for r in test if r['group']==group and r['camera_set']==cs and r['method']=='adaptive_validated']
            depths=[sum(r['accepted'] and r['diagnostics'].get('candidate_depth_mm')==depth for r in rr) for depth in [28,42,56]]
            backup=sum(r['accepted'] and r['diagnostics'].get('chosen_path')=='validated_fixed_weight_alternative' for r in rr)
            lines.append(f'|{group}|{cs}|'+ '|'.join(map(str,depths+[backup,sum(not r['accepted'] for r in rr)]))+'|')
    (args.output/'tables.md').write_text('\n'.join(lines)+'\n')
    for label,source in [('development',args.development),('test',args.test)]:
        (args.output/f'{label}_records.json.gz').write_bytes((source/'records.json.gz').read_bytes())
    methods=(['frozen_region','adaptive_validated','task_validated_output'] if args.primary_v2 else ['frozen_region','stable_local','adaptive_validated'])
    cameras=['all'] if args.primary_v2 else ['cam_a','cam_a,cam_b','all']
    fig,axes=plt.subplots(1,len(cameras),figsize=(6 if args.primary_v2 else 13,4),constrained_layout=True)
    for ax,cs in zip(np.atleast_1d(axes),cameras):
        for i,group in enumerate(['nominal','stress']):
            ss=[]
            for m in methods:
                rr=[r for r in test if r['group']==group and r['camera_set']==cs and r['method']==m]
                ss.append(sum(r['accepted'] and r['truth_pass'] for r in rr)/len(rr))
            ax.bar(np.arange(3)+(i-.5)*.35,ss,width=.35,label=group)
        ax.set(title=cs,ylabel='Correct-output fraction',ylim=(0,1.06),xticks=np.arange(3),xticklabels=(['old region','v1','task fallback'] if args.primary_v2 else ['old region','stable','adaptive']))
        ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(args.output/'correct_output_rates.png',dpi=160)
    print(args.output)


if __name__=='__main__':main()
