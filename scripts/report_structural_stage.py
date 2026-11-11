#!/usr/bin/env python3
"""Summarize EVERY frozen test and diagnose true-circle silhouette assumptions."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from neck_localization.geometry import circle_points
from neck_localization.stereo_robust import extract_stereo_masks
from neck_localization.structure_portfolio import target_contact
from scripts.run_depth_profile_baseline import load_frames


def load_rows(path):return json.loads(gzip.decompress(path.read_bytes()))


def render_root(source,layout):
    mapping=source/layout/'input_source.json'
    name=json.loads(mapping.read_text())['render_directory'] if mapping.exists() else 'render'
    return source/layout/name


def aggregate(rows):
    n=len(rows);accepted=[r for r in rows if r['accepted']]
    result=dict(n=n,correct=sum(r['accepted'] and r['truth_pass'] for r in rows),
        rejected=n-len(accepted),false_accepted=sum(r['accepted'] and not r['truth_pass'] for r in rows),
        solved=sum(r.get('estimate') is not None for r in rows),
        band_near_min=sum(bool(r['accepted'] and r['truth_pass'] and r.get('band') and
            r['band']['nominal_compatible'] and r['band_evaluation']['near_min_proxy']['contained']) for r in rows),
        band_full_neck=sum(bool(r['accepted'] and r['truth_pass'] and r.get('band') and
            r['band']['nominal_compatible'] and r['band_evaluation']['full_annotated_neck']['contained']) for r in rows))
    result.update(success_rate=result['correct']/n,rejection_rate=result['rejected']/n,
        false_accept_rate=result['false_accepted']/n)
    for key in ['center_mm','normal_deg','diameter_mm','normal_component_mm','tangent_mm','runtime_ms']:
        values=[abs(r[key]) for r in rows if r.get(key) is not None]
        for label,percent in [('median',50),('p95',95),('max',100)]:
            result[key+'_'+label]=float(np.percentile(values,percent)) if values else None
    result['reasons']={s:sum(s in r['reasons'] for r in rows) for s in sorted({s for r in rows for s in r['reasons']})}
    return result


def write_csv(path,rows):
    keys=sorted({k for r in rows for k in r})
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def run(source,output):
    assert (source/'completed.json').exists(),'Study incomplete'
    output.mkdir(parents=True,exist_ok=False)
    all_rows=[];summaries=[];groups=['main','matching','small_matching','phase_estimates']
    for layout in ['far','close']:
        for group in groups:
            rows=load_rows(source/layout/group/'records.json.gz')
            for r in rows:r['layout']=layout;r['study_group']=group
            all_rows.extend(rows)
            for mode,method in sorted({(r['condition_name'],r['method']) for r in rows}):
                selected=[r for r in rows if r['condition_name']==mode and r['method']==method]
                summaries.append(dict(layout=layout,mode=mode,method=method,**aggregate(selected)))
    (output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    write_csv(output/'summary.csv',summaries)
    write_csv(output/'cases.csv',[{k:v for k,v in r.items() if k not in ['diagnostics','estimate','band','band_evaluation']} for r in all_rows])
    # Counterfactual gate removal is post-hoc diagnosis, NOT a new estimator.
    counterfactual=[]
    for layout,mode,method in sorted({(r['layout'],r['condition_name'],r['method']) for r in all_rows if r['method'].startswith('portfolio')}):
        rows=[r for r in all_rows if (r['layout'],r['condition_name'],r['method'])==(layout,mode,method)]
        affected=[r for r in rows if r['reasons']==['candidate_inconsistent_with_rgb_silhouette']]
        counterfactual.append(dict(layout=layout,mode=mode,method=method,
            diagnostic_only=True,additional_correct=sum(r['truth_pass'] for r in affected),
            additional_wrong=sum(not r['truth_pass'] for r in affected)))
    (output/'gate_counterfactual.json').write_text(json.dumps(counterfactual,indent=2)+'\n')
    # Test the validation ASSUMPTION on an exact known circle, with both the
    # extracted RGB mask and an ideal renderer silhouette. Never used to infer.
    true_contacts=[]
    for layout in ['far','close']:
        config=json.loads((render_root(source,layout)/'effective_config.json').read_text())
        for case in config['cases']:
            name=case['name'];scene=render_root(source,layout)/'ideal_rgb/scenes'/name
            frames=load_frames(scene/'observations','rgb');masks,_=extract_stereo_masks(frames)
            truth=json.loads((scene/'evaluation_only/target.json').read_text())
            center=np.array(truth['center_m'])*1000;normal=np.array(truth['normal']);radius=truth['radius_m']*1000
            for label,mm in [('observed_RGB',masks),('ideal_silhouette_diagnostic',[f.depth_z_m>0 for f in frames])]:
                errors,pixels=target_contact(frames,mm,center,normal,radius)
                true_contacts.append(dict(layout=layout,case=name,mask=label,diagnostic_only=True,
                    true_circle_contact_rms_pixels=float(np.sqrt(np.mean(errors**2))),
                    maximum_abs_pixels=float(np.max(abs(errors))),signed_errors_pixels=errors.tolist(),contact_pixels=pixels))
    (output/'true_circle_contact_diagnostic.json').write_text(json.dumps(true_contacts,indent=2)+'\n')
    # Compact rates figure: no success-only error filtering.
    methods=['baseline','portfolio_plain','portfolio_robust','portfolio','portfolio_single','portfolio_anisotropic','portfolio_fixed']
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharey=True)
    for row,layout in enumerate(['far','close']):
        for col,mode in enumerate(['ideal_rgb','stereo_raw','narrow5']):
            ax=axes[row,col]
            rr=[next(s for s in summaries if (s['layout'],s['mode'],s['method'])==(layout,mode,m)) for m in methods]
            x=np.arange(len(rr));good=np.array([s['correct'] for s in rr]);bad=np.array([s['false_accepted'] for s in rr])
            ax.bar(x,good,color='#438b70',label='Correct output')
            ax.bar(x,bad,bottom=good,color='#d47762',label='Wrong accepted')
            ax.bar(x,[s['rejected'] for s in rr],bottom=good+bad,color='#b7c1ca',label='Rejected (failure)')
            ax.set_xticks(x,['old','plain','robust','region','single','variance','rim axis'],rotation=40,ha='right',fontsize=8)
            ax.set_ylim(0,12);ax.set_title(layout+' / '+mode);ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Frozen independent 12-jar test, paired fixed layouts; true K/T')
    fig.tight_layout();fig.savefig(output/'output_rates.png',dpi=160);plt.close(fig)
    # All cases in an atlas; the crops are evaluation presentation only.
    for layout in ['far','close']:
        rows=[r for r in all_rows if r['layout']==layout and r['condition_name']=='stereo_raw' and r['method']=='portfolio']
        fig,axes=plt.subplots(3,4,figsize=(15,10))
        for ax,r in zip(axes.ravel(),rows):
            scene=render_root(source,layout)/'stereo_raw/scenes'/r['case'];frames=load_frames(scene/'observations','rgb');f=frames[0]
            truth=json.loads((scene/'evaluation_only/target.json').read_text())
            ring=f.camera.project(circle_points(truth['center_m'],truth['normal'],truth['radius_m'],180))
            ax.imshow(f.rgb)
            ax.plot(ring[:,0],ring[:,1],color='#55e4f0',lw=.8,ls='--')
            n=f.camera.project(np.array([truth['center_m'],np.array(truth['center_m'])+.035*np.array(truth['normal'])]))
            ax.plot(n[:,0],n[:,1],color='#55e4f0',lw=.8)
            if r.get('estimate'):
                e=r['estimate'];p=f.camera.project(circle_points(e['center_m'],e['normal'],e['radius_m'],180))
                color='#e66c62' if r['accepted'] else '#e9b45e'
                ax.plot(p[:,0],p[:,1],color=color,lw=.8,ls='-' if r['accepted'] else '--')
                q=f.camera.project(np.array([e['center_m'],np.array(e['center_m'])+.035*np.array(e['normal'])]))
                ax.plot(q[:,0],q[:,1],color=color,lw=.8)
            ax.set_xlim(ring[:,0].min()-24,ring[:,0].max()+24)
            ax.set_ylim(ring[:,1].max()+34,ring[:,1].min()-36)
            title=r['case'].replace('structural_test_','')+(' / accepted' if r['accepted'] else ' / rejected')
            if r['center_mm'] is not None:title+=f"\nc={r['center_mm']:.2f}mm n={r['normal_deg']:.2f}deg d={r['diameter_mm']:.2f}mm"
            ax.set_title(title,fontsize=8);ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(layout+' / raw stereo / region portfolio: cyan truth, red accepted, orange REJECTED candidate\nAll 12 cases; camera A only, not evidence of full 3D accuracy')
        fig.tight_layout();fig.savefig(output/(layout+'_all_cases.png'),dpi=170);plt.close(fig)
    (output/'scope.json').write_text(json.dumps(dict(total_records=len(all_rows),layouts=2,independent_jar_geometries=12,
        paired_layouts_not_24_independent_jars=True,parameter_tuning_uses_only_old_development=True,
        no_calibrated_uncertainty=True,no_real_hardware=True,all_rejections_are_failures=True),indent=2)+'\n')
    print('summaries',len(summaries),'records',len(all_rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run,a.output)
