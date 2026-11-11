#!/usr/bin/env python3
"""Paired engineering audit, verifier negative controls and concise figures."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from neck_localization.geometry import Estimate
from neck_localization.task_surface_validation import revalidate_task_candidate,SURFACE_REASON
from neck_localization.visible_profile_baseline import revalidate_candidate
from scripts import evaluate_capability_depth as shared
from scripts.run_depth_profile_baseline import load_frames


def run(evaluations,output):
    output.mkdir(parents=True,exist_ok=False);rows=[];summary=[];changes=[];negatives=[];tracked={};regression=[]
    for name,directory in evaluations:
        provenance=shared._json(directory/'provenance.json');records=json.loads(gzip.decompress((directory/'records.json.gz').read_bytes()))
        for row in records:row['dataset']=name
        rows.extend(records)
        for item in shared._json(directory/'summary.json'):summary.append(dict(dataset=name,**item))
        cache=shared._json(provenance['foreground_cache']);entries={(e['layout'],e['case'],e['camera']):e for e in cache['entries']}
        pairs={}
        for row in records:pairs.setdefault((row['layout'],row['case'],row['condition']),{})[row['backend']]=row
        for key,group in pairs.items():
            base,task=group['visible_profile'],group['task_surface']
            if base.get('estimate')!=task.get('estimate') or base['accepted']!=task['accepted']:
                changes.append(dict(dataset=name,layout=key[0],case=key[1],condition=key[2],
                    baseline=dict(accepted=base['accepted'],center_mm=base['center_mm'],normal_deg=base['normal_deg'],reasons=base['reasons']),
                    task=dict(accepted=task['accepted'],center_mm=task['center_mm'],normal_deg=task['normal_deg'],reasons=task['reasons']),
                    candidate_changed=base['diagnostics'].get('candidate_index')!=task['diagnostics'].get('candidate_index')))
            # Verifier-only stress controls: do NOT claim re-fitted end-to-end
            # precision. Keep all existing non-RGB reasons for fair comparison.
            if key[2]!='clean' or not base.get('estimate'):continue
            masks=[];camera_entries=[]
            for cam in ['cam_a','cam_b','cam_c']:
                e=entries[(key[0],key[1],cam)];camera_entries.append(e)
                p=Path(e['mask_path']);tracked[str(p)]=shared.file_hash(p)
                masks.append(cv2.imread(str(p),0)>0)
            frames=load_frames(Path(camera_entries[0]['rgb_path']).parent,'rgb')
            for label,shift,scale in [('shift_x_150mm',[.15,0,0],1.),('shift_y_150mm',[0,.15,0],1.),('radius_times3',[0,0,0],3.)]:
                d=deepcopy(base['diagnostics']);shift=np.array(shift)
                d['axis_origin_m']=(np.array(d['axis_origin_m'])+shift).tolist()
                d['profile_radius_m']=(np.array(d['profile_radius_m'])*scale).tolist()
                original=Estimate(center=np.array(base['estimate']['center_m'])+shift,normal=np.array(base['estimate']['normal']),
                    radius=base['estimate']['radius_m']*scale,accepted=False,
                    reasons=[r for r in base['reasons'] if r!=SURFACE_REASON],diagnostics=d)
                context=revalidate_candidate(original,frames,masks)
                task_neg=revalidate_task_candidate(context,frames,masks)
                ungated=not [r for r in context.reasons if r!=SURFACE_REASON]
                negatives.append(dict(dataset=name,case=key[1],perturbation=label,
                    full_context_accept=context.accepted,task_accept=task_neg.accepted,no_RGB_accept=ungated,
                    reasons=task_neg.reasons,scope='verifier-only synthetic corruption; stale depth fit is deliberate; NOT end-to-end false-accept metric'))
        # Where prior records exist, assert the shared fitter reproduced both
        # unchanged controls exactly (not just their acceptance counts).
        prior={'fs12':'geometry_fs_sam2_audit_20261027_v1','ss12':'geometry_simsense_sam2_audit_20261027_v1',
            'fs8':'geometry_fs_sam2_confirmation_20261027_v1','ss8':'geometry_simsense_sam2_confirmation_20261027_v1'}.get(name)
        if prior:
            previous=json.loads(gzip.decompress((ROOT/'experiments/neck_localization/results'/prior/'records.json.gz').read_bytes()))
            index={(r['layout'],r['case'],r['condition'],r['backend']):r for r in previous}
            for row in records:
                if row['backend'] not in ['visible_profile','standard_sor']:continue
                old=index[(row['layout'],row['case'],row['condition'],row['backend'])]
                if row['accepted']!=old['accepted'] or row['reasons']!=old['reasons']:raise ValueError('frozen_control_decision_changed')
                differences={k:float(np.max(abs(np.asarray(row['estimate'][k])-old['estimate'][k])))
                    for k in ['center_m','normal','radius_m']}
                # Thread-count changes produce ~2e-10 unit-normal roundoff.
                # This is numerical replay tolerance, not a task gate change.
                for k in differences:np.testing.assert_allclose(row['estimate'][k],old['estimate'][k],rtol=0,atol=1e-9)
                regression.append(dict(dataset=name,case=row['case'],condition=row['condition'],backend=row['backend'],
                    matched=True,absolute_differences=differences,numerical_tolerance=1e-9))
    assert all(shared.file_hash(p)==h for p,h in tracked.items())
    for filename,value in [('summary',summary),('changed_cases',changes),('verifier_negatives',negatives),('baseline_replay',regression)]:
        shared._write_json(output/(filename+'.json'),value)
    (output/'records.json.gz').write_bytes(gzip.compress(json.dumps(rows,allow_nan=False).encode(),mtime=0))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for ax,metric,title in zip(axs,['false_reject_center_pass_4mm','false_accept_axial_band'],
        ['False rejects of <=4mm center candidates','Accepted but outside annotated axial band']):
        labels=[];a=[];b=[]
        for dataset in ['fs8','fs_new8','ss_new8']:
            for condition in ['clean','shared_noise']:
                g=[r for r in summary if r['dataset']==dataset and r['condition']==condition]
                if not g:continue
                labels.append(dataset+'/'+condition.replace('shared_noise','noise'))
                a.append(next(r[metric] for r in g if r['backend']=='visible_profile'))
                b.append(next(r[metric] for r in g if r['backend']=='task_surface'))
        x=np.arange(len(labels));ax.bar(x-.18,a,.36,label='Full-context gate');ax.bar(x+.18,b,.36,label='Task-surface gate')
        ax.set(title=title,xticks=x,xticklabels=labels,ylabel='Count / 8');ax.tick_params(axis='x',rotation=30);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'engineering_gate_comparison.png',dpi=160);plt.close(fig)
    shared._write_json(output/'validation.json',dict(passed=True,records=len(rows),baseline_records_replayed=len(regression),
        negative_controls=len(negatives),negative_task_rejected=sum(not r['task_accept'] for r in negatives),
        negatives_are_not_end_to_end=True,tracked_files=len(tracked),no_hardware=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',action='append',required=True,help='label=/path/to/result')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run([(v.split('=',1)[0],Path(v.split('=',1)[1])) for v in a.evaluation],a.output)
