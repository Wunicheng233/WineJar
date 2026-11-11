#!/usr/bin/env python3
"""Summarize a frozen bias compensation run without changing its records."""
import argparse,csv,gzip,json
from pathlib import Path
import numpy as np


def aggregate(rows,method):
    rr=[r for r in rows if r['method']==method]
    def good(r,policy):
        return bool(r['accepted'] and r['truth_pass'] and r.get('band') and r['band']['nominal_compatible']
                    and r['band_evaluation'][policy]['contained'])
    result=dict(method=method,n=len(rr),accepted=sum(r['accepted'] for r in rr),
        correct_circle=sum(r['accepted'] and r['truth_pass'] for r in rr),rejected=sum(not r['accepted'] for r in rr),
        wrong_accepted=sum(r['accepted'] and not r['truth_pass'] for r in rr),
        near_min_task_success=sum(good(r,'near_min_proxy') for r in rr),
        full_neck_task_success=sum(good(r,'full_annotated_neck') for r in rr),
        geometry_candidate_n=sum(r['center_mm'] is not None for r in rr))
    for metric in ['center_mm','normal_component_mm','tangent_mm','normal_deg','diameter_mm','runtime_ms']:
        vals=[abs(r[metric]) for r in rr if r.get(metric) is not None]
        for name,p in [('median',50),('p95',95),('max',100)]:
            result[metric+'_'+name]=float(np.percentile(vals,p)) if vals else None
    errors=[abs(e) for r in rr for e in r.get('evaluation_only_bias_error_mm',{}).values()]
    for name,p in [('median',50),('p95',95),('max',100)]:
        result['estimated_bias_error_mm_'+name]=float(np.percentile(errors,p)) if errors else None
    for policy in ['near_min_proxy','full_annotated_neck']:
        vals=[r['band_evaluation'][policy]['minimum_margin_mm'] for r in rr if r.get('band_evaluation')]
        result[policy+'_minimum_margin_mm']=min(vals) if vals else None
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=json.load(gzip.open(args.run/'records.json.gz','rt'))
    plan=json.loads((args.run/'plan.json').read_text())
    methods=['frozen_v2']+[m['name'] for m in plan['methods']]
    overall=[aggregate(rows,m) for m in methods]
    (args.output/'overall.json').write_text(json.dumps(overall,indent=2)+'\n')
    with (args.output/'overall.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(overall[0]));w.writeheader();w.writerows(overall)
    summary=json.loads((args.run/'summary.json').read_text())
    with (args.output/'conditions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,dict) else v for k,v in r.items()} for r in summary)
    basekeys=['method','condition_name','case','group','accepted','truth_pass','center_mm','normal_component_mm','tangent_mm','normal_deg','diameter_mm','runtime_ms','reasons']
    cases=[]
    for r in rows:
        item={k:r.get(k) for k in basekeys}
        d=r.get('diagnostics',{}).get('bias_compensation',{})
        for key in ['applied','estimated_bias_by_camera_mm','compensation_reasons','fit_evaluations','heldout_weighted_rms_before_mm','heldout_weighted_rms_after_mm']:
            value=d.get(key);item[key]=json.dumps(value) if isinstance(value,(dict,list)) else value
        item['data_condition']=d.get('data_information',{}).get('scaled_data_condition')
        item['evaluation_only_bias_error_mm']=json.dumps(r.get('evaluation_only_bias_error_mm'))
        for policy in ['near_min_proxy','full_annotated_neck']:
            item[policy+'_minimum_margin_mm']=(r.get('band_evaluation') or {}).get(policy,{}).get('minimum_margin_mm')
        cases.append(item)
    with (args.output/'cases.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conditions=plan['conditions']
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    matrix=np.array([[next(s['correct_circle']/s['n'] for s in summary if s['method']==m and s['condition_name']==c['name'] and s['group']=='all') for c in conditions] for m in methods])
    axes[0,0].imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    axes[0,0].set(xticks=np.arange(len(conditions)),xticklabels=[c['name'] for c in conditions],yticks=np.arange(len(methods)),yticklabels=methods,title='Correct output / all cases (including rejection failures)')
    axes[0,0].tick_params(axis='x',rotation=35,labelsize=7)
    for i in range(len(methods)):
        for j in range(len(conditions)):axes[0,0].text(j,i,f'{matrix[i,j]*100:.0f}%',ha='center',va='center',color='white' if matrix[i,j]>.5 else 'black',fontsize=8)
    x=np.arange(len(methods))
    for k,label in [('center_mm_p95','Centre P95'),('diameter_mm_p95','Diameter P95')]:
        axes[0,1].plot(x,[s[k] for s in overall],'o-',label=label)
    axes[0,1].set(xticks=x,xticklabels=methods,ylabel='Error [mm]',title='Solved candidates, including rejected')
    axes[0,1].legend(fontsize=8)
    for k,label in [('runtime_ms_median','Median'),('runtime_ms_p95','P95')]:
        axes[1,0].plot(x,[s[k] for s in overall],'o-',label=label)
    axes[1,0].set(xticks=x,xticklabels=methods,ylabel='Estimator runtime [ms]',title='Baseline timing reused; not a controlled speed benchmark')
    axes[1,0].legend(fontsize=8)
    for method in methods[1:]:
        vals=[abs(e) for r in rows if r['method']==method for e in r.get('evaluation_only_bias_error_mm',{}).values()]
        axes[1,1].hist(vals,bins=40,histtype='step',label=method)
    axes[1,1].set(xlabel='Absolute estimated bias error [mm]',ylabel='Camera estimates',title='Bias is a nuisance estimate, not validated device calibration')
    axes[1,1].legend(fontsize=8)
    for ax in [axes[0,1],axes[1,0]]:ax.tick_params(axis='x',labelsize=8);ax.grid(alpha=.2)
    fig.suptitle('Circular-section geometry + assumed additive depth bias; no real-device accuracy claim',fontsize=11)
    fig.savefig(args.output/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(overall,indent=2))


if __name__=='__main__':main()
