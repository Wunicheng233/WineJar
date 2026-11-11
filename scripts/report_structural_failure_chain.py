#!/usr/bin/env python3
"""Summarize paired diagnostic interventions, without calling them success rates."""
import argparse
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.report_structural_stage import render_root


def run(chain,sensor,source,output):
    output.mkdir(parents=True,exist_ok=False)
    rows=json.loads(gzip.decompress((chain/'records.json.gz').read_bytes()))
    sensor_rows=json.loads(gzip.decompress((sensor/'readouts.json.gz').read_bytes()))
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    methods=['frozen_baseline','readout_axis_weights','oracle_axis','oracle_depth_same_pixels','oracle_axis_and_depth']
    labels=['Frozen candidate','Axis weights for readout','Oracle axis only','Oracle depth only','Oracle axis + depth']
    for ax,layout in zip(axs,['far','close']):
        counts=[sum(r['truth_pass'] for r in rows if r['layout']==layout and r['condition_name']=='stereo_raw' and r['intervention']==m) for m in methods]
        bars=ax.barh(labels,counts,color=['#888888','#888888','#3d85bb','#3d85bb','#3d85bb'])
        ax.bar_label(bars,padding=3);ax.set(xlim=(0,13),title=layout+' / same 12 jars',xlabel='Candidates meeting geometric thresholds (of 12)');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.savefig(output/'paired_diagnostics.png',dpi=170);plt.close(fig)
    case='structural_test_stress_003';layout='far'
    original=next(r for r in rows if r['layout']==layout and r['case']==case and r['condition_name']=='stereo_raw' and r['intervention']=='frozen_baseline')
    d=original['diagnostics'];n=np.array(original['estimate']['normal']);a=np.array(d['axis_origin_m'])*1000
    truth=json.loads((render_root(source,layout)/'geometry'/case/'evaluation_only/target.json').read_text())
    offset=(np.array(truth['center_m'])*1000-a)@n
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    colors=['#d88a15','#3d85bb','#35a596']
    for color,name,label in zip(colors,['measured_common','ideal_IR_registered_common','ideal_RGB_common'],['Measured stereo + registration','Ideal IR + quantization + registration','Ideal RGB depth']):
        r=next(r for r in sensor_rows if r['layout']==layout and r['case']==case and r['intervention']==name)
        curve=r['curve'];e=r['estimate'];x=np.array(curve['grid_mm'])-offset
        ax.plot(x,curve['radius_mm'],color=color,lw=2,label=label)
        position=(np.array(e['center_m'])*1000-a)@n-offset
        ax.scatter([position],[e['radius_m']*1000],color=color,marker='x',s=80,zorder=5)
    ax.axvline(0,color='0.4',ls='--',label='True target axial position (evaluation only)')
    ax.scatter([0],[truth['radius_m']*1000],marker='+',color='black',s=100,zorder=6)
    ax.set(xlabel='Position along SAME estimated axis, relative to true center [mm]',ylabel='Fitted radius [mm]',title='Your screenshot: far / stress_003; same axis, pixels, weights and readout range')
    ax.grid(alpha=.2);ax.legend(fontsize=9);fig.savefig(output/'stress_003_depth_intervention.png',dpi=180);plt.close(fig)
    probes=json.loads((chain/'solver_probes.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for r in probes:
        label=r['layout']+'/'+r['case'].replace('structural_test_','')
        axs[0].scatter([z['objective']/r['original_objective'] for z in r['restarts']],[z['normal_deg'] for z in r['restarts']],label=label)
    axs[0].set(xlabel='Fixed-weight objective / original objective',ylabel='Normal error [degrees]',title='Lower objective can give WORSE geometry');axs[0].grid(alpha=.2);axs[0].legend(fontsize=7)
    data=json.loads((sensor/'camera_diagnostics.json').read_text())
    for layout,color in [('far','#d88a15'),('close','#3d85bb')]:
        g=[r for r in data if r['layout']==layout and r['case']==case and r['space']=='native_IR_all_visible_neck_common_valid']
        for r in g:
            bins=[b for b in r['axial_bin_bias'] if b['n']>=10]
            axs[1].plot([b['s_mm'] for b in bins],[b['median'] for b in bins],marker='.',label=layout+'/'+r['camera'])
    axs[1].axhline(0,color='0.4',lw=1);axs[1].set(xlabel='True axial position relative to target [mm]',ylabel='Measured - ideal radial distance [mm]',title='Native IR: bias varies with position and view');axs[1].grid(alpha=.2);axs[1].legend(fontsize=7)
    fig.savefig(output/'solver_and_native_bias.png',dpi=170);plt.close(fig)
    # Compact machine-readable matched transfer summary.
    summary=[]
    for layout in ['far','close']:
        for name in ['measured_common','ideal_IR_registered_common','ideal_RGB_common']:
            group=[r for r in sensor_rows if r['layout']==layout and r['intervention']==name]
            summary.append(dict(layout=layout,intervention=name,n=len(group),geometric_pass=sum(r['truth_pass'] for r in group),
                minimum_common_fraction=min(r['common_points']/r['original_points'] for r in group),
                diameter_median_mm=float(np.median([r['diameter_mm'] for r in group])),diameter_max_mm=max(r['diameter_mm'] for r in group)))
    (output/'sensor_summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['chain','sensor','run','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.chain,a.sensor,a.run,a.output)
