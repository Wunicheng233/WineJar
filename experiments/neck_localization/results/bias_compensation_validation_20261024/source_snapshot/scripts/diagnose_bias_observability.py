#!/usr/bin/env python3
"""Local DATA-only bias/shape identifiability and derivative diagnostics."""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize._numdiff import approx_derivative

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from neck_localization.bias_profile import BiasAxisSplineProblem,observation_rays
from neck_localization.reliable_output import _prepare
from neck_localization.support_feedback import WeightedAxisSplineProblem
from scripts.run_depth_profile_baseline import load_frames


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=ROOT/'experiments/neck_localization/runs/depth_bias_stage1_20261023_v1/zero'
    cases=['nominal_000','nominal_002','stress_000','stress_003']
    angles=[85,45,20,10,5]
    data=json.load(gzip.open(root/'records.json.gz','rt'))
    rows=[];derivatives=[]
    for case in cases:
        row=next(r for r in data if r['case']==case)
        frames=load_frames(root/'scenes'/case/'observations','rgb')
        prep=_prepare(frames,row['diagnostics']['candidate_depth_mm'])
        base=prep['problem'];ids=prep['ids'][prep['train']]
        rays=observation_rays(base.points/1000,ids,frames)
        x=np.r_[row['diagnostics']['axis_parameters'],[0,0,0]]
        a,n,_=base.axis(x[:4]);delta=base.points-a;s=delta@n;rad=delta-s[:,None]*n
        camera=np.array([f.camera.T_world_from_camera[:3,3]*1000 for f in frames])
        view=camera[ids]-base.points;view-=(view@n)[:,None]*n
        cosine=np.sum(rad*view,axis=1)/(np.linalg.norm(rad,axis=1)*np.linalg.norm(view,axis=1))
        masks=[cosine>np.cos(np.deg2rad(angle)) for angle in angles]
        each=min(int(np.min(np.bincount(ids[m],minlength=3))) for m in masks)
        for angle,mask in zip(angles,masks):
            for mode in ['natural_count','equal_count_per_camera']:
                indices=np.flatnonzero(mask)
                if mode=='equal_count_per_camera':
                    indices=np.concatenate([local[np.linspace(0,len(local)-1,each).astype(int)] for j in range(3)
                        if len(local:=np.flatnonzero(mask&(ids==j)))])
                b=WeightedAxisSplineProblem(base.points[indices],base.origin,base.n0,ids[indices],base.settings)
                b.set_weights(np.ones(len(indices)))
                p=BiasAxisSplineProblem(b,ids[indices],rays[indices])
                info=p.data_information(x)
                rows.append(dict(case=case,half_angle_deg=angle,mode=mode,n=len(indices),**info))
        # Derivative convergence on real observed points; no GT axis/label.
        base.set_weights(np.ones(len(base.points)))
        p=BiasAxisSplineProblem(base,ids,rays,sigma_bias_mm=2.)
        x=x+np.array([.1,-.1,.002,-.001,.3,-.2,.1])
        analytical=p.jac(x)
        for h in [1e-3,1e-4,1e-5,1e-6,1e-7]:
            numerical=approx_derivative(p.fun,x,method='3-point',abs_step=h)
            derivatives.append(dict(case=case,step=h,max_abs_difference_by_parameter=np.max(np.abs(analytical-numerical),axis=0).tolist()))
    (args.output/'observability.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'derivative_convergence.json').write_text(json.dumps(derivatives,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for mode in ['natural_count','equal_count_per_camera']:
        for case in cases:
            selected=[r for r in rows if r['case']==case and r['mode']==mode]
            axes[0].semilogy([r['half_angle_deg'] for r in selected],[r['scaled_data_condition'] for r in selected],
                              '.-' if mode=='natural_count' else '.--',label=case+' '+mode,alpha=.7)
    axes[0].set(xlabel='Retained facing half-angle [degrees]',ylabel='Scaled DATA condition (no priors)',title='Narrower views worsen bias/shape separation')
    axes[0].invert_xaxis();axes[0].legend(fontsize=6)
    for case in cases:
        selected=[r for r in derivatives if r['case']==case]
        axes[1].loglog([r['step'] for r in selected],[max(r['max_abs_difference_by_parameter']) for r in selected],'.-',label=case)
    axes[1].set(xlabel='Finite-difference step (mixed coordinates)',ylabel='Maximum Jacobian difference',title='Truncation then rounding: choose parameter-specific steps')
    axes[1].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(args.output/'observability_and_derivatives.png',dpi=150)
    plt.close(fig)
    print('Saved',len(rows),'view-geometry diagnostics and',len(derivatives),'derivative checks')


if __name__=='__main__':main()
