#!/usr/bin/env python3
"""Evaluation-only neck visibility and perfect-circle silhouette-gate audit."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from neck_localization.geometry import Camera,basis
from neck_localization.stereo_robust import extract_stereo_masks
from neck_localization.structure_portfolio import target_contact
from scripts.run_depth_profile_baseline import load_frames
from scripts.report_structural_stage import render_root


def run(source,output):
    rows=[];gates=[]
    for layout in ['far','close']:
        folder=render_root(source,layout)
        config=json.loads((folder/'effective_config.json').read_text())
        for case in config['cases']:
            name=case['name'];geometry=folder/'geometry'/name
            truth=json.loads((geometry/'evaluation_only/target.json').read_text())
            with np.load(geometry/'evaluation_only/profile.npz') as p:ss,rr=p['s_m'].copy(),p['radius_m'].copy()
            sensor=json.loads((geometry/'sensor.json').read_text());n=np.array(truth['normal']);u,v=basis(n)
            frames=load_frames(folder/'ideal_rgb/scenes'/name/'observations','rgb');masks,_=extract_stereo_masks(frames)
            for label,mm in [('observed_RGB',masks),('ideal_silhouette',[f.depth_z_m>0 for f in frames])]:
                error,_=target_contact(frames,mm,np.array(truth['center_m'])*1000,n,truth['radius_m']*1000)
                gates.append(dict(layout=layout,case=name,mask=label,rms_pixels=float(np.sqrt(np.mean(error**2))),
                    true_circle_would_fail=float(np.sqrt(np.mean(error**2)))>2,errors_pixels=error.tolist()))
            for item in sensor['cameras']:
                with np.load(folder/'sensor_diagnostics'/name/(item['name']+'_native.npz')) as native:
                    z=native['ideal_left_z'].copy()
                camera=Camera(item['name'],np.array(item['K_ir']),np.array(item['T_world_from_left_ir']),z.shape[1],z.shape[0])
                yy,xx=np.nonzero(z>0);q=camera.unproject(np.c_[xx,yy],z[yy,xx])
                d=q-truth['center_m'];s=d@n+truth['s_m'];radial=d-(d@n)[:,None]*n
                rho=np.linalg.norm(radial,axis=1);lo,hi=truth['search_interval_s_m']
                good=(s>=lo)&(s<=hi)&(abs(rho-np.interp(s,ss,rr))<.001)
                angles=np.sort(np.mod(np.arctan2(radial[good]@v,radial[good]@u),2*np.pi))
                coverage=float(np.degrees(2*np.pi-np.diff(np.r_[angles,angles[0]+2*np.pi]).max())) if len(angles)>1 else 0.
                ends=np.array(truth['center_m'])+(np.array([lo,hi])-truth['s_m'])[:,None]*n
                span=float(np.linalg.norm(np.diff(camera.project(ends),axis=0)))
                rows.append(dict(layout=layout,case=name,camera=item['name'],visible_native_outer_neck_pixels=int(good.sum()),
                    visible_azimuth_coverage_deg=coverage,projected_axis_span_pixels=span,
                    scope='true outer profile +/-1mm and true neck interval; evaluation ONLY, not estimator selection'))
    output.mkdir(parents=True,exist_ok=False)
    summary={layout:dict(native_outer_neck_pixels_total=sum(r['visible_native_outer_neck_pixels'] for r in rows if r['layout']==layout),
        zero_neck_cameras=sum(r['visible_native_outer_neck_pixels']==0 for r in rows if r['layout']==layout),
        perfect_circle_gate_fails_observed=sum(r['true_circle_would_fail'] for r in gates if r['layout']==layout and r['mask']=='observed_RGB'),
        perfect_circle_gate_fails_ideal=sum(r['true_circle_would_fail'] for r in gates if r['layout']==layout and r['mask']=='ideal_silhouette')) for layout in ['far','close']}
    (output/'visibility.json').write_text(json.dumps(dict(summary=summary,rows=rows,perfect_circle_gate_diagnostic=gates),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run,a.output)
