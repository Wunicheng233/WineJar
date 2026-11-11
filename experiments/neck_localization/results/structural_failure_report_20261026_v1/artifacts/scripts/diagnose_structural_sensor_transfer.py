#!/usr/bin/env python3
"""Oracle diagnostic separating stereo measurement and RGB registration effects.

Uses already saved data only. Same selected RGB rays and a common-valid subset
are used for all readouts; depth replacement is NOT a deployable correction.
"""
import argparse
from hashlib import sha256
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from neck_localization.geometry import Camera
from neck_localization.evaluation import score
from neck_localization.joint_profile import JointSettings
from dataclasses import replace
from scripts.diagnose_structural_failure_chain import read_curve,metric_stats
from scripts.report_structural_stage import render_root
from scripts.run_depth_profile_baseline import load_frames


def outer_labels(points,truth,ss,rr):
    n=np.array(truth['normal']);d=points-np.array(truth['center_m'])*1000
    axial=d@n;s=axial/1000+truth['s_m'];rho=np.linalg.norm(d-axial[:,None]*n,axis=1)
    lo,hi=truth['search_interval_s_m']
    return (s>=lo)&(s<=hi)&(abs(rho/1000-np.interp(s,ss,rr))<.001),rho,s


def run(source,output):
    output.mkdir(parents=True,exist_ok=False);camera_rows=[];readouts=[];input_hashes={}
    for layout in ['far','close']:
        root=render_root(source,layout)
        records=json.loads(gzip.decompress((source/layout/'main/records.json.gz').read_bytes()))
        for row in records:
            if row['condition_name']!='stereo_raw' or row['method']!='portfolio':continue
            case=row['case'];d=row['diagnostics'];points=np.array(d['fit_point_positions_m'])*1000;ids=np.array(d['fit_camera_ids'])
            truth=json.loads((root/'geometry'/case/'evaluation_only/target.json').read_text())
            with np.load(root/'geometry'/case/'evaluation_only/profile.npz') as f:ss,rr=f['s_m'].copy(),f['radius_m'].copy()
            ideal=load_frames(root/'ideal_rgb/scenes'/case/'observations','rgb')
            registered=load_frames(root/'ideal_ir_registered/scenes'/case/'observations','rgb')
            clean=np.zeros_like(points);reg=np.zeros_like(points);common=np.ones(len(points),bool)
            sensor=json.loads((root/'geometry'/case/'sensor.json').read_text())
            for i,(f,g,meta) in enumerate(zip(ideal,registered,sensor['cameras'])):
                use=ids==i;uv=np.rint(f.camera.project(points[use]/1000)).astype(int)
                zi=f.depth_z_m[uv[:,1],uv[:,0]];zr=g.depth_z_m[uv[:,1],uv[:,0]]
                clean[use]=f.camera.unproject(uv,zi)*1000;reg[use]=g.camera.unproject(uv,zr)*1000
                common[use]=(zi>0)&(zr>0)&np.isfinite(zi)&np.isfinite(zr)
                path=root/'sensor_diagnostics'/case/(f.camera.name+'_native.npz')
                input_hashes[str(path.resolve())]=sha256(path.read_bytes()).hexdigest()
                with np.load(path) as native:
                    z=native['ideal_left_z'];measured=native['native_float_z'];quantized=native['native_z16_m']
                cam=Camera(f.camera.name,np.array(meta['K_ir']),np.array(meta['T_world_from_left_ir']),z.shape[1],z.shape[0])
                yy,xx=np.nonzero((z>0)&(measured>0)&(quantized>0)&np.isfinite(measured));pixels=np.c_[xx,yy]
                pi=cam.unproject(pixels,z[yy,xx])*1000;pm=cam.unproject(pixels,measured[yy,xx])*1000;pq=cam.unproject(pixels,quantized[yy,xx])*1000
                label,ri,s=outer_labels(pi,truth,ss,rr);_,rm,_=outer_labels(pm,truth,ss,rr);_,rq,_=outer_labels(pq,truth,ss,rr)
                camera_rows.append(dict(layout=layout,case=case,camera=f.camera.name,space='native_IR_all_visible_neck_common_valid',
                    stereo_radial_mm=metric_stats((rm-ri)[label]),quantized_radial_mm=metric_stats((rq-ri)[label]),
                    stereo_optical_z_mm=metric_stats(((measured[yy,xx]-z[yy,xx])*1000)[label]),
                    axial_bin_bias=[dict(s_mm=float((b+.5)*2),**metric_stats((rm-ri)[label&(np.floor((s-truth['s_m'])*1000/2)==b)]))
                        for b in range(-8,8)]))
            label,ri,_=outer_labels(clean,truth,ss,rr);_,rm,_=outer_labels(points,truth,ss,rr);_,rg,_=outer_labels(reg,truth,ss,rr)
            for i,f in enumerate(ideal):
                use=label&common&(ids==i)
                camera_rows.append(dict(layout=layout,case=case,camera=f.camera.name,space='selected_RGB_rays_common_valid',
                    stereo_radial_mm=metric_stats((rm-ri)[use]),ideal_registered_radial_mm=metric_stats((rg-ri)[use])))
            joint=replace(JointSettings(),knot_spacing_mm=3.,smoothing_mm3=.01,max_nfev=90)
            origin=np.array(d['axis_origin_m'])*1000;normal=np.array(row['estimate']['normal']);weights=np.array(d['readout_weights'])[common]
            interval=(d['profile_s_m'][0]*1000,d['profile_s_m'][-1]*1000+.05)
            for mode,p in [('measured_common',points),('ideal_IR_registered_common',reg),('ideal_RGB_common',clean)]:
                estimate,curve=read_curve(p[common],ids[common],origin,normal,joint,weights,interval)
                readouts.append(dict(layout=layout,case=case,intervention=mode,diagnostic_only=True,
                    original_points=len(points),common_points=int(common.sum()),**score(estimate,truth),curve=curve))
            print(layout,case,'sensor stages audited',flush=True)
    assert all(sha256(Path(p).read_bytes()).hexdigest()==h for p,h in input_hashes.items())
    (output/'camera_diagnostics.json').write_text(json.dumps(camera_rows,indent=2)+'\n')
    (output/'readouts.json.gz').write_bytes(gzip.compress(json.dumps(readouts).encode(),mtime=0))
    (output/'provenance.json').write_text(json.dumps(dict(source=str(source.resolve()),input_native_hashes=input_hashes,
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),diagnostic_only=True,
        registered_branch='Ideal left IR Z + same Z16 quantization + SDK RGB alignment',
        raw_branch='SimSense stereo + same quantization + SDK RGB alignment',
        no_new_render=True,no_new_estimator=True,prior_test_is_now_diagnostic=True),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run,a.output)
