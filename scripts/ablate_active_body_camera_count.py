#!/usr/bin/env python3
"""One downstream camera versus two, using identical saved SimSense frames."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from neck_localization.geometry import Camera
from neck_localization.station_tying import pose,move_circle
from neck_localization.body_relocalization import surface_normals
from neck_localization.body_partial_registration import register_observed_subset
from neck_localization.cross_station_uncertainty import evaluate_arrays
from neck_localization.experiment_io import read,write,digest
from scripts.run_body_relocalization_study import limit


def run(bundle):
    root=bundle/'active_stereo_probe';protocol=read(bundle/'body/protocol.json')
    base=ROOT/protocol['baseline'];first=ROOT/read(base/'protocol.json')['source_bundle'];estimates=read(base/'band_controls.json')['rows']
    records=[]
    for case in ['confirmation_nominal_003','confirmation_nominal_001']:
        body=bundle/'body/scenes/new8'/case;roi=read(body/'roi_from_observation.json')['body_roi_world_z_m']
        source=np.load(body/'recognition_body_templates.npz')['saved_simsense_body']
        meta=read(root/case/'sensor_metadata.json')['cameras'][0]
        K=np.array(meta['K_rgb']);C=np.array(meta['T_world_from_rgb']);camera=Camera(meta['name'],K,C,848,480)
        circle=next(r['estimate'] for r in estimates if (r['cohort'],r['case'],r['condition'],r['method'])==('new8',case,'clean','pose_fs_radius'))
        for i,delta in enumerate([-.002,-.0005,.0005,.002]):
            frame=np.load(root/case/f'stop_{i}'/(camera.name+'_depths.npz'))
            predictions=[]
            for kind in ['ideal_rgb','stereo_raw']:
                z=frame[kind];y,x=np.indices(z.shape);valid=np.isfinite(z)&(z>.1)&(z<2)
                w=camera.unproject(np.c_[x[valid],y[valid]],z[valid]);keep=(abs(w[:,0]-2.4)<.25)&(abs(w[:,1])<.25)&(w[:,2]>roi[0])&(w[:,2]<roi[1])
                q=limit((w[keep]-C[:3,3]) @C[:3,:3],protocol['body']);normals=surface_normals(q)
                fit=register_observed_subset(source,q,normals,np.linalg.inv(C) @pose(translation=[2.4,0,0]),C[:3,:3].T @[1,0,0],reciprocal=True)
                predictions.append((kind,fit,move_circle(circle,C @fit['transform'])))
            # Reference data enter evaluation after both predictions exist.
            folder=first/'new8/ss/clean/scenes'/case;truth=read(folder/'evaluation_only/target.json');length=read(folder/'model_manifest.json')['jar_spec']['neck_length_m']
            M=pose(translation=[2.4+delta,0,0]);moved=move_circle(truth,M);expected=move_circle(circle,M)
            for kind,fit,est in predictions:
                scored=evaluate_arrays(np.array(est['center_m'])[None],np.array(est['normal'])[None],np.array([est['radius_m']]),
                    np.array(moved['center_m'])[None],np.array(moved['normal'])[None],np.array([length]))
                records.append(dict(case=case,depth=kind,stop_error_mm=delta*1000,camera_count=1,method='reciprocal_axis',**fit,
                    **{k:v[0] for k,v in scored.items()},transport_only_center_mm=float(np.linalg.norm(np.array(est['center_m'])-expected['center_m'])*1000),
                    final_fastening_success=None))
    write(root/'one_camera_ablation.json',records)
    write(root/'one_camera_provenance.json',dict(script_sha256=digest(__file__),same_saved_camera0_frames_and_template=True,
        paired_two_camera_records_sha256=digest(root/'records.json'),no_new_rendering_or_threshold_change=True))
    print('one-camera ablation complete',len(records),'records')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    run(p.parse_args().bundle.resolve())
