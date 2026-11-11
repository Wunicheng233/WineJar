#!/usr/bin/env python3
"""Selected-pair SAPIEN/SimSense check of post-cover body transfer.

Uses the existing pinned sensor implementation; not factory characterization.
Only two already-seen jars. No FS inference, training, or physical devices.
"""
import argparse
from pathlib import Path
import sys
from time import perf_counter
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import sapien
from neck_localization.active_stereo import ActiveStereoRig,StereoSettings,initialize_renderer,add_jar_assets
from neck_localization.geometry import Camera
from neck_localization.station_tying import pose,move_circle
from neck_localization.body_relocalization import register_body,surface_normals
from neck_localization.body_partial_registration import register_observed_subset
from neck_localization.cross_station_uncertainty import evaluate_arrays
from neck_localization.experiment_io import read,write,digest
from scripts.run_body_relocalization_study import limit


def run(bundle):
    output=bundle/'active_stereo_probe'
    if output.exists():raise FileExistsError(output)
    output.mkdir()
    protocol=read(bundle/'body/protocol.json');settings=StereoSettings(seed=2026102853)
    initialize_renderer(settings)
    base=ROOT/protocol['baseline'];first=ROOT/read(base/'protocol.json')['source_bundle']
    estimates=read(base/'band_controls.json')['rows'];records=[];tracked={}
    for case in ['confirmation_nominal_003','confirmation_nominal_001']:
        source=first/'new8/ss/clean/scenes'/case;body=bundle/'body/scenes/new8'/case
        dest=output/case;dest.mkdir();(dest/'assets').symlink_to((source/'assets').resolve(),target_is_directory=True)
        scene=sapien.Scene();scene.set_ambient_light([.3,.3,.3]);scene.add_directional_light([1,-1,-2],[.6,.6,.6]);scene.add_directional_light([-1,1,-1],[.25,.25,.25])
        jar=add_jar_assets(scene,dest)
        xml=ET.parse(source/'scene.xml');cover=xml.getroot().find("worldbody/body[@name='loading_cover']")
        coverp=np.fromstring(cover.get('pos'),sep=' ');coverq=np.fromstring(cover.get('quat'),sep=' ')
        builder=scene.create_actor_builder();builder.add_visual_from_file(str((source/'loading_cover.obj').resolve()),material=[.62,.50,.25,1]);cap=builder.build_static(name='cover')
        for p in [source/'loading_cover.obj',source/'assets/jar.obj',source/'assets/jar_inside.obj',source/'assets/ceramic.png']:
            tracked[str(p.resolve())]=digest(p)
        cameras=[Camera(r['name'],np.array(r['K']),np.array(r['T_world_from_camera']),r['width'],r['height']) for r in read(body/'secondary_cameras.json')]
        rig=ActiveStereoRig(scene,cameras,settings)
        write(dest/'sensor_metadata.json',rig.metadata())
        C=rig.entries[0]['rgb'].T_world_from_camera
        p=np.load(body/'recognition_body_templates.npz')['saved_simsense_body']
        roi=read(body/'roi_from_observation.json')['body_roi_world_z_m']
        estimate=next(r['estimate'] for r in estimates if (r['cohort'],r['case'],r['method'],r['condition'])==('new8',case,'pose_fs_radius','clean'))
        try:
            for i,delta in enumerate([-.002,-.0005,.0005,.002]):
                M=pose(translation=[2.4+delta,0,0]);jar.set_pose(sapien.Pose(M));cap.set_pose(sapien.Pose(coverp+[2.4+delta,0,0],coverq))
                points={k:[] for k in ['ideal_rgb','stereo_raw']}
                framefolder=dest/f'stop_{i}';framefolder.mkdir()
                for j,entry in enumerate(rig.entries):
                    capture=rig.capture(j);cam=entry['rgb']
                    cv2.imwrite(str(framefolder/(cam.name+'_rgb.png')),cv2.cvtColor(capture['rgb'],cv2.COLOR_RGB2BGR))
                    np.savez_compressed(framefolder/(cam.name+'_depths.npz'),**capture['depths'])
                    for side,ir in enumerate(capture['ir_clean']):np.savez_compressed(framefolder/f'{cam.name}_ir_{side}.npz',ir=ir)
                    for kind in points:
                        z=capture['depths'][kind];y,x=np.indices(z.shape);valid=np.isfinite(z)&(z>.1)&(z<2)
                        world=cam.unproject(np.c_[x[valid],y[valid]],z[valid]);keep=(abs(world[:,0]-2.4)<.25)&(abs(world[:,1])<.25)&(world[:,2]>roi[0])&(world[:,2]<roi[1])
                        points[kind].append((world[keep]-C[:3,3]) @C[:3,:3])
                predictions=[]
                initial=np.linalg.inv(C) @pose(translation=[2.4,0,0])
                for kind,parts in points.items():
                    q=limit(np.concatenate(parts),protocol['body']);n=surface_normals(q) if len(q)>=50 else np.empty((0,3))
                    np.savez_compressed(framefolder/(kind+'_body_points.npz'),points_C=q,normals_C=n)
                    for method in ['nominal','axis','observed_subset_axis','reciprocal_axis']:
                        start=perf_counter()
                        if method=='nominal':fit=dict(accepted=True,reason='nominal',transform=initial)
                        elif method=='axis':fit=register_body(p,q,initial,mode='axis',direction=C[:3,:3].T @[1,0,0],normals=n)
                        else:fit=register_observed_subset(p,q,n,initial,C[:3,:3].T @[1,0,0],reciprocal=method=='reciprocal_axis')
                        prediction=move_circle(estimate,C @fit['transform'])
                        predictions.append((kind,method,fit,prediction,(perf_counter()-start)*1000,len(q)))
                # Exact target is loaded after all predictions, exclusively for scoring.
                truth=read(source/'evaluation_only/target.json');truth2=move_circle(truth,M)
                spec=read(source/'model_manifest.json')['jar_spec']
                for kind,method,fit,est,time_ms,count in predictions:
                    value=evaluate_arrays(np.array(est['center_m'])[None],np.array(est['normal'])[None],np.array([est['radius_m']]),
                        np.array(truth2['center_m'])[None],np.array(truth2['normal'])[None],np.array([spec['neck_length_m']]))
                    expected=move_circle(estimate,M)
                    records.append(dict(case=case,stop_error_mm=delta*1000,depth=kind,method=method,**fit,estimate=est,
                        **{k:v[0] for k,v in value.items()},transport_only_center_mm=float(np.linalg.norm(np.array(est['center_m'])-expected['center_m'])*1000),
                        elapsed_ms=time_ms,target_points=count,final_fastening_success=None))
                print('active body',case,i,'complete',flush=True)
        finally:rig.close()
    write(output/'records.json',records)
    for p,h in tracked.items():assert digest(p)==h,p
    write(output/'provenance.json',dict(source_sha256=tracked|{str(Path(__file__).resolve()):digest(__file__)},
        sensor='existing_SAPIEN_SimSense_D435_preset_32spp',isolated_emitters=True,
        materials_noise_not_measured_D435i=True,not_new_independent_jars=True,no_target_depth_or_label_to_registration=True,
        no_devices_training_or_official_filter_enabled_as_default=True))
    print('active-stereo selected-body check complete',len(records),'records')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    run(p.parse_args().bundle.resolve())
