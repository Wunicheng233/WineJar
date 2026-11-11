#!/usr/bin/env python3
"""Render an independent post-cover body view; compare transport refinements.

Secondary MuJoCo ideal-Z is a geometry diagnostic, not an active-stereo model.
The saved SimSense template control intentionally exposes cross-sensor bias.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import os
import sys
from time import perf_counter
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from neck_localization.geometry import Camera,look_at
from neck_localization.station_tying import pose,move_circle
from neck_localization.body_relocalization import register_body,voxel_points,surface_normals
from neck_localization.cross_station_uncertainty import evaluate_arrays
from neck_localization.experiment_io import read,write,digest,quantiles


def cameras_from_model(model,names,w,h):
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    result=[]
    for name in names:
        i=model.camera(name).id
        f=h/(2*np.tan(np.deg2rad(model.cam_fovy[i])/2))
        K=np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1.]])
        T=pose(data.cam_xmat[i].reshape(3,3) @ np.diag([1.,-1.,-1.]),data.cam_xpos[i])
        result.append(Camera(name,K,T,w,h))
    return result


def prepare_scene(folder,output,settings):
    root=ET.parse(folder/'scene.xml').getroot()
    world=root.find('worldbody')
    for body in list(world.findall('body')):
        if body.get('name') not in ('jar','loading_cover'):world.remove(body)
    for site in list(world.findall('site')):world.remove(site)
    for tag in ('actuator','keyframe','sensor','equality','contact','tendon','default'):
        element=root.find(tag)
        if element is not None:root.remove(element)
    asset=root.find('asset')
    for item in list(asset):
        if item.get('name','').startswith('right_'):asset.remove(item)
    w,h=settings['resolution']
    root.find('visual/global').set('offwidth',str(w));root.find('visual/global').set('offheight',str(h))
    for j,position in enumerate(settings['camera_positions_relative_station_m']):
        camera=look_at('body_'+str(j),np.array(position)+[2.4,0,0],np.array(settings['camera_target_relative_station_m'])+[2.4,0,0],w,h,50)
        R=camera.T_world_from_camera[:3,:3] @ np.diag([1.,-1.,-1.])
        ET.SubElement(world,'camera',name=camera.name,pos=' '.join(map(str,camera.T_world_from_camera[:3,3])),
            xyaxes=' '.join(map(str,R[:,:2].T.ravel())),fovy='50')
    path=output/'scene.xml';ET.ElementTree(root).write(path)
    return mujoco.MjModel.from_xml_path(str(path))


def capture(model,data,renderer,cameras,folder,*,bias=0.,roi=(.08,.4),station=0.):
    points=[]
    reference=cameras[0].T_world_from_camera
    for camera in cameras:
        renderer.update_scene(data,camera=camera.name)
        rgb=renderer.render().copy()
        renderer.enable_depth_rendering();z=renderer.render().copy();renderer.disable_depth_rendering()
        if folder is not None:
            folder.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(folder/(camera.name+'_rgb.png')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            np.savez_compressed(folder/(camera.name+'_z_m.npz'),z=z)
        y,x=np.indices(z.shape)
        good=np.isfinite(z)&(z>.1)&(z<2.)
        uv=np.c_[x[good],y[good]]
        world=camera.unproject(uv,z[good]+bias)
        keep=(abs(world[:,0]-station)<.25)&(abs(world[:,1])<.25)&(world[:,2]>roi[0])&(world[:,2]<roi[1])
        points.append((world[keep]-reference[:3,3]) @ reference[:3,:3])
    return np.concatenate(points)


def saved_template(folder,roi,tracked):
    cp=folder/'observations/camera_true_parameters.json';tracked[str(cp.resolve())]=digest(cp)
    points=[]
    for item in read(cp)['cameras']:
        camera=Camera(item['name'],np.array(item['K']),np.array(item['T_world_from_camera']),item['width'],item['height'])
        dp=folder/'observations'/(camera.name+'_depth_z_m.npz')
        mp=folder/'foreground_masks'/(camera.name+'_foreground.png')
        tracked[str(dp.resolve())],tracked[str(mp.resolve())]=digest(dp),digest(mp)
        raw=np.load(dp);z=raw[raw.files[0]]
        mask=cv2.imread(str(mp),cv2.IMREAD_GRAYSCALE)>0
        y,x=np.nonzero(mask&np.isfinite(z)&(z>.1)&(z<2))
        p=camera.unproject(np.c_[x,y],z[y,x])
        good=(abs(p[:,0])<.25)&(abs(p[:,1])<.25)&(p[:,2]>roi[0])&(p[:,2]<roi[1])
        points.append(p[good])
    return np.concatenate(points)


def limit(points,settings):
    p=voxel_points(points,settings['voxel_m'])
    if len(p)>settings['maximum_template_points']:
        p=p[np.linspace(0,len(p)-1,settings['maximum_template_points'],dtype=int)]
    return p


def run(config,output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    protocol=read(config);settings=deepcopy(protocol['body'])
    # Separately locked extension of the preregistered mechanism controls.
    settings['conditions']+=['upstream_frame_bias_2mm']
    settings['roi_policy']='observed_mouth_initializer_z_minus_250_to_60mm; narrow_control_minus_120_to_80mm'
    settings['body_height_roi_W_m']=None
    settings['lower_strip_roi_W_m']=None
    settings['source_controls']=['mujoco_ideal_body_diagnostic','saved_simsense_body']
    write(output/'protocol.json',protocol|dict(body=settings))
    baseline=ROOT/protocol['baseline'];original=read(baseline/'viewer/provenance.json')['input_and_source_sha256']
    for p,h in original.items():assert digest(p)==h,p
    tracked={str(p.resolve()):digest(p) for p in [config,Path(__file__),ROOT/'neck_localization/body_relocalization.py']}
    source=ROOT/read(baseline/'protocol.json')['source_bundle']
    band=read(baseline/'band_controls.json')['rows'];station=read(source/'config.json')
    B=np.array(station['B_T_W'])
    records=[]
    for cohort in settings['cohorts']:
        cases=sorted({r['case'] for r in band if r['cohort']==cohort})
        for case in cases:
            folder=source/cohort/'ss/clean/scenes'/case
            op=folder/'observation_t0.json';tracked[str(op.resolve())]=digest(op)
            mouth_z=float(read(op)['mouth_seed_m'][2])
            settings['body_height_roi_W_m']=[mouth_z-.250,mouth_z-.060]
            settings['lower_strip_roi_W_m']=[mouth_z-.120,mouth_z-.080]
            dest=output/'scenes'/cohort/case;dest.mkdir(parents=True)
            write(dest/'roi_from_observation.json',dict(mouth_initializer_z_m=mouth_z,
                body_roi_world_z_m=settings['body_height_roi_W_m'],narrow_roi_world_z_m=settings['lower_strip_roi_W_m'],
                neck_or_true_height_used=False))
            model=prepare_scene(folder,dest,settings)
            data=mujoco.MjData(model);mujoco.mj_forward(model,data)
            w,h=settings['resolution']
            first=cameras_from_model(model,['cam_a','cam_b','cam_c'],w,h)
            second=cameras_from_model(model,['body_0','body_1'],w,h)
            W_T_C=second[0].T_world_from_camera
            write(dest/'secondary_cameras.json',[dict(name=c.name,K=c.K,T_world_from_camera=c.T_world_from_camera,width=w,height=h) for c in second])
            first_C=first[0].T_world_from_camera
            base_row=next(r for r in band if (r['cohort'],r['case'],r['condition'],r['method'])==(cohort,case,'clean','pose_fs_radius'))
            truth=read(folder/'evaluation_only/target.json');spec=read(folder/'model_manifest.json')['jar_spec']
            # Freeze the recognition estimate before loading evaluation truth
            # into the scoring loop; registration never receives either value.
            estimate=deepcopy(base_row['estimate'])
            jar_id=model.body_mocapid[model.body('jar').id]
            cover_id=model.body_mocapid[model.body('loading_cover').id]
            cover_origin=data.mocap_pos[cover_id].copy();cover_rotation=Rotation.from_quat(data.mocap_quat[cover_id][[1,2,3,0]]).as_matrix()
            with mujoco.Renderer(model,h,w) as renderer:
                model.geom_rgba[model.geom('cover_visual').id,3]=0
                ideal_C=capture(model,data,renderer,first,dest/'recognition_ideal_body',roi=settings['body_height_roi_W_m'])
                ideal=ideal_C @ first_C[:3,:3].T+first_C[:3,3]
                saved=saved_template(folder,settings['body_height_roi_W_m'],tracked)
                templates={'mujoco_ideal_body_diagnostic':limit(ideal,settings),'saved_simsense_body':limit(saved,settings)}
                np.savez_compressed(dest/'recognition_body_templates.npz',**templates)
                for stop_id,delta in enumerate(settings['stop_errors_m']):
                    cache={}
                    for rotate in (False,True):
                        R=Rotation.from_euler('y',1,degrees=True).as_matrix() if rotate else np.eye(3)
                        M=pose(R,[2.4+delta,0,0])
                        data.mocap_pos[jar_id]=M[:3,3];data.mocap_quat[jar_id]=Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
                        data.mocap_pos[cover_id]=R @ cover_origin+M[:3,3]
                        data.mocap_quat[cover_id]=Rotation.from_matrix(R @ cover_rotation).as_quat()[[3,0,1,2]]
                        model.geom_rgba[model.geom('cover_visual').id,3]=1
                        mujoco.mj_forward(model,data)
                        rendered=dest/f'stop_{stop_id}'/('tilted' if rotate else 'nominal')
                        for condition in settings['conditions']:
                            if (condition=='rotation_mismatch_1deg')!=rotate:continue
                            roi=settings['lower_strip_roi_W_m'] if condition=='lower_strip' else settings['body_height_roi_W_m']
                            bias=.001 if condition=='depth_bias_1mm' else .003 if condition=='depth_bias_3mm' else 0.
                            target=limit(capture(model,data,renderer,second,rendered if condition in ('ideal','rotation_mismatch_1deg') else None,
                                bias=bias,roi=roi,station=2.4),settings)
                            normals=surface_normals(target) if len(target)>=50 else None
                            np.savez_compressed(dest/f'target_{stop_id}_{condition}.npz',points_C=target,**({'normals_C':normals} if normals is not None else {}))
                            nominal_M=pose(translation=[2.4,0,0])
                            initial=np.linalg.inv(W_T_C) @ nominal_M
                            local_calibration=B @ W_T_C
                            if condition=='local_extrinsic_1mm':local_calibration[:3,3]+=B[:3,:3] @ np.array([.001,0,0])
                            H=pose(translation=[0,0,.002]) if condition=='upstream_frame_bias_2mm' else np.eye(4)
                            input_circle=move_circle(estimate,H)
                            moved_truth=move_circle(truth,M)
                            for template_name,template in templates.items():
                                p=template.copy()
                                if condition=='lower_strip':p=p[(p[:,2]>roi[0])&(p[:,2]<roi[1])]
                                p=p @ H[:3,:3].T+H[:3,3]
                                for method in settings['methods']:
                                    started=perf_counter()
                                    fit=dict(accepted=True,reason='nominal_no_secondary_measurement',transform=initial)
                                    if method!='nominal':
                                        fit=register_body(p,target,initial,mode=method,direction=W_T_C[:3,:3].T @ [1,0,0],normals=normals)
                                    elapsed=(perf_counter()-started)*1000
                                    if method=='nominal':
                                        actual_circle=move_circle(input_circle,nominal_M)
                                    else:
                                        world_actual=np.linalg.inv(B) @ local_calibration @ fit['transform']
                                        actual_circle=move_circle(input_circle,world_actual)
                                    scored=evaluate_arrays(np.array(actual_circle['center_m'])[None],np.array(actual_circle['normal'])[None],
                                        np.array([actual_circle['radius_m']]),np.array(moved_truth['center_m'])[None],np.array(moved_truth['normal'])[None],np.array([spec['neck_length_m']]))
                                    row=dict(cohort=cohort,case=case,condition=condition,source_control=template_name,stop_error_mm=delta*1000,method=method,
                                        **{k:v for k,v in fit.items() if k not in ('history_rms_m',)},estimate=actual_circle,
                                        **{k:v[0] for k,v in scored.items()},elapsed_ms=elapsed,
                                        source_points=len(p),target_points=len(target),diameter_error_mm=base_row['diameter_mm'],
                                        final_fastening_success=None,registration_acceptance_is_not_process_release=True)
                                    records.append(row)
            print('body registration',cohort,case,'complete',flush=True)
    write(output/'records.json',records)
    summary=[]
    for condition in settings['conditions']:
        for template in settings['source_controls']:
            for method in settings['methods']:
                rows=[r for r in records if (r['condition'],r['source_control'],r['method'])==(condition,template,method)]
                accepted=[r for r in rows if r['accepted']]
                summary.append(dict(condition=condition,source_control=template,method=method,total=len(rows),accepted=len(accepted),
                    center_false_accepts=sum(r['accepted'] and r['center_mm']>4 for r in rows),
                    center_false_rejects=sum(not r['accepted'] and r['center_mm']<=4 for r in rows),
                    accepted_band_contained=sum(r['accepted'] and r['band_contained'] for r in rows),
                    **{key:quantiles([r[key] for r in accepted]) if accepted else None for key in ['center_mm','axial_mm','tangent_mm','normal_deg','elapsed_ms']},
                    final_fastening_success=None))
    write(output/'summary.json',summary)
    for p,h in tracked.items():assert digest(p)==h,p
    for p,h in original.items():assert digest(p)==h,p
    write(output/'provenance.json',dict(source_sha256=tracked,previous_protected_inputs=len(original),unchanged=True,
        target_neck_not_used_by_registration=True,no_target_geometry_or_pose_to_estimator=True,
        source_and_target_independently_rendered=True,secondary_depth_not_active_stereo=True,
        upstream_depth_control_is_saved_simsense=True,all_injected_errors_assumed=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'experiments/neck_localization/configs/pose_transfer_20261028_v1.json')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.config.resolve(),a.output.resolve())
