#!/usr/bin/env python3
"""Add the selected SAPIEN body observations to the original viewer bundle."""
import argparse
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from neck_localization.experiment_io import read,write,digest


def run(bundle):
    viewer=bundle/'viewer';probe=bundle/'active_stereo_probe';folder=viewer/'active_stereo'
    if folder.exists():raise FileExistsError(folder)
    scenes=folder/'scenes';scenes.mkdir(parents=True)
    allrows=read(probe/'records.json');records=[]
    for case in ['confirmation_nominal_003','confirmation_nominal_001']:
        key='new8__'+case;dst=scenes/key
        shutil.copytree(viewer/'condition_1/scenes'/key,dst,symlinks=True)
        cameras=[]
        meta=read(probe/case/'sensor_metadata.json')
        roi=read(bundle/'body/scenes/new8'/case/'roi_from_observation.json')['body_roi_world_z_m']
        for camera in meta['cameras']:
            name=camera['name'];T=np.array(camera['T_world_from_rgb']);K=np.array(camera['K_rgb'])
            rgb=probe/case/'stop_3'/(name+'_rgb.png');image=cv2.imread(str(rgb));h,w=image.shape[:2]
            cameras.append(dict(name=name,K=K,T_world_from_camera=T,width=w,height=h))
            target=dst/'observations'/(name+'_rgb.png');target.unlink();target.symlink_to(rgb.resolve())
            z=np.load(probe/case/'stop_3'/(name+'_depths.npz'))['stereo_raw']
            np.savez_compressed(dst/'observations'/(name+'_depth_z_m.npz'),depth_z_m=z)
            y,x=np.indices(z.shape);valid=np.isfinite(z)&(z>.1)&(z<2)
            p=(np.c_[x[valid],y[valid],np.ones(valid.sum())] @np.linalg.inv(K).T*z[valid,None]) @T[:3,:3].T+T[:3,3]
            keep=(abs(p[:,0]-2.4)<.25)&(abs(p[:,1])<.25)&(p[:,2]>roi[0])&(p[:,2]<roi[1])
            mask=np.zeros(z.shape,np.uint8);mask[valid]=keep.astype(np.uint8)*255
            cv2.imwrite(str(dst/'foreground_masks'/(name+'_foreground.png')),mask)
        write(dst/'observations/camera_true_parameters.json',dict(cameras=cameras,kind='saved_SAPIEN_SimSense_observation_cameras_for_display'))
        radius=read(dst/'evaluation_only/target.json')['radius_m']
        for r in allrows:
            if (r['case'],r['stop_error_mm'],r['depth'])!=(case,2.,'stereo_raw'):continue
            records.append(r|dict(case=key,evidence='rgb',camera_set='all',condition_name='主动双目下游·两例探针',
                diameter_mm=abs(r['estimate']['radius_m']-radius)*2000,reasons=[] if r['accepted'] else [r['reason']],
                band=r['estimate']|dict(width_m=.003),diagnostics={},truth_pass=bool(r['center_mm']<=4),
                secondary_body_observation=True,neck_has_not_been_reobserved=True,not_independent_new_case=True))
    write(folder/'records.json',records);write(folder/'settings.json',dict(preprocessing=dict(condition='SAPIEN主动双目·两例探针')))
    datasets=read(viewer/'datasets.json');datasets.append(dict(condition='body_active',label='主动双目两例',scenes=str(scenes.resolve()),records=str((folder/'records.json').resolve()),method='reciprocal_axis'))
    write(viewer/'datasets.json',datasets)
    write(viewer/'active_stereo_append.json',dict(script_sha256=digest(__file__),sensor_records_sha256=digest(probe/'records.json'),
        original_four_datasets_unchanged=True,observation_views=2,case_count=2))
    print('active stereo viewer append complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    run(p.parse_args().bundle.resolve())
