#!/usr/bin/env python3
"""Read-only body-transfer datasets for the existing MuJoCo InspectionViewer."""
import argparse
from copy import deepcopy
import gzip
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from neck_localization.station_tying import pose,move_circle
from neck_localization.experiment_io import read,write,digest


def run(bundle):
    viewer=bundle/'viewer'
    if viewer.exists():raise FileExistsError(viewer)
    viewer.mkdir()
    body=bundle/'body';protocol=read(body/'protocol.json')
    baseline=ROOT/protocol['baseline'];source=ROOT/read(baseline/'protocol.json')['source_bundle']
    rows=read(body/'records.json')+read(body/'partial_view_ablation/records.json')
    datasets=[]
    groups=[('ideal','mujoco_ideal_body_diagnostic','理想坛身对照'),('ideal','saved_simsense_body','双目坛身模板'),
        ('depth_bias_3mm','saved_simsense_body','下游深度假设偏差3mm'),('lower_strip','saved_simsense_body','窄可见区域压力对照')]
    for j,(condition,template,label) in enumerate(groups):
        folder=viewer/f'condition_{j}';folder.mkdir();scenes=folder/'scenes';scenes.mkdir()
        selected=[r for r in rows if (r['condition'],r['source_control'],r['stop_error_mm'])==(condition,template,2.)]
        records=[]
        for r in selected:
            case=r['cohort']+'__'+r['case'];dst=scenes/case
            body_scene=body/'scenes'/r['cohort']/r['case']
            first_scene=source/r['cohort']/'ss/clean/scenes'/r['case']
            if not dst.exists():
                dst.mkdir();obs=dst/'observations';obs.mkdir();masks=dst/'foreground_masks';masks.mkdir()
                tree=ET.parse(body_scene/'scene.xml');world=tree.getroot().find('worldbody')
                # Visualization uses exact physical scene placement; estimates
                # are already frozen and remain in their recorded world frame.
                world.find("body[@name='jar']").set('pos','2.402 0 0')
                cover=world.find("body[@name='loading_cover']")
                cp=np.fromstring(cover.get('pos'),sep=' ')+[2.402,0,0]
                cover.set('pos',' '.join(map(str,cp)));cover.find('geom').set('rgba','.62 .50 .25 1')
                tree.write(dst/'scene.xml')
                cameras=read(body_scene/'secondary_cameras.json')
                write(obs/'camera_true_parameters.json',dict(cameras=cameras,kind='evaluation_renderer_pose_for_display_not_new_handeye_measurement'))
                roi=read(body_scene/'roi_from_observation.json')['narrow_roi_world_z_m' if condition=='lower_strip' else 'body_roi_world_z_m']
                for c in cameras:
                    image=body_scene/'stop_3/nominal'/(c['name']+'_rgb.png')
                    (obs/(c['name']+'_rgb.png')).symlink_to(image.resolve())
                    z=np.load(body_scene/'stop_3/nominal'/(c['name']+'_z_m.npz'))['z']
                    if condition=='depth_bias_3mm':z=z+.003
                    np.savez_compressed(obs/(c['name']+'_depth_z_m.npz'),depth_z_m=z)
                    y,x=np.indices(z.shape);good=np.isfinite(z)&(z>.1)&(z<2)
                    uv=np.c_[x[good],y[good],np.ones(good.sum())]
                    T=np.array(c['T_world_from_camera']);p=(uv @ np.linalg.inv(c['K']).T*z[good,None]) @ T[:3,:3].T+T[:3,3]
                    keep=(abs(p[:,0]-2.4)<.25)&(abs(p[:,1])<.25)&(p[:,2]>roi[0])&(p[:,2]<roi[1])
                    mask=np.zeros(z.shape,np.uint8);mask[good]=keep.astype(np.uint8)*255
                    cv2.imwrite(str(masks/(c['name']+'_foreground.png')),mask)
                evaluation=dst/'evaluation_only';evaluation.mkdir()
                truth=read(first_scene/'evaluation_only/target.json');truth.update(move_circle(truth,pose(translation=[2.402,0,0])))
                write(evaluation/'target.json',truth)
                for name in ('model_manifest.json',): (dst/name).symlink_to((first_scene/name).resolve())
            record=deepcopy(r)
            record.update(case=case,evidence='rgb',camera_set='all',condition_name=label,
                diameter_mm=r['diameter_error_mm'],reasons=[] if r['accepted'] else [r['reason']],
                band=r['estimate']|dict(width_m=.003),diagnostics={},truth_pass=bool(r['center_mm']<=4),
                truth_pass_definition='center_4mm_only_not_process_release',
                secondary_body_observation=True,neck_has_not_been_reobserved=True)
            records.append(record)
        write(folder/'records.json',records)
        write(folder/'settings.json',dict(preprocessing=dict(condition=label)))
        datasets.append(dict(condition=f'body_{j}',label=label,scenes=str(scenes.resolve()),records=str((folder/'records.json').resolve()),method='observed_subset_axis'))
    write(viewer/'datasets.json',datasets)
    write(viewer/'provenance.json',dict(source_record_sha256={str((body/'records.json').resolve()):digest(body/'records.json'),
        str((body/'partial_view_ablation/records.json').resolve()):digest(body/'partial_view_ablation/records.json')},
        exporter_sha256=digest(__file__),old_viewer_source_unchanged=True,
        no_robot_motion_in_this_inspection_scene=True,exact_placement_used_only_by_visualization=True))
    print('body transfer viewer exported:',len(datasets),'conditions')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    run(p.parse_args().bundle.resolve())
