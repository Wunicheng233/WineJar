#!/usr/bin/env python3
"""Export actual perturbed-FK arrivals to the original MuJoCo viewer format."""
import argparse
from pathlib import Path
import sys
import gzip
import copy
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from neck_localization.robot_error_model import VirtualRobot
from neck_localization.inspection_viewer import read_records
from neck_localization.experiment_io import read,write,digest


def run(bundle,output,seed,condition,include_simple=False):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    inputs=bundle/'confirmation_inputs';protocol=read(inputs/'protocol.json');base=ROOT/protocol['source_viewer']
    evaluations=[]
    evaluation_folders=['confirmation_evaluation_baselines','confirmation_evaluation_poe','confirmation_evaluation_poe_block']
    if include_simple:evaluation_folders.append('simplicity_evaluation')
    for name in evaluation_folders:
        evaluations.extend(read(bundle/name/'records.json'))
    evaluations=[r for r in evaluations if r['seed']==seed and r['condition']==condition and r['tool_policy']=='measured_board_jig']
    truth=read(inputs/'evaluation_only/test'/str(seed)/condition/'truth.json');datasets=[]
    for method in sorted({r['method'] for r in evaluations}):
        root=output/method;scene_root=root/'scenes';scene_root.mkdir(parents=True);rows=[]
        for e in [r for r in evaluations if r['method']==method]:
            cohort,case=e['cohort'],e['case'];alias=cohort+'__'+case
            source=base/cohort/'fusion/clean/scenes'/case;dest=scene_root/alias;dest.mkdir()
            source_records=base/cohort/'fusion/clean/records.json.gz'
            row=copy.deepcopy(next(r for r in read_records(source_records) if r['case']==case))
            row.update(case=alias,method=method,calibration_arrival=e,
                actual_joint_coordinates=(np.array(e['q_command'])*truth['actual_robot_parameters']['joint_scale']+truth['actual_robot_parameters']['joint_zero_rad']).tolist())
            for name in ['observations','evaluation_only','foreground_masks','model_manifest.json','plan.json','ik_path.json','scene_metadata.json','evaluation.json']:
                p=source/name
                if p.exists():(dest/name).symlink_to(p.resolve(),target_is_directory=p.is_dir())
            robot=VirtualRobot((source/'scene.xml').read_text(),truth['actual_robot_parameters'])
            # mj_saveLastXML rounds unrelated camera parameters. Preserve the
            # original observed camera/geometry text and update only robot links.
            tree=ET.parse(source/'scene.xml')
            for i in range(1,7):
                name=f'right_link{i}';b=robot.model.body(name).id
                node=tree.getroot().find(f".//body[@name='{name}']")
                node.set('pos',' '.join(format(x,'.17g') for x in robot.model.body_pos[b]))
                node.set('quat',' '.join(format(x,'.17g') for x in robot.model.body_quat[b]))
            gun=tree.getroot().find(".//body[@name='tying_guide']")
            G=np.array(truth['F_T_G_true']);quat=Rotation.from_matrix(G[:3,:3]).as_quat()[[3,0,1,2]]
            gun.set('pos',' '.join(map(str,G[:3,3])));gun.set('quat',' '.join(map(str,quat)))
            tree.write(dest/'scene.xml',encoding='unicode')
            rows.append(row)
        p=root/'records.json.gz';p.write_bytes(gzip.compress(__import__('json').dumps(rows,ensure_ascii=False).encode(),mtime=0))
        write(root/'settings.json',dict(preprocessing=dict(condition=condition),depth_display_range_m=[.2,.9]))
        label={'pixel_huber':'原像素Huber','uncertain_vce':'不确定性标定','camera_poe':'关节映射','camera_poe_block':'关节映射·整图加权',
               'park':'Park闭式法','fixed_block':'固定外参·整图加权'}[method]
        datasets.append(dict(condition=method,label=label,method=method,scenes=str(scene_root),records=str(p)))
    write(output/'datasets.json',datasets);write(output/'config.json',read(base/'config.json'))
    write(output/'provenance.json',dict(seed=seed,condition=condition,all_cases_and_methods_exported=True,
        supplementary_simple_controls_included=include_simple,
        static_arrival_only_no_validated_motion_or_collision=True,true_kinematic_geometry_used_only_for_rendering_and_scoring=True,
        source_script_sha256=digest(__file__)))
    print('arrival viewer exported',len(datasets),'methods')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=2026102981);p.add_argument('--condition',default='kinematic_bad_image')
    p.add_argument('--include-simple-controls',action='store_true')
    a=p.parse_args();run(a.bundle.resolve(),a.output.resolve(),a.seed,a.condition,a.include_simple_controls)
