#!/usr/bin/env python3
"""Actual MuJoCo RGB board check. Texture/detection in vision, render in stereo.

No hardware. Board rendering/robot geometry truth is used by the simulator,
never by corner extraction or calibration. Original scenes remain untouched.
"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from neck_localization.experiment_io import read,write


def board_definition():
    import cv2
    return cv2.aruco.CharucoBoard((8,6),.020,.014,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))


def make_texture(output):
    import cv2
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);board=board_definition()
    cv2.imwrite(str(output/'board.png'),board.generateImage((1600,1200),marginSize=0,borderBits=1))
    points=board.getChessboardCorners().astype(float);points[:,0]-=.08;points[:,1]=.06-points[:,1]
    write(output/'board_points.json',dict(points_P_m=points,ID_order=list(range(len(points))),physical_square_m=.020,
        P_X_print_right_P_Y_print_up_P_Z_front=True))
    (output/'board.obj').write_text('v -.08 -.06 0\nv .08 -.06 0\nv .08 .06 0\nv -.08 .06 0\n'
        'v -.08 -.06 -.001\nv .08 -.06 -.001\nv .08 .06 -.001\nv -.08 .06 -.001\n'
        'vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n'
        'f 8/4 7/3 6/2\nf 8/4 6/2 5/1\nf 1/1 5/1 6/2\nf 1/1 6/2 2/2\n'
        'f 2/2 6/2 7/3\nf 2/2 7/3 3/3\nf 3/3 7/3 8/4\nf 3/3 8/4 4/4\n'
        'f 4/4 8/4 5/1\nf 4/4 5/1 1/1\n')


def render(inputs,output):
    import os
    os.environ.setdefault('MUJOCO_GL','egl')
    import cv2,mujoco,xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    from neck_localization.robot_error_model import VirtualRobot
    from neck_localization.station_tying import pose
    from scripts.prepare_uncertain_handeye import project
    folder=output/'rendered'
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir()
    geometry=read(inputs/'geometry.json');C=np.array(geometry['W_T_C']);K=np.array(geometry['K']);Y=np.array(geometry['nominal_F_T_P'])
    points=np.array(read(output/'board_points.json')['points_P_m'])
    params=read(inputs/'evaluation_only/test/2026102981/kinematic_5/truth.json')['actual_robot_parameters']
    robot=VirtualRobot((inputs/'robot_nominal.xml').read_text(),params)
    mujoco.mj_saveLastXML(str(folder/'robot_geometry.xml'),robot.model)
    root=ET.parse(folder/'robot_geometry.xml').getroot();flange=root.find(".//body[@name='right_link6']")
    for child in list(flange.findall('body')):flange.remove(child)
    q=Rotation.from_matrix(Y[:3,:3]).as_quat()[[3,0,1,2]]
    body=ET.SubElement(flange,'body',name='calibration_board',pos=' '.join(map(str,Y[:3,3])),quat=' '.join(map(str,q)))
    asset=root.find('asset')
    ET.SubElement(asset,'mesh',name='board_mesh',file=str((output/'board.obj').resolve()))
    ET.SubElement(asset,'texture',name='board_texture',type='2d',file=str((output/'board.png').resolve()))
    ET.SubElement(asset,'material',name='board_material',texture='board_texture',texuniform='false',rgba='1 1 1 1',emission='.7',specular='0')
    ET.SubElement(body,'geom',name='board_front',type='mesh',mesh='board_mesh',material='board_material',contype='0',conaffinity='0')
    R=C[:3,:3] @np.diag([1,-1,-1]);world=root.find('worldbody')
    ET.SubElement(world,'camera',name='calibration_camera',pos=' '.join(map(str,C[:3,3])),xyaxes=' '.join(map(str,R[:,:2].T.ravel())),fovy='50')
    root.find('visual/global').set('offwidth','640');root.find('visual/global').set('offheight','480')
    path=folder/'scene.xml';ET.ElementTree(root).write(path,encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    qadr=model.jnt_qposadr[[model.joint(f'right_joint{i}').id for i in range(1,7)]]
    source=np.load(inputs/'measurements/test/2026102981/kinematic_5/observations.npz')
    nominal=VirtualRobot((inputs/'robot_nominal.xml').read_text());B=np.array(geometry['B_T_W']);manifest=[]
    with mujoco.Renderer(model,480,640) as renderer:
        for i,q in enumerate(source['q_reported']):
            data.qpos[qadr]=q*robot.scale+robot.zero;mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera='calibration_camera');rgb=renderer.render().copy()
            cv2.imwrite(str(folder/f'{i:02}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            true_pose=robot.forward(q) @Y;uv,_=project(np.linalg.inv(C) @true_pose,points,K)
            manifest.append(dict(frame=i,B_T_F_reported=B @nominal.forward(q),q_reported=q))
            write(folder/f'{i:02}_evaluation_only.json',dict(projected_true_corners=uv))
    write(folder/'observations.json',dict(frames=manifest,K=K,nominal_points=points,
        measured_P_T_G=source['measured_P_T_G'],rendered_input_has_no_added_corner_Gaussian_noise=True))
    write(folder/'evaluation_only.json',dict(source_parameter_truth=params,original_truth_folder=str(inputs/'evaluation_only/test/2026102981/kinematic_5')))
    print('actual MuJoCo board images',len(manifest))


def detect(inputs,output):
    import cv2
    from neck_localization.fixed_camera_calibration import solve_eye_to_hand
    from neck_localization.uncertain_handeye import solve_uncertain_eye_to_hand
    from neck_localization.poe_camera_calibration import fit_camera_poe
    board=board_definition();detector=cv2.aruco.CharucoDetector(board)
    observed=read(output/'rendered/observations.json');nominal=read(inputs/'nominal_poe.json');frames=[];allrows=[]
    for frame in observed['frames']:
        p=output/'rendered'/f"{frame['frame']:02}.png";image=cv2.imread(str(p))
        corners,ids,_,_=detector.detectBoard(image)
        count=0 if ids is None else len(ids)
        row=dict(frame=frame['frame'],detected_corners=count)
        if count==35:
            order=np.argsort(ids.ravel());uv=corners.reshape(-1,2)[order]
            cv2.cornerSubPix(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),uv,(3,3),(-1,-1),
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,50,.0001))
            frames.append(frame|dict(identified_pixels=uv));row['used']=True
            cv2.aruco.drawDetectedCornersCharuco(image,uv[:,None].copy(),np.arange(35,dtype=np.int32)[:,None])
            cv2.imwrite(str(output/'rendered'/f"{frame['frame']:02}_detected.png"),image)
        else:row['used']=False
        allrows.append(row)
    write(output/'detection.json',dict(frames=allrows,complete=len(frames),only_images_used_for_extraction=True))
    if len(frames)<24:
        print('insufficient full board observations; retain images and report limitation',len(frames));return
    n=18;A=np.array([f['B_T_F_reported'] for f in frames]);uv=np.array([f['identified_pixels'] for f in frames]);points=np.array(observed['nominal_points']);K=np.array(observed['K'])
    records=[]
    for method in ('pixel_huber','uncertain_vce','camera_poe_block'):
        if method=='pixel_huber':result=solve_eye_to_hand(A[:n],uv[:n],points,K,method=method)
        elif method=='uncertain_vce':result=solve_uncertain_eye_to_hand(A[:n],uv[:n],points,K)
        else:
            init=records[0];result=fit_camera_poe(np.array([f['q_reported'] for f in frames[:n]]),uv[:n],points,K,
                np.array(nominal['axes_B']),np.array(nominal['points_B']),np.array(nominal['B_T_F0']),init['B_T_C'],init['F_T_P'],block_robust=True)
            result.update(B_T_C=init['B_T_C'],F_T_P=init['F_T_P'])
        records.append(dict(phase='test',seed=2026102981,condition='kinematic_5',method=method,**result,
            source_observations=str((inputs/'measurements/test/2026102981/kinematic_5/observations.npz').resolve()),
            source_rendered_board=True,holdout_pixel_rms=0.,elapsed_ms=0.))
    # Holdout is scored by a separate replay, not assigned a fabricated zero.
    from neck_localization.fixed_camera_calibration import projection
    from neck_localization.poe_camera_calibration import CameraPOE
    for r in records:
        T=CameraPOE(**r['camera_poe']).forward(np.array([f['q_reported'] for f in frames[n:]])) if 'camera_poe' in r else np.linalg.inv(r['B_T_C']) @A[n:] @r['F_T_P']
        r['holdout_pixel_rms']=float(np.sqrt(np.mean([(projection(t,points,K)-pixels)**2 for t,pixels in zip(T,uv[n:])])))
    write(output/'records.json',records)
    # Evaluation-only corner sampling errors cannot affect any fitted model.
    errors=[]
    for f in frames:
        truth=read(output/'rendered'/f"{f['frame']:02}_evaluation_only.json")
        errors.extend(np.linalg.norm(np.array(f['identified_pixels'])-truth['projected_true_corners'],axis=1))
    write(output/'corner_scoring.json',dict(median_px=float(np.median(errors)),p95_px=float(np.percentile(errors,95)),max_px=float(np.max(errors))))
    print('rendered board calibration',len(frames),'frames',[(r['method'],r['holdout_pixel_rms']) for r in records])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['texture','render','detect'],required=True)
    a=p.parse_args();f=a.output.resolve();i=a.inputs.resolve()
    if a.stage=='texture':make_texture(f)
    elif a.stage=='render':render(i,f)
    else:detect(i,f)
