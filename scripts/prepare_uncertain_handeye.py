#!/usr/bin/env python3
"""Generate reachable virtual robot/image pairs, with separate evaluation truth."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from neck_localization.robot_error_model import VirtualRobot,draw_parameters
from neck_localization.station_tying import pose,frame
from neck_localization.tying_scene import OfflineArm
from neck_localization.experiment_io import read,write,digest


def project(T,points,K):
    p=points @ T[:3,:3].T+T[:3,3]
    return p[:,:2]/p[:,2,None]*[K[0,0],K[1,1]]+K[:2,2],p


def run(config,output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    protocol=read(config);base=ROOT/protocol['source_viewer']
    cameras=read(ROOT/protocol['body_camera_file']);camera=cameras[0]
    C=np.array(camera['T_world_from_camera']);K=np.array(camera['K'])
    station=read(base/'config.json');B=np.array(station['B_T_W'])
    from neck_localization.station_tying import GunSpec
    F_G=GunSpec(**station['gun']).T_F_G
    scene=base/'new8/fusion/clean/scenes/confirmation_nominal_003/scene.xml'
    xml=scene.read_text();nominal=VirtualRobot(xml)
    arm=OfflineArm(nominal.model)
    start=pose(frame([0,0,1],[0,1,0]),[2.4,0,.36])
    result=arm.solve(start)
    if not result['passed']:raise RuntimeError('calibration centre not reachable in nominal model')
    centre=np.array(result['q_rad']);F0=nominal.forward(centre)
    P0=pose(C[:3,:3] @np.diag([1,-1,-1]),[2.4,0,.36])
    Y=np.linalg.inv(F0) @P0
    a,b=np.meshgrid(np.arange(7)-3,np.arange(5)-2);board=np.c_[a.ravel(),b.ravel(),np.zeros(a.size)]*.020
    rng=np.random.default_rng(protocol['pose_pool_seed'])
    pool=[];attempts=0
    while len(pool)<protocol['pool_poses'] and attempts<10000:
        attempts+=1;q=centre+rng.uniform(-1,1,6)*np.deg2rad([30,25,25,45,40,45])
        if np.any(q<arm.limits[:,0]+.02) or np.any(q>arm.limits[:,1]-.02):continue
        A=nominal.forward(q);T=np.linalg.inv(C) @A @Y
        uv,p=project(T,board,K)
        normal=T[:3,2]
        if (p[:,2].min()>.15 and np.all(uv>[14,14]) and np.all(uv<[camera['width']-15,camera['height']-15])
            and np.linalg.norm(uv.max(0)-uv.min(0))>90 and np.dot(normal,-T[:3,3])/np.linalg.norm(T[:3,3])>.45):
            pool.append(q)
    if len(pool)<protocol['pool_poses']:raise RuntimeError(f'only {len(pool)} visible joint-limited calibration poses')
    pool=np.array(pool)
    write(output/'protocol.json',protocol)
    write(output/'geometry.json',dict(xml_source=str(scene.resolve()),B_T_W=B,W_T_C=C,K=K,
        nominal_F_T_G=F_G,nominal_F_T_P=Y,pose_pool=pool,attempts=attempts,board_points=board,
        nominal_IK_and_board_visibility_checked=True,collision_dynamics_and_real_robot_not_checked=True))
    (output/'robot_nominal.xml').write_text(xml)
    for phase,seedlist in protocol['sessions'].items():
        for seed in seedlist:
            rr=np.random.default_rng(seed);indices=rr.permutation(len(pool))[:protocol['training_poses']+protocol['holdout_poses']]
            q=pool[indices];nom=np.array([nominal.forward(v) for v in q])
            pixels_noise=rr.normal(0,protocol['pixel_std'],(len(q),len(board),2))
            eps_t=rr.normal(size=(len(q),3));eps_r=rr.normal(size=(len(q),3))
            true_jig_error=pose(Rotation.from_rotvec(rr.normal(0,np.deg2rad(.15),3)).as_matrix(),rr.normal(0,.0006,3))
            gun_true=F_G @true_jig_error
            metrology_error=pose(Rotation.from_rotvec(rr.normal(0,np.deg2rad(.03),3)).as_matrix(),rr.normal(0,.0001,3))
            measured_jig=np.linalg.inv(Y) @gun_true @metrology_error
            for condition in protocol['conditions']:
                factor={'kinematic_1':1,'kinematic_5':5,'kinematic_bad_image':5,'wrong_K':1}.get(condition,0)
                params=draw_parameters(seed+781,factor)
                actual=VirtualRobot(xml,params)
                trueposes=np.array([actual.forward(v) for v in q])
                reported=nom.copy()
                if condition=='random_pose':
                    reported[:,:3,3]+=eps_t*.0005
                    reported[:,:3,:3]=Rotation.from_rotvec(eps_r*np.deg2rad(.1)).as_matrix() @reported[:,:3,:3]
                images=np.array([project(np.linalg.inv(C) @A @Y,board,K)[0] for A in trueposes])
                if condition!='ideal':images+=pixels_noise
                if condition=='kinematic_bad_image':images[5]+=[4,-3]
                measured_K=K.copy()
                if condition=='wrong_K':measured_K[0,0]*=1.002
                folder=output/'measurements'/phase/str(seed)/condition;folder.mkdir(parents=True)
                np.savez_compressed(folder/'observations.npz',reported_B_T_F=B @reported,identified_pixels=images,
                    nominal_board_points=board,K=measured_K,q_reported=q,measured_P_T_G=measured_jig,
                    training_count=protocol['training_poses'])
                truth=output/'evaluation_only'/phase/str(seed)/condition;truth.mkdir(parents=True)
                write(truth/'truth.json',dict(B_T_C=B @C,F_T_P=Y,F_T_G_true=gun_true,
                    nominal_F_T_G=F_G,actual_robot_parameters=params,q=q,true_W_T_F=trueposes,
                    observation_model='Projected identified corners, not a camera render or hardware data',
                    board_jig_metrology_noise_m=.0001,board_jig_metrology_noise_deg=.03,
                    all_physical_error_scales_unmeasured_xarm_assumptions=True))
            print('prepared',phase,seed,flush=True)
    write(output/'provenance.json',dict(source_sha256={str(p.resolve()):digest(p) for p in [config,scene,Path(__file__),ROOT/'neck_localization/robot_error_model.py']},
        no_estimator_reads_evaluation_only=True,no_devices_training_or_installs=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.config.resolve(),a.output.resolve())
