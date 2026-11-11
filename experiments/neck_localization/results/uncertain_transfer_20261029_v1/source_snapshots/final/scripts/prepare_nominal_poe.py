#!/usr/bin/env python3
"""Extract only nominal robot geometry, not physical perturbation truth."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from neck_localization.robot_error_model import VirtualRobot
from neck_localization.experiment_io import read,write

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);a=p.parse_args();folder=a.inputs.resolve()
    robot=VirtualRobot((folder/'robot_nominal.xml').read_text());B=np.array(read(folder/'geometry.json')['B_T_W'])
    F=robot.forward(np.zeros(6));axes=[];points=[]
    for i in range(1,7):
        j=robot.model.joint(f'right_joint{i}').id
        axes.append(B[:3,:3] @robot.data.xaxis[j]);points.append(B[:3,:3] @robot.data.xanchor[j]+B[:3,3])
    path=folder/'nominal_poe.json'
    if path.exists():raise FileExistsError(path)
    write(path,dict(axes_B=axes,points_B=points,B_T_F0=B @F,nominal_geometry_only=True))
    from neck_localization.poe_camera_calibration import CameraPOE
    model=CameraPOE(axes,points,B @F)
    for q in np.random.default_rng(55).uniform(-1,1,(15,6)):
        np.testing.assert_allclose(model.forward(q),B @robot.forward(q),atol=2e-12)
    print('nominal PoE matches MuJoCo FK for 15 independent configurations')
