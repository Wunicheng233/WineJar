"""Offline structured FK mismatch. Never imports a controller or edits XML.

Parameter scales follow the mechanism used in Ulrich/Hillemann's experiment,
not measured xArm accuracy. Body-frame perturbations are an XML-model analogue
of link-offset/twist errors, not an identification of xArm DH tolerances.
"""
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .station_tying import pose


class VirtualRobot:
    def __init__(self, xml, parameters=None):
        root = ET.fromstring(xml)
        p = parameters or {}
        for i in range(6):
            body = root.find(f".//body[@name='right_link{i+1}']")
            if 'link_translation_m' in p:
                t = np.fromstring(body.get('pos','0 0 0'),sep=' ')+np.array(p['link_translation_m'][i])
                body.set('pos',' '.join(map(str,t)))
            if 'link_rotation_rad' in p:
                q = np.fromstring(body.get('quat','1 0 0 0'),sep=' ')
                R = Rotation.from_rotvec(p['link_rotation_rad'][i])*Rotation.from_quat(q[[1,2,3,0]])
                q = R.as_quat()[[3,0,1,2]]
                body.set('quat',' '.join(map(str,q)))
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        ids = [self.model.joint(f'right_joint{i}').id for i in range(1,7)]
        self.qadr = self.model.jnt_qposadr[ids]
        self.flange = self.model.body('right_link6').id
        self.zero = np.asarray(p.get('joint_zero_rad',np.zeros(6)))
        self.scale = np.asarray(p.get('joint_scale',np.ones(6)))

    def forward(self, q):
        self.data.qpos[self.qadr] = np.asarray(q)*self.scale+self.zero
        mujoco.mj_forward(self.model,self.data)
        return pose(self.data.xmat[self.flange].reshape(3,3),self.data.xpos[self.flange])


def draw_parameters(seed, factor=1.):
    rng = np.random.default_rng(seed)
    return dict(link_translation_m=rng.normal(0,.00005,(6,3))*factor,
        link_rotation_rad=rng.normal(0,np.deg2rad(.02),(6,3))*factor,
        joint_zero_rad=rng.normal(0,np.deg2rad(.02),6)*factor,
        joint_scale=1+rng.normal(0,.0001,6)*factor)
