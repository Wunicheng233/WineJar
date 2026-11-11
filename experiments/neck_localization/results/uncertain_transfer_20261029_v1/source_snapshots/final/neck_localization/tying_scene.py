"""MuJoCo kinematic tying scene, using existing xArm meshes read-only.

No controller, live SDK, dynamics, vacuum, or device connection is imported.
"""
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .station_tying import GunSpec, jaw_arcs, pose
from .jar_model import JarSpec, centerline, neck_normal
from .tying_cover import cover_mesh


def numbers(x):
    return ' '.join(f'{v:.12g}' for v in np.asarray(x).ravel())


def quat(r):
    q = Rotation.from_matrix(r).as_quat()
    return q[[3, 0, 1, 2]]


def build_scene(source_xml, destination, spec, config, repository):
    """Asset/GT geometry only constructs environment, never the target plan."""
    source_xml, destination = Path(source_xml), Path(destination)
    tree = ET.parse(source_xml)
    root = tree.getroot()
    compiler = root.find('compiler')
    meshdir = source_xml.parent/compiler.get('meshdir', '')
    texturedir = source_xml.parent/compiler.get('texturedir', compiler.get('meshdir', ''))
    for element in root.find('asset'):
        if element.get('file'):
            base = texturedir if element.tag == 'texture' else meshdir
            element.set('file', str((base/element.get('file')).resolve()))
    compiler.attrib.pop('meshdir', None)
    compiler.attrib.pop('texturedir', None)
    compiler.set('autolimits', 'true')
    compiler.set('angle', 'radian')
    world = root.find('worldbody')
    jar = world.find("body[@name='jar']")
    jar.set('mocap', 'true')
    for name in ('static_conveyor', 'conveyor'):
        old = world.find(f"geom[@name='{name}']")
        if old is not None:
            world.remove(old)
    final = np.asarray(config['M_W_t2_t0'])[:3, 3]
    ET.SubElement(world, 'geom', name='three_station_conveyor', type='box',
                  pos=numbers(final/2+[0, 0, -.025]), size=numbers([final[0]/2+.55, .38, .025]),
                  rgba='.09 .12 .15 1', contype='0', conaffinity='0')
    for label, p in [('recognition', np.zeros(3)), ('loading', config['loading_displacement_W_m']), ('tying', final)]:
        ET.SubElement(world, 'site', name='station_'+label, pos=numbers(np.array(p)+[0, -.35, .012]),
                      type='box', size='.1 .025 .01', rgba='.15 .65 .70 1')
    # Loading surrogate follows the outer neck/lip with an assumed radial
    # allowance; a wide solid cylinder would hide a geometrically impossible
    # band inside itself. Still no cloth deformation/tension is simulated.
    cover = ET.SubElement(world, 'body', name='loading_cover', mocap='true')
    cm = cover_mesh(spec, config)
    start = cm['start_s_m']
    cover_position, cover_rotation = cm['origin'], cm['rotation']
    cover_path = destination.parent/'loading_cover.obj'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with cover_path.open('w') as stream:
        stream.write('# Assumed conforming occluder, NOT measured flexible material\n')
        for v in cm['vertices']:
            stream.write('v '+numbers(v)+'\n')
        for f in cm['faces']:
            stream.write('f '+numbers(f+1)+'\n')
    ET.SubElement(root.find('asset'), 'mesh', name='loading_cover_mesh', file=str(cover_path.resolve()))
    cover.set('pos', numbers(cover_position))
    cover.set('quat', numbers(quat(cover_rotation)))
    ET.SubElement(cover, 'geom', name='cover_visual', type='mesh', mesh='loading_cover_mesh',
                  rgba='.62 .50 .25 0', contype='0', conaffinity='0')
    bare = Path(repository)/'mujoco_xarm6/scene/xarm6_right_bare.xml'
    arm = ET.parse(bare).getroot()
    for asset in arm.find('asset'):
        item = copy.deepcopy(asset)
        if item.get('file'):
            item.set('file', str((bare.parent/'assets'/item.get('file')).resolve()))
        root.find('asset').append(item)
    default = root.find('default')
    if default is None:
        default = ET.SubElement(root, 'default')
    for item in arm.find('default'):
        default.append(copy.deepcopy(item))
    base = copy.deepcopy(arm.find('worldbody/body'))
    W_T_B = np.linalg.inv(np.asarray(config['B_T_W']))
    base.attrib.pop('euler', None)
    base.set('pos', numbers(W_T_B[:3, 3]))
    base.set('quat', numbers(quat(W_T_B[:3, :3])))
    for geom in base.iter('geom'):
        geom.set('contype', '0')
        geom.set('conaffinity', '0')
    flange = base.find(".//body[@name='right_link6']")
    old = flange.find("body[@name='right_bare_end_effector']")
    flange.remove(old)
    gun = GunSpec(**config['gun'])
    mount = ET.SubElement(flange, 'body', name='tying_guide',
                          pos=numbers(gun.T_F_G[:3, 3]), quat=numbers(quat(gun.T_F_G[:3, :3])))
    ET.SubElement(mount, 'site', name='guide_tcp', pos='0 0 0', size='.002', rgba='1 .3 .1 1')
    for name, a, b, radius in [('mount', [-gun.flange_behind_m, 0, gun.flange_above_m], [-.11, 0, .025], .012),
                               ('housing', [-.145, 0, .025], [-.09, 0, .025], .019)]:
        ET.SubElement(mount, 'geom', name='gun_'+name, type='capsule', fromto=numbers([a, b]),
                      size=str(radius), rgba='.25 .30 .35 1', contype='0', conaffinity='0')
    for side, arc in zip((1, -1), jaw_arcs(gun, 0)):
        hinge = np.array([-gun.arc_radius_m, 0, 0])
        child = ET.SubElement(mount, 'body', name=f'jaw_{side}', pos=numbers(hinge))
        ET.SubElement(child, 'joint', name=f'jaw_joint_{side}', type='hinge', axis=f'0 0 {side}',
                      range=numbers([0, np.deg2rad(gun.max_opening_deg/2)]))
        for i in range(len(arc)-1):
            ET.SubElement(child, 'geom', name=f'jaw_{side}_{i}', type='capsule',
                          fromto=numbers([arc[i]-hinge, arc[i+1]-hinge]),
                          size=str(gun.head_tube_radius_m), rgba='.83 .63 .22 1',
                          contype='0', conaffinity='0', mass='.002')
    world.append(base)
    bpos = W_T_B[:3, 3]
    ET.SubElement(world, 'geom', name='assumed_robot_pedestal', type='box',
                  pos=numbers([bpos[0], bpos[1], bpos[2]/2]), size=numbers([.12, .12, bpos[2]/2]),
                  rgba='.24 .27 .32 1', contype='0', conaffinity='0')
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding='unicode')
    return dict(cover_position_m=cover_position.tolist(), cover_rotation=cover_rotation.tolist(),
                arm_source=str(bare.resolve()), cover_start_s_m=start)


class OfflineArm:
    """Bounded FK/Jacobian IK. Reports failure; never warps the TCP to the target."""
    def __init__(self, model):
        self.model, self.data = model, mujoco.MjData(model)
        ids = np.array([model.joint(f'right_joint{i}').id for i in range(1, 7)])
        self.qadr, self.dadr = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
        self.limits = model.jnt_range[ids]
        self.site = model.site('guide_tcp').id

    def forward(self, q):
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)
        return pose(self.data.site_xmat[self.site].reshape(3, 3), self.data.site_xpos[self.site])

    def solve(self, target, seed=None):
        target = np.asarray(target, float)
        def residual(q):
            current = self.forward(q)
            angle = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
            return np.r_[current[:3, 3]-target[:3, 3], -.15*angle]
        seeds = [np.array([0, -.65, -1.5, 0, .8, 0]), np.array([0, .2, -1.8, 0, 1.5, 0]),
                 np.array([np.pi, -.5, -1.4, 0, 1., 0])]
        if seed is not None:
            seeds.insert(0, seed)
        best = None
        for initial in seeds:
            solution = least_squares(residual, np.clip(initial, self.limits[:, 0]+1e-8, self.limits[:, 1]-1e-8),
                bounds=(self.limits[:, 0], self.limits[:, 1]), max_nfev=160,
                ftol=1e-11, xtol=1e-11, gtol=1e-11)
            actual = self.forward(solution.x)
            pos = float(np.linalg.norm(actual[:3, 3]-target[:3, 3]))
            angle = float(np.rad2deg(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()))
            value = dict(q_rad=solution.x.tolist(), position_error_m=pos, angle_error_deg=angle,
                         passed=pos < .0001 and angle < .05)
            if best is None or pos+.001*angle < best['position_error_m']+.001*best['angle_error_deg']:
                best = value
            if value['passed']:
                break
        return best

    def descent_path(self, plan, samples=41):
        if not plan.get('waypoints'):
            return dict(passed=False, reason='no_plan', samples=[])
        above, end = np.array(plan['W_T_above']), np.array(plan['W_T_G_t2'])
        results, seed = [], None
        for u in np.linspace(0, 1, samples):
            target = above.copy()
            target[:3, 3] = (1-u)*above[:3, 3]+u*end[:3, 3]
            row = self.solve(target, seed)
            row['fraction'] = float(u)
            results.append(row)
            seed = np.array(row['q_rad'])
        jump = max(np.max(abs(np.array(b['q_rad'])-a['q_rad'])) for a, b in zip(results, results[1:]))
        return dict(passed=bool(all(r['passed'] for r in results) and jump < .15),
                    max_joint_step_rad=float(jump), samples=results,
                    scope='sampled_FK_IK_and_joint_limits_no_arm_collision_or_dynamics')
