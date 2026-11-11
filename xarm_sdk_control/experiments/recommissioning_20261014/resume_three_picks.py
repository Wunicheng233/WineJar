"""Attended three-material lift/return check; live limits are never overridden."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from xarm.wrapper import XArmAPI
from xarm_sdk_control.experiments.recommissioning_20261014.context import BASE, CAL, get, ready, digest
from xarm_sdk_control.hand_eye_calibration import pose_mm_deg_to_transform
from xarm_sdk_control.end_effector_config import load_end_effector_config
from xarm_sdk_control.suction_sensor import SuctionSensor
from xarm_sdk_control.vacuum_control import VacuumController
from xarm_sdk_control.realsense_capture import RealSenseCaptureSession

FRAME = BASE / 'resumed_suction/20261014_202934_361'
PLAN = FRAME / 'pick_plan.json'
RUN = FRAME / 'execution.json'
MATERIAL = 'paper_ceramic'
COMPRESSION_MM = 0.0





def targets():
    vision = json.loads((FRAME / 'vision_result.json').read_text())
    meta = json.loads((FRAME / 'frame.json').read_text())
    depth = cv2.imread(str(FRAME / 'depth_raw.png'), -1)
    K = np.array(meta['intrinsics']['camera_matrix'])
    D = np.array(meta['intrinsics']['distortion_coefficients'])
    T = np.array(json.loads(CAL.read_text())['base_T_camera'])
    overlay = cv2.imread(str(FRAME / 'color.png'))
    result = []
    # Explicitly reviewed current scene regions, not a general object classifier.
    regions = [('paper', (200, 420, 200, 355), 400),
               ('ceramic', (110, 200, 220, 310), 400),
               ('bamboo', (80, 465, 80, 200), 300)]
    if MATERIAL == 'bamboo':
        regions = [('bamboo', (100, 440, 180, 300), 400)]
    for name, (x0, x1, y0, y1), transit in regions:
        candidates = []
        for item in vision['detections']:
            b = item['box_xyxy']
            u, v = (b[0]+b[2])/2, (b[1]+b[3])/2
            if x0 < u < x1 and y0 < v < y1 and not item['touches_image_edge']:
                if name == 'bamboo' and (b[2]-b[0])/(b[3]-b[1]) < 2:
                    continue
                candidates.append(item)
        if not candidates:
            raise RuntimeError(f'Missing reviewed region: {name}')
        item = max(candidates, key=lambda x: x['score'])
        mask = cv2.imread(str(FRAME / item['mask_path']), 0)
        if name == 'ceramic':
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            center = np.array(cv2.fitEllipse(max(contours, key=cv2.contourArea))[0])
        else:
            ys, xs = np.nonzero(mask > 127)
            center = np.array([xs.mean(), ys.mean()])
        u, v = np.rint(center).astype(int)
        patch = depth[v-3:v+4, u-3:u+4]
        valid = patch[patch > 0]
        if len(valid) < 35 or np.ptp(valid) > 20:
            raise RuntimeError(f'Unstable contact depth: {name}')
        z = float(np.median(valid))*meta['stream']['depth_scale_m_per_unit']
        ray = cv2.undistortPoints(center.reshape(1, 1, 2), K, D).reshape(2)
        point = (T @ np.r_[np.r_[ray, 1]*z, 1])[:3]*1000
        result.append(dict(name=name, pixel=center.tolist(), surface_mm=point.tolist(),
                           transit_flange_z_mm=transit, mask=item['mask_path']))
        cv2.drawMarker(overlay, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(overlay, name, (u+8, v), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
    cv2.imwrite(str(FRAME / 'reviewed_targets.png'), overlay)
    return result


def build(arm):
    pose, tcp = ready(arm)
    if not np.allclose(tcp, [0, 0, 116, 0, 0, 0], atol=.01):
        raise RuntimeError('Unexpected active TCP, rebuild conversion')
    R = pose_mm_deg_to_transform(pose)[:3, :3]
    # Nominal cup location = reported TCP + R*(physical tool - active TCP).
    delta = R @ (np.array([0, 0, 116.])-np.array(tcp[:3]))
    reviewed = targets()
    # Current bamboo area is rejected by live controller checks at approach
    # and contact. Preserve the target evidence, but execute no bamboo motion.
    items = reviewed if MATERIAL == 'bamboo' else [item for item in reviewed if item['name'] != 'bamboo']
    steps = []
    for item in items:
        contact = np.r_[np.array(item['surface_mm'])-delta, pose[3:]]
        high = contact.copy()
        high[2] = item['transit_flange_z_mm']+(R @ np.array(tcp[:3]))[2]
        previous = np.array(pose if not steps else steps[-1]['pose'])
        level = previous.copy()
        level[2] = high[2]
        pre = contact.copy(); pre[2] += 30
        lift = contact.copy(); lift[2] += 50
        release = contact.copy(); release[2] += 5
        contact[2] -= COMPRESSION_MM
        for label, p, speed, event in [
            ('level', level, 40, None), ('above', high, 40, None),
            ('approach', pre, 25, None), ('contact', contact, 5, 'acquire'),
            ('lift', lift, 15, 'inspect'), ('return', release, 10, 'release'),
            ('retract', high, 30, None)]:
            steps.append(dict(material=item['name'], label=label, pose=p.tolist(), speed=speed, event=event))
    previous = np.array(pose)
    last_angles = np.array(get(arm.get_servo_angle())[:6])
    checks = []
    for step in steps:
        end = np.array(step['pose'])
        for f in np.linspace(0, 1, max(2, int(np.ceil(np.linalg.norm(end[:3]-previous[:3])/10))+1))[1:]:
            p = (previous + f*(end-previous)).tolist()
            angles = get(arm.get_inverse_kinematics(p, input_is_radian=False, return_is_radian=False))
            tl = get(arm.is_tcp_limit(p, is_radian=False))
            jl = get(arm.is_joint_limit(angles[:6], is_radian=False))
            jump = float(np.max(np.abs(np.array(angles[:6])-last_angles)))
            if tl or jl or jump > 20:
                raise RuntimeError(f"Live check rejected {step['material']}/{step['label']}: {p}, tcp={tl}, joint={jl}, jump={jump}")
            checks.append(dict(pose=p, tcp_limit=tl, joint_limit=jl, jump=jump))
            last_angles = np.array(angles[:6])
        previous = end
    files = [CAL, FRAME/'color.png', FRAME/'depth_raw.png', FRAME/'frame.json', FRAME/'vision_result.json']
    files += [FRAME/item['mask'] for item in items]
    plan = dict(start_pose=pose, tcp=tcp, targets=items, steps=steps, checks=checks,
                extra_compression_mm=COMPRESSION_MM,
                deferred=[] if MATERIAL == 'bamboo' else [dict(target=reviewed[2], reason='live TCP limit rejects approach/contact; reposition required')],
                inputs={str(p):digest(p) for p in files}, passed=True)
    PLAN.write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(dict(targets=items, checks=len(checks), max_joint_step=max(c['jump'] for c in checks), tcp=tcp), indent=2), flush=True)


def execute(arm):
    plan = json.loads(PLAN.read_text())
    if not plan['passed'] or RUN.exists():
        raise RuntimeError('Unapproved plan or existing execution; refusing replay')
    for p, h in plan['inputs'].items():
        if digest(Path(p)) != h:
            raise RuntimeError(f'Input changed: {p}')
    pose, tcp = ready(arm)
    if not np.allclose(tcp, plan['tcp'], atol=.01) or np.max(np.abs(np.array(pose)-plan['start_pose'])) > .5:
        raise RuntimeError('Start pose or active TCP changed')
    tool = load_end_effector_config()
    sensor = SuctionSensor(arm, tool.suction_sensor)
    vacuum = VacuumController(arm, tool.vacuum_io)
    record = dict(status='running', steps=[], materials={})
    # Exclusive file creation prevents a second execution, including failed runs.
    with RUN.open('x') as f:
        json.dump(record, f)
    def save():
        RUN.write_text(json.dumps(record, indent=2)+'\n')
    holding = False
    moving = False
    camera = RealSenseCaptureSession(serial='013222073103')
    failed = set()
    try:
        camera.start()
        vacuum.configure()
        vacuum.safe_shutdown()
        if sensor.read() != 1:
            raise RuntimeError('CI0 does not show released baseline')
        if arm.set_state(0) != 0:
            raise RuntimeError('Cannot enter ready state')
        for step in plan['steps']:
            name = step['material']
            if name in failed and step['label'] in ('lift', 'return'):
                continue
            moving = True
            code = arm.set_position(*step['pose'], speed=step['speed'], mvacc=80,
                                    wait=True, timeout=60, radius=-1, is_radian=False)
            entry = dict(step, code=code, actual=get(arm.get_position()))
            record['steps'].append(entry); save()
            if code != 0 or get(arm.get_err_warn_code()) != [0, 0]:
                raise RuntimeError(f'Motion failed: {name}/{step["label"]}')
            moving = False
            if step['event'] == 'acquire':
                if sensor.read() != 1:
                    raise RuntimeError('Unexpected CI0 before acquisition')
                vacuum.prepare_for_pick()
                # Conservatively retain vacuum on any uncertain acquisition failure.
                holding = True
                vacuum.acquire(.9)
                confirmed = sensor.wait_for_suction()
                record['materials'][name] = dict(acquire_confirmed=confirmed, acquire=vars(sensor.last_wait))
                if not confirmed:
                    # Still at supported source: release and skip loaded motion.
                    vacuum.release(.5); holding = False; failed.add(name)
                    record['materials'][name]['result'] = 'no_seal_at_measured_depth'
                record['materials'][name]['contact_frame'] = str(camera.capture(FRAME/f'{name}_contact'))
                print(f'{name}: acquire={confirmed}', flush=True)
            elif step['event'] == 'inspect':
                values = []
                until = time.monotonic()+1
                while time.monotonic() < until:
                    values.append(sensor.read()); time.sleep(.05)
                stable = all(v == 0 for v in values)
                record['materials'][name].update(lift_ci0=values, lift_pressure_stable=stable)
                record['materials'][name]['lift_frame'] = str(camera.capture(FRAME/f'{name}_lift'))
                print(f'{name}: lifted 50 mm, pressure stable={stable}; returning to source', flush=True)
            elif step['event'] == 'release':
                vacuum.release(.5); holding = False
                record['materials'][name]['release_ci0'] = sensor.read()
                record['materials'][name]['result'] = 'returned_after_lift'
            save()
        vacuum.safe_shutdown()
        record['status'] = 'complete'
        record['final_pose'] = get(arm.get_position())
        record['final_frame'] = str(camera.capture(FRAME/'after'))
        save()
    except BaseException as exc:
        record.update(status='stopped', error=repr(exc), may_be_holding=holding)
        if moving:
            record['stop_code'] = arm.set_state(4)
        save()
        raise
    finally:
        if not holding:
            vacuum.safe_shutdown()
        else:
            print('Possible held material: vacuum left on, no further motion.', flush=True)
        camera.stop()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--execute', action='store_true')
    p.add_argument('--frame', type=Path)
    p.add_argument('--material', choices=['paper_ceramic', 'bamboo'], default='paper_ceramic')
    p.add_argument('--compression-mm', type=float, choices=[0.0, 3.0, 8.0], default=0.0)
    args = p.parse_args()
    MATERIAL = args.material
    COMPRESSION_MM = args.compression_mm
    if MATERIAL == 'bamboo' and args.frame is None:
        p.error('Relocated bamboo requires a freshly reviewed --frame')
    if args.frame is not None:
        FRAME = args.frame.resolve()
        PLAN = FRAME / 'pick_plan.json'
        RUN = FRAME / 'execution.json'
    if COMPRESSION_MM:
        if MATERIAL != 'bamboo':
            p.error('Extra compression is only reviewed for the relocated bamboo')
        PLAN = FRAME / f'pick_plan_compression{COMPRESSION_MM:g}.json'
        RUN = FRAME / f'execution_compression{COMPRESSION_MM:g}.json'
    arm = XArmAPI('192.168.1.212', is_radian=False)
    try:
        time.sleep(1)
        execute(arm) if args.execute else build(arm)
    finally:
        arm.disconnect()
