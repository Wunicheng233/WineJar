"""Attended loading in the relocated workcell; explicit reviewed RGB-D targets.

Each leg is planned against live controller limits, then executed once. No old
home pose, old bias, automatic error clearing, or historical limit overrides.
"""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from xarm.wrapper import XArmAPI
from xarm_sdk_control.experiments.recommissioning_20261014.context import CAL, get, ready, digest
from xarm_sdk_control.end_effector_config import load_end_effector_config
from xarm_sdk_control.suction_sensor import SuctionSensor
from xarm_sdk_control.vacuum_control import VacuumController
from xarm_sdk_control.realsense_capture import RealSenseCaptureSession


OBSERVATION = [419.163, -267.385, 284.064, 178.45756, 1.119502, -.049905]

def preflight(arm, start, steps):
    previous = np.array(start, dtype=float)
    joints = np.array(get(arm.get_servo_angle())[:6])
    checks = []
    for s in steps:
        end = np.array(s['pose'])
        count = max(1, int(np.ceil(np.linalg.norm(end[:3]-previous[:3])/10)),
                    int(np.ceil(np.max(np.abs(end[3:]-previous[3:]))/2)))
        for f in np.linspace(0, 1, count+1)[1:]:
            p = (previous+f*(end-previous)).tolist()
            angles = np.array(get(arm.get_inverse_kinematics(p, input_is_radian=False, return_is_radian=False))[:6])
            tl = get(arm.is_tcp_limit(p, is_radian=False))
            jl = get(arm.is_joint_limit(angles.tolist(), is_radian=False))
            jump = float(np.max(np.abs(angles-joints)))
            if tl or jl or jump > 20:
                raise RuntimeError(f'Live route rejection {s["label"]}: tcp={tl}, joint={jl}, jump={jump}, pose={p}')
            checks.append(dict(label=s['label'], pose=p, tcp_limit=tl, joint_limit=jl, joint_step=jump))
            joints = angles
        previous = end
    return checks

def point(frame, pixel):
    m = json.loads((frame/'frame.json').read_text())
    a = cv2.imread(str(frame/'depth_raw.png'), -1)
    u,v = np.rint(pixel).astype(int)
    if not 4 <= u < a.shape[1]-4 or not 4 <= v < a.shape[0]-4:
        raise RuntimeError('Target outside frame interior')
    patch = a[v-3:v+4,u-3:u+4]; valid = patch[patch>0]
    if len(valid) < 40 or np.ptp(valid)>20:
        raise RuntimeError(f'Unstable depth at {pixel}: {valid}')
    z = float(np.median(valid))*m['stream']['depth_scale_m_per_unit']
    ray = cv2.undistortPoints(np.array(pixel,dtype=float).reshape(1,1,2),
        np.array(m['intrinsics']['camera_matrix']),np.array(m['intrinsics']['distortion_coefficients'])).reshape(2)
    T = np.array(json.loads(CAL.read_text())['base_T_camera'])
    return ((T@np.r_[np.r_[ray,1]*z,1])[:3]*1000).tolist()

def plan(args, arm):
    pose,tcp = ready(arm)
    if not np.allclose(tcp,[0,0,116,0,0,0],atol=.01):
        raise RuntimeError('Active TCP must be the verified 116 mm')
    steps=[]
    def add(label,p,speed,event=None):
        steps.append(dict(label=label,pose=list(map(float,p)),speed=speed,event=event))
    high = OBSERVATION[2]
    if args.material == 'clear':
        up = pose.copy(); up[2]=high
        add('level',up,30); add('observation',OBSERVATION,40)
        source=destination=None
    else:
        source=point(args.frame,args.source)
        destination=point(args.frame,args.destination)
        if np.linalg.norm(np.array(source[:2])-destination[:2])<100:
            raise RuntimeError('Source/destination too close')
        ori=OBSERVATION[3:]
        place=ori.copy(); place[2]+=args.yaw
        comp=8 if args.material.startswith('shell') else 0
        thickness={'paper':.5,'ceramic':5,'shell_1':3,'shell_2':3}[args.material]
        def at(xyz,z,angles): return [xyz[0],xyz[1],z,*angles]
        up=pose.copy();up[2]=high
        add('level',up,30)
        add('source_above',at(source,high,ori),40)
        add('source_approach',at(source,source[2]+30,ori),25)
        add('source_contact',at(source,source[2]-comp,ori),5,'acquire')
        add('source_inspection',at(source,source[2]+50,ori),15,'inspect')
        add('source_lift',at(source,high,ori),25)
        if args.yaw:
            add('rotate_at_height',at(source,high,place),15)
        add('destination_above',at(destination,high,place),30)
        release=destination[2]+thickness+5
        add('destination_approach',at(destination,release+30,place),20)
        add('destination_release',at(destination,release,place),5,'release')
        add('destination_retract',at(destination,high,place),25)
        add('restore_yaw_at_height',at(destination,high,ori),15)
        add('observation',OBSERVATION,40)
    checks=preflight(arm,pose,steps)
    inputs={str(CAL):digest(CAL)}
    if args.frame:
        for name in ['frame.json','depth_raw.png','color.png']:
            p=args.frame/name;inputs[str(p)]=digest(p)
    result=dict(material=args.material,start_pose=pose,tcp=tcp,source=source,
        destination=destination,source_pixel=args.source,destination_pixel=args.destination,
        extra_compression_mm=8 if args.material.startswith('shell') else 0,
        placement_yaw_delta=args.yaw,steps=steps,checks=checks,inputs=inputs,passed=True)
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'plan.json'
    if path.exists(): raise RuntimeError('Plan already exists; choose a new output directory')
    path.write_text(json.dumps(result,indent=2)+'\n')
    if args.frame:
        im=cv2.imread(str(args.frame/'color.png'))
        for label,pixel in [('source',args.source),('destination',args.destination)]:
            p=tuple(np.rint(pixel).astype(int));cv2.drawMarker(im,p,(0,0,255),cv2.MARKER_CROSS,18,2)
            cv2.putText(im,label,(p[0]+8,p[1]),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
        cv2.imwrite(str(args.output/'targets.png'),im)
    print(json.dumps({k:result[k] for k in ['material','source','destination','extra_compression_mm','placement_yaw_delta']},indent=2))
    print(f'Live checks passed: {len(checks)}; max joint step {max(c["joint_step"] for c in checks):.3f}',flush=True)

def execute(args,arm):
    p=json.loads((args.output/'plan.json').read_text());pose,tcp=ready(arm)
    if not p['passed'] or not np.allclose(tcp,p['tcp'],atol=.01) or np.max(np.abs(np.array(pose)-p['start_pose']))>.5:
        raise RuntimeError('Start state changed')
    for name,h in p['inputs'].items():
        if digest(Path(name))!=h: raise RuntimeError(f'Input changed: {name}')
    preflight(arm,pose,p['steps'])
    cfg=load_end_effector_config();sensor=SuctionSensor(arm,cfg.suction_sensor);vacuum=VacuumController(arm,cfg.vacuum_io)
    record=dict(status='running',material=p['material'],steps=[])
    path=args.output/'execution.json'
    with path.open('x') as f: json.dump(record,f)
    def save(): path.write_text(json.dumps(record,indent=2)+'\n')
    held=False;moving=False
    camera=RealSenseCaptureSession(serial='013222073103')
    try:
        camera.start()
        if sensor.read()!=1: raise RuntimeError('Expected released CI0 before task')
        if p['material']!='clear': vacuum.configure();vacuum.safe_shutdown()
        if arm.set_state(0)!=0: raise RuntimeError('Cannot enter ready state')
        for step in p['steps']:
            if held and not sensor.is_suction_ok(): raise RuntimeError('Vacuum lost before next loaded move')
            moving=True
            code=arm.set_position(*step['pose'],speed=step['speed'],mvacc=80,wait=True,timeout=60,radius=-1,is_radian=False)
            actual=get(arm.get_position());err=get(arm.get_err_warn_code())
            record['steps'].append(dict(step,code=code,actual=actual));save()
            if code!=0 or err!=[0,0] or np.max(np.abs(np.array(actual)-step['pose']))>1:
                raise RuntimeError(f'Motion failed: {step["label"]}, code={code}, errors={err}')
            moving=False
            if held and not sensor.is_suction_ok(): raise RuntimeError('Vacuum lost during loaded move')
            if step['event']=='acquire':
                if sensor.read()!=1: raise RuntimeError('CI0 baseline changed')
                vacuum.prepare_for_pick();held=True;vacuum.acquire(.9)
                ok=sensor.wait_for_suction();record['acquire']=vars(sensor.last_wait);save()
                print(f'{p["material"]}: acquire={ok}',flush=True)
                if not ok:
                    vacuum.release(.5);held=False
                    # No lateral motion after a failed acquisition. Use the checked vertical segment.
                    retreat=next(s for s in p['steps'] if s['label']=='source_above')
                    moving=True
                    rc=arm.set_position(*retreat['pose'],speed=20,mvacc=80,wait=True,timeout=60,radius=-1,is_radian=False)
                    if rc!=0: raise RuntimeError(f'Failed source retreat: {rc}')
                    moving=False
                    record['failed_acquire_retreat']=get(arm.get_position())
                    raise RuntimeError('No suction; released at source and retracted')
            elif step['event']=='inspect':
                readings=[]
                for _ in range(20): readings.append(sensor.read());time.sleep(.05)
                record['lift_ci0']=readings
                record['lift_frame']=str(camera.capture(args.output/'lift'))
                if any(v!=0 for v in readings): raise RuntimeError('Unstable vacuum at inspection')
                print(f'{p["material"]}: lifted, pressure stable',flush=True)
            elif step['event']=='release':
                vacuum.release(.5);held=False
                record['release_ci0']=sensor.read()
                if record['release_ci0']!=1: raise RuntimeError('Release not confirmed')
                print(f'{p["material"]}: placed and released',flush=True)
            save()
        record['status']='complete';record['final_pose']=get(arm.get_position())
        record['final_frame']=str(camera.capture(args.output/'after'));save()
    except BaseException as exc:
        if moving or held: record['stop_code']=arm.set_state(4)
        record.update(status='stopped',error=repr(exc),may_be_holding=held);save();raise
    finally:
        if not held and p['material']!='clear':vacuum.safe_shutdown()
        if held: print('Stopped with possible held material; vacuum remains ON.',flush=True)
        camera.stop()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--execute',action='store_true')
    a.add_argument('--material',choices=['clear','paper','ceramic','shell_1','shell_2'],required=True)
    a.add_argument('--frame',type=Path);a.add_argument('--source',nargs=2,type=float)
    a.add_argument('--destination',nargs=2,type=float);a.add_argument('--yaw',type=float,default=0)
    a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    args.output=args.output.resolve()
    if args.frame:args.frame=args.frame.resolve()
    if not args.execute and args.material!='clear' and (not args.frame or not args.source or not args.destination):
        a.error('Material planning requires a fresh frame and reviewed source/destination pixels')
    if abs(args.yaw)>90: a.error('Yaw outside reviewed +/-90 degree range')
    arm=XArmAPI('192.168.1.212',is_radian=False)
    try:
        time.sleep(1)
        execute(args,arm) if args.execute else plan(args,arm)
    finally: arm.disconnect()
