#!/usr/bin/env python3
"""Rebuild the latest 8-scene/40-condition frozen comparison in a new folder.

Run inside embodied-sim. No devices/network are used. Original plans and source
hashes are retained; only observation/output paths are rebased in copied plans.
"""
import argparse,json,os,subprocess,sys
from pathlib import Path
from hashlib import sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    plans=out/'plans';plans.mkdir()
    raw_path=ROOT/'experiments/neck_localization/configs/bias_compensation_stage2_raw_v1.json'
    test_path=ROOT/'experiments/neck_localization/configs/bias_compensation_stage2_test_v1.json'
    raw=json.loads(raw_path.read_text());raw['scenes']=str(out/'scenes')
    test=json.loads(test_path.read_text());test['input_run']=str(out/'raw')
    (plans/'raw.json').write_text(json.dumps(raw,indent=2)+'\n')
    (plans/'test.json').write_text(json.dumps(test,indent=2)+'\n')
    (plans/'original_plan_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in [raw_path,test_path]},indent=2)+'\n')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl')
    def run(script,*args):
        command=[sys.executable,'-u',str(ROOT/script),*map(str,args)]
        print('Running',script,flush=True);subprocess.run(command,cwd=ROOT,env=env,check=True)
    run('scripts/prepare_neck_scene.py','--config',ROOT/'experiments/neck_localization/configs/bias_compensation_stage2_scenes_v1.json','--output',out/'scenes')
    run('scripts/run_depth_bias_stage1.py','--plan',plans/'raw.json','--output',out/'raw')
    run('scripts/run_bias_compensation_stage2.py','--plan',plans/'test.json','--output',out/'comparison')
    run('scripts/report_bias_compensation.py','--run',out/'comparison','--output',out/'report')
    print('Reproduction finished:',out,flush=True)


if __name__=='__main__':main()
