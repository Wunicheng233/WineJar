#!/usr/bin/env python3
"""Re-render the locked eight jars and reproduce the sensor/correction study.

By default uses the separately trained, locked GP and rendered-board artifacts.
Their hashes are checked; their own training commands are in the stage report.
Every output lives in a NEW directory. No physical devices are accessed.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from neck_localization.scene_preparation import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--skip-board-comparison',action='store_true')
    p.add_argument('--check-artifacts',action='store_true',help='Only verify locked calibration artifacts and source hashes')
    args=p.parse_args()
    if not args.output and not args.check_artifacts:
        p.error('--output is required unless --check-artifacts is set')
    freeze=ROOT/'experiments/neck_localization/results/active_stereo_20261025/freeze.json'
    config=ROOT/'experiments/neck_localization/configs/active_stereo_test_v1.json'
    artifacts=freeze.parent/'artifacts'
    model=artifacts/'sim_d435_preset.npz'
    board=artifacts/'estimated_camera_parameters.json'
    plan=json.loads(freeze.read_text())
    assert sha256(model.read_bytes()).hexdigest()==plan['gp_model_sha256']
    # The board comparison was a separately recorded calibration experiment;
    # keep its final artifact fixed as well as the pre-jar-test sensor freeze.
    assert sha256(board.read_bytes()).hexdigest()=='9f9d25bccbcc333a0d8dc14cfc2c94e9f82d8c1f2d4eb4a29134a10323c63bf7'
    for name,digest in plan['gp_extra_sources'].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for name,digest in plan['evaluation_sources'].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    assert sha256(config.read_bytes()).hexdigest()==plan['test_config_sha256']
    if args.check_artifacts:
        print('Locked GP, board parameters, test config and source hashes verified; no rendering performed.')
        return
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    def run(script,*arguments):
        subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,arguments)],cwd=ROOT,check=True)
    run('run_active_stereo_stage1.py','--config',config,'--freeze',freeze,'--output',output/'observations')
    run('evaluate_active_stereo.py','--source',output/'observations','--freeze',freeze,'--output',output/'adapted')
    run('apply_active_depth_correction.py','--source',output/'observations','--model',model,'--output',output/'gp_observations')
    run('evaluate_active_stereo.py','--source',output/'gp_observations','--freeze',freeze,'--output',output/'gp_adapted')
    if not args.skip_board_comparison:
        run('evaluate_active_stereo.py','--source',output/'observations','--freeze',freeze,
            '--estimated-parameters',board,'--output',output/'board_adapted')
    write_json(output/'reproduction.json',dict(completed=True,freeze_sha256=sha256(freeze.read_bytes()).hexdigest(),
        gp_model_sha256=sha256(model.read_bytes()).hexdigest(),
        board_parameters_sha256=sha256(board.read_bytes()).hexdigest() if not args.skip_board_comparison else None,
        scope='fresh render and evaluation of eight fixed test jars; uses locked independent calibration artifacts',
        no_hardware=True))


if __name__=='__main__':main()
