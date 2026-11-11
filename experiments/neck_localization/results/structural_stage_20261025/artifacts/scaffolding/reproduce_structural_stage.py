#!/usr/bin/env python3
"""Frozen twelve-jar, two-layout structural study; offline GPU+CPU only.

Each output directory must be new. Uses pre-test source/config freezes and the
locked independent-plane disparity model. Does not tune parameters on tests.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def execute(script,*args):
    if '--output' in args:
        destination=Path(args[args.index('--output')+1])
        marker=('render_completed.json' if script=='render_structural_inputs.py' else
                'summary.json' if script in ['run_active_stereo_stage1.py','run_structural_study.py','run_structural_study_safe.py'] else
                'replay.json' if script=='rematch_structural_stereo.py' else 'provenance.json')
        if (destination/marker).exists():
            print('REUSE completed',script,destination,flush=True)
            return
    command=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)]
    print('RUN',script,*map(str,args),flush=True)
    subprocess.run(command,cwd=ROOT,check=True)


def run(output,layouts,resume=False):
    output=output.resolve();output.mkdir(parents=True,exist_ok=resume)
    plan=dict(layouts=layouts,test_seed=2026102519,n_jars_each=12,
        no_hardware=True,method_parameters_frozen=True,
        layouts_semantics='far original fixed rig; close ONE analytic sampling/visibility choice, not camera search',
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    if resume:
        previous=json.loads((output/'execution_plan.json').read_text())
        assert previous['layouts']==layouts
        (output/'execution_resume.json').write_text(json.dumps(plan,indent=2)+'\n')
    else:
        (output/'execution_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    config_root=ROOT/'experiments/neck_localization/configs'
    phase_model=ROOT/'experiments/neck_localization/results/disparity_calibration_20261025_v1/model.json'
    for layout in layouts:
        name='structural_'+('close_' if layout=='close' else '')+'test_20261025_v1'
        config=config_root/(name+'.json');freeze=config_root/(name+'_freeze.json')
        render_freeze=config_root/(name+'_render_freeze.json')
        base=output/layout;base.mkdir(exist_ok=resume)
        # Fresh reproductions always separate capture from inference. A
        # completed historical far capture can be reused during this run's
        # documented resume; the aborted legacy close capture is never reused.
        legacy=base/'render'
        if (legacy/'summary.json').exists():
            render=legacy
            print('REUSE completed legacy capture',render,flush=True)
        else:
            render=base/'render_inputs'
            execute('render_structural_inputs.py','--config',config,'--freeze',render_freeze,'--output',render)
        (base/'input_source.json').write_text(json.dumps(dict(render_directory=render.name),indent=2)+'\n')
        execute('rematch_structural_stereo.py','--run',render,'--output',base/'rematched')
        execute('apply_disparity_calibration.py','--run',render,'--model',phase_model,'--output',base/'phase')
        methods=['baseline','portfolio_plain','portfolio_robust','portfolio','portfolio_single','portfolio_anisotropic','portfolio_fixed']
        execute('run_structural_study_safe.py','--run',render,'--output',base/'main','--freeze',freeze,
                '--modes','ideal_rgb','stereo_raw','--methods',*methods)
        execute('run_structural_study_safe.py','--run',base/'rematched','--output',base/'matching', '--freeze',freeze,
                '--modes','narrow5','--methods',*methods)
        execute('run_structural_study_safe.py','--run',base/'rematched','--output',base/'small_matching','--freeze',freeze,
                '--modes','local3','--methods','baseline','portfolio')
        execute('run_structural_study_safe.py','--run',base/'phase','--output',base/'phase_estimates','--freeze',freeze,
                '--modes','phase_calibrated','--methods','baseline','portfolio','portfolio_fixed')
    (output/'completed.json').write_text(json.dumps(dict(completed=True,layouts=layouts),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layouts',nargs='+',choices=['far','close'],default=['far','close'])
    p.add_argument('--resume',action='store_true',help='Reuse completed groups only; does not overwrite partial groups')
    a=p.parse_args();run(a.output,a.layouts,a.resume)
