#!/usr/bin/env python3
"""Open the new twelve-jar structural comparisons, using mouse navigation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'experiments/neck_localization/results/structural_stage_20261025_viewer'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--viewer',type=Path,default=DEFAULT)
    p.add_argument('--condition',default='close/stereo_raw')
    p.add_argument('--case',default='structural_test_nominal_000')
    p.add_argument('--method',default='portfolio')
    p.add_argument('--ground-truth',action='store_true')
    p.add_argument('--snapshot',type=Path)
    a=p.parse_args();manifest=a.viewer/'datasets.json'
    datasets=json.loads(manifest.read_text());dataset=next(d for d in datasets if d['condition']==a.condition)
    cmd=[sys.executable,str(ROOT/'scripts/view_neck_localization.py'),'--scenes',dataset['scenes'],
        '--records',dataset['records'],'--case',a.case,'--method',a.method,
        '--comparison-manifest',str(manifest),'--point-cloud','--large-neck','--overlay','--rejected-candidate']
    if a.ground_truth:cmd.append('--ground-truth')
    if a.snapshot:cmd+=['--snapshot',str(a.snapshot.resolve())]
    subprocess.run(cmd,cwd=ROOT,check=True)


if __name__=='__main__':main()
