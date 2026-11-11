#!/usr/bin/env python3
"""Open saved structural/capability comparisons in the original MuJoCo window."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'experiments/neck_localization/results/structural_stage_20261025_viewer'
CAPABILITY_DEFAULT=ROOT/'experiments/neck_localization/results/capability_structural_viewer_20261027_v1'


def select_dataset(viewer=None, condition=None):
    """Use the explicit bundle when supplied; otherwise prefer the new audit."""
    viewer = Path(viewer) if viewer is not None else (CAPABILITY_DEFAULT
        if (CAPABILITY_DEFAULT/'datasets.json').exists() else DEFAULT)
    manifest=viewer/'datasets.json'
    datasets=json.loads(manifest.read_text())
    if not datasets:
        raise ValueError('Empty viewer datasets: '+str(manifest))
    conditions={d['condition'] for d in datasets}
    if condition is None:
        condition=next((c for c in ['close/foundation_stereo/clean','close/foundation_stereo_sam/clean',
                                    'close/stereo_raw'] if c in conditions),
                       datasets[0]['condition'])
    if condition not in conditions:
        raise ValueError('Unknown condition '+condition+'; available: '+', '.join(sorted(conditions)))
    return manifest,datasets,next(d for d in datasets if d['condition']==condition)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--viewer',type=Path,help='Explicit saved bundle; preserves access to older structural results')
    p.add_argument('--condition',help='Saved condition; default is close FoundationStereo when available')
    p.add_argument('--case',default='structural_test_nominal_000')
    p.add_argument('--method',help='Saved readout method; defaults to bundle recommendation')
    p.add_argument('--ground-truth',action='store_true')
    p.add_argument('--snapshot',type=Path)
    a=p.parse_args();manifest,datasets,dataset=select_dataset(a.viewer,a.condition)
    cmd=[sys.executable,str(ROOT/'scripts/view_neck_localization.py'),'--scenes',dataset['scenes'],
        '--records',dataset['records'],'--case',a.case,'--method',a.method or dataset.get('method','portfolio'),
        '--comparison-manifest',str(manifest),'--large-neck','--overlay','--rejected-candidate']
    if a.ground_truth:cmd.append('--ground-truth')
    if a.snapshot:cmd+=['--snapshot',str(a.snapshot.resolve())]
    subprocess.run(cmd,cwd=ROOT,check=True)


if __name__=='__main__':main()
