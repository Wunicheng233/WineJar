#!/usr/bin/env python3
"""Inspect the eight-jar study or the explicitly separate native-IR diagnostic."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition',choices=['ideal_rgb','ideal_ir_registered','stereo_raw','stereo_spatial','stereo_gp','stereo_gp_spatial',
        'native_ideal','native_ideal_z16','native_raw','native_spatial'],default='stereo_raw')
    p.add_argument('--parameters',choices=['true','board'],default='true')
    p.add_argument('--case')
    p.add_argument('--ground-truth',action='store_true')
    p.add_argument('--weights',action='store_true',help='Show saved actual fitting points/weights')
    p.add_argument('--snapshot',type=Path)
    args=p.parse_args()
    native=args.condition.startswith('native_')
    method='native_ir_rgb_mapped' if native else 'stereo_rgb_adapted'
    case=args.case or ('development_nominal_000' if native else 'test_nominal_000')
    if native:
        if args.parameters=='board':
            p.error('Native-IR diagnostics use true IR/RGB parameters only')
        name='native_stereo_viewer_20261025_v2'
    elif args.parameters=='board':
        if args.condition.startswith('stereo_gp'):
            p.error('The board-parameter comparison currently covers the four non-GP inputs only')
        name='active_stereo_board_calibrated_test_20261025_v1'
    elif args.condition.startswith('stereo_gp'):
        name='active_stereo_gp_adapted_test_20261025_v1'
    else:
        name='active_stereo_adapted_test_20261025_v1'
    base=ROOT/'experiments/neck_localization/runs'/name
    directory=base/args.condition if native else base/'stereo_rgb_adapted'/args.condition
    command=[sys.executable,str(ROOT/'scripts/view_neck_localization.py'),
        '--scenes',str(directory/'scenes'),'--records',str(directory/'records.json.gz'),
        '--case',case,'--method',method,'--point-cloud','--large-neck','--overlay','--band']
    if args.ground_truth:command.append('--ground-truth')
    if args.weights:command.append('--weights')
    if args.snapshot:command+=['--snapshot',str(args.snapshot.resolve())]
    subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':main()
