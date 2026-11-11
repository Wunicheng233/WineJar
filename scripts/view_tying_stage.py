#!/usr/bin/env python3
"""Original MuJoCo viewer with three-station timeline. Offline saved predictions."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get('MUJOCO_GL') == 'egl':
    del os.environ['MUJOCO_GL']
from neck_localization.tying_viewer import TyingViewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', type=Path, default=ROOT/'experiments/neck_localization/results/three_station_tying_20261028_v5')
    parser.add_argument('--condition', default='new8/fs/clean')
    parser.add_argument('--case', default='confirmation_nominal_001')
    parser.add_argument('--ground-truth', action='store_true')
    parser.add_argument('--point-cloud', action='store_true')
    parser.add_argument('--large-neck', action='store_true')
    parser.add_argument('--phase', choices=['recognize', 'cover', 'above', 'descend', 'close', 'tighten', 'retract'])
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--hidden', action='store_true')
    args = parser.parse_args()
    datasets = json.loads((args.viewer/'datasets.json').read_text())
    selected = next(d for d in datasets if d['condition'] == args.condition)
    viewer = TyingViewer(selected['scenes'], selected['records'], args.case, method=selected['method'],
        datasets=datasets, station_config=json.loads((args.viewer/'config.json').read_text()),
        ground_truth=args.ground_truth, point_cloud=args.point_cloud, large_neck=args.large_neck,
        visible=not args.hidden, overlay=True, band=True)
    try:
        if args.phase:
            viewer._activate('stage:'+args.phase)
        if args.play:
            viewer._activate('stage:play')
        if args.snapshot:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            viewer.snapshot(args.snapshot)
        else:
            viewer.run()
    finally:
        viewer.close()


if __name__ == '__main__':
    main()
