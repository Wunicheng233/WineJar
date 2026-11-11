#!/usr/bin/env python3
"""Offline factorial comparison; observations to estimator, truth only to scorer."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neck_localization.support_feedback import FeedbackSettings, estimate_support_feedback
from neck_localization.joint_profile import JointSettings
from neck_localization.evaluation import score, summarize
from neck_localization.scene_preparation import write_json
from scripts.run_depth_profile_baseline import load_frames
from scripts.run_joint_profile import source_hashes as old_source_hashes

MODES = ['fixed', 'huber', 'region']
CAMERA_MODES = ['balanced', 'equal_trust']
CAMERA_SETS = ['cam_a', 'cam_a,cam_b', 'all']


def source_hashes():
    names = ['neck_localization/support_feedback.py', 'neck_localization/joint_profile.py',
             'neck_localization/depth_profile.py', 'neck_localization/geometry.py',
             'neck_localization/evaluation.py', 'scripts/run_depth_profile_baseline.py',
             'scripts/run_support_feedback.py', 'neck_localization/jar_model.py',
             'neck_localization/scene_preparation.py']
    return {n: sha256((ROOT/n).read_bytes()).hexdigest() for n in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['development','test'], default='development')
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--camera-sets', nargs='+', default=CAMERA_SETS)
    args = parser.parse_args()
    locked_sources = source_hashes()
    old = json.loads((ROOT/'experiments/neck_localization/results/joint_profile_20261023/freeze.json').read_text())
    assert old['sources'] == old_source_hashes(), 'Original estimator changed'
    if args.split == 'test':
        assert args.freeze and len(args.scenes) == 1 and not args.cases and args.camera_sets == CAMERA_SETS
        frozen = json.loads(args.freeze.read_text())
        assert frozen['sources'] == source_hashes()
        assert frozen['feedback_settings'] == asdict(FeedbackSettings())
        assert frozen['joint_settings'] == asdict(JointSettings())
        assert frozen['test_config_sha256'] == sha256((args.scenes[0]/'effective_config.json').read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    # Flat settings preserve compatibility with the viewer's observation reconstruction.
    write_json(args.output/'settings.json', dict(**asdict(JointSettings()), feedback=asdict(FeedbackSettings())))
    rows = []
    for scene_root in args.scenes:
        for camera_file in sorted(scene_root.glob('*/observations/camera_true_parameters.json')):
            folder = camera_file.parent.parent
            if args.cases and folder.name not in args.cases: continue
            for camera_set in args.camera_sets:
                frames = load_frames(folder/'observations', 'rgb', None if camera_set == 'all' else camera_set.split(','))
                for mode in MODES:
                    for camera_mode in CAMERA_MODES:
                        est = estimate_support_feedback(frames, mode=mode, camera_mode=camera_mode)
                        # Only after estimator returns, open evaluation truth.
                        truth = json.loads((folder/'evaluation_only/target.json').read_text())
                        row = dict(case=folder.name, split=args.split, geometry='photo_short_neck', evidence='rgb',
                            camera_set=camera_set, factor='ideal_true_cameras/'+camera_set,
                            method=f'{mode}_{camera_mode}', **score(est, truth))
                        row.setdefault('diagnostics', est.diagnostics)
                        rows.append(row)
                        write_json(args.output/'records.json', rows)
                        print(json.dumps({k:row[k] for k in ['case','camera_set','method','accepted','center_mm','normal_deg','reasons']}), flush=True)
    if not rows: raise ValueError('No scenes found')
    assert locked_sources == source_hashes(), 'Sources changed during run; keep as exploratory, rerun frozen version'
    write_json(args.output/'summary.json', summarize(rows))
    write_json(args.output/'provenance.json', dict(sources=source_hashes(), split=args.split,
        scene_roots=[str(s.resolve()) for s in args.scenes], feedback_settings=asdict(FeedbackSettings()),
        joint_settings=asdict(JointSettings()), scope='ideal depth; same fixed initial candidate set; no camera search/no noise/no truth in estimators'))


if __name__ == '__main__': main()
