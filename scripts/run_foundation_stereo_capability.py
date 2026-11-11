#!/usr/bin/env python3
"""Run official pretrained FoundationStereo on frozen observation-only pairs."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from neck_localization.baseline_capability import file_sha256, verify_pair, disparity_to_depth
from neck_localization.foundation_stereo_baseline import FoundationStereoRunner


def run(args):
    import torch
    torch.set_num_threads(6)
    manifest = json.loads(args.manifest.read_text())
    selected = [r for r in manifest['pairs'] if not args.layouts or r['layout'] in args.layouts]
    if args.limit:
        selected = selected[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model = FoundationStereoRunner(args.repo, args.checkpoint, iterations=args.iterations)
    load_ms = (time.perf_counter() - started) * 1000
    for rec in selected:
        verify_pair(rec)
        try:
            images = [cv2.imread(rec[k], cv2.IMREAD_GRAYSCALE) for k in ('left_path','right_path')]
            disparity, timing = model.predict(*images)
            depth, valid = disparity_to_depth(disparity, rec['K_ir'][0][0], rec['baseline_m'])
            path = args.output / 'predictions' / (rec['pair_id'] + '.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, disparity_px=disparity, native_float_z=depth, valid_mask=valid)
            row = dict(pair_id=rec['pair_id'], status='ok', **timing,
                input_sha256=rec['input_sha256'], prediction_sha256=file_sha256(path))
        except Exception as error:
            row = dict(pair_id=rec['pair_id'], status='error', error=repr(error), traceback=traceback.format_exc())
            torch.cuda.empty_cache()
        with (args.output / 'predictions.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(rec['pair_id'], row['status'], row.get('inference_ms'), row.get('error'), flush=True)
        if row['status'] == 'error' and args.limit:
            break
    provenance = dict(method='official FoundationStereo pretrained', checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=file_sha256(args.checkpoint), config=model.config, iterations=args.iterations,
        model_load_ms=load_ms, torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), manifest_sha256=file_sha256(args.manifest),
        source_sha256={str(p.relative_to(ROOT)):file_sha256(p) for p in
            [Path(__file__), ROOT/'neck_localization/foundation_stereo_baseline.py', ROOT/'neck_localization/baseline_capability.py']},
        upstream_source_sha256={str(p.relative_to(args.repo)):file_sha256(p) for p in args.repo.rglob('*.py')},
        no_hardware=True, no_ground_truth_input=True, weights_load='strict',
        adaptation='headless exact Utils helpers; bundled DINO architecture; skip redundant backbone pretraining before strict full state load',
        runtime='per-pair GPU forward+output transfer; first call includes warmup; model loading and IO separate')
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--layouts', nargs='*', default=[])
    p.add_argument('--iterations', type=int, default=32)
    p.add_argument('--limit', type=int, default=0)
    run(p.parse_args())
