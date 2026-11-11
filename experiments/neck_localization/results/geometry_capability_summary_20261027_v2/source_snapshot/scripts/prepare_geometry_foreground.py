#!/usr/bin/env python3
"""Cache observation-only SAM2 whole-jar silhouettes for all depth frontends.

Protocol is fixed before audit use: RGB chroma-key whole-object bbox + 8 px
padding, official pretrained SAM2 tiny, FP32, highest predicted mask score,
largest external contour filled. No target, neck ROI, depth or CAD is read.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from neck_localization.stereo_robust import blue_screen_foreground

CHECKPOINT = ROOT/'vision/models/sam2-hiera-tiny/sam2_hiera_tiny.pt'
CONFIG = 'configs/sam2/sam2_hiera_t.yaml'
PADDING_PIXELS = 8


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def prompt_from_rgb(rgb):
    coarse = blue_screen_foreground(rgb)
    yy, xx = np.nonzero(coarse)
    if not len(xx):
        raise ValueError('empty_RGB_whole_object_foreground')
    bbox = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
    h, w = coarse.shape
    padded = [max(0, bbox[0]-PADDING_PIXELS), max(0, bbox[1]-PADDING_PIXELS),
              min(w-1, bbox[2]+PADDING_PIXELS), min(h-1, bbox[3]+PADDING_PIXELS)]
    return bbox, padded


def select_silhouette(masks, scores):
    masks, scores = np.asarray(masks), np.asarray(scores)
    if masks.ndim != 3 or scores.ndim != 1 or len(masks) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('invalid_SAM2_multimask_outputs')
    selected = int(np.argmax(scores))
    raw = (masks[selected] > 0).astype(np.uint8)
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('highest_score_SAM2_mask_is_empty')
    silhouette = np.zeros(raw.shape, np.uint8)
    cv2.drawContours(silhouette, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    return selected, silhouette.astype(bool)


def selected_inputs(args):
    if args.manifest is not None:
        rows = json.loads(args.manifest.read_text())['pairs']
        images = {}
        for pair in rows:
            if args.layouts and pair['layout'] not in args.layouts:
                continue
            if args.cases and pair['case'] not in args.cases:
                continue
            key = (pair['layout'], pair['case'], pair['camera'])
            path = Path(pair['scene_dir'])/'observations'/(pair['camera']+'_rgb.png')
            if key in images and images[key] != path.resolve():
                raise ValueError('same_layout_case_camera_maps_to_different_RGB_paths')
            images[key] = path.resolve()
        return images
    root = args.render_root.resolve()
    config = json.loads((root/'effective_config.json').read_text())
    names = sorted(c['name'] for c in config['cases'] if not args.cases or c['name'] in args.cases)
    return {(args.layout, name, camera):root/'stereo_raw/scenes'/name/'observations'/(camera+'_rgb.png')
            for name in names for camera in ('cam_a', 'cam_b', 'cam_c')}


def run(args):
    # Explicit local existence checks prevent a hidden network fallback.
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    images = selected_inputs(args)
    if not images:
        raise ValueError('no_selected_RGB_inputs')
    args.output.mkdir(parents=True, exist_ok=False)
    import sam2
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    package = Path(sam2.__file__).parent
    config_path = package/CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    source_paths = [Path(__file__), ROOT/'neck_localization/stereo_robust.py']
    source_hashes = {str(p.resolve()):file_hash(p) for p in source_paths}
    official_sources = {str(p.relative_to(package)):file_hash(p) for p in sorted(package.rglob('*.py'))}
    model_hash = file_hash(args.checkpoint)
    input_hashes = {str(p):file_hash(p) for p in images.values()}
    torch.manual_seed(0)
    np.random.seed(0)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_sam2(config_file=CONFIG, ckpt_path=str(args.checkpoint.resolve()),
                       device=args.device)
    predictor = SAM2ImagePredictor(model)
    entries = []
    for (layout, case, camera), rgb_path in sorted(images.items()):
        entry = dict(layout=layout, case=case, camera=camera,
            rgb_path=str(rgb_path), rgb_sha256=input_hashes[str(rgb_path)])
        try:
            rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
            coarse_bbox, box = prompt_from_rgb(rgb)
            started = perf_counter()
            with torch.inference_mode():
                predictor.set_image(rgb)
                masks, scores, _ = predictor.predict(box=np.asarray(box, np.float32),
                    point_coords=None, point_labels=None, multimask_output=True)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            elapsed = (perf_counter()-started)*1000
            selected, silhouette = select_silhouette(masks, scores)
            mask_path = args.output/'masks'/layout/case/(camera+'.png')
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mask_path), silhouette.astype(np.uint8)*255):
                raise OSError('mask_PNG_write_failed')
            candidates_path = mask_path.with_name(camera+'_candidates.npz')
            np.savez_compressed(candidates_path, masks=np.asarray(masks)>0,
                predicted_scores=np.asarray(scores), selected_index=selected)
            entry.update(status='ok', prompt_box_xyxy=box,
                source_coarse_mask_bbox_xyxy=coarse_bbox, selected_index=selected,
                predicted_scores=np.asarray(scores, float).tolist(),
                selected_score=float(scores[selected]),
                mask_path=str(mask_path.resolve()), mask_sha256=file_hash(mask_path),
                mask_pixels_sha256=sha256(silhouette.astype(np.uint8).tobytes()).hexdigest(),
                candidate_masks_path=str(candidates_path.resolve()),
                candidate_masks_sha256=file_hash(candidates_path),
                raw_selected_area_pixels=int((np.asarray(masks[selected])>0).sum()),
                filled_area_pixels=int(silhouette.sum()), image_width=rgb.shape[1],
                image_height=rgb.shape[0], inference_ms=elapsed)
        except Exception as error:
            entry.update(status='error', error=f'{type(error).__name__}:{error}',
                         traceback=traceback.format_exc())
        entries.append(entry)
        with (args.output/'entries.jsonl').open('a') as stream:
            stream.write(json.dumps(entry, allow_nan=False)+'\n')
        print(layout, case, camera, entry['status'], entry.get('selected_score'), flush=True)
    unchanged = (source_hashes == {str(p.resolve()):file_hash(p) for p in source_paths}
                 and all(file_hash(p)==h for p,h in input_hashes.items())
                 and file_hash(args.checkpoint)==model_hash)
    manifest = dict(schema=1, entries=entries, no_hardware=True, no_training=True,
        no_ground_truth_read=True, no_depth_or_CAD_read=True,
        reusable_across_depth_frontends=True, source_sha256=source_hashes,
        input_sha256=input_hashes, official_SAM2_source_sha256=official_sources,
        protocol=dict(name='sam2_tiny_RGB_whole_jar_bbox_fixed8px_highest_score_filled_external',
            source='original_RGB_only', bbox='whole coarse RGB foreground bbox; image-clipped fixed pixel padding',
            padding_pixels=PADDING_PIXELS, points=None, neck_ROI=False,
            multimask_output=True, selection='argmax SAM2 predicted score, no external quality criterion',
            postprocess='largest external contour filled; no morphology or depth filtering',
            precision='float32', seed=0),
        model=dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=model_hash,
            config=CONFIG, config_path=str(config_path), config_sha256=file_hash(config_path),
            official_implementation='https://github.com/facebookresearch/sam2',
            torch_version=torch.__version__, device=args.device,
            gpu_name=torch.cuda.get_device_name() if args.device.startswith('cuda') else None),
        inputs_and_adapter_unchanged=unchanged,
        successful_images=sum(e['status']=='ok' for e in entries), total_images=len(entries))
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    if not unchanged:
        raise RuntimeError('input_adapter_or_checkpoint_changed_during_preparation')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest', type=Path)
    source.add_argument('--render-root', type=Path)
    p.add_argument('--layout', default='development', help='cache label for --render-root')
    p.add_argument('--layouts', nargs='*', default=['development'], help='filters for --manifest')
    p.add_argument('--cases', nargs='*', default=[])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    p.add_argument('--device', default='cuda')
    run(p.parse_args())
