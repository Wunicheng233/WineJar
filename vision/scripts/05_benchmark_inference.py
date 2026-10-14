#!/usr/bin/env python3
"""Offline GPU precision comparison; reads frames without changing their results."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment


def compare_outputs(reference, candidate):
    """Match same-label boxes before comparing masks; report unmatched detections."""
    old, old_masks, old_points = reference
    new, new_masks, new_points = candidate
    overlaps = np.zeros((len(old), len(new)))
    for i, a in enumerate(old):
        for j, b in enumerate(new):
            if a["label"] != b["label"]:
                continue
            ax1, ay1, ax2, ay2 = a["box_xyxy"]
            bx1, by1, bx2, by2 = b["box_xyxy"]
            intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
            union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - intersection
            overlaps[i, j] = intersection / union if union > 0 else 0
    rows, columns = linear_sum_assignment(-overlaps)
    matches = []
    for i, j in zip(rows, columns):
        if overlaps[i, j] < .5:
            continue
        a, b = old_masks[i] > 0, new_masks[j] > 0
        union = np.count_nonzero(a | b)
        matches.append(dict(
            reference_index=int(i), candidate_index=int(j), box_iou=float(overlaps[i, j]),
            mask_iou=float(np.count_nonzero(a & b) / union) if union else 1.,
            score_delta=float(new[j]["score"] - old[i]["score"]),
            point_validity_equal=(old_points[i] is None) == (new_points[j] is None),
            point_delta_mm=None if old_points[i] is None or new_points[j] is None else
            float(np.linalg.norm(np.subtract(old_points[i], new_points[j])) * 1000),
        ))
    matched_old = {m["reference_index"] for m in matches}
    matched_new = {m["candidate_index"] for m in matches}
    return dict(
        reference_count=len(old), candidate_count=len(new),
        exact_detections=old == new,
        exact_masks=len(old_masks) == len(new_masks) and all(np.array_equal(a, b) for a, b in zip(old_masks, new_masks)),
        unmatched_reference=[i for i in range(len(old)) if i not in matched_old],
        unmatched_candidate=[i for i in range(len(new)) if i not in matched_new], matches=matches,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON list of frame_dir, text_prompt, box_threshold, text_threshold.")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.output.exists():
        parser.error("Use positive repeats and a new output path")
    import cv2
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("This comparison requires a CUDA GPU with bfloat16 support")
    source = Path(__file__).with_name("01_offline_grounded_sam2.py")
    spec = importlib.util.spec_from_file_location("offline_benchmark", source)
    vision = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = vision
    spec.loader.exec_module(vision)
    device = torch.device("cuda")
    processor, detector = vision.load_grounding_dino(vision.DEFAULT_MODEL_ID, device, True)
    predictor = vision.build_sam2_predictor(device, vision.DEFAULT_SAM2_CONFIG,
                                          vision.DEFAULT_SAM2_CHECKPOINT, vision.DEFAULT_SAM2_MODEL_ID)
    report = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                  numpy=np.__version__, opencv=cv2.__version__, gpu=torch.cuda.get_device_name(),
                  torch_threads=torch.get_num_threads(), opencv_threads=cv2.getNumThreads(), repeats=args.repeats,
                  detector_model=vision.DEFAULT_MODEL_ID, sam2_model=vision.DEFAULT_SAM2_MODEL_ID,
                  sam2_config=vision.DEFAULT_SAM2_CONFIG,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(), frames=[])
    modes = ("fp32", "sam2_bf16", "all_bf16")
    for entry in json.loads(args.manifest.read_text()):
        folder = Path(entry["frame_dir"]).expanduser().resolve()
        metadata, color, depth, pil = vision.load_frame(folder)

        def infer(mode):
            torch.cuda.synchronize()
            started = time.perf_counter()
            with (torch.autocast("cuda", dtype=torch.bfloat16) if mode == "all_bf16" else nullcontext()):
                detections = vision.infer_grounding_dino_loaded(
                    pil, entry["text_prompt"], processor, detector,
                    entry["box_threshold"], entry["text_threshold"], device,
                )
            torch.cuda.synchronize()
            grounding_seconds = time.perf_counter() - started
            started = time.perf_counter()
            with (torch.autocast("cuda", dtype=torch.bfloat16) if mode == "all_bf16" else nullcontext()):
                masks = vision.infer_sam2_masks(
                    predictor, pil, detections, precision="bfloat16" if mode == "sam2_bf16" else "float32",
                )
            torch.cuda.synchronize()
            sam_seconds = time.perf_counter() - started
            points = [vision.mask_geometry(mask, depth, metadata["stream"]["depth_scale_m_per_unit"],
                                           metadata["intrinsics"], vision.centroid_mode_for_label(detection["label"]))[2]
                      for mask, detection in zip(masks, detections)]
            return (detections, masks, points), dict(grounding_dino=grounding_seconds, sam2=sam_seconds,
                                                   total=grounding_seconds + sam_seconds)

        reference, _ = infer("fp32")
        # Warm each mode before timed, interleaved rounds.
        for mode in modes:
            infer(mode)
        samples = {mode: [] for mode in modes}
        outputs = {}
        for repeat in range(args.repeats):
            order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
            for mode in order:
                output, timing = infer(mode)
                comparison = compare_outputs(reference, output)
                samples[mode].append(dict(timing_seconds=timing, comparison=comparison))
                outputs[mode] = dict(detections=output[0], points_camera_m=output[2])
        medians = {mode: {stage: statistics.median(s["timing_seconds"][stage] for s in values)
                          for stage in ("grounding_dino", "sam2", "total")} for mode, values in samples.items()}
        report["frames"].append(dict(input=entry, color_sha256=hashlib.sha256(color.tobytes()).hexdigest(),
                                      depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest(),
                                      median_seconds=medians, samples=samples, outputs=outputs))
        print(entry["text_prompt"], json.dumps(medians), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
