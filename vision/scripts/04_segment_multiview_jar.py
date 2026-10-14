#!/usr/bin/env python3
"""Segment a static jar capture using the existing GroundingDINO/SAM2 models."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt", default="ceramic jar. pottery vase.")
    parser.add_argument("--min-score", type=float, default=0.25)
    args = parser.parse_args()
    capture = args.capture.expanduser().resolve()
    metadata = json.loads((capture / "capture.json").read_text())
    if metadata.get("type") != "fixed_multiview_rgbd_capture":
        parser.error("Expected a fixed multiview RGB-D capture")
    if not 0 < args.min_score < 1:
        parser.error("--min-score must be between zero and one")
    module_path = Path(__file__).with_name("01_offline_grounded_sam2.py")
    spec = importlib.util.spec_from_file_location("winejar_grounded_sam2", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, detector = module.load_grounding_dino(
        module.DEFAULT_MODEL_ID, device, True
    )
    predictor = module.build_sam2_predictor(
        device,
        module.DEFAULT_SAM2_CONFIG,
        module.DEFAULT_SAM2_CHECKPOINT,
        module.DEFAULT_SAM2_MODEL_ID,
    )
    output = args.output_root.expanduser().resolve() / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "type": "multiview_jar_masks",
        "capture_dir": str(capture),
        "prompt": args.prompt,
        "cameras": {},
    }
    for frame_path in sorted(capture.glob("*/frame.json")):
        cid = frame_path.parent.name
        color = cv2.imread(str(frame_path.parent / "color.png"))
        if color is None:
            raise ValueError(f"{cid}: unreadable color image")
        image = Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
        detections = module.infer_grounding_dino_loaded(
            image, args.prompt, processor, detector, args.min_score, 0.15, device
        )
        if not detections:
            raise ValueError(
                f"{cid}: no jar detected; no geometric result will be produced"
            )
        detections.sort(key=lambda d: d["score"], reverse=True)
        # Duplicate labels for the same jar are common. Disjoint targets are ambiguous.
        box = np.array(detections[0]["box_xyxy"])
        for detection in detections[1:]:
            other = np.array(detection["box_xyxy"])
            intersection = np.prod(
                np.maximum(
                    0, np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2])
                )
            )
            union = (
                np.prod(box[2:] - box[:2])
                + np.prod(other[2:] - other[:2])
                - intersection
            )
            if (
                intersection / max(union, 1) < 0.3
                and detection["score"] > detections[0]["score"] - 0.15
            ):
                raise ValueError(
                    f"{cid}: multiple plausible jars; restrict the scene before measuring"
                )
        with torch.inference_mode():
            mask = module.infer_sam2_masks(predictor, image, detections[:1])[0] * 255
        camera_dir = output / cid
        camera_dir.mkdir()
        cv2.imwrite(str(camera_dir / "mask.png"), mask)
        overlay = color.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
        cv2.imwrite(str(camera_dir / "overlay.png"), overlay)
        result["cameras"][cid] = {
            "selected": detections[0],
            "detections": detections,
            "area_px": int(np.count_nonzero(mask)),
        }
        print(
            f"{cid}: {detections[0]['label']} score={detections[0]['score']:.3f}, area={np.count_nonzero(mask)}",
            flush=True,
        )
    (output / "segmentation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved jar masks: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
