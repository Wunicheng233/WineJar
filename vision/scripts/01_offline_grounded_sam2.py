#!/usr/bin/env python3
"""Run GroundingDINO + SAM2 on one saved RGB-D frame folder.

This script intentionally has no robot-control dependency. It consumes the frame
schema written by xarm_sdk_control/scripts/13_capture_realsense_frame.py.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_SAM2_CONFIG = "configs/sam2/sam2_hiera_t.yaml"
DEFAULT_SAM2_CHECKPOINT = ROOT / "models" / "sam2-hiera-tiny" / "sam2_hiera_tiny.pt"
DEFAULT_SAM2_MODEL_ID = "facebook/sam2-hiera-tiny"


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box_xyxy: list[float]
    mask_path: str | None
    centroid_px: list[float] | None
    median_depth_m: float | None
    point_camera_m: list[float] | None
    point_world_m: list[float] | None = None
    centroid_method: str = "nearest_surface"
    touches_image_edge: bool = False
    edge_sides: tuple[str, ...] = ()


def load_frame(frame_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, Image.Image]:
    metadata_path = frame_dir / "frame.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing frame metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    color_path = frame_dir / metadata["files"]["color_bgr"]
    depth_path = frame_dir / metadata["files"]["depth_raw"]
    color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if color_bgr is None:
        raise RuntimeError(f"Failed to read color image: {color_path}")
    if depth_raw is None:
        raise RuntimeError(f"Failed to read depth image: {depth_path}")

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(color_rgb)
    return metadata, color_bgr, depth_raw, image_pil


def infer_grounding_dino(
    image: Image.Image,
    text_prompt: str,
    model_id: str,
    box_threshold: float,
    text_threshold: float,
    device: torch.device,
    local_files_only: bool,
) -> tuple[list[dict[str, Any]], Any]:
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()

    text = text_prompt.strip()
    if not text.endswith("."):
        text += "."

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )[0]

    detections = []
    labels = results.get("text_labels", results["labels"])
    for label, score, box in zip(labels, results["scores"], results["boxes"]):
        detections.append(
            {
                "label": str(label),
                "score": float(score.detach().cpu()),
                "box_xyxy": [float(v) for v in box.detach().cpu().tolist()],
            }
        )
    return detections, processor


def build_sam2_predictor(
    device: torch.device,
    config_file: str,
    checkpoint: Path | None,
    model_id: str,
) -> Any:
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:  # pragma: no cover - depends on optional install.
        raise RuntimeError(
            "SAM2 is not importable. Install it with: "
            "pip install -r vision/requirements-sam2.txt"
        ) from exc

    if checkpoint is not None and checkpoint.exists():
        model = build_sam2(
            config_file=config_file,
            ckpt_path=str(checkpoint),
            device=str(device),
        )
        return SAM2ImagePredictor(model)

    print(f"SAM2 local checkpoint not found; falling back to Hugging Face: {model_id}")
    return SAM2ImagePredictor.from_pretrained(model_id)


def infer_sam2_masks(
    predictor: Any,
    image_pil: Image.Image,
    detections: list[dict[str, Any]],
) -> list[np.ndarray]:
    if not detections:
        return []

    image_rgb = np.asarray(image_pil).copy()
    predictor.set_image(image_rgb)
    boxes = np.array([det["box_xyxy"] for det in detections], dtype=np.float32)
    masks, scores, _ = predictor.predict(
        box=boxes,
        multimask_output=False,
    )
    masks = np.asarray(masks)
    if masks.ndim == 4:
        masks = masks[:, 0]
    return [(mask > 0).astype(np.uint8) for mask in masks]


def mask_geometry(
    mask: np.ndarray,
    depth_raw: np.ndarray,
    depth_scale_m: float,
    intrinsics: dict[str, Any],
    centroid_mode: str = "nearest_surface",
) -> tuple[list[float] | None, float | None, list[float] | None]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None, None

    depths_m = depth_raw[ys, xs].astype(np.float32) * float(depth_scale_m)
    valid = depths_m > 0
    if not np.any(valid):
        return [float(np.mean(xs)), float(np.mean(ys))], None, None

    if centroid_mode == "geometric":
        centroid_u = float(np.mean(xs))
        centroid_v = float(np.mean(ys))
        center_u = int(round(centroid_u))
        center_v = int(round(centroid_v))
        radius = 6
        y0, y1 = max(0, center_v - radius), min(mask.shape[0], center_v + radius + 1)
        x0, x1 = max(0, center_u - radius), min(mask.shape[1], center_u + radius + 1)
        local_mask = mask[y0:y1, x0:x1] > 0
        local_depths_m = (
            depth_raw[y0:y1, x0:x1][local_mask].astype(np.float32) * float(depth_scale_m)
        )
        local_valid = local_depths_m > 0
        depth_values = local_depths_m[local_valid]
        if depth_values.size < 9:
            depth_values = depths_m[valid]
        median_depth_m = float(np.median(depth_values))
    elif centroid_mode == "nearest_surface":
        # Isolate a raised object when SAM also includes its lower support.
        valid_depths = depths_m[valid]
        near_depth_m = float(np.percentile(valid_depths, 10.0))
        foreground = valid & (depths_m <= near_depth_m + 0.006)
        if np.count_nonzero(foreground) < 9:
            foreground = valid
        centroid_u = float(np.mean(xs[foreground]))
        centroid_v = float(np.mean(ys[foreground]))
        median_depth_m = float(np.median(depths_m[foreground]))
    else:
        raise ValueError(f"Unsupported centroid mode: {centroid_mode}")
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])
    x_m = (centroid_u - ppx) * median_depth_m / fx
    y_m = (centroid_v - ppy) * median_depth_m / fy
    return [centroid_u, centroid_v], median_depth_m, [x_m, y_m, median_depth_m]


def centroid_mode_for_label(label: str) -> str:
    return "geometric" if "bamboo" in label.lower() else "nearest_surface"


def mask_edge_sides(mask: np.ndarray, margin_px: int = 2) -> tuple[str, ...]:
    binary = mask > 0
    if not np.any(binary):
        return ()
    edges = []
    if np.any(binary[: margin_px + 1, :]):
        edges.append("top")
    if np.any(binary[-(margin_px + 1) :, :]):
        edges.append("bottom")
    if np.any(binary[:, : margin_px + 1]):
        edges.append("left")
    if np.any(binary[:, -(margin_px + 1) :]):
        edges.append("right")
    return tuple(edges)


def mask_touches_image_edge(mask: np.ndarray, margin_px: int = 2) -> bool:
    return bool(mask_edge_sides(mask, margin_px))


def draw_overlay(color_bgr: np.ndarray, detections: list[Detection], masks: list[np.ndarray]) -> np.ndarray:
    annotated = color_bgr.copy()
    palette = [
        (0, 255, 0),
        (0, 200, 255),
        (255, 0, 180),
        (255, 180, 0),
        (180, 255, 0),
    ]
    for idx, det in enumerate(detections):
        color = palette[idx % len(palette)]
        x1, y1, x2, y2 = [int(round(v)) for v in det.box_xyxy]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        clipping_text = " CLIPPED" if det.touches_image_edge else ""
        label = f"{det.label} {det.score:.2f}{clipping_text}"
        cv2.putText(annotated, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if idx < len(masks):
            mask = masks[idx] > 0
            overlay = np.zeros_like(annotated)
            overlay[mask] = color
            annotated = cv2.addWeighted(annotated, 1.0, overlay, 0.35, 0)
        if det.centroid_px is not None:
            u, v = [int(round(x)) for x in det.centroid_px]
            cv2.circle(annotated, (u, v), 4, color, -1)
    return annotated


def camera_point_to_world(
    point_camera_m: list[float] | None,
    metadata: dict[str, Any],
) -> list[float] | None:
    if point_camera_m is None:
        return None
    transform = metadata.get("camera", {}).get("opencv_camera_to_world")
    if transform is None:
        return None
    point_h = np.array([point_camera_m[0], point_camera_m[1], point_camera_m[2], 1.0])
    point_world = np.array(transform, dtype=float) @ point_h
    return [float(v) for v in point_world[:3]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--text-prompt", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--sam2-model-id", default=DEFAULT_SAM2_MODEL_ID)
    parser.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--box-threshold", type=float, default=0.45)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument(
        "--output-min-score",
        type=float,
        default=0.45,
        help="Discard detections below this confidence before SAM2 and visualization.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access. By default, use the local model cache only.",
    )
    args = parser.parse_args()

    frame_dir = args.frame_dir.expanduser().resolve()
    metadata, color_bgr, depth_raw, image_pil = load_frame(frame_dir)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    print(f"Frame: {frame_dir}")

    raw_detections, _ = infer_grounding_dino(
        image=image_pil,
        text_prompt=args.text_prompt,
        model_id=args.model_id,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device,
        local_files_only=not args.allow_model_download,
    )
    raw_detections = [
        detection
        for detection in raw_detections
        if float(detection.get("score", 0.0)) >= args.output_min_score
    ]
    print(f"Detections: {len(raw_detections)}")

    predictor = build_sam2_predictor(
        device=device,
        config_file=args.sam2_config,
        checkpoint=args.sam2_checkpoint.expanduser().resolve() if args.sam2_checkpoint else None,
        model_id=args.sam2_model_id,
    )
    masks = infer_sam2_masks(predictor, image_pil, raw_detections)

    depth_scale_m = metadata["stream"]["depth_scale_m_per_unit"]
    intrinsics = metadata["intrinsics"]
    detections: list[Detection] = []
    for idx, raw in enumerate(raw_detections):
        mask = masks[idx] if idx < len(masks) else np.zeros(depth_raw.shape, dtype=np.uint8)
        mask_name = f"mask_{idx:02d}_{raw['label'].replace(' ', '_')}.png"
        cv2.imwrite(str(frame_dir / mask_name), mask * 255)
        centroid_method = centroid_mode_for_label(raw["label"])
        edge_sides = mask_edge_sides(mask)
        centroid_px, median_depth_m, point_camera_m = mask_geometry(
            mask,
            depth_raw,
            depth_scale_m,
            intrinsics,
            centroid_mode=centroid_method,
        )
        point_world_m = camera_point_to_world(point_camera_m, metadata)
        detections.append(
            Detection(
                label=raw["label"],
                score=raw["score"],
                box_xyxy=raw["box_xyxy"],
                mask_path=mask_name,
                centroid_px=centroid_px,
                median_depth_m=median_depth_m,
                point_camera_m=point_camera_m,
                point_world_m=point_world_m,
                centroid_method=centroid_method,
                touches_image_edge=bool(edge_sides),
                edge_sides=edge_sides,
            )
        )

    annotated = draw_overlay(color_bgr, detections, masks)
    cv2.imwrite(str(frame_dir / "annotated.png"), annotated)

    result = {
        "schema_version": 1,
        "frame_dir": str(frame_dir),
        "text_prompt": args.text_prompt,
        "model_id": args.model_id,
        "sam2": {
            "model_id": args.sam2_model_id,
            "config": args.sam2_config,
            "checkpoint": str(args.sam2_checkpoint) if args.sam2_checkpoint else None,
        },
        "device": str(device),
        "coordinate_convention": metadata["stream"]["coordinate_convention"],
        "detections": [det.__dict__ for det in detections],
        "files": {
            "annotated": "annotated.png",
            "result": "vision_result.json",
        },
    }
    (frame_dir / "vision_result.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved: {frame_dir / 'vision_result.json'}")
    print(f"Saved: {frame_dir / 'annotated.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
