#!/usr/bin/env python3
"""Realtime RealSense preview with periodic GroundingDINO + SAM2 overlay.

This script does not connect to the robot. It is for camera placement, prompt
tuning, and segmentation quality checks.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from importlib.machinery import SourceFileLoader


OFFLINE = SourceFileLoader(
    "offline_grounded_sam2",
    str(Path(__file__).with_name("01_offline_grounded_sam2.py")),
).load_module()

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "realtime_captures"


class GroundedSam2Realtime:
    def __init__(
        self,
        text_prompt: str,
        model_id: str,
        sam2_model_id: str,
        sam2_config: str,
        sam2_checkpoint: Path | None,
        box_threshold: float,
        text_threshold: float,
        device: torch.device,
    ) -> None:
        self.text_prompt = text_prompt.strip()
        if not self.text_prompt.endswith("."):
            self.text_prompt += "."
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

        print(f"Loading GroundingDINO: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.detector.eval()

        print("Loading SAM2")
        self.sam2_predictor = OFFLINE.build_sam2_predictor(
            device=device,
            config_file=sam2_config,
            checkpoint=sam2_checkpoint,
            model_id=sam2_model_id,
        )

    def detect(self, color_bgr: np.ndarray) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(color_rgb)

        inputs = self.processor(images=image_pil, text=self.text_prompt, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            outputs = self.detector(**inputs)

        target_sizes = torch.tensor([image_pil.size[::-1]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )[0]

        raw_detections = []
        labels = results.get("text_labels", results.get("labels", []))
        for label, score, box in zip(labels, results["scores"], results["boxes"]):
            raw_detections.append(
                {
                    "label": str(label),
                    "score": float(score.detach().cpu()),
                    "box_xyxy": [float(v) for v in box.detach().cpu().tolist()],
                }
            )
        masks = OFFLINE.infer_sam2_masks(self.sam2_predictor, image_pil, raw_detections)
        return raw_detections, masks


def camera_info(device: rs.device) -> dict[str, str]:
    fields = {
        "name": rs.camera_info.name,
        "serial": rs.camera_info.serial_number,
        "firmware": rs.camera_info.firmware_version,
        "usb_type": rs.camera_info.usb_type_descriptor,
    }
    result = {}
    for key, field in fields.items():
        result[key] = device.get_info(field) if device.supports(field) else "unknown"
    return result


def choose_device(context: rs.context, serial: str | None) -> rs.device:
    devices = context.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device detected")
    if serial is None:
        return devices[0]
    for device in devices:
        if device.get_info(rs.camera_info.serial_number) == serial:
            return device
    available = [d.get_info(rs.camera_info.serial_number) for d in devices]
    raise RuntimeError(f"RealSense serial {serial!r} not found; available: {available}")


def intrinsics_to_dict(intrinsics: rs.intrinsics) -> dict[str, Any]:
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "distortion_model": str(intrinsics.model),
        "distortion_coefficients": list(intrinsics.coeffs),
    }


def detections_from_raw(
    raw_detections: list[dict[str, Any]],
    masks: list[np.ndarray],
    depth_raw: np.ndarray,
    depth_scale_m: float,
    intrinsics: dict[str, Any],
) -> list[Any]:
    detections = []
    for idx, raw in enumerate(raw_detections):
        mask = masks[idx] if idx < len(masks) else np.zeros(depth_raw.shape, dtype=np.uint8)
        centroid_method = OFFLINE.centroid_mode_for_label(raw["label"])
        edge_sides = OFFLINE.mask_edge_sides(mask)
        centroid_px, median_depth_m, point_camera_m = OFFLINE.mask_geometry(
            mask,
            depth_raw,
            depth_scale_m,
            intrinsics,
            centroid_mode=centroid_method,
        )
        detections.append(
            OFFLINE.Detection(
                label=raw["label"],
                score=raw["score"],
                box_xyxy=raw["box_xyxy"],
                mask_path=None,
                centroid_px=centroid_px,
                median_depth_m=median_depth_m,
                point_camera_m=point_camera_m,
                centroid_method=centroid_method,
                touches_image_edge=bool(edge_sides),
                edge_sides=edge_sides,
            )
        )
    return detections


def save_snapshot(
    output_root: Path,
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    annotated: np.ndarray,
    detections: list[Any],
    masks: list[np.ndarray],
    metadata: dict[str, Any],
) -> Path:
    timestamp = datetime.now().astimezone()
    output_dir = output_root.expanduser().resolve() / timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_dir.mkdir(parents=True, exist_ok=False)

    cv2.imwrite(str(output_dir / "color.png"), color_bgr)
    cv2.imwrite(str(output_dir / "depth_raw.png"), depth_raw)
    cv2.imwrite(str(output_dir / "annotated.png"), annotated)
    for idx, mask in enumerate(masks):
        cv2.imwrite(str(output_dir / f"mask_{idx:02d}.png"), (mask > 0).astype(np.uint8) * 255)

    result = {
        "schema_version": 1,
        "captured_at": timestamp.isoformat(timespec="milliseconds"),
        "metadata": metadata,
        "detections": [asdict(det) for det in detections],
        "files": {
            "color": "color.png",
            "depth_raw": "depth_raw.png",
            "annotated": "annotated.png",
        },
    }
    (output_dir / "vision_result.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-prompt", required=True)
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--infer-every-sec", type=float, default=1.0)
    parser.add_argument("--box-threshold", type=float, default=0.45)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--model-id", default=OFFLINE.DEFAULT_MODEL_ID)
    parser.add_argument("--sam2-model-id", default=OFFLINE.DEFAULT_SAM2_MODEL_ID)
    parser.add_argument("--sam2-config", default=OFFLINE.DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=OFFLINE.DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    context = rs.context()
    device = choose_device(context, args.serial)
    info = camera_info(device)
    print(
        f"Using {info['name']}, serial={info['serial']}, "
        f"firmware={info['firmware']}, USB={info['usb_type']}"
    )
    if not info["usb_type"].startswith("3"):
        print("WARNING: RealSense is not connected as USB3.x; RGB-D streaming may be unstable.")

    torch_device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Inference device: {torch_device}")
    realtime = GroundedSam2Realtime(
        text_prompt=args.text_prompt,
        model_id=args.model_id,
        sam2_model_id=args.sam2_model_id,
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint.expanduser().resolve() if args.sam2_checkpoint else None,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=torch_device,
    )

    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(info["serial"])
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align_to_color = rs.align(rs.stream.color)

    print("Controls: q/Esc quit, s save current frame/result.")
    started = False
    last_infer_t = 0.0
    last_detections: list[Any] = []
    last_masks: list[np.ndarray] = []
    last_annotated: np.ndarray | None = None

    try:
        profile = pipeline.start(config)
        started = True
        depth_scale_m = profile.get_device().first_depth_sensor().get_depth_scale()

        while True:
            frames = align_to_color.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            intrinsics = intrinsics_to_dict(
                color_frame.profile.as_video_stream_profile().intrinsics
            )
            now = time.monotonic()

            if now - last_infer_t >= args.infer_every_sec:
                started_infer = time.monotonic()
                raw_detections, last_masks = realtime.detect(color_bgr)
                last_detections = detections_from_raw(
                    raw_detections,
                    last_masks,
                    depth_raw,
                    depth_scale_m,
                    intrinsics,
                )
                last_infer_t = now
                elapsed = time.monotonic() - started_infer
                print(f"detections={len(last_detections)} inference={elapsed:.2f}s")

            last_annotated = OFFLINE.draw_overlay(color_bgr, last_detections, last_masks)
            status = f"{args.text_prompt} | detections={len(last_detections)} | q quit | s save"
            cv2.putText(
                last_annotated,
                status,
                (10, last_annotated.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.imshow("WineJar GroundingDINO + SAM2 RealSense Preview", last_annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and last_annotated is not None:
                metadata = {
                    "camera": info,
                    "stream": {
                        "width": args.width,
                        "height": args.height,
                        "fps": args.fps,
                        "depth_scale_m_per_unit": depth_scale_m,
                        "coordinate_convention": "+x right, +y down, +z forward",
                    },
                    "intrinsics": intrinsics,
                    "text_prompt": args.text_prompt,
                    "box_threshold": args.box_threshold,
                    "text_threshold": args.text_threshold,
                }
                output_dir = save_snapshot(
                    args.output_root,
                    color_bgr,
                    depth_raw,
                    last_annotated,
                    last_detections,
                    last_masks,
                    metadata,
                )
                print(f"Saved snapshot: {output_dir}")
    finally:
        if started:
            pipeline.stop()
            print("RealSense pipeline stopped.")
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
