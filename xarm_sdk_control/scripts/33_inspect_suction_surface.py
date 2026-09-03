#!/usr/bin/env python3
"""Capture or load one wrist RGB-D frame and visualize ranked dual-cup poses.

This diagnostic does not connect to or move the robot. Target segmentation is
provided by the existing Grounded-SAM2 service, or by an explicit mask path.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_sdk_control.realsense_capture import RealSenseCaptureSession  # noqa: E402
from xarm_sdk_control.suction_surface import (  # noqa: E402
    analyze_suction_surface,
    candidate_to_dict,
    draw_suction_candidates,
    load_surface_profiles,
)
from xarm_sdk_control.visual_targeting import (  # noqa: E402
    DEFAULT_VISION_SOCKET,
    detection_object_kind,
    run_vision,
)


DEFAULT_OUTPUT_ROOT = ROOT / "data" / "suction_surface"
TARGETS = {
    "lotus": ("lotus", "lotus leaf."),
    "bamboo": ("bamboo", "bamboo leaf."),
    "ceramic": ("ceramic", "ceramic disk."),
    "paper": ("paper", "white paper."),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=tuple(TARGETS), required=True)
    parser.add_argument("--serial", default="011422070172", help="Wrist RealSense serial.")
    parser.add_argument("--frame-dir", type=Path, help="Analyze an existing saved RGB-D frame.")
    parser.add_argument("--mask-path", type=Path, help="Use this object mask instead of Grounded-SAM2.")
    parser.add_argument("--min-score", type=float, default=0.30)
    parser.add_argument("--vision-socket", type=Path, default=DEFAULT_VISION_SOCKET)
    parser.add_argument("--require-vision-service", action="store_true")
    parser.add_argument("--vision-env", default="winejar-vision")
    parser.add_argument("--grid-step-px", type=int, default=6)
    parser.add_argument("--angle-step-deg", type=float, default=15.0)
    parser.add_argument("--preferred-angle-deg", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def load_frame(frame_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    resolved = frame_dir.expanduser().resolve()
    metadata = json.loads((resolved / "frame.json").read_text(encoding="utf-8"))
    color = cv2.imread(str(resolved / metadata["files"]["color_bgr"]), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(resolved / metadata["files"]["depth_raw"]), cv2.IMREAD_UNCHANGED)
    if color is None or depth_raw is None:
        raise ValueError(f"Unreadable RGB-D files in {resolved}")
    intrinsics = metadata["intrinsics"]
    camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    depth_scale = float(metadata["stream"]["depth_scale_m_per_unit"])
    return color, depth_raw, camera_matrix, depth_scale, metadata


def select_mask(
    args: argparse.Namespace,
    frame_dir: Path,
    color_shape: tuple[int, ...],
) -> tuple[np.ndarray, dict | None]:
    if args.mask_path is not None:
        mask_path = args.mask_path.expanduser().resolve()
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        if mask.shape != color_shape[:2]:
            raise ValueError("Explicit mask dimensions do not match the color frame")
        return mask, None

    kind, prompt = TARGETS[args.target]
    result = run_vision(
        frame_dir,
        args.vision_env,
        args.min_score,
        text_prompt=prompt,
        service_socket=args.vision_socket,
        require_service=args.require_vision_service,
    )
    candidates = [
        detection
        for detection in result.get("detections", [])
        if detection_object_kind(detection) == kind and detection.get("mask_path")
    ]
    if not candidates:
        raise RuntimeError(f"No segmented {args.target} candidate was returned")
    detection = max(candidates, key=lambda item: float(item.get("score", 0.0)))
    mask_path = frame_dir / str(detection["mask_path"])
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    return mask, detection


def main() -> int:
    args = parse_args()
    if min(args.grid_step_px, args.angle_step_deg, args.top_k) <= 0:
        raise ValueError("Search resolution and --top-k must be positive")
    if args.frame_dir is None:
        with RealSenseCaptureSession(serial=args.serial) as camera:
            frame_dir = camera.capture(args.output_root)
    else:
        frame_dir = args.frame_dir.expanduser().resolve()

    color, depth_raw, camera_matrix, depth_scale, frame_metadata = load_frame(frame_dir)
    mask, detection = select_mask(args, frame_dir, color.shape)
    profiles = load_surface_profiles()
    analysis = analyze_suction_surface(
        color,
        depth_raw.astype(np.float32) * depth_scale,
        mask,
        camera_matrix,
        profiles[args.target],
        grid_step_px=args.grid_step_px,
        angle_step_deg=args.angle_step_deg,
        preferred_angle_deg=args.preferred_angle_deg,
        max_candidates=args.top_k,
    )
    overlay = draw_suction_candidates(color, analysis, count=min(3, args.top_k))
    cv2.putText(
        overlay,
        f"{args.target}: {len(analysis.candidates)} valid poses | red=surface defect",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    overlay_path = frame_dir / f"suction_candidates_{args.target}.png"
    defect_path = frame_dir / f"suction_defect_mask_{args.target}.png"
    roughness_path = frame_dir / f"suction_roughness_{args.target}.png"
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(defect_path), analysis.defect_mask * 255)
    roughness_preview = np.clip(
        analysis.roughness_mm / profiles[args.target].max_roughness_mm * 255.0,
        0,
        255,
    ).astype(np.uint8)
    cv2.imwrite(str(roughness_path), cv2.applyColorMap(roughness_preview, cv2.COLORMAP_TURBO))
    output = {
        "schema_version": 1,
        "type": "dual_suction_surface_analysis",
        "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "target": args.target,
        "camera_serial": frame_metadata.get("camera", {}).get("serial"),
        "frame_dir": str(frame_dir),
        "source_detection": detection,
        "profile": vars(profiles[args.target]),
        "suction_geometry_mm": {"cup_diameter": 8.0, "cup_center_spacing": 48.0},
        "search": {
            "grid_step_px": args.grid_step_px,
            "angle_step_deg": args.angle_step_deg,
            "preferred_angle_deg": args.preferred_angle_deg,
            "reference_depth_m": analysis.reference_depth_m,
            "cup_radius_px": analysis.cup_radius_px,
            "cup_spacing_px": analysis.cup_spacing_px,
        },
        "candidate_count": len(analysis.candidates),
        "candidates": [candidate_to_dict(candidate) for candidate in analysis.candidates],
        "files": {
            "overlay": overlay_path.name,
            "defect_mask": defect_path.name,
            "roughness": roughness_path.name,
        },
    }
    result_path = frame_dir / f"suction_surface_result_{args.target}.json"
    result_path.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"Analyzed frame: {frame_dir}")
    print(
        f"Physical footprint: radius={analysis.cup_radius_px:.1f}px, "
        f"spacing={analysis.cup_spacing_px:.1f}px at Z={analysis.reference_depth_m:.3f}m"
    )
    if analysis.candidates:
        best = analysis.candidates[0]
        print(
            f"Best candidate: center={best.center_px}, angle={best.angle_deg:.1f}deg, "
            f"score={best.score:.3f}, camera point={best.center_camera_m}"
        )
    else:
        print("No pose passed all conservative surface checks.")
    print(f"Saved overlay: {overlay_path}")
    print(f"Saved result: {result_path}")
    if not args.no_display:
        cv2.namedWindow("WineJar suction surface inspection", cv2.WINDOW_NORMAL)
        cv2.imshow("WineJar suction surface inspection", overlay)
        print("Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0 if analysis.candidates else 2


if __name__ == "__main__":
    raise SystemExit(main())
