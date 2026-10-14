#!/usr/bin/env python3
"""Uncalibrated RGB preview; no depth inference or robot connection."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from xarm_sdk_control.realsense_capture import camera_info, choose_device
from xarm_sdk_control.run_output import new_timestamped_run_dir

CAMERAS = {
    "overhead": "013222073103",
    "side_a": "142122070566",
    "side_b": "011422070041",
}
WINDOW = "WineJar live RGB | 0: all | 1/2/3: single | S: snapshot | Q: close"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "data/multiview_neck/live_previews"
    )
    args = parser.parse_args()
    context = rs.context()
    pipelines, images, received, counters, devices = {}, {}, {}, {}, {}
    selected = 0
    saved_initial = False
    try:
        for cid, serial in CAMERAS.items():
            devices[cid] = camera_info(choose_device(context, serial))
        for cid, serial in CAMERAS.items():
            pipeline = rs.pipeline(context)
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            pipelines[cid] = pipeline
            images[cid] = np.zeros((480, 640, 3), np.uint8)
            received[cid], counters[cid] = 0.0, 0
            print(f"Opened {cid}: {serial}", flush=True)

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1536, 420)
        print("Live preview running; no calibration or robot commands", flush=True)
        while True:
            for cid, pipeline in pipelines.items():
                frames = pipeline.poll_for_frames()
                frame = frames.get_color_frame() if frames else None
                if frame:
                    images[cid] = np.asanyarray(frame.get_data()).copy()
                    received[cid] = time.monotonic()
                    counters[cid] += 1
            now = time.monotonic()
            panels = []
            for index, (cid, serial) in enumerate(CAMERAS.items(), 1):
                panel = cv2.copyMakeBorder(
                    images[cid], 32, 0, 0, 0, cv2.BORDER_CONSTANT
                )
                stale = now - received[cid] > 2
                label = f"{index} {cid} | {serial}" + (
                    " | NO LIVE FRAME" if stale else ""
                )
                cv2.putText(
                    panel,
                    label,
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (50, 80, 255) if stale else (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
                panels.append(panel)
            overview = np.hstack(panels)
            cv2.imshow(WINDOW, overview if selected == 0 else panels[selected - 1])
            key = cv2.waitKey(5) & 0xFF
            if (
                key in (27, ord("q"), ord("Q"))
                or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
            ):
                break
            if key in map(ord, "0123"):
                selected = key - ord("0")
                cv2.resizeWindow(
                    WINDOW,
                    1536 if selected == 0 else 960,
                    420 if selected == 0 else 768,
                )
            ready = all(
                counters[cid] >= 30 and now - received[cid] < 1 for cid in CAMERAS
            )
            if (not saved_initial and ready) or key in (ord("s"), ord("S")):
                if not ready:
                    print(
                        "Snapshot skipped: waiting for fresh frames from all cameras",
                        flush=True,
                    )
                    continue
                run = new_timestamped_run_dir(args.output_root.expanduser().resolve())
                for name, image in {**images, "overview": overview}.items():
                    if not cv2.imwrite(str(run / f"{name}.png"), image):
                        raise OSError(f"Could not save {name}")
                (run / "preview.json").write_text(
                    json.dumps(
                        {
                            "type": "uncalibrated_rgb_preview",
                            "cameras": devices,
                            "note": "Latest RGB frames, not synchronized or valid for metric measurement",
                            "frame_age_ms": {
                                cid: (now - received[cid]) * 1000 for cid in CAMERAS
                            },
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(f"Snapshot saved: {run}", flush=True)
                saved_initial = True
            time.sleep(0.005)
    except (RuntimeError, OSError, cv2.error) as error:
        print(f"ERROR: {error}", flush=True)
        return 2
    finally:
        for pipeline in pipelines.values():
            pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
