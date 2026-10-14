#!/usr/bin/env python3
"""Live wrist RGB-D preview and snapshots; no robot connection or motion."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from xarm_sdk_control.realsense_capture import RealSenseCaptureSession, make_depth_preview

WINDOW = 'WineJar wrist | S save | D depth | G grid | Q/Esc quit'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', default='011422070172')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'data/wrist_live_preview')
    parser.add_argument('--duration', type=float, default=0, help='Auto-close after N seconds; 0 means until Q')
    parser.add_argument('--save-first', action='store_true', help='Save first displayed RGB-D frame')
    args = parser.parse_args()
    if args.duration < 0:
        parser.error('--duration must be non-negative')
    if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        print('ERROR: No desktop display. Run this script in a terminal on the camera computer.')
        return 2
    depth_visible, grid, first = False, True, True
    try:
        with RealSenseCaptureSession(serial=args.serial) as camera:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 960, 768)
            started = previous = time.monotonic()
            fps = 0.0
            print('Preview only; robot not connected. Click window for keyboard controls.', flush=True)
            while True:
                # Read and save the SAME aligned frame using the existing capture schema.
                color, raw, timestamp, intrinsics = camera._read_frame_arrays()
                now = time.monotonic()
                instantaneous = 1 / max(now - previous, 1e-6)
                fps = instantaneous if first else 0.9 * fps + 0.1 * instantaneous
                previous = now
                scale = camera.profile.get_device().first_depth_sensor().get_depth_scale()
                panel = color.copy()
                if grid:
                    h, w = panel.shape[:2]
                    for n in [1, 2]:
                        cv2.line(panel, (w*n//3, 0), (w*n//3, h-1), (0, 180, 0), 1)
                        cv2.line(panel, (0, h*n//3), (w-1, h*n//3), (0, 180, 0), 1)
                    cv2.rectangle(panel, (15, 15), (w-16, h-16), (0, 180, 255), 1)
                if depth_visible:
                    panel = np.hstack([panel, make_depth_preview(raw, scale)])
                panel = cv2.copyMakeBorder(panel, 32, 0, 0, 0, cv2.BORDER_CONSTANT)
                cv2.putText(panel, f'{args.serial} | {fps:.1f} fps | RGB-D 640x480 | NO ROBOT CONTROL', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW, panel)
                key = cv2.waitKey(1) & 0xff
                if (first and args.save_first) or key in (ord('s'), ord('S')):
                    saved = camera._save_frame(args.output_root, color, raw, timestamp, intrinsics)
                    print(f'SNAPSHOT: {saved}', flush=True)
                first = False
                if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord('d'), ord('D')):
                    depth_visible = not depth_visible
                    cv2.resizeWindow(WINDOW, 1280 if depth_visible else 960, 512 if depth_visible else 768)
                if key in (ord('g'), ord('G')):
                    grid = not grid
                if args.duration and now - started >= args.duration:
                    break
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError, cv2.error) as exc:
        print(f'ERROR: {exc}\nIf camera is busy, close other camera capture/preview programs.', flush=True)
        return 2
    finally:
        cv2.destroyAllWindows()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
