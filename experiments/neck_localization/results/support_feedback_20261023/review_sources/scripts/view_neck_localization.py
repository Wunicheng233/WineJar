#!/usr/bin/env python3
"""Open one MuJoCo window with the camera rig, result and actual input frames."""
import os
# This is an onscreen GLFW viewer; headless benchmark EGL remains unchanged.
os.environ.pop("MUJOCO_GL", None)
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neck_localization.inspection_viewer import InspectionViewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, default=ROOT/"experiments/neck_localization/runs/short_neck_heldout_20261022_v1")
    parser.add_argument("--records", type=Path, default=ROOT/"experiments/neck_localization/runs/depth_profile_heldout_20261022_v1/records.json")
    parser.add_argument("--case", default="broad_reverse_tilt")
    parser.add_argument("--camera-set", choices=["cam_a", "cam_a,cam_b", "all"], default="all")
    parser.add_argument("--snapshot", type=Path, help="Save one inspection screenshot, then exit")
    parser.add_argument("--overlay", action="store_true", help="Display thin result lines on the camera images")
    parser.add_argument("--ground-truth", action="store_true", help="Explicit evaluation-only truth comparison")
    parser.add_argument("--large-neck", action="store_true", help="Start with the enlarged, independently rotatable neck view (N toggles)")
    parser.add_argument("--point-cloud", action="store_true", help="Show observation-derived foreground point clouds (P toggles)")
    parser.add_argument("--weights", action="store_true", help="Show saved actual point weights; implies local point cloud (W toggles)")
    parser.add_argument("--method", default="rim_guided_profile", help="Saved method name; M cycles available methods")
    args = parser.parse_args()
    viewer = InspectionViewer(args.scenes, args.records, args.case, args.camera_set,
                              visible=args.snapshot is None, overlay=args.overlay, ground_truth=args.ground_truth,
                              large_neck=args.large_neck, point_cloud=args.point_cloud, method=args.method)
    if args.weights:
        viewer.cloud_weights = viewer.cloud_local = viewer.point_cloud = True
    try:
        if args.snapshot:
            viewer.snapshot(args.snapshot)
            print(str(args.snapshot.resolve()))
        else:
            viewer.run()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
