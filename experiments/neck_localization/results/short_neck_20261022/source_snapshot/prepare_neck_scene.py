#!/usr/bin/env python3
"""Export and preview parameterized jar scenes; no localization or hardware calls."""
import os
os.environ.setdefault("MUJOCO_GL","egl")
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from neck_localization.scene_preparation import prepare_scene_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"experiments/neck_localization/configs/preparation_v2.json")
    parser.add_argument("--output",type=Path,required=True,help="New output directory; refuses overwrite")
    args=parser.parse_args()
    manifest=prepare_scene_bundle(json.loads(args.config.read_text()),args.output)
    print(json.dumps({"output":str(args.output.resolve()),"stage":manifest["stage"],"cases":len(manifest["cases"])}))


if __name__=="__main__":
    main()
