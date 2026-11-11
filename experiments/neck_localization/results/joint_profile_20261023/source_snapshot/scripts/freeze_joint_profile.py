#!/usr/bin/env python3
"""Freeze source/settings after development, before rendering/scoring new test."""
from dataclasses import asdict
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neck_localization.joint_profile import JointSettings
from scripts.run_joint_profile import source_hashes, METHODS, CAMERA_SETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-config", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = dict(sources=source_hashes(), settings=asdict(JointSettings()),
                test_config_sha256=sha256(args.test_config.read_bytes()).hexdigest(),
                development_records_sha256=sha256((args.development/"records.json").read_bytes()).hexdigest(),
                methods=METHODS, camera_sets=CAMERA_SETS,
                policy="No estimator/settings changes after new test is evaluated; inspected old holdout is development only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2)+"\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
