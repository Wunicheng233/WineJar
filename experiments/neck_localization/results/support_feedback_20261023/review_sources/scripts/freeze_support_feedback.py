#!/usr/bin/env python3
"""Freeze completed development before new support-feedback test evaluation."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neck_localization.support_feedback import FeedbackSettings
from neck_localization.joint_profile import JointSettings
from neck_localization.scene_preparation import write_json
from scripts.run_support_feedback import source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--test-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    provenance = json.loads((args.development/'provenance.json').read_text())
    assert provenance['sources'] == source_hashes(), 'Development source differs'
    assert provenance['feedback_settings'] == asdict(FeedbackSettings())
    assert provenance['joint_settings'] == asdict(JointSettings())
    plan_path = ROOT/'experiments/neck_localization/results/support_feedback_20261023/plan.json'
    plan = json.loads(plan_path.read_text())
    config_hash = sha256(args.test_config.read_bytes()).hexdigest()
    assert config_hash == plan['test_config_sha256'], 'Test specifications changed after plan'
    write_json(args.output, dict(sources=source_hashes(), feedback_settings=asdict(FeedbackSettings()),
        joint_settings=asdict(JointSettings()), test_config_sha256=config_hash,
        development_records_sha256=sha256((args.development/'records.json').read_bytes()).hexdigest(),
        plan_sha256=sha256(plan_path.read_bytes()).hexdigest(),
        policy='No source/settings changes after test; previously inspected 14 scenes are development only'))
    print(args.output)


if __name__ == '__main__': main()
