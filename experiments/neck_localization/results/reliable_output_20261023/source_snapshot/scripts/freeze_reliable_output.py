#!/usr/bin/env python3
"""Freeze success-rate pipeline after development and before unseen evaluation."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from neck_localization.reliable_output import ReliabilitySettings
from neck_localization.scene_preparation import write_json
from scripts.run_reliable_output import source_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development',type=Path,required=True)
    p.add_argument('--test-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    provenance=json.loads((args.development/'provenance.json').read_text())
    assert provenance['sources']==source_hashes()
    settings=json.loads(json.dumps(asdict(ReliabilitySettings())))
    assert provenance['settings']==settings
    planpath=ROOT/'experiments/neck_localization/results/reliable_output_20261023/plan.json'
    plan=json.loads(planpath.read_text())
    testhash=sha256(args.test_config.read_bytes()).hexdigest()
    assert testhash==plan['test_config_sha256']
    write_json(args.output,dict(sources=source_hashes(),settings=settings,test_config_sha256=testhash,
        development_records_sha256=sha256((args.development/'records.json.gz').read_bytes()).hexdigest(),
        plan_sha256=sha256(planpath.read_bytes()).hexdigest(),policy='No tuning after test; no real-system claims'))
    print(args.output)


if __name__=='__main__':main()
