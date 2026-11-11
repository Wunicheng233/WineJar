#!/usr/bin/env python3
"""Audit paired seeds, unchanged baselines, new viewer routes and exact replay."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get('MUJOCO_GL') == 'egl':
    del os.environ['MUJOCO_GL']
import numpy as np
from scripts.run_cross_station_study import digest, read, write
from scripts.verify_three_station_tying import run as verify_viewer
from neck_localization.tying_viewer import TyingViewer


def run(bundle):
    protocol = read(bundle/'protocol.json')
    source = ROOT/protocol['source_bundle']
    provenance = read(bundle/'provenance.json')
    for p, h in provenance['source_sha256'].items():
        assert digest(p) == h
    original = read(source/'provenance.json')['input_and_source_sha256']
    for p, h in original.items():
        assert digest(p) == h
    # Ordinary and outlier cases share exactly the same latent calibration;
    # only one supplied survey measurement has been corrupted.
    latent = read(bundle/'evaluation_only_session_latents.json')
    by_key = {(r['scenario'], r['session']): r for r in latent}
    for s in range(protocol['calibration_sessions']):
        a, b = [by_key[(name, s)] for name in ('combined', 'combined_survey_outlier')]
        assert {k:v for k,v in a.items() if k != 'scenario'} == {k:v for k,v in b.items() if k != 'scenario'}
    with np.load(bundle/'conditional_samples.npz') as data:
        axes = read(bundle/'sample_axes.json')
        ordinary, outlier = [axes['scenarios'].index(n) for n in ('combined', 'combined_survey_outlier')]
        for key in data.files:
            np.testing.assert_array_equal(data[key][ordinary, :2], data[key][outlier, :2])
    # New integer-normal input compatibility fix changed no saved predictions,
    # samples, or tool certificates relative to the paired v2 development run.
    development = bundle.parent/'cross_station_20261028_v2'
    replay = None
    if development.exists() and bundle.name != development.name:
        with np.load(bundle/'conditional_samples.npz') as a, np.load(development/'conditional_samples.npz') as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
        assert read(bundle/'band_controls.json') == read(development/'band_controls.json')
        assert read(bundle/'tool_scan.json') == read(development/'tool_scan.json')
        replay = 'bitwise_numeric_arrays_and_json_results_equal_to_paired_v2'
    # Reuse the original actual GLFW/FK audit. No new framework is introduced.
    write(bundle/'viewer/provenance.json', dict(input_and_source_sha256=original | provenance['source_sha256']))
    verify_viewer(bundle/'viewer', bundle/'interaction')
    datasets = read(bundle/'viewer/datasets.json')
    d = next(x for x in datasets if x['condition'] == 'new8/fusion/clean')
    viewer = TyingViewer(d['scenes'], d['records'], 'confirmation_nominal_003', method=d['method'], datasets=datasets,
        station_config=read(bundle/'viewer/config.json'), ground_truth=True, overlay=True, band=True, visible=False)
    try:
        viewer._activate('stage:close')
        viewer.snapshot(bundle/'interaction/fusion_geometry_pass.png')
        viewer.case_index = viewer.cases.index('confirmation_nominal_001')
        viewer._load_case()
        viewer._activate('stage:close')
        viewer.snapshot(bundle/'interaction/fusion_retained_band_failure.png')
    finally:
        viewer.close()
    for p, h in original.items():
        assert digest(p) == h
    write(bundle/'verification.json', dict(passed=True, calibration_outlier_inputs_paired=True,
        nominal_strategies_identical_in_paired_outlier_control=True, previous_465_inputs_unchanged=True,
        numeric_replay=replay, final_fastening_not_validated=True,
        no_runtime_carrier_or_body_observation=True, verification_script_sha256=digest(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    run(args.bundle.resolve())
