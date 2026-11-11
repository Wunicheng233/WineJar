#!/usr/bin/env python3
"""Lock eight new same-family nominal jars without rendering or inference.

Uses the unchanged nominal sampler and frozen close-camera layout. This does
not create a new shape family, model a factory distribution, run a simulator,
load target labels, select cases by outcomes, or change method parameters.
The emitted renderer freeze is consumed by render_structural_inputs.py.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neck_localization.active_stereo import StereoSettings
from neck_localization.jar_model import JarSpec
from neck_localization.random_audit import random_audit_config
from scripts.run_active_stereo_stage1 import sources

DEFAULT_TEMPLATE = ROOT/'experiments/neck_localization/configs/structural_close_test_20261025_v1.json'
DEFAULT_OUTPUT = ROOT/'experiments/neck_localization/configs/geometry_confirmation_close_20261027_v1.json'
DEFAULT_RENDER = ROOT/'experiments/neck_localization/runs/geometry_confirmation_close_20261027_v1'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n'


def prepare(template_path=DEFAULT_TEMPLATE, output=DEFAULT_OUTPUT, *, seed=2026102707, render_output=DEFAULT_RENDER):
    template_path, output, render_output = map(lambda p: Path(p).resolve(), (template_path, output, render_output))
    freeze_path = output.with_name(output.stem+'_render_freeze.json')
    protocol_path = output.with_name(output.stem+'_protocol.json')
    if any(path.exists() for path in (output, freeze_path, protocol_path)):
        raise FileExistsError('All three preparation outputs must be new')
    template = json.loads(template_path.read_text())
    config = random_audit_config(template, seed=seed, nominal=8, stress=0)
    config['dataset_split'] = 'locked_new_same_family_nominal_confirmation'
    config['generation_protocol'] = dict(
        source='unchanged neck_localization.random_audit.random_audit_config nominal branch',
        seed=seed, nominal=8, stress=0, filtering_by_estimator_outcome=False,
        neck_length_m=[.008, .014], minimum_confirmed_physical_neck_m=.006,
        camera_layout='unchanged existing close layout',
        material='unchanged mottled_gray procedural family; new seeded texture realizations',
        family='same circular cross-section, flat/shallow neck family; NOT cross-family generalization')
    config['audit_assumptions'].update(
        cameras='Unchanged close layout and D435 preset; true K/T, same synthetic active stereo measurement settings',
        noise='No newly injected noise or layout search; render keeps prior default active stereo noise for reference',
        minimum_neck_length='existing nominal range 8–14 mm; no clipping, rejection, or outcome-based resampling',
        interpretation='eight new instances within the SAME circular-section family; not independent factory validation')
    previous = {json.dumps(case['jar'], sort_keys=True) for case in template['cases']}
    for index, case in enumerate(config['cases']):
        case['name'] = f'confirmation_nominal_{index:03d}'
        assert case['group'] == 'nominal' and case['material'] == 'mottled_gray'
        assert .008 <= case['jar']['neck_length_m'] <= .014
        assert json.dumps(case['jar'], sort_keys=True) not in previous
        JarSpec(**case['jar'])  # Parameter validity only; no mesh/truth generated.
    assert config['scene'] == template['scene']
    assert len(config['cases']) == 8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encode(config))
    frozen_sources = sources()
    freeze = dict(sources=frozen_sources, test_config_sha256=digest(output),
        settings=asdict(StereoSettings()), no_rendering_yet=True,
        scope='renderer-only source/settings snapshot; does not freeze/evaluate estimators',
        preparation_script_sha256=digest(__file__),
        geometry_sampler_sha256=digest(ROOT/'neck_localization/random_audit.py'))
    freeze_path.write_text(encode(freeze))
    command = [sys.executable, str(ROOT/'scripts/render_structural_inputs.py'),
               '--config', str(output), '--freeze', str(freeze_path), '--output', str(render_output)]
    protocol = dict(schema=1, stage='configuration_locked_not_rendered',
        config=str(output), config_sha256=digest(output), renderer_freeze=str(freeze_path),
        renderer_freeze_sha256=digest(freeze_path),
        template=str(template_path), template_sha256=digest(template_path),
        seed=seed, cases=[case['name'] for case in config['cases']],
        shape_family='same circular cross-section family', nominal_count=8, stress_count=0,
        no_truth_read=True, no_hardware=True, no_rendering=True, no_inference=True,
        no_result_based_case_filtering=True, no_method_tuning=True,
        same_scene=config['scene'] == template['scene'], same_sensor_settings=True,
        scalar_ranges=dict(height_m=[.57,.65], body_radius_m=[.165,.195], neck_radius_m=[.048,.057],
            neck_length_m=[.008,.014], tilt_magnitude_deg=[0.,7.], neck_offset_component_m=[-.005,.005],
            lip_above_neck_radius_m=[.008,.014], lip_height_m=[.005,.007], shallow_relief_m=[.0012,.0024]),
        render_only_command=shlex.join(command),
        rendering='Existing render_structural_inputs.py exports geometry and all four saved depth references but invokes no estimator',
        expected_inputs=dict(cases=8, cameras_per_case=3, clean_IR_pairs=24, RGB_frames=24),
        estimated_render_seconds=[30,90], estimated_output_megabytes=200,
        estimate_basis='prior 12-case close render took approximately 38 seconds and occupied 298 MiB; not a timing guarantee')
    protocol_path.write_text(encode(protocol))
    print(encode(dict(config=str(output), freeze=str(freeze_path), protocol=str(protocol_path),
                      render_command=shlex.join(command))).strip())
    return config, freeze, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seed', type=int, default=2026102707)
    parser.add_argument('--render-output', type=Path, default=DEFAULT_RENDER)
    args = parser.parse_args()
    prepare(args.template, args.output, seed=args.seed, render_output=args.render_output)


if __name__ == '__main__':
    main()
