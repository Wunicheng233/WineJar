#!/usr/bin/env python3
"""Summarize saved geometry audits without rerunning or selecting estimates."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'experiments/neck_localization/results'
INPUTS = {
    'opened_fs': ('foundation_stereo', 'opened_12', 'geometry_fs_audit_20261027_v1'),
    'opened_simsense': ('simsense', 'opened_12', 'geometry_simsense_audit_20261027_v1'),
    'ideal_control': ('ideal_native_oracle', 'opened_12', 'geometry_ideal_control_20261027_v1'),
    'confirmation_fs': ('foundation_stereo', 'new_8_first_frozen_evaluation', 'geometry_fs_confirmation_20261027_v1'),
    'confirmation_simsense': ('simsense', 'new_8_first_frozen_evaluation', 'geometry_simsense_confirmation_20261027_v1'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    """Every row, including rejection/error, remains in the denominator."""
    result = {'n': len(rows), 'accepted': sum(r['accepted'] for r in rows),
              'input_or_backend_errors': sum(r['status'] != 'ok' for r in rows)}
    for name in ['center_pass_3mm', 'center_pass_4mm', 'axis_pass_2deg',
                 'pose_pass_3mm_2deg', 'pose_pass_4mm_2deg']:
        result['candidate_' + name] = sum(r[name] for r in rows)
        result['output_' + name] = sum(r['accepted'] and r[name] for r in rows)
    result['full_circle_output'] = sum(r['accepted'] and r['truth_pass'] for r in rows)
    result['common_band_accepted'] = sum(r['accepted_with_common_band_gate'] for r in rows)
    for label in ['candidate', 'output']:
        result[label + '_axial_band_contained'] = sum(
            bool((r.get('band_evaluation') or {}).get('full_annotated_neck', {}).get('contained'))
            and (label == 'candidate' or r['accepted']) for r in rows)
    for name in ['center_mm', 'normal_deg', 'diameter_mm']:
        values = [r[name] for r in rows if r[name] is not None]
        result[name + '_max'] = max(values) if values else None
    result['rejections'] = [{k: r[k] for k in ['case', 'reasons']} for r in rows if not r['accepted']]
    result['outside_primary_center_4mm'] = [{k: r[k] for k in ['case', 'center_mm', 'accepted']}
                                           for r in rows if not r['center_pass_4mm']]
    return result


def run(output, extra=()):
    output.mkdir(parents=True, exist_ok=False)
    inputs = dict(INPUTS)
    for frontend, scope, directory in extra:
        inputs[frontend + '_' + scope] = (frontend, scope, directory)
    tables, sources, paired = [], {}, []
    for label, (frontend, scope, directory) in inputs.items():
        directory = Path(directory)
        if not directory.is_absolute():
            directory = RESULTS / directory
        validation = json.loads((directory / 'validation.json').read_text())
        if not validation['passed'] or validation['input_or_backend_errors']:
            raise ValueError(f'invalid_results:{directory}')
        rows = json.loads(gzip.decompress((directory / 'records.json.gz').read_bytes()))
        for name in ['records.json.gz', 'validation.json', 'provenance.json', 'summary.json']:
            path = directory / name
            sources[str(path)] = digest(path)
        for key in sorted({(r['layout'], r['condition'], r['backend']) for r in rows}):
            chosen = [r for r in rows if (r['layout'], r['condition'], r['backend']) == key]
            tables.append(dict(frontend=frontend, scope=scope, layout=key[0],
                               condition=key[1], backend=key[2], **summarize(chosen)))
        for key in sorted({(r['layout'], r['case'], r['condition']) for r in rows}):
            chosen = [r for r in rows if (r['layout'], r['case'], r['condition']) == key]
            by_method = {r['backend']: r for r in chosen}
            if len(by_method) != 2:
                continue
            paired.append(dict(frontend=frontend, scope=scope, layout=key[0], case=key[1],
                condition=key[2], methods={m: {k: r[k] for k in [
                    'accepted', 'center_mm', 'normal_deg', 'diameter_mm', 'truth_pass', 'reasons']}
                    for m, r in by_method.items()}))
    (output / 'summary.json').write_text(json.dumps(tables, indent=2) + '\n')
    (output / 'paired.json').write_text(json.dumps(paired, indent=2) + '\n')
    columns = ['frontend', 'scope', 'layout', 'condition', 'backend', 'n', 'accepted',
               'candidate_center_pass_3mm', 'candidate_center_pass_4mm',
               'output_center_pass_3mm', 'output_center_pass_4mm', 'candidate_axis_pass_2deg',
               'output_pose_pass_4mm_2deg', 'full_circle_output', 'common_band_accepted',
               'candidate_axial_band_contained', 'output_axial_band_contained',
               'center_mm_max', 'normal_deg_max', 'diameter_mm_max']
    with (output / 'summary.tsv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(tables)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    for name in ['scripts/run_geometry_capability_stage.py',
                 'scripts/summarize_geometry_capability.py',
                 'neck_localization/visible_profile_baseline.py',
                 'neck_localization/sor_baseline.py']:
        path = ROOT / name
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        sources[name] = digest(path)
    context = dict(
        no_training=True, foundation_stereo='official pretrained checkpoint; no winejar fine-tuning',
        primary_center_requirement_mm=4, preferred_center_mm=3,
        axis_analysis_deg=2, diameter_analysis_mm=2,
        axis_and_diameter_are_not_factory_acceptance_requirements=True,
        nominal_band_width_mm=3,
        axial_containment='evaluation in annotated axis interval only; not contact, collision, slip, or physical closure success',
        acceptance='native validators retained; observed nominal band gate separately reported; no calibrated accuracy confidence',
        scope='12 previously viewed jars in two layouts, plus 8 frozen-config new circular-section jars in close layout; neither cross-family nor real-device validation',
        confirmation='first v1 scoring after methods/configuration freeze; later mask adaptations are paired diagnostic evaluation of already opened new8',
        role_metadata_correction='run_geometry_capability_stage v1 labels non-oracle rows method_capability_audit; split, manifest, scope and this sidecar distinguish new8 confirmation. Numeric records unchanged.',
        source_sha256=sources)
    (output / 'evaluation_context.json').write_text(json.dumps(context, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), groups=len(tables), paired_cases=len(paired))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extra', nargs=3, action='append', default=[], metavar=('FRONTEND', 'SCOPE', 'DIRECTORY'))
    arguments = parser.parse_args()
    run(arguments.output, arguments.extra)
