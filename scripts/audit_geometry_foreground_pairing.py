#!/usr/bin/env python3
"""Read-only audit of completed paired SAM2 foreground geometry results.

Reads saved evaluations, hashes and metrics only, never target/profile files
for inference or parameter selection. Writes a separate review artifact.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'experiments/neck_localization/results'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def key(row):
    return tuple(row[k] for k in ('layout', 'case', 'condition', 'backend'))


def read_stage(path):
    validation, provenance = load(path/'validation.json'), load(path/'provenance.json')
    if not validation['passed'] or validation['input_or_backend_errors']:
        raise ValueError('stage_validation_or_inference_failure:'+str(path))
    rows = json.loads(gzip.decompress((path/'records.json.gz').read_bytes()))
    if len({key(r) for r in rows}) != len(rows):
        raise ValueError('duplicate_record_key')
    return rows, provenance, validation


def contained(row):
    return bool((row.get('band_evaluation') or {}).get('full_annotated_neck', {}).get('contained'))


def summarize(rows):
    def passes(row, threshold):
        return row.get('center_mm') is not None and row['center_mm'] <= threshold
    solved = [r for r in rows if r.get('center_mm') is not None]
    return dict(n=len(rows), solved=len(solved), native_accepted=sum(r['accepted'] for r in rows),
        candidate_center_3mm=sum(passes(r, 3) for r in rows),
        candidate_center_4mm=sum(passes(r, 4) for r in rows),
        accepted_center_3mm=sum(r['accepted'] and passes(r, 3) for r in rows),
        accepted_center_4mm=sum(r['accepted'] and passes(r, 4) for r in rows),
        accepted_wrong_center_4mm=sum(r['accepted'] and not passes(r, 4) for r in rows),
        accepted_pose_4mm_2deg=sum(r['accepted'] and passes(r, 4) and r['normal_deg'] <= 2 for r in rows),
        accepted_full_circle_research_thresholds=sum(r['accepted'] and r['truth_pass'] for r in rows),
        candidate_axial_band_contained=sum(contained(r) for r in rows),
        accepted_axial_band_contained=sum(r['accepted'] and contained(r) for r in rows),
        accepted_common_band_axially_contained=sum(r['accepted_with_common_band_gate'] and contained(r) for r in rows),
        maximum_candidate_center_mm=max((r['center_mm'] for r in solved), default=None),
        maximum_candidate_axis_deg=max((r['normal_deg'] for r in solved), default=None),
        reasons={s:sum(s in r['reasons'] for r in rows) for s in sorted({s for r in rows for s in r['reasons']})})


def compact(row):
    return dict(**{k:row.get(k) for k in ['accepted', 'center_mm', 'normal_deg', 'diameter_mm',
        'reasons', 'observed_band_usable', 'accepted_with_common_band_gate']},
        annotated_axial_band_contained=contained(row),
        minimum_annotated_axial_band_margin_mm=(row.get('band_evaluation') or {}).get('full_annotated_neck',{}).get('minimum_margin_mm'))


def run(output):
    stages, reports, summaries, case_changes = {}, [], [], []
    source_reference = None
    for dataset, expected in [('audit', 48), ('confirmation', 32)]:
        for frontend in ['fs', 'simsense']:
            path = RESULTS/f'geometry_{frontend}_sam2_{dataset}_20261027_v1'
            rows, provenance, validation = read_stage(path)
            if len(rows) != expected:
                raise ValueError('wrong_record_count:'+str(path))
            source_hashes = provenance['source_sha256']
            cache_path = Path(provenance['foreground_cache'])
            if digest(cache_path) != provenance['input_sha256'][str(cache_path)]:
                raise ValueError('foreground_manifest_changed')
            cache = load(cache_path)
            for entry in cache['entries']:
                if entry['status'] != 'ok':
                    raise ValueError('failed_foreground_cache_entry')
                if (digest(entry['mask_path']) != entry['mask_sha256'] or
                        digest(entry['rgb_path']) != entry['rgb_sha256']):
                    raise ValueError('foreground_mask_or_RGB_changed')
            changed_sources = [p for p,h in source_hashes.items() if digest(ROOT/p) != h]
            if changed_sources:
                raise ValueError('source_changed:'+str(changed_sources))
            if source_reference is None:
                source_reference = source_hashes
            if source_reference != source_hashes:
                raise ValueError('paired_stages_have_different_sources')
            for row in rows:
                if row['evaluation_role'] != 'paired_fixed_foreground_adaptation_not_new_holdout':
                    raise ValueError('misleading_evaluation_role')
                if row['split'] != 'previously_scored_objects_fixed_foreground_adaptation':
                    raise ValueError('misleading_split')
                for threshold in [3, 4]:
                    actual = row.get('center_mm') is not None and row['center_mm'] <= threshold
                    if actual != row[f'center_pass_{threshold}mm']:
                        raise ValueError('saved_primary_center_metric_mismatch')
            old_path = RESULTS/f'geometry_{frontend}_{dataset}_20261027_v1'
            old_rows, old_provenance, _ = read_stage(old_path)
            selected_layouts = {r['layout'] for r in rows}
            old_rows = [r for r in old_rows if r['layout'] in selected_layouts]
            old_map = {key(r):r for r in old_rows}
            if set(old_map) != {key(r) for r in rows}:
                raise ValueError('v1_SAM2_unpaired_case_set')
            for condition, backend in sorted({(r['condition'], r['backend']) for r in rows}):
                chosen = [r for r in rows if (r['condition'],r['backend']) == (condition, backend)]
                before = [old_map[key(r)] for r in chosen]
                summaries.append(dict(dataset=dataset, frontend=frontend,
                    condition=condition, backend=backend, v1=summarize(before), SAM2=summarize(chosen)))
            if dataset == 'confirmation':
                for row in rows:
                    if row['case'].endswith(('_004','_005')):
                        case_changes.append(dict(frontend=frontend, case=row['case'],
                            condition=row['condition'], backend=row['backend'],
                            v1=compact(old_map[key(row)]), SAM2=compact(row)))
            reports.append(dict(stage=str(path), record_count=len(rows), validation=validation,
                all_record_roles_verified=True, current_sources_match_frozen=True,
                provenance_sha256=digest(path/'provenance.json'),
                validation_sha256=digest(path/'validation.json'),
                records_sha256=digest(path/'records.json.gz'),
                foreground_cache=provenance['foreground_cache']))
            stages[(dataset,frontend)] = (rows,provenance)
        fs, fs_provenance = stages[(dataset,'fs')]
        sim, sim_provenance = stages[(dataset,'simsense')]
        if fs_provenance['foreground_cache'] != sim_provenance['foreground_cache']:
            raise ValueError('paired_frontends_use_different_foreground_cache')
        sim_map = {key(r):r for r in sim}
        for row in fs:
            a = row['diagnostics']['foreground']
            b = sim_map[key(row)]['diagnostics']['foreground']
            if a != b:
                raise ValueError('FS_SimSense_actual_foreground_differs')
    result = dict(schema=1, passed=True, role='independent_read_only_saved_result_review',
        no_inference=True, no_training=True, no_parameter_selection=True, no_target_profile_read=True,
        interpretation='Primary 3/4 mm center capability, native acceptance, 2 degree axis analysis, '
            '2 mm diameter research threshold and annotated axial band containment are distinct.',
        evaluation_context='SAM2 protocol fixed using development images; both prior 12 and prior 8 '
            'are previously scored objects, this paired adaptation is not another independent test.',
        verified_same_mask_for_each_FS_SimSense_pair=True,
        all_four_stages_same_frozen_source_hashes=True,
        reference_source_sha256=source_reference, stages=reports,
        summaries=summaries, confirmation_004_005_changes=case_changes,
        review_script_sha256=digest(Path(__file__)))
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for row in summaries:
        print(row['dataset'],row['frontend'],row['condition'],row['backend'],
              'center4',row['v1']['candidate_center_4mm'],'->',row['SAM2']['candidate_center_4mm'],
              'accepted',row['v1']['native_accepted'],'->',row['SAM2']['native_accepted'],
              'axial',row['v1']['candidate_axial_band_contained'],'->',row['SAM2']['candidate_axial_band_contained'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
