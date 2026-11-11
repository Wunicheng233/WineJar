#!/usr/bin/env python3
"""Read-only paired band audit: separate frontend and geometry effects."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from neck_localization.band_geometry import evaluate_band
from scripts.run_three_station_tying import COHORTS, read, write, digest


def run(bundle, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    entries = [e for e in read(bundle/'filter_manifest.json')['entries'] if e['retained']]
    allowed = {(e['cohort'], e['case']): e for e in entries}
    records, hashes = [], {}
    for cohort, (fs, ss, _, _) in COHORTS.items():
        for frontend, name in [('fs', fs), ('ss', ss)]:
            path = ROOT/'experiments/neck_localization/results'/f'task_surface_{name}_20261028_v1/records.json.gz'
            hashes[str(path)] = digest(path)
            for row in read(path):
                if row['backend'] not in ('visible_profile', 'standard_sor') or (cohort, row['case']) not in allowed:
                    continue
                target_path = Path(allowed[(cohort, row['case'])]['source_model']).parent/'evaluation_only/target.json'
                hashes[str(target_path)] = digest(target_path)
                truth = read(target_path)
                score = evaluate_band(row['band'], truth, truth['search_interval_s_m'])
                records.append(dict(cohort=cohort, frontend=frontend, backend=row['backend'],
                    condition=row['condition'], case=row['case'], accepted=row['accepted'],
                    center_mm=row['center_mm'], normal_deg=row['normal_deg'], diameter_mm=row['diameter_mm'],
                    physical_band=score, original_near_min_proxy=row['band_evaluation']))
    summary = []
    for frontend in ('fs', 'ss'):
        for backend in ('visible_profile', 'standard_sor'):
            for condition in ('clean', 'shared_noise'):
                group = [r for r in records if (r['frontend'], r['backend'], r['condition']) == (frontend, backend, condition)]
                summary.append(dict(frontend=frontend, backend=backend, condition=condition, total=len(group),
                    accepted=sum(r['accepted'] for r in group), physical_band_contained=sum(r['physical_band']['contained'] for r in group),
                    center_false_rejects=sum(not r['accepted'] and r['center_mm'] <= 4 for r in group),
                    center_false_accepts=sum(r['accepted'] and r['center_mm'] > 4 for r in group),
                    center_mm=dict(zip(('median', 'p95', 'max'), np.percentile([r['center_mm'] for r in group], [50, 95, 100]).tolist())),
                    max_normal_deg=max(r['normal_deg'] for r in group), max_diameter_mm=max(r['diameter_mm'] for r in group)))
    write(output/'records.json', records)
    write(output/'summary.json', dict(groups=summary, observations=len(records), distinct_objects=len(entries),
        scope='posthoc_same_saved_cases_no_new_independent_test_no_robot_or_radial_contact_validation'))
    assert all(digest(p) == h for p, h in hashes.items())
    write(output/'provenance.json', dict(input_sha256=hashes, inputs_unchanged=True,
        source_sha256={str(Path(__file__).resolve()): digest(__file__)}))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.viewer.resolve(), args.output.resolve())
