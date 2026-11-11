#!/usr/bin/env python3
"""Select portable research evidence for Git without deleting local data.

Produces an explicit NUL-delimited result pathspec and an auditable catalog.
Does not stage, commit, push, run estimators, or modify historical results.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'experiments/neck_localization/results'
EXPLICIT={
    'random_geometry_audit_20261023/records.json.gz',
    'support_feedback_20261023/report/test_records.json.gz',
    'short_neck_20261022/heldout/records.json.gz',
    'joint_profile_20261023/report/test_records.json.gz',
}


def select(path):
    if path.is_symlink():return False,'local_symlink'
    rel=path.relative_to(RESULTS);parts=rel.parts
    if 'scenes' in parts:return False,'full_scene_inputs_kept_locally'
    if 'replay' in parts[0] or 'initial' in parts[0]:return False,'duplicate_or_preliminary_bundle'
    size=path.stat().st_size
    if str(rel) in EXPLICIT:return True,'full_primary_camera_comparison_records'
    if path.suffix in {'.png','.svg','.csv','.tsv','.md','.py','.sh','.yml','.yaml','.txt'}:
        return True,'report_visual_table_or_frozen_source'
    if path.suffix=='.json' and size<=1024**2 and not path.name.endswith('.partial.json'):
        return True,'compact_json_evidence'
    if path.name.endswith('.json.gz') and size<=512*1024:return True,'compact_compressed_records'
    return False,'large_pointwise_input_or_redundant_journal'


def run(asset_root,pathspec):
    selected=[];omitted={};catalog={}
    for p in sorted(RESULTS.rglob('*')):
        if not p.is_file() and not p.is_symlink():continue
        if p.name=='.gitignore':continue
        use,reason=select(p)
        if not use:
            omitted[reason]=omitted.get(reason,0)+1;continue
        relative=p.relative_to(ROOT);selected.append(relative)
        study=p.relative_to(RESULTS).parts[0]
        row=catalog.setdefault(study,dict(study=study,published_files=0,published_bytes=0,figures=0,
            tables=0,summary_json=[],reports=[]))
        row['published_files']+=1;row['published_bytes']+=p.stat().st_size
        row['figures']+=p.suffix in {'.png','.svg'};row['tables']+=p.suffix in {'.csv','.tsv'}
        if p.name in {'summary.json','confirmation_summary.json','fit_summary.json','method_summaries.json'}:
            row['summary_json'].append(str(relative))
        if p.suffix=='.md':row['reports'].append(str(relative))
    data=asset_root/'data';data.mkdir(parents=True,exist_ok=True)
    catalog_path=data/'result_catalog.csv'
    with catalog_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['study','published_files','published_bytes','figures','tables','summary_json','reports'])
        writer.writeheader()
        writer.writerows([{**r,'summary_json':';'.join(r['summary_json']),'reports':';'.join(r['reports'])} for r in catalog.values()])
    manifest=dict(date='2026-11-11',policy='Selected evidence only; local raw data, symlinks and duplicates are not removed.',
        selected_result_files=[str(p) for p in selected],
        selected_result_bytes=sum((ROOT/p).stat().st_size for p in selected),
        omitted_file_counts=omitted,full_primary_camera_records=sorted(EXPLICIT),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        historical_absolute_paths_are_provenance_not_portable_runtime_configuration=True,
        default_portable_runtime='scripts/view_camera_count_comparison.py')
    (asset_root/'publication_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    pathspec.parent.mkdir(parents=True,exist_ok=True)
    pathspec.write_bytes(b'\0'.join(str(p).encode() for p in selected)+b'\0')
    print('Selected',len(selected),'result files;',round(manifest['selected_result_bytes']/1048576,1),'MiB;',len(catalog),'study bundles')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=ROOT/'experiments/presentation_assets_20261111')
    p.add_argument('--pathspec',type=Path,default=ROOT/'tmp/publication_20261111/results.pathspec')
    a=p.parse_args();run(a.assets.resolve(),a.pathspec.resolve())
