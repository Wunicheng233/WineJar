#!/usr/bin/env python3
"""Fixed-input paired validation-domain study; all GT reads after inference."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import gzip
import json
import multiprocessing
from pathlib import Path
import sys
from time import perf_counter
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
from neck_localization.band_geometry import plan_band,evaluate_band
from neck_localization.geometry import Estimate
from neck_localization.masked_geometry_baseline import estimate_masked_sor
from neck_localization.structure_portfolio import PortfolioSettings
from neck_localization.task_surface_validation import estimate_paired_surfaces
from scripts import evaluate_capability_depth as shared
from scripts.run_masked_geometry_capability_stage import load_foreground,pose_fields,aggregate

METHODS=['visible_profile','task_surface','no_rgb_gate','standard_sor']


def sources():
    paths=[ROOT/'neck_localization'/name for name in ['visible_profile_baseline.py','masked_geometry_baseline.py',
        'sor_baseline.py','task_surface_validation.py']]+[Path(__file__),ROOT/'scripts/run_masked_geometry_capability_stage.py']
    return dict(shared.source_hashes(),**{str(p.relative_to(ROOT)):shared.file_hash(p) for p in paths})


def worker(job):
    pairs,predictions,cache,role=job
    cv2.setNumThreads(1);predictions=Path(predictions);tracked={};checks=[];frames=[]
    index=shared.load_prediction_index(predictions);first=pairs[0];pending={};error=None
    started=perf_counter()
    try:
        for pair in sorted(pairs,key=lambda p:p['camera']):
            depth,info=shared.read_prediction(pair,predictions,index,tracked)
            frame,_,registration=shared.prepare_frame(pair,depth,tracked);frames.append(frame)
            checks.append(dict(pair_id=pair['pair_id'],prediction=info,registration=registration))
        if tuple(f.camera.name for f in frames)!=shared.ALL_CAMERAS:raise ValueError('three_cameras_required')
        masks,foreground=load_foreground(cache,pairs,frames,tracked)
        pending=estimate_paired_surfaces(frames,masks,foreground)
        pending['standard_sor']=estimate_masked_sor(frames,masks,foreground)
        for estimate in pending.values():shared._validate_estimate(estimate)
    except Exception as exception:
        error=repr(exception);checks.append(dict(error=error,traceback=traceback.format_exc()))
        pending={m:Estimate(reasons=[error]) for m in METHODS}
    elapsed=(perf_counter()-started)*1000
    # All methods have returned before any evaluation geometry is read.
    truth,_=shared._load_truth(first,tracked)
    rows=[]
    for method in METHODS:
        estimate=pending[method];metrics=shared.score(estimate,truth);metrics['fit_condition']=metrics.pop('condition')
        row=dict(layout=first['layout'],case=first['case'],condition=first['condition'],split=role,
            backend=method,camera_set='all',status='error' if error else 'ok',**metrics)
        row.update(pose_fields(row));row.update(band=None,band_evaluation=None,observed_band_usable=False,
            evaluation_role=role,paired_group_total_ms=elapsed)
        if row.get('estimate'):
            band=plan_band(row['estimate'],estimate.diagnostics);row['band']=band
            row['observed_band_usable']=bool(band['nominal_compatible'])
            row['band_evaluation']={name:evaluate_band(band,truth,truth[field]) for name,field in
                [('near_min_proxy','near_min_interval_s_m'),('full_annotated_neck','search_interval_s_m')]}
        row['accepted_with_common_band_gate']=bool(row['accepted'] and row['observed_band_usable'])
        row['acceptance_contract']='task_local_3mm_surface_all_frozen_nonRGB_gates' if method=='task_surface' else method
        rows.append(row)
    return rows,tracked,checks


def run(args):
    frozen=sources()
    if args.freeze:
        protocol=shared._json(args.freeze)
        if protocol['source_sha256']!=frozen:raise ValueError('method_changed_after_freeze')
    manifest=shared._json(args.manifest)
    groups={}
    for pair in manifest['pairs']:
        if args.layouts and pair['layout'] not in args.layouts:continue
        groups.setdefault((pair['layout'],pair['case'],pair['condition']),[]).append(pair)
    if not groups:raise ValueError('no_input_groups')
    args.output.mkdir(parents=True,exist_ok=False)
    tracked={str(p.resolve()):shared.file_hash(p) for p in [args.manifest,args.foreground_cache,args.predictions/'predictions.jsonl']}
    rows=[];checks=[]
    jobs=[(groups[k],str(args.predictions.resolve()),str(args.foreground_cache.resolve()),args.role) for k in sorted(groups)]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker,j) for j in jobs]
        for future in as_completed(futures):
            group,hashes,check=future.result();rows.extend(group);tracked.update(hashes);checks.extend(check)
            with (args.output/'records.jsonl').open('a') as f:
                for row in group:f.write(json.dumps(row,allow_nan=False)+'\n')
            print(group[0]['case'],group[0]['condition'],[(r['backend'],r['accepted'],round(r.get('center_mm') or 0,3)) for r in group],flush=True)
    rows.sort(key=lambda r:(r['layout'],r['case'],r['condition'],r['backend']))
    summary=[]
    for key in sorted({(r['layout'],r['condition'],r['backend']) for r in rows}):
        group=[r for r in rows if (r['layout'],r['condition'],r['backend'])==key]
        result=dict(zip(('layout','condition','backend'),key),**aggregate(group))
        for metric in ['center_pass_4mm','pose_pass_4mm_2deg']:
            result['false_reject_'+metric]=sum(not r['accepted'] and r[metric] for r in group)
            result['false_accept_'+metric]=sum(r['accepted'] and not r[metric] for r in group)
        result['false_accept_axial_band']=sum(r['accepted'] and not bool((r.get('band_evaluation') or {}).get('full_annotated_neck',{}).get('contained')) for r in group)
        summary.append(result)
    changed=[p for p,h in tracked.items() if shared.file_hash(p)!=h]
    valid=not changed and sources()==frozen
    (args.output/'records.json.gz').write_bytes(gzip.compress(json.dumps(rows,allow_nan=False).encode(),mtime=0))
    for name,value in [('summary',summary),('pair_checks',checks),('validation',dict(passed=valid,changed_inputs=changed,
        frozen_sources_unchanged=sources()==frozen,groups=len(groups),records=len(rows),errors=sum(r['status']!='ok' for r in rows))),
        ('provenance',dict(manifest=str(args.manifest.resolve()),predictions=str(args.predictions.resolve()),
            foreground_cache=str(args.foreground_cache.resolve()),source_sha256=frozen,input_sha256=tracked,
            methods=METHODS,role=args.role,settings=asdict(PortfolioSettings()),width_m=.003,
            no_hardware=True,no_training=True,no_renderer=True,no_stereo_inference=True,
            truth_access='after all four outputs per group; no estimator GT access',
            runtime='paired validators share fitted candidates; group total is not individual deployment latency'))]:
        shared._write_json(args.output/(name+'.json'),value)
    if not valid:raise RuntimeError('sources_or_inputs_changed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['manifest','predictions','foreground-cache','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--layouts',nargs='*',default=['close']);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--freeze',type=Path);p.add_argument('--role',default='opened_engineering_development')
    run(p.parse_args())
