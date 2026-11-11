#!/usr/bin/env python3
"""Paired engineering-corrected / standard geometry on frozen saved depth.

No model training, stereo inference, rendering or hardware access. Every group
finishes both geometry calls before opening evaluation-only target/profile.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import hashlib
import importlib
import json
import multiprocessing
from pathlib import Path
import sys
from time import perf_counter
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from neck_localization.band_geometry import evaluate_band, plan_band
from neck_localization.geometry import Estimate
from neck_localization.structure_portfolio import PortfolioSettings
from scripts import evaluate_capability_depth as shared

MODULES={'visible_profile':('visible_profile_baseline','estimate_visible_profile','PortfolioSettings'),
         'standard_sor':('sor_baseline','estimate_sor','SORSettings')}


def backend(name):
    module_name,function_name,settings_name=MODULES[name]
    module=importlib.import_module('neck_localization.'+module_name)
    settings=PortfolioSettings() if settings_name=='PortfolioSettings' else getattr(module,settings_name)()
    return getattr(module,function_name),settings


def sources(methods):
    paths=[ROOT/'neck_localization'/(MODULES[m][0]+'.py') for m in methods]+[Path(__file__)]
    return dict(shared.source_hashes(),**{str(p.relative_to(ROOT)):shared.file_hash(p) for p in paths})


def pose_fields(row):
    c,n=row.get('center_mm'),row.get('normal_deg')
    return dict(center_pass_3mm=c is not None and c<=3,
                center_pass_4mm=c is not None and c<=4,
                axis_pass_2deg=n is not None and n<=2,
                pose_pass_3mm_2deg=c is not None and n is not None and c<=3 and n<=2,
                pose_pass_4mm_2deg=c is not None and n is not None and c<=4 and n<=2)


def worker(job):
    cv2.setNumThreads(1)
    pairs,predictions,methods=job
    predictions=Path(predictions)
    first=pairs[0]
    index=shared.load_prediction_index(predictions)
    frames,tracked,checks=[],{},[]
    error=None
    try:
        for pair in sorted(pairs,key=lambda p:p['camera']):
            depth,info=shared.read_prediction(pair,predictions,index,tracked)
            frame,_,registration=shared.prepare_frame(pair,depth,tracked)
            frames.append(frame)
            checks.append(dict(pair_id=pair['pair_id'],status='ok',prediction=info,registration=registration))
        if tuple(f.camera.name for f in frames)!=shared.ALL_CAMERAS:
            raise ValueError('all_three_camera_inputs_required')
    except Exception as exception:
        error=f'{type(exception).__name__}:{exception}'
        checks.append(dict(status='input_error',error=error,traceback=traceback.format_exc()))
    pending=[]
    for method in methods:
        started=perf_counter()
        status='ok'
        estimate=Estimate()
        if error:
            status='input_error';estimate.reasons=[error]
        else:
            try:
                infer,settings=backend(method)
                estimate=infer(frames,settings=settings)
                shared._validate_estimate(estimate)
            except Exception as exception:
                status='backend_error'
                estimate=Estimate(reasons=[f'{type(exception).__name__}:{exception}'],
                    diagnostics=dict(traceback=traceback.format_exc()))
        estimate.elapsed_ms=(perf_counter()-started)*1000
        pending.append((method,status,estimate))
    # These reads occur only after ALL methods above have returned.
    truth,_=shared._load_truth(first,tracked)
    records=[]
    for method,status,estimate in pending:
        metrics=shared.score(estimate,truth)
        metrics['fit_condition']=metrics.pop('condition')
        row=dict(layout=first['layout'],case=first['case'],condition=first['condition'],
            split=first['split'],camera_set='all',backend=method,status=status,**metrics)
        row.update(pose_fields(row))
        row['evaluation_role']='oracle_measurement_diagnostic_not_deployed_input' if first.get('evaluation_role','').startswith('oracle') else 'method_capability_audit'
        row['band'],row['band_evaluation']=None,None
        row['observed_band_usable']=False
        if row.get('estimate') and 'near_min_interval_from_axis_origin_m' in estimate.diagnostics:
            try:
                band=plan_band(row['estimate'],estimate.diagnostics)
                row['band']=band
                row['observed_band_usable']=bool(band['nominal_compatible'])
                row['band_evaluation']={label:evaluate_band(band,truth,truth[field]) for label,field in
                    [('near_min_proxy','near_min_interval_s_m'),('full_annotated_neck','search_interval_s_m')]}
            except Exception as exception:
                row['band_error']=f'{type(exception).__name__}:{exception}'
        # Different native validators are kept intact. This explicit common
        # task gate reports the same 3 mm observed-band requirement separately.
        row['accepted_with_common_band_gate']=bool(row['accepted'] and row['observed_band_usable'])
        row['acceptance_contract']=('frozen_noncontact_gates_plus_finite_surface_containment'
            if method=='visible_profile' else 'standard_sor_numerical_support_and_holdout_gates_no_native_band_gate')
        records.append(row)
    return records,checks,tracked


def aggregate(rows):
    out=shared.aggregate(rows)
    out['full_circle_correct_output']=out.pop('correct_output')
    out['full_circle_wrong_accepted']=out.pop('wrong_accepted')
    out['full_circle_correct_output_rate']=out.pop('correct_output_rate')
    out['full_circle_wrong_accepted_rate']=out.pop('wrong_accepted_rate')
    for key in ['center_pass_3mm','center_pass_4mm','axis_pass_2deg','pose_pass_3mm_2deg','pose_pass_4mm_2deg']:
        out[key]=sum(bool(r[key]) for r in rows)
        out['accepted_'+key]=sum(bool(r['accepted'] and r[key]) for r in rows)
        out['common_band_accepted_'+key]=sum(bool(r['accepted_with_common_band_gate'] and r[key]) for r in rows)
    out['common_band_accepted']=sum(r['accepted_with_common_band_gate'] for r in rows)
    contained=lambda r:bool((r.get('band_evaluation') or {}).get('full_annotated_neck',{}).get('contained'))
    out['candidate_axial_band_contained']=sum(contained(r) for r in rows)
    out['accepted_axial_band_contained']=sum(r['accepted'] and contained(r) for r in rows)
    out['common_band_accepted_axial_band_contained']=sum(r['accepted_with_common_band_gate'] and contained(r) for r in rows)
    return out


def run(args):
    manifest=shared._json(args.manifest)
    prediction_provenance=shared._json(args.predictions/'provenance.json')
    oracle=bool(prediction_provenance.get('oracle'))
    groups={}
    for pair in manifest['pairs']:
        if args.layouts and pair['layout'] not in args.layouts:continue
        if args.conditions and pair['condition'] not in args.conditions:continue
        if args.cases and pair['case'] not in args.cases:continue
        pair=dict(pair)
        if oracle:pair['evaluation_role']='oracle_measurement_diagnostic_not_deployed_input'
        groups.setdefault((pair['layout'],pair['case'],pair['condition']),[]).append(pair)
    if not groups:raise ValueError('no_selected_inputs')
    args.output.mkdir(parents=True,exist_ok=False)
    source_hashes=sources(args.methods)
    settings={name:asdict(backend(name)[1]) for name in args.methods}
    tracked={str(args.manifest.resolve()):shared.file_hash(args.manifest),
        str((args.predictions/'predictions.jsonl').resolve()):shared.file_hash(args.predictions/'predictions.jsonl')}
    records,checks=[],[]
    jobs=[(groups[key],str(args.predictions.resolve()),args.methods) for key in sorted(groups)]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(worker,job):job[0][0] for job in jobs}
        for future in as_completed(futures):
            rows,checked,hashes=future.result()
            records.extend(rows);checks.extend(checked);tracked.update(hashes)
            with (args.output/'records.jsonl').open('a') as stream:
                for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
            first=rows[0]
            print(first['layout'],first['case'],first['condition'],[(r['backend'],r['accepted'],r['center_pass_3mm'],r['truth_pass']) for r in rows],flush=True)
    records.sort(key=lambda r:(r['layout'],r['case'],r['condition'],r['backend']))
    summaries=[]
    for key in sorted({(r['layout'],r['condition'],r['backend']) for r in records}):
        chosen=[r for r in records if (r['layout'],r['condition'],r['backend'])==key]
        summaries.append(dict(zip(('layout','condition','backend'),key),camera_set='all',**aggregate(chosen)))
    shared._write_json(args.output/'summary.json',summaries)
    shared._write_json(args.output/'pair_checks.json',checks)
    (args.output/'records.json.gz').write_bytes(gzip.compress(json.dumps(records,allow_nan=False).encode(),mtime=0))
    changed=[p for p,h in tracked.items() if not Path(p).exists() or shared.file_hash(p)!=h]
    valid=not changed and source_hashes==sources(args.methods)
    shared._write_json(args.output/'validation.json',dict(passed=valid,changed_inputs=changed,
        frozen_sources_unchanged=source_hashes==sources(args.methods),groups=len(groups),records=len(records),
        input_or_backend_errors=sum(r['status']!='ok' for r in records)))
    shared._write_json(args.output/'provenance.json',dict(schema=1,manifest=str(args.manifest.resolve()),
        predictions=str(args.predictions.resolve()),source_sha256=source_hashes,input_sha256=tracked,
        methods=args.methods,settings=settings,layouts=args.layouts,conditions=args.conditions,
        no_hardware=True,no_training=True,no_stereo_inference=True,
        oracle_measurement_control=oracle,
        evaluation_thresholds=dict(center_primary_max_mm=4,center_preferred_mm=3,axis_analysis_deg=2,diameter_analysis_mm=2),
        acceptance='Native validators retained; common 3mm nominal band gate and actual annotated axial containment reported separately',
        truth_access='after both methods per case/condition; no GT inputs or candidate selection',
        runtime='CPU workers contend; timings not deployment latency claims',
        geometry_families=args.scope))
    if not valid:raise RuntimeError('inputs_or_sources_changed_during_run')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods',choices=list(MODULES),nargs='+',default=list(MODULES))
    p.add_argument('--layouts',nargs='*',default=[])
    p.add_argument('--conditions',nargs='*',default=[])
    p.add_argument('--cases',nargs='*',default=[])
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--scope',default='previously viewed synthetic circular-section jars, no cross-family or real-device claim')
    run(p.parse_args())
