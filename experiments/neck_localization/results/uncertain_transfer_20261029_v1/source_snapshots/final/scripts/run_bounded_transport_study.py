#!/usr/bin/env python3
"""Compare full visual updates with bounded/shrunk updates on frozen inputs."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from neck_localization.bounded_transport import bounded_translation,gaussian_shrinkage
from neck_localization.station_tying import pose,move_circle
from neck_localization.cross_station_uncertainty import evaluate_arrays
from neck_localization.experiment_io import read,write,digest,quantiles


def run(source,output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    original=read(source/'body/records.json');partial=read(source/'body/partial_view_ablation/records.json')
    base=ROOT/read(source/'body/protocol.json')['baseline'];geometry=ROOT/read(base/'protocol.json')['source_bundle']
    controls={tuple(r[k] for k in ('cohort','case','condition','source_control','stop_error_mm')):r for r in original if r['method']=='nominal'}
    predictions=[]
    for r in partial:
        if r['method']!='reciprocal_axis' or r['source_control']!='saved_simsense_body':continue
        key=tuple(r[k] for k in ('cohort','case','condition','source_control','stop_error_mm'));control=controls[key]
        measured=r['estimate']['center_m'][0]-control['estimate']['center_m'][0]
        # These fixed bounds are scanned across all conditions; the estimator
        # never receives condition, actual stop error, or saved scoring errors.
        for a in (.0005,.002):
            for b in (.0005,.001,.003,.005):
                for method in ('nominal','full_visual','bounded','shrinkage'):
                    if method=='nominal':update=dict(delta_m=0.,status='nominal')
                    elif method=='full_visual':update=dict(delta_m=measured if r['accepted'] else 0.,status='visual_candidate')
                    elif method=='bounded':update=bounded_translation(measured,a,b,measurement_available=r['accepted'])
                    else:update=gaussian_shrinkage(measured,a,b) if r['accepted'] else dict(delta_m=0.,status='nominal_fallback')
                    estimate={k:np.array(v).copy() if isinstance(v,list) else v for k,v in control['estimate'].items()}
                    estimate['center_m'][0]+=update['delta_m']
                    predictions.append(dict(cohort=r['cohort'],case=r['case'],condition=r['condition'],stop_error_mm=r['stop_error_mm'],
                        prior_half_mm=a*1000,measurement_bound_mm=b*1000,method=method,update=update,estimate=estimate))
    # Prediction production has ended. Only this scoring pass reads true rings.
    records=[]
    for r in predictions:
        folder=geometry/r['cohort']/'ss/clean/scenes'/r['case']
        truth=read(folder/'evaluation_only/target.json');length=read(folder/'model_manifest.json')['jar_spec']['neck_length_m']
        R=Rotation.from_euler('y',1,degrees=True).as_matrix() if r['condition']=='rotation_mismatch_1deg' else np.eye(3)
        target=move_circle(truth,pose(R,[2.4+r['stop_error_mm']/1000,0,0]))
        e=r['estimate'];v=evaluate_arrays(np.array(e['center_m'])[None],np.array(e['normal'])[None],np.array([e['radius_m']]),
            np.array(target['center_m'])[None],np.array(target['normal'])[None],np.array([length]))
        records.append(r|dict(**{k:x[0] for k,x in v.items()},
            transport_only_mm=abs(r['update']['delta_m']*1000-r['stop_error_mm']),
            actual_stop_within_assumed_prior=abs(r['stop_error_mm'])<=r['prior_half_mm'],final_fastening_success=None))
    summary=[]
    for key in sorted({tuple(r[k] for k in ('condition','prior_half_mm','measurement_bound_mm','method','actual_stop_within_assumed_prior')) for r in records}):
        g=[r for r in records if tuple(r[k] for k in ('condition','prior_half_mm','measurement_bound_mm','method','actual_stop_within_assumed_prior'))==key]
        summary.append(dict(zip(('condition','prior_half_mm','measurement_bound_mm','method','actual_stop_within_assumed_prior'),key))|
            dict(total=len(g),**{k:quantiles([r[k] for r in g]) for k in ('center_mm','transport_only_mm')},
                band_contained=sum(r['band_contained'] for r in g),fallback=sum(r['update']['status'].startswith('nominal_fallback') for r in g),
                final_fastening_success=None))
    write(output/'records.json',records);write(output/'summary.json',summary)
    write(output/'provenance.json',dict(source_sha256={str(p.resolve()):digest(p) for p in [source/'body/records.json',source/'body/partial_view_ablation/records.json',Path(__file__),ROOT/'neck_localization/bounded_transport.py']},
        error_bound_scan_not_measured_factory_bound=True,condition_and_truth_never_given_to_update_function=True,
        no_new_rejection_of_jar_or_claimed_confidence=True,old_seen_geometry_and_noise_conditions=True))
    print('bounded transport complete',len(records),'records')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source.resolve(),a.output.resolve())
