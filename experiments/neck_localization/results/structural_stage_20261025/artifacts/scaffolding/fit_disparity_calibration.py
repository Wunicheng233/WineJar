#!/usr/bin/env python3
"""Fit periodic disparity baseline on training planes; separately report splits."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from neck_localization.depth_correction import load_calibration_manifest
from neck_localization.disparity_calibration import fit_correction, DisparitySettings


def run(manifest, output):
    metadata, frames=load_calibration_manifest(manifest)
    sensor=json.loads((manifest.parent/'sensor.json').read_text())
    camera=sensor['cameras'][0]; fb=camera['K_ir'][0][0]*camera['baseline_m']
    output.mkdir(parents=True,exist_ok=False)
    model, detail=fit_correction(frames,fb)
    model.save(output/'model.json')
    rows=[]
    for f in frames:
        old=f['measured_depth']; new=model.correct(old)
        mask=f['valid_mask']&(old>0)&(f['reference_depth']>0)
        before=(old[mask]-f['reference_depth'][mask])*1000
        after=(new[mask]-f['reference_depth'][mask])*1000
        rows.append(dict(frame=f['frame_id'],split=f['split'],n=int(mask.sum()),
            before_rmse_mm=float(np.sqrt(np.mean(before**2))),after_rmse_mm=float(np.sqrt(np.mean(after**2))),
            before_mean_mm=float(before.mean()),after_mean_mm=float(after.mean())))
    report=dict(training=detail,frames=rows,model='periodic disparity engineering baseline, not official calibration',
        manifest=str(manifest.resolve()),manifest_sha256=sha256(manifest.read_bytes()).hexdigest(),
        no_jar_data=True, sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'neck_localization/disparity_calibration.py']})
    report['summary']={split:dict(n=sum(r['split']==split for r in rows),
        before_rmse_mm=float(np.sqrt(np.mean([r['before_rmse_mm']**2 for r in rows if r['split']==split]))),
        after_rmse_mm=float(np.sqrt(np.mean([r['after_rmse_mm']**2 for r in rows if r['split']==split])))) for split in ['train','validation','test']}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.manifest,a.output)
