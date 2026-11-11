#!/usr/bin/env python3
"""Follow-up plane diagnosis: feature/count controls and mapping monotonicity.

These plane cases have already been inspected. All variant fits still use only
the TRAIN split; this is explanatory follow-up, not a fresh calibration test.
No variant replaces the locked model in the jar experiment.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import least_squares
from neck_localization.depth_correction import load_calibration_manifest,GaussianProcessDepthCorrection
from neck_localization.disparity_calibration import DisparitySettings,DisparityCorrection,features


def run(manifest,output):
    _,frames=load_calibration_manifest(manifest)
    sensor=json.loads((manifest.parent/'sensor.json').read_text());c=sensor['cameras'][0];fb=c['K_ir'][0][0]*c['baseline_m']
    rng=np.random.default_rng(2026102507);d=[];y=[]
    for f in frames:
        if f['split']!='train':continue
        valid=f['valid_mask']&(f['measured_depth']>0)&(f['reference_depth']>0)
        ids=np.flatnonzero(valid);ids=rng.choice(ids,min(len(ids),128),replace=False)
        observed=fb/f['measured_depth'].ravel()[ids];reference=fb/f['reference_depth'].ravel()[ids]
        d.extend(observed);y.extend(reference-observed)
    d,y=np.asarray(d),np.asarray(y);subset=np.random.default_rng(2026102507).choice(len(d),512,replace=False)
    variants={};reports=[]
    for harmonic in [0,2]:
        for n in [512,len(d)]:
            ids=subset if n==512 else np.arange(len(d));X=features(d[ids],harmonic)
            fit=least_squares(lambda b:np.r_[X@b-y[ids],.02*b],np.zeros(X.shape[1]),loss='soft_l1',f_scale=.04)
            label=f'harmonic{harmonic}_n{n}';variants[label]=DisparityCorrection(fit.x,fb,replace(DisparitySettings(),harmonics=harmonic))
    gp_path=ROOT/'experiments/neck_localization/results/active_stereo_20261025/artifacts/sim_d435_preset.npz'
    gp=GaussianProcessDepthCorrection.load(gp_path)
    accumulator={k:[] for k in ['raw','GP_frozen_512',*variants]}
    for f in frames:
        if f['split']!='test':continue
        old=f['measured_depth'];mask=f['valid_mask']&(old>0)&(f['reference_depth']>0)
        corrected={'raw':old,'GP_frozen_512':gp.correct_depth(old,f['K'],f['temperature_c'])}
        corrected.update({k:m.correct(old) for k,m in variants.items()})
        for label,z in corrected.items():
            err=(z[mask]-f['reference_depth'][mask])*1000
            accumulator[label].append(err)
    for label,errors in accumulator.items():
        pooled=np.concatenate(errors)
        reports.append(dict(method=label,pixels=len(pooled),pooled_rmse_mm=float(np.sqrt(np.mean(pooled**2))),
            frame_equal_rmse_mm=float(np.sqrt(np.mean([np.mean(e**2) for e in errors]))),
            pooled_abs_p95_mm=float(np.percentile(abs(pooled),95)),mean_mm=float(pooled.mean())))
    derivative=[]
    phase=np.linspace(0,1,10001)
    for label,model in variants.items():
        b=model.coefficients;g=np.ones_like(phase)+b[1]/15
        for k in range(1,model.settings.harmonics+1):
            g+=2*np.pi*k*(b[2*k]*np.cos(2*np.pi*k*phase)-b[2*k+1]*np.sin(2*np.pi*k*phase))
        derivative.append(dict(method=label,min_corrected_disparity_derivative=float(g.min()),
            negative_fraction=float(np.mean(g<0)),nonmonotone_mapping=bool(np.any(g<0))))
    output.mkdir(parents=True,exist_ok=False)
    (output/'diagnosis.json').write_text(json.dumps(dict(scope='follow-up diagnosis on opened plane test; no jar data; locked jar model unchanged',
        metrics=reports,mapping_derivative=derivative,training_points=len(d),
        inference='feature and sample-count effects separated for simple models; GP also differs in objective/kernel/hyperparameters'),indent=2)+'\n')
    print(json.dumps(dict(metrics=reports,mapping_derivative=derivative),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.manifest,a.output)
