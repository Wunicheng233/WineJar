#!/usr/bin/env python3
"""Analytic/Monte-Carlo diagnostic for a shared translation covariance term.

No 4096 new estimator trials: translates existing output using an exact gauge
identity checked separately by the explicit extrinsic perturbation experiment.
Known Gaussian priors below are ASSUMPTIONS, not measured device uncertainty.
"""
import argparse,gzip,json
from pathlib import Path
import numpy as np
from scipy.stats import chi2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    source=root/'experiments/neck_localization/runs/depth_bias_stage1_20261023_v1/zero/records.json.gz'
    rows=json.load(gzip.open(source,'rt'))
    count=4096;seed=2026102467
    directions=np.random.default_rng(seed).normal(size=(count,3))
    threshold=float(chi2.ppf(.95,3))
    results=[]
    for row in rows:
        truth=json.loads((root/'experiments/neck_localization/runs/reliable_output_v2_test_scenes'/row['case']/'evaluation_only/target.json').read_text())
        delta=(np.array(row['estimate']['center_m'])-truth['center_m'])*1000
        n=np.array(truth['normal'])
        estimates_n=np.array(row['estimate']['normal'])
        radius=row['estimate']['radius_m']*1000
        nominal_half=1.5*abs(estimates_n@n)+radius*np.linalg.norm(np.cross(estimates_n,n))
        interval=np.array(truth['search_interval_s_m'])*1000-truth['s_m']*1000
        npoints=row['diagnostics']['fit_points']
        for sigma in [.5,1.,2.]:
            errors=delta+sigma*directions
            squared=np.sum(errors**2,axis=1)/sigma**2
            axial=errors@n
            fits=(axial-nominal_half>=interval[0])&(axial+nominal_half<=interval[1])
            results.append(dict(case=row['case'],assumed_common_translation_sigma_mm=sigma,n=count,
                center_coverage95_common_term=float(np.mean(squared<=threshold)),
                naive_independent_point_coverage95=float(np.mean(squared*npoints<=threshold)),
                zero_estimator_bias_oracle_coverage95=float(np.mean(np.sum(directions**2,axis=1)<=threshold)),
                axial_band_containment_probability_under_assumed_distribution=float(np.mean(fits)),
                naive_point_count=npoints,unmodelled_fixed_center_error_mm=float(np.linalg.norm(delta))))
    (args.output/'coverage.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.output/'protocol.json').write_text(json.dumps(dict(seed=seed,n_per_case=count,source=str(source),
        scope='posthoc covariance demonstration; no new estimator performance claim, no measured covariance',
        perturbation='common isotropic world-translation only; normal fixed, no rotation or detector covariance',
        actual_input_translation_identity='explicitly checked by extrinsic_shared_x5mm diagnostic',
        warning='common term alone omits existing estimator bias; empirical coverage can be below95'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    for field,label in [('center_coverage95_common_term','Common term + existing estimator bias'),('naive_independent_point_coverage95','Wrongly divided by fitted point count'),('zero_estimator_bias_oracle_coverage95','Diagnostic ideal estimator (no fixed bias)')]:
        means=[np.mean([r[field] for r in results if r['assumed_common_translation_sigma_mm']==sigma]) for sigma in [.5,1,2]]
        ax.plot([.5,1,2],means,'o-',label=label)
    ax.axhline(.95,ls=':',color='grey');ax.set(xlabel='Assumed common translation std per axis [mm]',ylabel='3D 95% ellipsoid coverage',ylim=(-.03,1.03))
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(args.output/'common_covariance_coverage.png',dpi=150)
    print('Saved shared-covariance diagnostic; not a calibrated system confidence output')


if __name__=='__main__':main()
