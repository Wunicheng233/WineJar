#!/usr/bin/env python3
"""Compact engineering tables/figures, preserving every trial and failure."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from neck_localization.experiment_io import read,write,quantiles,digest


def run(bundle):
    output=bundle/'analysis'
    if output.exists():raise FileExistsError(output)
    output.mkdir()
    records=[];fits=[]
    for name in ['baselines','poe','poe_block']:
        records.extend(read(bundle/f'confirmation_evaluation_{name}/records.json'))
        fits.extend(read(bundle/f'confirmation_{name}/records.json'))
    write(output/'confirmation_records.json',records)
    summary=[]
    for cond in sorted({r['condition'] for r in records}):
        for method in sorted({r['method'] for r in records}):
            g=[r for r in records if (r['condition'],r['method'],r['tool_policy'])==(cond,method,'measured_board_jig')]
            if not g:continue
            summary.append(dict(condition=cond,method=method,total=len(g),
                IK_solved=sum(r['nominal_IK_passed'] for r in g),
                **{k:quantiles([r[k] for r in g]) for k in ['center_mm','normal_deg','calibration_and_execution_only_mm','band_margin_mm']},
                band_contained=sum(r['band_contained'] for r in g),
                all_stage_outputs_are_candidates_not_process_acceptance=True,final_fastening_success=None,
                band_failures=[{k:r[k] for k in ('seed','cohort','case','band_margin_mm','center_mm')} for r in g if not r['band_contained']]))
    write(output/'confirmation_summary.json',summary)
    fit_summary=[]
    for cond in sorted({r['condition'] for r in fits}):
        for method in sorted({r['method'] for r in fits}):
            g=[r for r in fits if (r['condition'],r['method'])==(cond,method)]
            if g:fit_summary.append(dict(condition=cond,method=method,total=len(g),accepted=sum(r['accepted'] for r in g),
                elapsed_ms=quantiles([r['elapsed_ms'] for r in g]),holdout_pixel_rms=quantiles([r['holdout_pixel_rms'] for r in g])))
    write(output/'fit_summary.json',fit_summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'pixel_huber':'Pixel Huber','uncertain_vce':'Uncertainty model','camera_poe':'Camera-joint mapping','camera_poe_block':'Mapping + view weights'}
    methods=list(labels);conditions=['image_only','kinematic_5','kinematic_bad_image','wrong_K']
    titles=['Image noise','Structured FK bias','FK bias + bad view','Wrong focal length']
    fig,axes=plt.subplots(1,4,figsize=(16,4.6),sharey=True)
    for ax,c,title in zip(axes,conditions,titles):
        vals=[next(r['calibration_and_execution_only_mm']['p95'] for r in summary if r['condition']==c and r['method']==m) for m in methods]
        bars=ax.bar(range(4),vals,color=['#8b99ac','#e1ad45','#3a96b6','#27986d'])
        ax.set_xticks(range(4),['Pixel','Uncertain','Mapping','+ weights'],rotation=25,ha='right');ax.set_title(title)
        ax.spines[['top','right']].set_visible(False);ax.bar_label(bars,fmt='%.2f',padding=3);ax.set_ylim(0,max(4.5,max(vals)*1.2))
    axes[0].set_ylabel('Added calibration/execution error P95 [mm]')
    fig.suptitle('New calibration campaigns, 11 previously seen jars; simulation only',fontsize=14)
    fig.tight_layout();fig.savefig(output/'calibration_execution_comparison.png',dpi=180);plt.close(fig)
    # Keep the whole set of process failures, including introduced failures.
    worst=sorted([r for r in records if r['tool_policy']=='measured_board_jig'],key=lambda r:r['center_mm'],reverse=True)
    write(output/'retained_worst_cases.json',worst[:30])
    write(output/'summary_provenance.json',dict(source_sha256={str(p.resolve()):digest(p) for p in [Path(__file__)]},
        aggregate_distribution_is_conditional_not_factory_probability=True,previously_seen_jars=True))
    print('summarized',len(records),'actual FK task arrivals',len(fits),'calibrations')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);a=p.parse_args();run(a.bundle.resolve())
