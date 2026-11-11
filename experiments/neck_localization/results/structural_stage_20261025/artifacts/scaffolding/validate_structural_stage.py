#!/usr/bin/env python3
"""Audit frozen inputs/fit points and repeat a bounded subset without GT access."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from neck_localization.stereo_robust import estimate_stereo_robust
from neck_localization.structure_portfolio import estimate_portfolio
from scripts.report_structural_stage import load_rows,render_root
from scripts.run_depth_profile_baseline import load_frames
from scripts.run_structural_study import sources


def run(source,output):
    assert (source/'completed.json').exists()
    paths={};point_checks=[];repeats=[];freeze_checks=[];replay=[]
    groups=['main','matching','small_matching','phase_estimates']
    total=0
    for layout in ['far','close']:
        stem='structural_'+('close_' if layout=='close' else '')+'test_20261025_v1'
        freeze=json.loads((ROOT/'experiments/neck_localization/configs'/(stem+'_freeze.json')).read_text())
        assert freeze['sources']==sources(),'Frozen algorithm changed'
        for folder in ['render','rematched','phase']:
            config=(render_root(source,layout) if folder=='render' else source/layout/folder)/'effective_config.json'
            assert sha256(config.read_bytes()).hexdigest()==freeze['config_sha256']
        freeze_checks.append(dict(layout=layout,source_hashes_match=True,config_bytes_match=True))
        rr=json.loads((source/layout/'rematched/replay.json').read_text())
        replay.extend([dict(layout=layout,**r) for r in rr if r['preset']=='default7'])
        assert all(r['replay_max_difference_m']==0. for r in replay)
        for group in groups:
            rows=load_rows(source/layout/group/'records.json.gz');total+=len(rows)
            cache={}
            for row in rows:
                scene=source/layout/group/row['condition_name']/'scenes'/row['case']
                key=(row['condition_name'],row['case'])
                if key not in cache:
                    cache[key]=load_frames(scene/'observations','rgb')
                    for p in (scene/'observations').iterdir():
                        if p.suffix in ['.png','.npz','.json']:
                            path=p.resolve();paths[str(path)]=sha256(path.read_bytes()).hexdigest()
                frames=cache[key];d=row.get('diagnostics',{})
                if not d.get('fit_point_positions_m'):continue
                points=np.asarray(d['fit_point_positions_m']);ids=np.asarray(d['fit_camera_ids']);names=d['fit_camera_names']
                weights=np.asarray(d['fit_point_weights'])
                assert len(weights)==len(points) and np.isfinite(weights).all() and (weights>0).all()
                maximum=0.
                for frame in frames:
                    use=ids==names.index(frame.camera.name);q=points[use]
                    if not len(q):continue
                    uv=np.rint(frame.camera.project(q)).astype(int)
                    assert ((uv[:,0]>=0)&(uv[:,0]<frame.camera.width)&(uv[:,1]>=0)&(uv[:,1]<frame.camera.height)).all()
                    back=frame.camera.unproject(uv,frame.depth_z_m[uv[:,1],uv[:,0]])
                    maximum=max(maximum,float(np.max(abs(back-q))))
                assert maximum<1e-7,'Fit coordinates do not come from saved actual depth'
                point_checks.append(dict(layout=layout,group=group,case=row['case'],method=row['method'],
                    condition=row['condition_name'],n=len(points),maximum_coordinate_difference_m=maximum))
            # Bound repeated inference; forbid reading any evaluation artefact.
            if group=='main':
                for method in ['baseline','portfolio']:
                    row=next(r for r in rows if r['condition_name']=='stereo_raw' and r['method']==method and
                             r['case']=='structural_test_nominal_000')
                    frames=cache[('stereo_raw',row['case'])]
                    original=Path.open
                    def guarded(path,*args,**kwargs):
                        if 'evaluation_only' in str(path):raise AssertionError('GT accessed during inference')
                        return original(path,*args,**kwargs)
                    with patch.object(Path,'open',guarded):
                        estimate=(estimate_stereo_robust(frames) if method=='baseline' else estimate_portfolio(frames))
                    assert estimate.accepted==row['accepted'] and estimate.reasons==row['reasons']
                    if row.get('estimate'):
                        np.testing.assert_allclose(estimate.center,row['estimate']['center_m'],atol=1e-10,rtol=0)
                        np.testing.assert_allclose(estimate.normal,row['estimate']['normal'],atol=1e-10,rtol=0)
                        np.testing.assert_allclose(estimate.radius,row['estimate']['radius_m'],atol=1e-10,rtol=0)
                    else:assert estimate.center is None
                    repeats.append(dict(layout=layout,case=row['case'],method=method,matched=True,
                        no_evaluation_file_access=True,runtime_ms=estimate.elapsed_ms))
    after={p:sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    assert after==paths,'An observation changed during validation'
    report=dict(total_test_records=total,freeze_checks=freeze_checks,replay_native_exact_matches=len(replay),
        observed_files_unchanged=len(paths),observation_sha256=paths,fit_groups=len(point_checks),
        total_fit_points=sum(r['n'] for r in point_checks),point_checks=point_checks,repeated_inference=repeats,
        no_full_second_render_claim=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ['point_checks','observation_sha256']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run,a.output)
