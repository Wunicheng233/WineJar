#!/usr/bin/env python3
"""Oracle same-pixel depth intervention on the frozen AXIS objective.

Keep selected rays, original spline basis, regularization and final axis weights
fixed. Compare measured and ideal XYZ; no production estimator changes.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import least_squares
from neck_localization.support_feedback import WeightedAxisSplineProblem
from scripts.diagnose_structural_failure_chain import replay_capture,ideal_correspondences,axis_at_z
from scripts.report_structural_stage import render_root
from scripts.run_depth_profile_baseline import load_frames


def run(source,output):
    output.mkdir(parents=True,exist_ok=False);rows=[]
    for layout,case in [('far','structural_test_nominal_005'),('far','structural_test_stress_003'),
                        ('close','structural_test_nominal_000'),('close','structural_test_stress_003')]:
        root=render_root(source,layout);scene=root/'stereo_raw/scenes'/case
        frames=load_frames(scene/'observations','rgb');result,v=replay_capture(frames)
        truth=json.loads((scene/'evaluation_only/target.json').read_text())
        with np.load(scene/'evaluation_only/profile.npz') as f:ss,rr=f['s_m'].copy(),f['radius_m'].copy()
        old=v['problem'];points=v['points'][v['training']];ids=v['ids'][v['training']]
        clean,valid,_,_=ideal_correspondences(points,ids,load_frames(root/'ideal_rgb/scenes'/case/'observations','rgb'),truth,ss,rr)
        assert valid.all(),'Must retain exact same point count'
        a,n=axis_at_z(truth,old.origin[2]);true_x=np.r_[a[:2]-old.origin[:2],old.U.T@n/(old.n0@n)]
        bounds=np.array([15.,15.,.22,.22]);original_x=v['x']
        for label,xyz in [('measured',points),('ideal_same_pixels',clean)]:
            problem=WeightedAxisSplineProblem(xyz,old.origin,old.n0,ids,v['joint'])
            problem.basis=old.basis;problem.derivative=old.derivative;problem.regularizer=old.regularizer
            problem.set_weights(v['axis_used_weights'])
            record=dict(layout=layout,case=case,intervention=label,diagnostic_only=True,
                old_axis_cost=float(problem.evaluate(original_x)['objective']),true_axis_cost=float(problem.evaluate(true_x)['objective']),fits=[])
            starts=[('old_axis',original_x),('zero',np.zeros(4))]
            if np.all(abs(true_x)<bounds):starts.append(('true_axis',true_x))
            for name,start in starts:
                fit=least_squares(problem.fun,start,jac=problem.jac,bounds=(-bounds,bounds),x_scale=[5,5,.1,.1],
                    max_nfev=400,ftol=1e-7,xtol=1e-7,gtol=1e-6)
                st=problem.evaluate(fit.x)
                record['fits'].append(dict(start=name,success=bool(fit.success),cost=float(st['objective']),
                    normal_deg=float(np.degrees(np.arccos(np.clip(st['normal']@n,-1,1)))),nfev=fit.nfev))
            rows.append(record)
            print(layout,case,label,'old/true cost',record['old_axis_cost'],record['true_axis_cost'],record['fits'],flush=True)
    (output/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
    (output/'provenance.json').write_text(json.dumps(dict(source=str(source.resolve()),diagnostic_only=True,
        same_points_and_basis_and_weights=True,script_sha256=sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run,a.output)
