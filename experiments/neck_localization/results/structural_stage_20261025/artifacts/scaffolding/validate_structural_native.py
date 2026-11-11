#!/usr/bin/env python3
"""Check ancillary native fit points against their saved native depth images."""
import argparse
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_depth_profile_baseline import load_frames


def run(source,output):
    rows=json.loads(gzip.decompress((source/'records.json.gz').read_bytes()));checks=[];cache={}
    for row in rows:
        key=(row['layout'],row['condition_name'],row['case'])
        if key not in cache:cache[key]=load_frames(source/key[0]/key[1]/'scenes'/key[2]/'observations','rgb')
        d=row.get('diagnostics',{});points=d.get('fit_point_positions_m')
        if not points:continue
        points=np.array(points);ids=np.array(d['fit_camera_ids']);names=d['fit_camera_names'];maximum=0.
        weights=np.array(d['fit_point_weights']);assert len(weights)==len(points) and (weights>0).all()
        for f in cache[key]:
            q=points[ids==names.index(f.camera.name)]
            if not len(q):continue
            uv=np.rint(f.camera.project(q)).astype(int)
            back=f.camera.unproject(uv,f.depth_z_m[uv[:,1],uv[:,0]])
            maximum=max(maximum,float(np.max(abs(back-q))))
        assert maximum<1e-7
        checks.append(dict(layout=key[0],mode=key[1],case=key[2],method=row['method'],n=len(points),maximum_error_m=maximum))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(dict(records=len(rows),fit_groups=len(checks),
        fit_points=sum(c['n'] for c in checks),checks=checks,scope='native geometry, not mesh/true target points'),stream,indent=2)
    print('records',len(rows),'fit groups',len(checks),'points',sum(c['n'] for c in checks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run,a.output)
