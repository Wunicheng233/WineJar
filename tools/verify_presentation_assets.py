#!/usr/bin/env python3
"""Verify published figures/tables; optionally relocate and render the viewer.

Default uses the Python standard library only, with no inference or hardware.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


def verify(assets,check_viewer=False):
    manifest=json.loads((assets/'manifest.json').read_text())
    for name,sha in manifest['asset_sha256'].items():
        p=assets/name
        assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==sha,name
    assert not any(p.is_symlink() for p in assets.rglob('*'))
    rows=list(csv.DictReader((assets/'data/camera_count_records.csv').open()))
    summary=list(csv.DictReader((assets/'data/camera_count_summary.csv').open()))
    assert len(rows)==216 and len({r['case'] for r in rows})==36 and len(summary)==12
    for item in summary:
        group=[r for r in rows if all(r[k]==item[k] for k in ('group','method','camera_set'))]
        actual=dict(total=len(group),success=sum(r['accepted']=='True' and r['truth_pass']=='True' for r in group),
                    rejected=sum(r['accepted']=='False' for r in group),
                    false_accepted=sum(r['accepted']=='True' and r['truth_pass']=='False' for r in group))
        assert all(int(item[k])==v for k,v in actual.items()),item
    dimensions={}
    for p in (assets/'camera_count').glob('*.png'):
        dimensions[p.name]=struct.unpack('>II',p.read_bytes()[16:24])
    assert dimensions['nominal_014_comparison.png']==(3200,1800)
    assert dimensions['stress_010_comparison.png']==(3200,1800)
    for p in (assets/'viewer/scenes').glob('*/scene.xml'):
        tree=ET.parse(p)
        for asset in tree.findall('.//asset/*'):
            if asset.get('file'):assert not Path(asset.get('file')).is_absolute(),p
    result=dict(passed=True,hashed_asset_files=len(manifest['asset_sha256']),records=216,
                summary_groups=12,figures=dimensions,no_local_symlinks=True,
                no_new_estimation=True,hardware_access=False)
    if check_viewer:
        import os
        os.environ.pop('MUJOCO_GL',None)
        sys.path.insert(0,str(ROOT))
        from neck_localization.inspection_viewer import InspectionViewer
        routes=[]
        with tempfile.TemporaryDirectory(prefix='winejar_portable_viewer_') as folder:
            copied=Path(folder)/'viewer';shutil.copytree(assets/'viewer',copied)
            v=InspectionViewer(copied/'scenes',copied/'records.json.gz','nominal_014',
                method='region_equal_trust',visible=False,ground_truth=True,overlay=True,large_neck=True)
            try:
                for case in ['nominal_014','stress_010']:
                    v.case_index=v.cases.index(case);v._load_case()
                    for camera in ['cam_a','cam_a,cam_b','all']:
                        v.camera_set=camera
                        for cloud in [False,True]:
                            v.point_cloud=cloud;v._cloud_cache_key=None;assert v.draw() is not None
                        routes.append([case,camera])
            finally:v.close()
        result.update(relocated_viewer_passed=True,actual_GL_routes=routes,rendered_model_and_cloud_views=12)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=ROOT/'experiments/presentation_assets_20261111')
    p.add_argument('--check-viewer',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();r=verify(a.assets.resolve(),a.check_viewer)
    if a.output:
        if a.output.exists():raise FileExistsError(a.output)
        a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(r,ensure_ascii=False))
