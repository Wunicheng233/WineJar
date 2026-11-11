#!/usr/bin/env python3
"""Post-evaluation viewer bundles; preserve actual observations and RGB masks."""
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from neck_localization.stereo_robust import extract_stereo_masks
from scripts.run_depth_profile_baseline import load_frames


def run(source,output):
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    datasets=[];checks=[]
    groups=[('main','ideal_rgb','理想'),('main','stereo_raw','原始'),('matching','narrow5','窄窗'),
            ('small_matching','local3','小窗'),('phase_estimates','phase_calibrated','相位校正')]
    for layout in ['far','close']:
        for group,mode,label in groups:
            original=source/layout/group/mode
            if not (original/'records.json.gz').exists():continue
            rows=json.loads(gzip.decompress((original/'records.json.gz').read_bytes()))
            destination=output/layout/mode;destination.mkdir(parents=True)
            (destination/'records.json.gz').write_bytes((original/'records.json.gz').read_bytes())
            for name in sorted({r['case'] for r in rows}):
                scene=original/'scenes'/name;target=destination/'scenes'/name;target.mkdir(parents=True)
                for child in ['assets','evaluation_only','model_manifest.json','observations']:
                    (target/child).symlink_to((scene/child).resolve(),target_is_directory=(scene/child).is_dir())
                shutil.copy2(scene/'scene.xml',target/'scene.xml')
                frames=load_frames(scene/'observations','rgb');masks,metadata=extract_stereo_masks(frames)
                (target/'foreground_masks').mkdir()
                for f,m,d in zip(frames,masks,metadata['cameras']):
                    cv2.imwrite(str(target/'foreground_masks'/(f.camera.name+'_foreground.png')),m.astype(np.uint8)*255)
                    row=next(r for r in rows if r['case']==name)
                    saved=row.get('diagnostics',{}).get('foreground',{}).get('cameras',[])
                    if saved:
                        expected=next(c['mask_sha256'] for c in saved if c['camera']==f.camera.name)
                        assert expected==d['mask_sha256']
                    checks.append(dict(layout=layout,mode=mode,case=name,camera=f.camera.name,mask_sha256=d['mask_sha256'],
                        source_rgb_sha256=d['rgb_sha256'],rgb_file_sha256=sha256((scene/'observations'/(f.camera.name+'_rgb.png')).read_bytes()).hexdigest(),
                        depth_file_sha256=sha256((scene/'observations'/(f.camera.name+'_depth_z_m.npz')).read_bytes()).hexdigest()))
            (destination/'settings.json').write_text(json.dumps(dict(preprocessing=dict(kind='frozen_structural_study',condition=f'{layout}/{mode}'),
                depth_display_range_m=[.18,.85] if layout=='close' else [.45,1.35]),indent=2)+'\n')
            datasets.append(dict(condition=f'{layout}/{mode}',label=('近' if layout=='close' else '远')+'·'+label,
                scenes=str(destination/'scenes'),records=str(destination/'records.json.gz'),method='portfolio'))
    (output/'datasets.json').write_text(json.dumps(datasets,ensure_ascii=False,indent=2)+'\n')
    (output/'validation.json').write_text(json.dumps(dict(source=str(source.resolve()),export_only=True,
        same_records=True,observations_symlinks=True,mask_checks=checks),indent=2)+'\n')
    print('exported',len(datasets),'datasets',len(checks),'mask/input checks',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run,a.output)
