#!/usr/bin/env python3
"""Publish saved 1/2/3-camera evidence and a self-contained two-case viewer.

Renders with the original MuJoCo InspectionViewer. No estimator reruns or
synthetic result lines: all results/rejections come from the frozen records.
Evaluation truth is used only for overlays and a shared inspection viewpoint.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.pop('MUJOCO_GL', None)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from neck_localization.inspection_viewer import InspectionViewer, read_records, FONT

CAMERAS=[('cam_a','单台 A'),('cam_a,cam_b','双台 A+B'),('all','三台 A+B+C')]
CASES=['nominal_014','stress_010']
METHOD='region_equal_trust'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def font(size):
    return ImageFont.truetype(str(FONT),size)


def run(scenes, records, output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    rows=read_records(records)
    assert len(rows)==216 and len({r['case'] for r in rows})==36
    exported=[]
    fields=['case','group','method','camera_set','accepted','truth_pass','center_mm',
            'normal_deg','diameter_mm','runtime_ms','normal_component_mm','tangent_mm','reasons']
    for r in rows:
        exported.append({k:r.get(k) for k in fields})
    write(output/'data/camera_count_records.json',exported)
    with (output/'data/camera_count_records.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows([{**r,'reasons':';'.join(r['reasons'])} for r in exported])
    summary=[]
    for group in ('nominal','stress'):
        for method in ('fixed_equal_trust','region_equal_trust'):
            for camera,label in CAMERAS:
                g=[r for r in rows if (r['group'],r['method'],r['camera_set'])==(group,method,camera)]
                summary.append(dict(group=group,method=method,camera_set=camera,total=len(g),
                    success=sum(r['accepted'] and r['truth_pass'] for r in g),
                    rejected=sum(not r['accepted'] for r in g),
                    false_accepted=sum(r['accepted'] and not r['truth_pass'] for r in g)))
    write(output/'data/camera_count_summary.json',summary)
    with (output/'data/camera_count_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)

    viewer=output/'viewer';viewer.mkdir()
    for case in CASES:
        src=scenes/case;dest=viewer/'scenes'/case;dest.mkdir(parents=True)
        for name in ('scene.xml','model_manifest.json'):
            shutil.copy2(src/name,dest/name)
        for sub in ('assets','observations','evaluation_only'):
            (dest/sub).mkdir()
        for p in (src/'assets').iterdir():
            if p.is_file():shutil.copy2(p,dest/'assets'/p.name)
        for p in (src/'observations').iterdir():
            if p.name.endswith('_rgb.png') or p.name.endswith('_depth_z_m.npz') or p.name=='camera_true_parameters.json':
                shutil.copy2(p,dest/'observations'/p.name)
        shutil.copy2(src/'evaluation_only/target.json',dest/'evaluation_only/target.json')
    subset=[r for r in rows if r['case'] in CASES]
    (viewer/'records.json.gz').write_bytes(gzip.compress(json.dumps(subset,ensure_ascii=False).encode(),mtime=0))
    settings=scenes.parent/'results/settings.json'
    if settings.exists():shutil.copy2(settings,viewer/'settings.json')

    for case in CASES:
        v=InspectionViewer(viewer/'scenes',viewer/'records.json.gz',case,method=METHOD,
                           visible=False,ground_truth=True,overlay=True,large_neck=True)
        try:
            truth=json.loads((viewer/'scenes'/case/'evaluation_only/target.json').read_text())
            # Identical evaluation viewpoint for all three inputs. Never fed
            # back to initialization, fitting or selection of predictions.
            v.neck_camera.lookat[:]=truth['center_m']
            v.neck_camera.distance=.25
            v.neck_camera.azimuth=130
            v.neck_camera.elevation=-15
            canvas=Image.new('RGB',(3200,1800),(14,22,35));d=ImageDraw.Draw(canvas)
            d.text((64,30),'单 / 双 / 三相机：同一酒坛的保存结果',font=font(50),fill=(240,246,255))
            d.text((64,104),f'{case}  ·  同一算法、同一观察角  ·  早期理想深度仿真',font=font(28),fill=(166,189,213))
            tile_w,tile_h=1008,540
            sample_metadata=[]
            for i,(camera,label) in enumerate(CAMERAS):
                x=64+i*1032;v.camera_set=camera;v._cloud_cache_key=None
                v.point_cloud=False
                model=v._render_view(v.neck_camera,v.neck_scene,tile_w,tile_h)
                canvas.paste(Image.fromarray(model),(x,220))
                row=v.row;accepted=row['accepted']
                d.text((x,165),label,font=font(36),fill=(221,234,249))
                status='接受输出' if accepted else '拒绝 / 没有可靠输出'
                d.text((x,782),status,font=font(30),fill=(119,222,172) if accepted else (255,185,111))
                if row['center_mm'] is not None:
                    d.text((x,831),f"圆心 {row['center_mm']:.3f} mm   法向 {row['normal_deg']:.3f}°",font=font(28),fill=(225,234,245))
                else:
                    d.text((x,831),'原因：核心范围触及候选边界',font=font(27),fill=(212,192,170))
                v.point_cloud=True;v.cloud_crop=True;v.cloud_source_colors=True
                cloud=v._render_view(v.neck_camera,v.neck_scene,tile_w,tile_h)
                canvas.paste(Image.fromarray(cloud),(x,980))
                d.text((x,923),'实际参与的观测点云',font=font(30),fill=(200,219,240))
                sample_metadata.append(dict(camera_set=camera,accepted=accepted,
                    center_mm=row['center_mm'],normal_deg=row['normal_deg'],reasons=row['reasons'],
                    observation_points_before_display_thinning=int(v._cloud_data[2])))
            d.text((64,1560),'红色细线：已接受的估计圆 / 法向    青色虚线：评估真值    单机无输出时不补画估计',font=font(28),fill=(199,219,238))
            d.text((64,1615),'点云来源：A 蓝色 · B 绿色 · C 金色；仅保存的RGB前景与有效深度，不使用网格补点。',font=font(27),fill=(163,190,213))
            d.text((64,1670),'仅用于说明早期算法的输入与结果；不是最新算法、真实相机或扎带成功率证明。',font=font(27),fill=(243,196,119))
            dest=output/'camera_count'/f'{case}_comparison.png';dest.parent.mkdir(exist_ok=True)
            canvas.save(dest)
            write(dest.with_suffix('.json'),dict(case=case,source_records_sha256=digest(records),
                same_viewpoint_for_all_camera_sets=True,truth_for_display_only=True,
                no_estimator_rerun=True,method=METHOD,results=sample_metadata))
        finally:v.close()

    # A chart of all cases, not only the two illustrations.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    fp=FontProperties(fname=str(FONT))
    fig,axes=plt.subplots(1,2,figsize=(14,7),sharey=True)
    for ax,group,title in zip(axes,['nominal','stress'],['常规假设范围（24例）','压力范围（12例）']):
        selected=[next(r for r in summary if (r['group'],r['method'],r['camera_set'])==(group,METHOD,c)) for c,_ in CAMERAS]
        bars=ax.bar(range(3),[100*r['success']/r['total'] for r in selected],color=['#588fd2','#51a688','#e3b25e'])
        ax.set_xticks(range(3),[label for _,label in CAMERAS],fontproperties=fp)
        ax.set_title(title,fontproperties=fp,fontsize=16);ax.set_ylim(0,113)
        ax.set_ylabel('通过率 (%)',fontproperties=fp);ax.spines[['top','right']].set_visible(False)
        for bar,row in zip(bars,selected):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f"{row['success']}/{row['total']}",ha='center',fontsize=16)
    fig.suptitle('早期理想仿真：相机数量主要影响有效输出率',fontproperties=fp,fontsize=20)
    fig.text(.5,.04,'成功 = 算法接受且圆心≤4 mm、法向≤2°、直径误差≤2 mm；后两项为当时分析阈值。\n36例均保留，包含当前工况以外的颈长；不等于真实工厂或最终扎带成功率。',ha='center',fontproperties=fp,fontsize=11)
    fig.tight_layout(rect=[0,.14,1,.93]);fig.savefig(output/'camera_count/all_36_cases.png',dpi=220);plt.close(fig)
    manifest=dict(created='2026-11-11',study_date='2026-10-23',no_new_estimation=True,
        source_records=str(records.relative_to(ROOT)),source_records_sha256=digest(records),
        illustrative_cases=CASES,case_selection='nominal_014: accepted in all sets; stress_010: single rejected, double/triple accepted; both physical necks 6–10 mm',
        all_216_records_exported_without_fit_point_diagnostics=True,
        images_are_simulation_not_factory_measurements=True,
        portable_viewer_contains_real_files_no_symlinks=True)
    manifest['asset_sha256']={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    write(output/'manifest.json',manifest)
    print('exported',len(rows),'records, 3 comparison figures, portable cases:',','.join(CASES))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes',type=Path,default=ROOT/'experiments/neck_localization/runs/random_geometry_audit_20261023_v1/scenes')
    p.add_argument('--records',type=Path,default=ROOT/'experiments/neck_localization/results/random_geometry_audit_20261023/records.json.gz')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.scenes.resolve(),a.records.resolve(),a.output.resolve())
