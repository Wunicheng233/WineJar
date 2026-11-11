#!/usr/bin/env python3
"""Audit complete pose chains; separate transport error from neck estimation."""
import argparse
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from neck_localization.experiment_io import read,write,digest,quantiles
from neck_localization.station_tying import pose,frame,move_circle,GunSpec
from neck_localization.fixed_camera_calibration import projection
from scripts.run_fixed_camera_calibration_study import dataset


def run(bundle,calibration):
    output=bundle/'analysis';output.mkdir(exist_ok=True)
    cal=read(calibration/'records.json');protocol=read(calibration/'protocol.json')
    base=ROOT/protocol['baseline'];old=ROOT/read(base/'protocol.json')['source_bundle']
    gun=GunSpec(**read(old/'config.json')['gun']);actual_mount=gun.T_F_G
    chain=[]
    for r in cal:
        if 'B_T_C' not in r:continue
        d=dataset(protocol,r['session'],'diverse',r['condition'])
        X=np.array(r['B_T_C']);Y=np.array(r['F_T_P']);trueX=d['truth_X'];trueY=d['truth_Y']
        effectiveX=np.linalg.inv(d['gauge_relabel']) @ X
        P_T_G=np.linalg.inv(trueY) @ actual_mount
        # This relation is a declared exact metrology prior ONLY for diagnosis;
        # in hardware it needs a rigid measured calibration fixture.
        estimated_mount=Y @ P_T_G
        n=trueX[:3,:3].T @ [0,0,1]
        errors={name:[] for name in ['camera_only_exact_tool_diagnostic','joint_calibration_exact_board_to_gun_jig']}
        angles={name:[] for name in errors}
        for p in d['query_points_camera']:
            C_T_G=pose(frame(n),p);wanted=trueX @ C_T_G
            for name,mount in [('camera_only_exact_tool_diagnostic',actual_mount),('joint_calibration_exact_board_to_gun_jig',estimated_mount)]:
                actual=effectiveX @ C_T_G @ np.linalg.inv(mount) @ actual_mount
                errors[name].append(np.linalg.norm(actual[:3,3]-wanted[:3,3])*1000)
                angles[name].append(np.rad2deg(np.arccos(np.clip(actual[:3,2] @ wanted[:3,2],-1,1))))
        diverse_holdout=np.sqrt(np.mean([(projection(np.linalg.inv(X) @ B @ Y,d['board'],d['K'])-uv)**2
            for B,uv in zip(d['robot'][18:],d['pixels'][18:])]))
        chain.append(dict(layout=r['layout'],condition=r['condition'],session=r['session'],method=r['method'],accepted=r['accepted'],
            training_pixel_rms=r['pixel_rms'],same_family_holdout_pixel_rms=r['holdout_pixel_rms'],diverse_holdout_pixel_rms=float(diverse_holdout),
            **{name:dict(max_center_mm=float(max(errors[name])),max_normal_deg=float(max(angles[name]))) for name in errors},
            covariance_correlation_preserved_by_same_joint_estimate=True,not_real_tcp_calibration=True))
    write(output/'calibration_full_chain.json',chain)
    cal_summary=[]
    for layout in protocol['calibration']['layouts']:
        for condition in protocol['calibration']['conditions']:
            for method in protocol['calibration']['methods']:
                rows=[r for r in chain if (r['layout'],r['condition'],r['method'])==(layout,condition,method)]
                if not rows:continue
                cal_summary.append(dict(layout=layout,condition=condition,method=method,
                    **{key:quantiles([r[key] for r in rows]) for key in ['training_pixel_rms','same_family_holdout_pixel_rms','diverse_holdout_pixel_rms']},
                    **{name:quantiles([r[name]['max_center_mm'] for r in rows]) for name in ['camera_only_exact_tool_diagnostic','joint_calibration_exact_board_to_gun_jig']}))
    write(output/'calibration_full_chain_summary.json',cal_summary)
    rows=read(bundle/'body/records.json')+read(bundle/'body/partial_view_ablation/records.json')
    initial=read(base/'band_controls.json')['rows']
    for r in rows:
        original=next(x['estimate'] for x in initial if (x['cohort'],x['case'],x['condition'],x['method'])==(r['cohort'],r['case'],'clean','pose_fs_radius'))
        R=Rotation.from_euler('y',1,degrees=True).as_matrix() if r['condition']=='rotation_mismatch_1deg' else np.eye(3)
        M=pose(R,[2.4+r['stop_error_mm']/1000,0,0])
        expected=move_circle(original,M)
        # Exact *transport of the original estimate* separates additional
        # relocation error from the retained neck-estimation error.
        r['transport_only_center_mm']=float(np.linalg.norm(np.array(r['estimate']['center_m'])-expected['center_m'])*1000)
        r['transport_only_normal_deg']=float(np.rad2deg(np.arccos(np.clip(np.array(r['estimate']['normal']) @ expected['normal'],-1,1))))
    write(output/'body_with_transport_diagnostics.json',rows)
    summaries=[]
    for condition in sorted({r['condition'] for r in rows}):
        for source in sorted({r['source_control'] for r in rows}):
            for method in sorted({r['method'] for r in rows}):
                group=[r for r in rows if (r['condition'],r['source_control'],r['method'])==(condition,source,method)]
                accepted=[r for r in group if r['accepted']]
                summaries.append(dict(condition=condition,source_control=source,method=method,total=len(group),accepted=len(accepted),
                    center_false_accepts=sum(r['accepted'] and r['center_mm']>4 for r in group),
                    center_false_rejects=sum(not r['accepted'] and r['center_mm']<=4 for r in group),
                    accepted_band_contained=sum(r['accepted'] and r['band_contained'] for r in group),
                    **{k:quantiles([r[k] for r in accepted]) if accepted else None for k in ['center_mm','normal_deg','tangent_mm','transport_only_center_mm','transport_only_normal_deg','elapsed_ms']},
                    final_fastening_success=None))
    write(output/'body_summary.json',summaries)
    with (output/'body_summary.csv').open('w',newline='') as f:
        fields=['condition','source_control','method','total','accepted','center_false_accepts','center_false_rejects','accepted_band_contained','center_p95_mm','transport_only_p95_mm','normal_p95_deg','elapsed_p95_ms']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in summaries:
            w.writerow({k:r[k] for k in fields[:8]}|dict(center_p95_mm=None if r['center_mm'] is None else r['center_mm']['p95'],
                transport_only_p95_mm=None if r['transport_only_center_mm'] is None else r['transport_only_center_mm']['p95'],
                normal_p95_deg=None if r['normal_deg'] is None else r['normal_deg']['p95'],elapsed_p95_ms=None if r['elapsed_ms'] is None else r['elapsed_ms']['p95']))
    write(output/'retained_worst_cases.json',sorted(rows,key=lambda x:x['center_mm'],reverse=True)[:20])
    plot(bundle,summaries,calibration,output)
    write(output/'provenance.json',dict(script_sha256=digest(__file__),body_records_sha256=digest(bundle/'body/records.json'),
        calibration_records_sha256=digest(calibration/'records.json'),no_new_fitting_or_selection=True))
    print('pose transfer summary complete',len(rows),'body records')


def plot(bundle,body,calibration,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties,fontManager
    f='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fontManager.addfont(f);plt.rcParams['font.family']=FontProperties(fname=f).get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(1,3,figsize=(14,4.8),constrained_layout=True)
    raw=read(calibration/'summary.json');methods=['park','horaud','pixel_ls','pixel_huber'];labels=['Park','Horaud','像素联合LS','像素联合Huber']
    for condition,label in [('nominal','假设随机误差'),('one_bad_image','一张异常图')]:
        q=[next(r for r in raw if (r['layout'],r['condition'],r['method'])==('diverse',condition,m)) for m in methods]
        axs[0].plot(labels,[r['physical_query_max_mm']['p95'] for r in q],'o-',label=label)
    axs[0].set_title('固定相机标定：独立点误差P95 / mm');axs[0].legend(fontsize=9)
    names=['nominal','axis','translation','rigid','observed_subset_axis','reciprocal_axis'];short=['名义输送','前向1D','前向3D','前向6D','观测→模板1D','互为最近邻1D']
    for source,label in [('mujoco_ideal_body_diagnostic','理想坛身模板'),('saved_simsense_body','保存双目坛身模板')]:
        q=[next(r for r in body if (r['condition'],r['source_control'],r['method'])==('ideal',source,m)) for m in names]
        axs[1].plot(short,[r['transport_only_center_mm']['p95'] for r in q],'o-',label=label)
    axs[1].set_yscale('symlog',linthresh=1);axs[1].set_title('新增传递误差P95 / mm（理想下游深度）');axs[1].legend(fontsize=8)
    for condition,label in [('ideal','理想下游深度'),('depth_bias_3mm','假设深度偏差+3mm'),('local_extrinsic_1mm','假设局部标定偏差1mm')]:
        q=[next(r for r in body if (r['condition'],r['source_control'],r['method'])==(condition,'saved_simsense_body',m)) for m in ['nominal','axis','observed_subset_axis','reciprocal_axis']]
        axs[2].plot(['名义输送','前向1D','观测→模板1D','互为最近邻1D'],[r['center_mm']['p95'] for r in q],'o-',label=label)
    axs[2].set_title('实际圆心误差P95 / mm（保留颈部误差）');axs[2].legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=20)
    fig.suptitle('标定与坛身确认：成熟方法配对对照；11只已看模型，全部扰动幅度为假设',fontsize=13)
    fig.savefig(output/'engineering_comparison.png',dpi=160);fig.savefig(output/'engineering_comparison.svg');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    a=p.parse_args();run(a.bundle.resolve(),a.calibration.resolve())
