#!/usr/bin/env python3
"""Postprocess a frozen random audit: all cases, worst cases and review gallery."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from neck_localization.geometry import Camera
from neck_localization.evaluation import summarize
from neck_localization.inspection_viewer import draw_result,ESTIMATE_COLOR,TRUTH_COLOR


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    args=p.parse_args()
    out=args.audit/'report';out.mkdir(exist_ok=False)
    rows=json.loads(gzip.decompress((args.audit/'results/records.json.gz').read_bytes()))
    lines=['# 冻结算法的随机几何审计','',
        '24例假设常规范围、12例压力范围；不是实测工厂分布。固定相机、准确K/T、理想深度。',
        '每例两方法×三视图组合；全部尝试计入分母，误差包含已求出但拒绝的候选，未求出不填零。',
        '单元格为中位数 / P95 / 最大；4 mm/2°/直径2 mm为分析阈值，后两项尚未工艺定案。','',
        '|组|相机|方法|成功/拒绝/错误接受/总数|圆心mm|法向°|直径mm|耗时ms|',
        '|---|---|---|---|---|---|---|---|']
    for s in summarize(rows):
        group=[r for r in rows if r['split']==s['split'] and r['method']==s['method'] and r['factor']==s['factor']]
        counts=[sum(r['accepted'] and r['truth_pass'] for r in group),sum(not r['accepted'] for r in group),sum(r['accepted'] and not r['truth_pass'] for r in group),len(group)]
        cells=[]
        for m in ['center_mm','normal_deg','diameter_mm','runtime_ms']:
            cells.append(' / '.join(f'{s[m+"_"+q]:.3f}' if s[m+'_'+q] is not None else '—' for q in ['median','p95','max']))
        lines.append('|'+group[0]['group']+'|'+group[0]['camera_set']+'|'+s['method']+'|'+'/'.join(map(str,counts))+'|'+'|'.join(cells)+'|')
    lines += ['', '## 拒绝和错误接受','']
    for r in rows:
        if not r['accepted'] or not r['truth_pass']:
            lines.append(f'- {r["case"]} / {r["camera_set"]} / {r["method"]}: '+
                ('拒绝 '+','.join(r['reasons']) if not r['accepted'] else '错误接受')+
                f'; center={r["center_mm"]}, normal={r["normal_deg"]}, diameter={r["diameter_mm"]}')
    index={}
    for group in ['nominal','stress']:
        for cs in ['cam_a','cam_a,cam_b','all']:
            candidates=[r for r in rows if r['group']==group and r['camera_set']==cs and r['method']=='region_equal_trust']
            accepted=[r for r in candidates if r['accepted'] and r['center_mm'] is not None]
            index[group+'/'+cs]=dict(
                worst_accepted_center=max(accepted,key=lambda r:r['center_mm'])['case'] if accepted else None,
                worst_accepted_normal=max(accepted,key=lambda r:r['normal_deg'])['case'] if accepted else None,
                rejected=[r['case'] for r in candidates if not r['accepted']],
                false_accepted=[r['case'] for r in candidates if r['accepted'] and not r['truth_pass']])
    (out/'review_index.json').write_text(json.dumps(index,indent=2)+'\n')
    lines += ['', '精确参数见config.json和scenes/*/model_manifest.json；按结果分类的查看索引见review_index.json。',
              '图库红线为区域策略三视图估计，青色虚线为评估真值；没有结果时只画真值并标拒绝。',
              '本次只审计现有圆截面模型家族，不包含椭圆、不规则凹槽、真实反光或深度偏差。',
              '样本来自假设随机分布，经验P95与零失败均不能作为工厂性能保证。']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    cases=sorted({r['case'] for r in rows})
    tile_w,tile_h=320,295
    gallery=Image.new('RGB',(tile_w*6,tile_h*((len(cases)+5)//6)),(22,29,40))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    for i,case in enumerate(cases):
        row=next(r for r in rows if r['case']==case and r['camera_set']=='all' and r['method']=='region_equal_trust')
        folder=args.audit/'scenes'/case
        img=cv2.cvtColor(cv2.imread(str(folder/'review/neck.png')),cv2.COLOR_BGR2RGB)
        model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        cid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,'review_neck')
        h,w=img.shape[:2]; focal=h/(2*np.tan(np.deg2rad(model.cam_fovy[cid])/2))
        T=np.eye(4);T[:3,:3]=data.cam_xmat[cid].reshape(3,3) @ np.diag([1,-1,-1]);T[:3,3]=data.cam_xpos[cid]
        camera=Camera('review_neck',np.array([[focal,0,(w-1)/2],[0,focal,(h-1)/2],[0,0,1]]),T,w,h)
        if row.get('estimate') is not None:
            draw_result(img,camera.project,row['estimate'],ESTIMATE_COLOR)
        # Truth used only in this human review gallery AFTER estimation.
        truth=json.loads((folder/'evaluation_only/target.json').read_text())
        draw_result(img,camera.project,truth,TRUTH_COLOR,dashed=True)
        tile=Image.new('RGB',(tile_w,tile_h),(22,29,40));tile.paste(Image.fromarray(img).resize((320,240)),(0,0))
        d=ImageDraw.Draw(tile)
        status='ACCEPT' if row['accepted'] else 'REJECT'
        if row['accepted'] and not row['truth_pass']:status='FALSE ACCEPT'
        d.text((5,242),case+'  '+status,font=font,fill=(245,210,130))
        metrics=(f'C={row["center_mm"]:.2f}mm  N={row["normal_deg"]:.2f}deg' if row['center_mm'] is not None else 'No reliable candidate')
        d.text((5,260),metrics,font=font,fill=(220,230,240))
        gallery.paste(tile,((i%6)*tile_w,(i//6)*tile_h))
    gallery.save(out/'gallery.png')
    print(out)


if __name__=='__main__':main()
