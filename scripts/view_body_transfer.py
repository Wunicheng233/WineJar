#!/usr/bin/env python3
"""Existing MuJoCo sidebar/close-up/input viewer, body-transfer diagnostics."""
import argparse
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.pop('MUJOCO_GL',None)
import numpy as np
from PIL import Image,ImageDraw
from neck_localization.inspection_viewer import InspectionViewer,METHOD_LABELS
from neck_localization.experiment_io import read

# Display names in this diagnostic entry point only; original files/defaults
# and the estimator implementation remain untouched.
METHOD_LABELS.update(nominal='名义输送',axis='前向1D',translation='前向3D',rigid='前向6D',
                     observed_subset_axis='观测→模板1D',reciprocal_axis='互近邻1D')


class BodyTransferViewer(InspectionViewer):
    def _reset_camera(self):
        super()._reset_camera();self.camera.lookat[:]=[2.4,0,.35]

    def _text(self,image,items):
        layout=self._layout(image.shape[1],image.shape[0]);x=layout.main.x+18
        return super()._text(image,items+[(x,88,'扎口工位坛身确认；口颈已遮盖。右侧为实际保存的两路下游输入。',12,(241,208,114)),
            (x,110,'本页比较传递圆环；配准候选不是工艺放行，没有机械臂动作。',12,(177,192,211))])

    def _draw_sidebar(self,image,layout):
        image=super()._draw_sidebar(image,layout);canvas=Image.fromarray(image);draw=ImageDraw.Draw(canvas)
        h,w=layout.sidebar.height,layout.sidebar.width
        draw.rectangle((0,h-160,w-1,h-1),fill=(20,29,42))
        for y,text in [(h-140,'已生成配准候选' if self.row['accepted'] else '配准候选未形成'),
            (h-115,'不作为最终工艺放行'),(h-80,'本页点云：坛身观测'),(h-60,'不是颈部拟合点或最终内点'),(h-40,'红：传递估计  青：评估真值'),(h-20,f'圆心误差 {self.row["center_mm"]:.3f} mm')]:
            draw.text((10,y),text,font=self._font(11),fill=(177,201,218))
        return np.asarray(canvas)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--viewer',type=Path,default=ROOT/'experiments/neck_localization/results/pose_transfer_20261028_v3/viewer')
    p.add_argument('--condition',default='body_1');p.add_argument('--case',default='new8__confirmation_nominal_003')
    p.add_argument('--method',default='observed_subset_axis');p.add_argument('--ground-truth',action='store_true');p.add_argument('--point-cloud',action='store_true')
    p.add_argument('--snapshot',type=Path);p.add_argument('--hidden',action='store_true')
    a=p.parse_args();datasets=read(a.viewer/'datasets.json')
    for d,label in zip(datasets,['理想模板','双目模板','深度偏差3mm','窄区观测']):d['label']=label
    d=next(r for r in datasets if r['condition']==a.condition)
    v=BodyTransferViewer(d['scenes'],d['records'],a.case,method=a.method,datasets=datasets,
        ground_truth=a.ground_truth,point_cloud=a.point_cloud,visible=not a.hidden,overlay=True,band=True)
    try:
        if a.snapshot:a.snapshot.parent.mkdir(parents=True,exist_ok=True);v.snapshot(a.snapshot)
        else:v.run()
    finally:v.close()


if __name__=='__main__':main()
