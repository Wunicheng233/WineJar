#!/usr/bin/env python3
"""Original MuJoCo scene/navigation with actual simulated tool-arrival overlay."""
import argparse
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.pop('MUJOCO_GL',None)
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from neck_localization.tying_viewer import TyingViewer
from neck_localization.inspection_viewer import InspectionViewer,draw_result,project_scene,METHOD_LABELS
from neck_localization.experiment_io import read

METHOD_LABELS.update(pixel_huber='原像素Huber',uncertain_vce='不确定性标定',camera_poe='关节映射',camera_poe_block='关节映射·整图加权',
                     park='Park闭式法',fixed_block='固定外参·整图加权')


class ArrivalViewer(TyingViewer):
    def _apply_stage(self):
        self.clock_s=23.
        super()._apply_stage()
        if 'actual_joint_coordinates' in self.row:
            self.data.qpos[self.joint_addresses]=self.row['actual_joint_coordinates']
            mujoco.mj_forward(self.model,self.data)
            i=self.model.site('guide_tcp').id
            actual=np.array(self.row['calibration_arrival']['actual_guide_W'])
            np.testing.assert_allclose(self.data.site_xpos[i],actual[:3,3],atol=1e-7)
            np.testing.assert_allclose(self.data.site_xmat[i].reshape(3,3),actual[:3,:3],atol=1e-7)

    def _draw_annotations(self,image,scene=None,camera=None):
        super()._draw_annotations(image,scene,camera)
        if camera is None:
            T=np.array(self.row['calibration_arrival']['actual_guide_W'])
            project=lambda p:project_scene(scene,p,image.shape[1],image.shape[0])
            draw_result(image,project,dict(center_m=T[:3,3],normal=T[:3,2],radius_m=.075),(255,198,80),arrow=True)

    def _draw_sidebar(self,image,layout):
        image=InspectionViewer._draw_sidebar(self,image,layout)
        canvas=Image.fromarray(image);draw=ImageDraw.Draw(canvas);w=layout.sidebar.width;h=layout.sidebar.height
        draw.rectangle((0,72,w-7,94),fill=(20,29,42))
        draw.text((12,74),'标定方案对照（同一输入）',font=self._font(12),fill=(177,192,211))
        draw.rectangle((0,h-160,w-1,h-1),fill=(20,29,42))
        r=self.row['calibration_arrival']
        for y,t in [(h-142,'数值候选，未作工艺放行'),(h-115,'本页仅显示静态枪头到位'),
                    (h-88,'拟合点/右图均为识别t0'),(h-61,'不重新观测被盖住的颈部'),
                    (h-34,f"枪头圆心误差 {r['center_mm']:.3f} mm")]:
            draw.text((12,y),t,font=self._font(12),fill=(241,208,114))
        return np.asarray(canvas)

    def _text(self,image,items):
        layout=self._layout(image.shape[1],image.shape[0]);x=layout.main.x+18;r=self.row['calibration_arrival']
        return InspectionViewer._text(self,image,items+[
            (x,91,'静态到位诊断：假设的真实连杆/关节偏差；未模拟运动过程或实机',12,(241,208,114)),
            (x,117,f"枪头圆心误差 {r['center_mm']:.3f}mm  方向 {r['normal_deg']:.3f}°  带余量 {r['band_margin_mm']:+.3f}mm",13,(255,198,80)),
            (x,141,'红：视觉目标圆  青：颈部真值  黄：枪头实际内圆/轴',12,(177,192,211)),
            (layout.inputs.x+12,image.shape[0]-18,'右侧：识别t0保存输入，未重新看到遮盖后的颈部',11,(177,192,211))])

    def _activate(self,action):
        if action.startswith('stage:'):return
        super()._activate(action)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--viewer',type=Path,default=ROOT/'experiments/neck_localization/results/uncertain_transfer_20261029_v1/arrival_viewer_v2')
    p.add_argument('--method',default='camera_poe_block');p.add_argument('--case',default='new8__confirmation_nominal_003')
    p.add_argument('--snapshot',type=Path);p.add_argument('--hidden',action='store_true');p.add_argument('--ground-truth',action='store_true');p.add_argument('--large-neck',action='store_true')
    a=p.parse_args();datasets=read(a.viewer/'datasets.json');d=next(x for x in datasets if x['method']==a.method)
    v=ArrivalViewer(d['scenes'],d['records'],a.case,method=a.method,datasets=datasets,station_config=read(a.viewer/'config.json'),
        visible=not a.hidden,ground_truth=a.ground_truth,overlay=True,band=False,large_neck=a.large_neck)
    try:
        if a.snapshot:a.snapshot.parent.mkdir(parents=True,exist_ok=True);v.snapshot(a.snapshot)
        else:v.run()
    finally:v.close()


if __name__=='__main__':main()
