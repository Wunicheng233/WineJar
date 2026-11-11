# 作业区验证配对查看束

沿用原MuJoCo；不重算定位、不连设备。`fs_old8`是旧开发8例，`fs_test8`/`ss_test8`是10-28新锁定8例。每种输入含原`visible_profile`、新`task_surface`、删除RGB的消融`no_rgb_gate`、强对照`standard_sor`。默认方案未换。

```bash
conda run --no-capture-output -n winejar-stereo \
 python /home/wunicheng/Git/WineJar/scripts/view_structural_study.py --ground-truth \
 --viewer /home/wunicheng/Git/WineJar/experiments/neck_localization/results/task_surface_viewer_20261028_v1 \
 --condition close/fs_old8/clean --case confirmation_nominal_006 --method task_surface
```

点击“可见母线拟合”对照旧误拒。查看新带失败：`--condition close/fs_test8/shared_noise --case confirmation_nominal_001`，左栏开“带宽边界”。接受仅定位层状态，不能当作可执行扎口。

验证含48场景/144实际RGB-D、192保存结果和实际拟合点逐点核对；`validation.json`为导出审计。`interaction/validation.json`为6条件×4算法的实际GL/注入鼠标回调、四张截图与输入不变检查，窗口已关闭。

查看器只有原交互与细线叠加；不会伪造点云、用真值点代替观察或修改估计。路径是绝对路径/符号链接；迁移需同时携带所引用runs或重建导出。
