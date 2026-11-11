# 原 MuJoCo 查看器：固定 SAM2 适配配对回放

本束独立展示同一批 RGB 和双目预测上的 SAM2 前景适配；原来的无参入口和 12 条件束保持不变。
请显式指定本束：

```bash
conda run --no-capture-output -n winejar-stereo \
  python /home/wunicheng/Git/WineJar/scripts/view_structural_study.py --ground-truth \
  --viewer /home/wunicheng/Git/WineJar/experiments/neck_localization/results/capability_structural_viewer_sam2_20261027_v1
```

默认是旧 12 坛中的 `structural_test_nominal_000`、近距 FS+SAM2、clean、visible_profile，三维显示实体。
侧栏“FS+SAM2 / 双目+SAM2”是旧 12 坛；“FS8+SAM2 / 双目8+SAM2”是已看新 8 坛。
共 8 条件、80 场景、240 份实际相机深度和 160 条保存读出。
两种方法都使用各自推理时保存的同一份 SAM2 RGB 前景，模型和定位器不会在查看时重算。
所有拟合点逐点核对为这些实际深度中的观测；前景不是网格或真值掩码。

SAM2 tiny 是现成权重和固定提示协议的普通适配；此处没有重新训练。
原 12 坛与新 8 坛均已看过，本束是配对诊断，不构成新的未见测试，也不意味着找到了论文创新。
默认四舍五入显示中心误差不会代替法向、拒绝和整带检查。

保留以下解释例：

| 条件/坛/visible_profile | 保存的结果 |
| --- | --- |
| 新 8/shared，`confirmation_nominal_004` | 已接受；中心 0.589 mm、法向 0.569°；此前 HSV 前景版本误拒绝 |
| 新 8/shared，`confirmation_nominal_005` | 已接受；中心 2.890 mm、法向 2.430°；3 mm 轴向带检查越界约 1.525 mm |
| 新 8/clean，`confirmation_nominal_006` | 中心 0.660 mm、法向 0.348°，仍被投影前景门槛拒绝；没有隐藏此例 |

这里的 2° 是研究分析线；整带结果仅为仿真轴向几何检查，不含接触、材料、工位传递或机器人执行。
点击“带宽边界”或按 `B` 可看黄色带边界及真值近最细范围；青虚线仅在打开评估真值时出现。

例如打开仍有整带问题的 `_005`：

```bash
conda run --no-capture-output -n winejar-stereo \
  python /home/wunicheng/Git/WineJar/scripts/view_structural_study.py --ground-truth \
  --viewer /home/wunicheng/Git/WineJar/experiments/neck_localization/results/capability_structural_viewer_sam2_20261027_v1 \
  --condition close/foundation_stereo_confirm_sam/shared_noise \
  --case confirmation_nominal_005 --method visible_profile
```

`validation.json` 保存导出来源、掩码/原始 RGB/深度散列及实际拟合点一致性检查。
`interaction/interaction_validation.json` 与 `interaction/*.png` 为实际 MuJoCo/GLFW 回调及呈现后 GL 缓冲区截图。
重做基本界面检查：

```bash
conda run --no-capture-output -n winejar-stereo \
  python /home/wunicheng/Git/WineJar/scripts/verify_capability_structural_viewer.py --sam2 \
  --bundle /home/wunicheng/Git/WineJar/experiments/neck_localization/results/capability_structural_viewer_sam2_20261027_v1 \
  --output /tmp/winejar_sam2_viewer_check_NEW
```

一次因缓存清单文件/目录接口差异中止的导出保留为旁边的 `*_interrupted_export`。
修复只影响读取清单的路径处理，未改任何估计、掩码或来源记录；本束重新完整导出后才作展示验证。
