# 原 MuJoCo 查看器：成熟方法能力审查

直接使用原入口：

```bash
conda run --no-capture-output -n winejar-stereo \
  python /home/wunicheng/Git/WineJar/scripts/view_structural_study.py --ground-truth
```

默认显示近距 FoundationStereo、clean 输入、`visible_profile` 修正版几何读出。
这只是优先展示顺序，不表示质量判断已经可靠。
默认三维视图显示实体；算法按钮紧随条件按钮，960 像素高的默认窗口中四方法均可直接点击。
侧栏可切换旧 12 坛的远/近布局、两种双目前端、clean/shared_noise，以及新 8 坛的近距对照。
`foundation_stereo_confirm` 与 `simsense_confirm` 是新 8 坛路由。

共有 12 个条件、128 个场景、384 份相机深度和 640 条保存读出。
旧 12 坛有 `portfolio`、`stereo_robust`、`visible_profile`、`standard_sor`；
新 8 坛有后两种。当前 bundle 使用本轮对应的 RGB 蓝幕前景，未替换为 SAM2。
蓝色陶瓷反光造成的前景漏检会误拒绝好候选，因此不能用这些拒绝证明学习方法失败。

“实体 / 全坛点云 / 拟合点”分别显示模型、从实际 RGB-D 提取的观测、保存的实际拟合支持；
没有保存拟合点时显示空集。权重与拟合像素来自原读出记录。
红线表示被算法接受的估计，橙虚线表示被拒绝的候选，青虚线仅为评估真值。
估计线是三维结果投影，不是新检测到的 RGB 边缘。
按 `M` 换方法、方向键换坛，`D` 换 RGB/深度，`P/L/W` 看全点云/拟合点/权重，`G` 开关评估真值；
侧栏可滚动，近景与全景可以独立旋转。

以下是后验选择的解释例，不能另算一批测试：

| 条件/案例/方法 | 需观察的问题 |
| --- | --- |
| 近 FS/shared，`structural_test_stress_001`，standard_sor | 已接受，但圆心误差 15.815 mm、法向 11.763° |
| 新 8 FS/shared，`confirmation_nominal_004`，visible_profile | 圆心误差 0.667 mm、法向 0.710°，被投影前景门槛拒绝 |
| 新 8 FS/shared，`confirmation_nominal_005`，visible_profile | 已接受；圆心 2.842 mm，法向 2.512°，超过研究用 2° 线 |

例如直接打开新 8 坛的 `_005`：

```bash
conda run --no-capture-output -n winejar-stereo \
  python /home/wunicheng/Git/WineJar/scripts/view_structural_study.py \
  --ground-truth --condition close/foundation_stereo_confirm/shared_noise \
  --case confirmation_nominal_005 --method visible_profile
```

显式保留此前结构研究：

```bash
conda run --no-capture-output -n winejar-stereo \
  python /home/wunicheng/Git/WineJar/scripts/view_structural_study.py \
  --ground-truth --viewer /home/wunicheng/Git/WineJar/experiments/neck_localization/results/structural_stage_20261025_viewer
```

`validation.json` 核对来源散列、384 份相机深度、711504 个保存拟合点与实际输入深度的逐点一致性，
并确认科学记录仅增加查看器路由字段。`interaction/interaction_validation.json` 记录实际 MuJoCo/GLFW 回调验证；
`interaction/*.png` 为呈现后的 GL 缓冲区截图。重做界面检查可运行：

```bash
conda run --no-capture-output -n winejar-stereo \
  python /home/wunicheng/Git/WineJar/scripts/verify_capability_structural_viewer.py \
  --output /tmp/winejar_capability_viewer_check_NEW
```

查看与导出均不重跑模型/定位器、不连接设备。这些仍是圆截面仿真族和准确标定输入，不能当作实机验收。
