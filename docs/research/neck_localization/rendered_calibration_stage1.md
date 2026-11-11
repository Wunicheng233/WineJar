# 实际渲染棋盘的 RGB 标定（2026-10-25）

本步已经跑通：**真实渲染棋盘 → OpenCV 图像角点 → RGB 内参 → 已知世界夹具的相机外参**。
三台相机共 63 张图全部实际检测成功。正式记录为
`experiments/neck_localization/runs/rendered_calibration_20261025_v3/`。

这补上了 RGB 相机参数求解，但**不是整个 RGB-D 系统全部标定**：IR 内参、IR–RGB 变换和深度链路仍用传感器预设；
标定板平面度、格子尺寸和夹具世界位姿没有附加误差，均为理想已知参考。

## 使用的成熟方法与实际输入

采用 [OpenCV 官方标准标定接口](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)：
`findChessboardCornersSB` 直接提取亚像素角点，`calibrateCamera` 求内参，`solvePnP` 求板到相机变换。
没有用投影点替换检测点，没有把真内参或板姿态传给内参优化初始化。

- 棋盘：9×6 内角点、25 mm 格子、白边和红/绿方向标记；尺寸是本实验定义，不是现场板测量值。
- 每机 15 张训练图：距离约 0.52–1.02 m，位置、倾角、面内旋转按种子 `2026102551` 事先固定。
- 每机 5 张新板姿态留出验证；不参与 `calibrateCamera`。
- 每机另 1 张固定世界参考面图用于外参；可理解为理想测量夹具上朝向三台相机的三个固定棋盘面。
  各面的世界位姿被视作独立测量后已知，不声称从无基准图像恢复世界坐标。
- 棋盘黑白格由 SAPIEN 实际几何光栅化产生 RGB。当前没有额外镜头畸变，所以标定明确固定径向/切向畸变为零；
  焦距和主点由检测数据求解，不固定为真值。

角点排序也来自图像：检测红/绿印刷标记像素中心，以已知印刷位置的单应性比较四种行列翻转，
恢复统一棋盘坐标。排序函数没有 K、相机真姿态或理想角点输入，避免棋盘对称性造成外参 180° 翻转。

## 坐标和真值隔离

单位米。列向量变换 `T_world_from_camera` 将相机坐标变到世界；相机采用 OpenCV 右/下/前。
SAPIEN 使用前/左/上，相机旋转明确转换，并逐次核对渲染器 model matrix。
SAPIEN 的投影 K 以像素角为原点，图像数组按整数像素中心：

\[
c_{x,OpenCV}=c_{x,SAPIEN}-0.5,\qquad c_{y,OpenCV}=c_{y,SAPIEN}-0.5.
\]

实际检测对真投影的整体 RMS 约 **0.111 px**，两轴平均偏差约 **(+0.005, 0.000) px**；
若漏掉上述半像素转换，对照 RMS 约 **0.712 px**。这项仅为坐标审计，未用于替换检测点。

内参求解只接收棋盘米制点和真实检测像素，`CALIB_USE_INTRINSIC_GUESS` 没有启用。
外参采用固定夹具给出的 `T_world_from_board`：

\[
T_{world\leftarrow camera}
=T_{world\leftarrow board}\,(T_{camera\leftarrow board})^{-1}.
\]

板真实姿态、理想角点、真实相机参数另外存放在 `evaluation_only.json` 和 `evaluation_only/`，只用于评估；
固定参考面位姿则是明确允许的独立理想夹具输入。棋盘自身尺寸也是标定的必要已知量。

## 正式结果

| 相机 | fx / fy 误差 | 主点 cx / cy 误差 | 训练重投影 RMS | 固定参考求相机位置误差 | 旋转误差 |
|---|---|---|---|---|---|
| A | +3.292 / +3.380 px | +0.168 / +0.101 px | 0.0989 px | 3.967 mm | 0.0800° |
| B | +3.164 / +3.241 px | +0.177 / +0.294 px | 0.0989 px | 3.818 mm | 0.0806° |
| C | +3.272 / +3.362 px | +0.158 / +0.106 px | 0.0988 px | 3.944 mm | 0.0789° |

真焦距约 605 px，估计约 608 px。固定最终世界外参后，新板姿态上的重投影 RMS 最大约 0.205 px。
若用每张新板独立求 PnP，位置误差约 3.14–6.25 mm；这不是把各帧平均后的标定结果。

**低重投影残差不能当作绝对精度。** 焦距与估计板距离/相机位置可以互相补偿，使图像投影仍然很好，
但把深度点转到世界时仍有误差。作为明确的理想诊断，用名义目标附近 27 个精确像素/精确 Z 点反投影，
三机最大误差约 4.014、3.866、3.948 mm；这还没有图像分割、深度测量误差，不是酒坛定位测试。
真正的任务影响需在同一批新酒坛观测上比较真参数和本次估计参数。

三台相机使用同一传感器模型与相同的相对板采样计划，所以采样误差高度相关；
不能把 3 组相近结果当成 3 个独立真实设备的统计结论。当前只有一个固定标定方案，没有作完整标定姿态优化。

## 调试证据与已有实现问题

保留 `v1` 和 `v2` 记录作为开发诊断：

- `v1`：对高对比光栅图叠加直方图均衡 `NORMALIZE_IMAGE` 后，SB 检测失败。同一保存图用 SB 不均衡即可检测，
  所以正式设置使用 `EXHAUSTIVE | ACCURACY`，不均衡图像。
- `v2`：SB 后额外调用 `cornerSubPix`，把例如 0.130 px 的角点误差放大到 0.465 px；
  标定焦距偏差约 −6 px，相机位置误差约 7.9 mm。
- `v3`：按照 OpenCV 文档保留 SB 的直接亚像素输出。文档明确说明 SB 自带亚像素定位，额外梯度细化不是必要步骤。
  这解决的是实现错误，不能把 `v2` 的退步归因于成熟标定方法本身。

正式主流程和参数在新酒坛测试前冻结；上述开发调整没有查看新酒坛的任务结果。
三机 63 次检测是无额外噪声渲染条件下的检测成功率，不代表工厂标定或生产识别成功率。

## 复现、结果文件和验证

```bash
cd /home/wunicheng/Git/WineJar
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
conda run --no-capture-output -n winejar-stereo \
  python scripts/run_rendered_calibration.py \
  --output experiments/neck_localization/runs/NEW_rendered_calibration

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
conda run --no-capture-output -n winejar-stereo \
  python -m unittest discover -s tests/neck_localization -p test_rendered_calibration.py -v
```

4 项通过：真实图像 SB 检测/标记排序、四种角点翻转、变换方向、无真值初始化的内参求解诊断。
集成运行核查了 63 张实际渲染图及所有 model matrix；不声称已第二遍完整复跑。

正式运行目录内：

- `estimated_camera_parameters.json`：根实验可直接读取的 RGB K、零畸变与 `T_world_from_camera`。
- `evaluation_only/camera_true_parameters.json`：真实相机参数，仅评估。
- `<camera>/<frame>/rgb.png`、`detected_corners.npy`、`detected_overlay.png`：实际图像和实际角点。
- `<camera>/<frame>/evaluation_only.json`：真实板位姿及理想角点，仅评估。
- `calibration_report.json`、`plan.json`、`provenance.json`：完整指标、姿态计划和源码哈希。

小汇总和预览位于 `experiments/neck_localization/results/rendered_calibration_20261025/`：
`summary.json`、`pixel_convention_audit.json`、`checkerboard_detection_preview.png`。
预览为棋盘裁剪后的展示图，不是算法输入图。

正式估计参数 SHA256：
`9f9d25bccbcc333a0d8dc14cfc2c94e9f82d8c1f2d4eb4a29134a10323c63bf7`。
未连接设备，未改变旧定位器、主动双目实现或任何实机配置。
