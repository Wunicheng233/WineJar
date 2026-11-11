# 主动双目落地检查点（2026-10-25，未结束）

用户已授权按讨论方向自主推进。本文件是过程检查点，最终结论以完成报告为准。
全程离线，没有连接相机或机器人，没有提交推送。

**已结束：**请改读[完成报告](active_stereo_results_20261025.md)，以下内容保留中间状态。

已落地：

- 独立环境`winejar-stereo`（克隆`embodied-sim`），SAPIEN3.0.3、pyrealsense2 2.59.1.11053。
- `neck_localization/active_stereo.py`：同一酒坛OBJ/贴图，左右IR投影/光追/SimSense匹配，原生IR深度，
  官方单帧视差域滤波、官方RGB对齐，再显式换算RGB光轴Z。没有真值补洞。
- 核查并修正RT Position保存最后Halton射线而非像素中心的问题；理想深度改用独立中心射线raster，
  renderer K的主点减0.5转成数组整数中心约定。已知倾斜平面误差小于0.001 mm仅为数值/坐标诊断。
- 原灰背景造成RGB前景迁移错误；改为声明过的蓝背景，另建RGB-only提取保留暗颈。
  旧数据/拒绝均保留，不能把这些实现问题算传感器模型缺陷。
- 4开发坛型的图像适配版：理想RGB深度4/4；IR采样/量化/对齐诊断、主动双目、主动双目+空间滤波均拒绝。
  未放宽门槛；真实拟合点、图像估计mask和失败原因均保存。
- 独立119平面（93训练/14验证/12测试），已移植作者BSD GP校正模型；12测试RMSE4.285→4.300 mm，未改善。
  模型冻结，未用酒坛真值训练。GP不是完整Basso标定，当前也未加入本机实际标定畸变。
- 六已知平面原生SimSense深度可由保存IR精确重放；1m正对平面的约−15mm偏差在关IR噪声/中值后仍存在，
  指向匹配/亚像素非线性，不能外推真实D435i。见专项诊断。

主要入口：

- `scripts/run_active_stereo_stage1.py`（渲染与冻结旧法）
- `scripts/evaluate_active_stereo.py`（图像适配/相机参数对照）
- `scripts/prepare_active_depth_calibration.py`、`scripts/fit_depth_correction.py`、`scripts/apply_active_depth_correction.py`
- 诊断报告：`active_stereo_plane_diagnosis.md`、`active_stereo_initialization_diagnosis.md`、`depth_correction_adapter_notes.md`

当前正在完成：8新坛型锁测（`results/active_stereo_20261025/freeze.json`已冻结）、
实际渲染棋盘检测标定（仍在审计角点流程）、小型不依赖深度的多视图轮廓可行性对照。
后两项未成为默认方案。当前定位器没有完整已校准置信区间，未验证实机有效性。

开发结果：`runs/active_stereo_rgb_adapted_development_20261025_v2/`；
新测试观测：`runs/active_stereo_test_20261025_v1/`；
平面数据/GP锁测：`runs/active_depth_calibration_20261024_v1/`、`runs/gp_depth_calibration_locked_test_20261024_v1/`。
路径均相对`experiments/neck_localization/`。
