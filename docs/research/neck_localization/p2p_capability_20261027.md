# P2P-Bridge 预训练能力审查：推理与配对输入审计

2026-10-27。仅离线；未读取机器人或相机。本页以 **v2成熟前景接口** 为当前记录，不填写真值定位成绩，后续以独立评估报告为准。12坛型已被查看，两布局是同12个物体，属于固定能力审查集。

v1曾沿用 `depth_profile.rgb_foreground` 的旧亮度门槛V>35，切掉暗颈，与冻结后端的前景接口不同。其原目录完整保留为接口诊断，不用于评价预训练模型能力。发现后只统一为现有 `extract_stereo_masks/blue_screen_foreground`，未根据审查真值调整裁剪或模型；v2新增实际RGB/mask、mask设置、前景模块与工位设置散列。

使用[官方 P2P-Bridge](https://github.com/matvogel/P2P-Bridge) 的 `PVDS_PUNet` XYZ 权重。上游提交 `eb26f07326134839bdc32370cb40fee941d183c9`，上游工作区无修改；checkpoint 423,653,217 字节，训练步数450000。303个 backbone 状态与608个 EMA 状态均以 `strict=True` 完整加载；采用 EMA、官方 OT-ODE 采样器、5步、seed42，未训练或微调。

- checkpoint SHA-256：`dffd1ad7cded121d8862f4a3a1ff7397a29123df8fd74f0d4e2e295d91639bae`
- `opt.yaml` SHA-256：`2d934c7da7a013b3c53db62cbe9f1569b1c79265cd69f0403e14d1c1351bd36a`
- 原生扩展 SHA-256：`5181ff691855a2dca11d9a39fedc7204d85331e9a5739c5dcbc158dfaee0c89a`

运行环境位于 `experiments/neck_localization/external_baselines/p2p_bridge/runtime`，继承视觉环境的 PyTorch 2.13.0+cu130；控制环境未改。原始 `pointnet2_batch` 的21个 C++/CUDA 源文件使用系统 g++13、隔离 CUDA13 编译工具和 `sm_89` 编译成功。此前 g++9/C++20 与 CCCL include 失败已修复；这些构建失败不计作算法失败。仅隔离了未调用的训练用 EMD 和无关导入，backbone、采样器及 CUDA 算子源码未改。完整源散列、配置和扩展散列保存在每次运行的 `provenance.json`。

输入为历史 `stereo_raw` 或 `ideal_rgb` 的普通 RGB-D 及同一相机参数，**不是 FoundationStereo 的共享 IR 条件**。`ideal_rgb` 是合成理想深度正对照，不能部署。三相机先分别按正式蓝背景前景与现有固定工位裁剪，再取观测世界Z第99.5百分位以下70 mm，合并后用固定 seed 均匀抽8192点；不读网格、目标帧、真值颈段或诊断mask。近距从 `close/input_source.json` 解析至 `render_inputs`，未使用旧中断的 `close/render`。

整云使用官方 object 归一化规则（包围盒中点、最大半径），FPS种子与2048点KNN patch沿用官方几何规则；8192点对应12个patch，patch均值居中后共享一个最大尺度。为保持来源审计，本适配器采用官方 room 的逐来源点平均替代 object 脚本末端的 FPS 重采样；外层 FPS/KNN 在 CPU NumPy/SciPy 实现，归一化用float64再转网络float32，每个patch独立batch1。故这是明确适配后的预训练基线，尚未证明与官方完整 object CLI 逐数相同；未覆盖点的预设规则是保留输入，本轮48条均无未覆盖点。

输出 NPZ 保存 `points_world_m`、完全相同裁剪/采样的 `input_points_world_m`，以及原始 `source_camera_ids`、`source_pixel_uv`、patch更新次数。source IDs 只是输入来源；恢复点可以移动，不能再当作原像素的深度。模型输入是三视合并云，不从中筛来源cam_a冒充单视模型。下游必须对恢复/未恢复两云使用相同重投影与冻结后端，另列裁剪/采样对全密度原输入的影响。

v2 prepare48均通过：各8192点、所选点Z跨度均>69 mm，实际保留的V≤35暗像素远距728–3862点/近距720–3394点，未使用真值标签。这是观测支持范围检查，不等于完整真实颈段已被测量。mask settings SHA-256为 `414016ae47eecab2e5c4455ea99b6c20606b083d6debac83e1c3692ce8a327cf`。数据位置均相对于仓库根目录：

v2两布局48条真实推理已全部 `inference_completed`。48条输出哈希、形状、有限性、三相机来源、前景元数据与prepare输入逐点相等核查全部通过；无未覆盖点。所有组推理完成后才交独立评估器开真值。

| 记录 | 路径 |
|---|---|
| 旧mask开发smoke，仅接口诊断 | `experiments/neck_localization/runs/p2p_development_smoke_20261027_v1/` |
| 远距12坛型×2条件 | `experiments/neck_localization/runs/p2p_capability_far_20261027_v2/` |
| 近距12坛型×2条件 | `experiments/neck_localization/runs/p2p_capability_close_20261027_v2/` |
| v2观测支持检查 | `experiments/neck_localization/runs/p2p_prepare_mask_audit_20261027_v2.json` |
| 哈希/配对输入与耗时审计 | `experiments/neck_localization/runs/p2p_capability_output_audit_20261027_v2.json` |

RTX4050 Laptop 6GB上，v2四个组的恢复耗时中位1.206–1.226秒/云，首次预热最大1.933秒；峰值分配246.18MiB。耗时包含归一化、12个patch循环和点平均，不含模型加载/文件读取。raw点位移中位远0.728/近0.615 mm，ideal中位远0.346/近0.380 mm；个别点移动最大59.03 mm。位移既非真值误差也非恢复收益，未根据其幅度删除点。

代码为 `neck_localization/point_restoration_baseline.py`、`scripts/run_point_restoration_capability.py`。4项CPU检查覆盖暗颈像素保留、米制归一化可逆、重复点来源聚合、重叠平均/未覆盖点保留，全部通过：

```bash
conda run -n winejar-stereo python -m unittest discover -s tests/neck_localization -p test_point_restoration_baseline.py -v
experiments/neck_localization/external_baselines/p2p_bridge/runtime/bin/python scripts/run_point_restoration_capability.py --layout far --output NEW_FAR_DIR
experiments/neck_localization/external_baselines/p2p_bridge/runtime/bin/python scripts/run_point_restoration_capability.py --layout close --output NEW_CLOSE_DIR
```

本页没有据此判断科学缺口或可发表性；预训练域外失败仍需与接口适配、观测采样、后端局限及可行的简单适应区分。
