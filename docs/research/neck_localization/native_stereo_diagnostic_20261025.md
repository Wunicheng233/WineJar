# 保留原生IR点几何的有界对照

日期：2026-10-25。只使用已经打开的4个开发坛型；不打开最终测试集，不生成新场景，不调整噪声或几何门槛。

## 做了什么

原流程将左IR深度注册到RGB像素网格，再根据RGB像素和深度重建点云。新增对照保留**左IR原生像素、光轴深度和内外参**；只把真实RGB颜色及RGB提取出的蓝幕前景mask映射到这些三维点。

这遵循常见彩色点云做法。RealSense官方示例从原深度计算点顶点，再通过颜色流取得纹理坐标；并非必须先把深度重采样到颜色网格。[官方点云示例](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/opencv_pointcloud_viewer.py)。本实现采用明确的最近邻取色和原生点z-buffer，**没有声称逐位复现SDK纹理映射**。

实现：`neck_localization/native_stereo.py`。

1. 对每个有效原生IR点，用IR内参反投影，再经准确的合成相机外参变换至RGB相机。
2. 投影到真实RGB图像，最近邻读取原始颜色以及既定蓝幕mask。颜色没有经过“涂白前景、涂蓝背景”等伪造。
3. 负深度、投影在RGB背面/画外的点排除。同一RGB像素若有多个原生点，保留测得RGB-Z最小的点，避免已知的同像素遮挡冲突。
4. 以**显式映射mask**调用冻结的几何portfolio，包括相同初始化、权重、求解器、多初值和接受门槛。没有再次对未映射的黑色占位像素做HSV前景判断。

实际RGB原图另存为 `CAMERA_source_rgb.png` 链接；`CAMERA_rgb.png` 清楚标记为RGB→IR点的取色图，不能当作原生IR相机拍摄的RGB画面。

## 条件与结果

输入：`experiments/neck_localization/runs/active_stereo_development_20261024_v3` 保存的原生深度和原RGB。没有使用目标尺寸或理想掩码提取实际数据。

| 条件 | 深度来源 | 候选数 | 接受且圆误差通过 | 错误接受 |
|---|---|---:|---:|---:|
| native_ideal | 原生左IR的浮点理想光轴深度；诊断 | 4/4 | **4/4** | 0 |
| native_ideal_z16 | 同一理想深度，采用既有1 mm量化；诊断 | 4/4 | **4/4** | 0 |
| native_raw | 已保存的SimSense输出、1 mm量化 | 0/4 | 0/4 | 0 |
| native_spatial | 同一原生深度经官方单帧空间滤波 | 0/4 | 0/4 | 0 |

原生理想深度两个诊断条件下，3 mm整带相对当前近最小区间代理的仿真检查均4/4通过；这不是实际扎带接触验收。

理想诊断误差（中位数 / P95 / 最大值）：

| 条件 | 圆心/mm | 法向/° | 直径/mm |
|---|---:|---:|---:|
| native_ideal | 0.013 / 0.521 / 0.611 | 0.004 / 0.012 / 0.013 | 0.006 / 0.042 / 0.049 |
| native_ideal_z16 | 0.058 / 0.491 / 0.568 | 0.086 / 0.127 / 0.129 | 0.033 / 0.056 / 0.058 |

几何估计耗时（中位数/最大值）分别约81/90 ms、84/91 ms；**不包括RGB→IR投影、输入导出、I/O或采集**。没有做受控速度比较，不据此宣称比注册流程快多少。

主动双目仿真两条件的全部失败：

| 坛型 | native_raw | native_spatial |
|---|---|---|
| nominal_000 | continuous_core_insufficient_extent | continuous_core_insufficient_extent |
| nominal_001 | continuous_core_insufficient_extent | continuous_core_touches_candidate_boundary |
| stress_000 | no_supported_near_upward_mouth_plane | no_supported_near_upward_mouth_plane |
| stress_001 | no_supported_near_upward_mouth_plane | no_supported_near_upward_mouth_plane |

后两例没有口沿初值，因此没有“实际拟合点”；前两例保留真实准备并调用求解器的点集。所有无可靠圆的记录均保持圆心/法向/半径缺失，没有用诊断参考补出结果。

原生raw顶6 mm种子数分别698/522/201/209，0.18 mm平面RANSAC最佳票数79/79/**48/56**；spatial分别83/77/**49/57**票。后两例直接低于既定60票门槛。对应RGB重采样后，点密度增加且能越过初始化门槛，但初始法向仍可能偏约16°。因此固定“点数”门槛还与分辨率/重复采样有关，不能把更多重采样点视为更多独立信息。原始统计见结果目录 `initial_seed_diagnostics.json`，未据此放宽门槛。

## 可以与不可以据此得出的结论

**已验证：** 既有1 mm深度量化并不必然导致当前几何门槛失败。原生理想点加同样量化时仍4/4，目标区训练/留出残差约0.23–0.27 mm；对应RGB注册理想深度的适配版本为0/4，目标残差约0.43–0.65 mm。这足以支持将“保留原生深度几何，仅映射RGB信息”作为下一阶段候选。

**尚未证明：** 两流程除了深度注册，还改变了点密度、取样位置、像素footprint及mask的可见性，因此不能把差异完全归因于SDK align中的一个实现细节。原生方法也没有解决SimSense实际匹配误差，raw/spatial仍0/4。

RGB→IR映射使用当前测量深度，因此错误深度会导致错误取色/前景标签。原生点z-buffer只能处理当前点云内的遮挡，无法知道RGB相机可见但原生IR点云未观察到的遮挡表面。这里没有读取理想RGB深度或真值可见性来补足信息。

下一步应优先研究实际双目原生点的口沿初始化与局部几何噪声，而非继续把理想点重采样问题误当成全部传感器误差。此4例只是开发诊断，不是新测试或真实相机有效性验证。

## 复现与交付

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
conda run --no-capture-output -n winejar-stereo \
python scripts/evaluate_native_stereo.py \
  --run experiments/neck_localization/runs/active_stereo_development_20261024_v3 \
  --output experiments/neck_localization/runs/native_stereo_development_NEW
```

完整结果：`experiments/neck_localization/runs/native_stereo_development_20261025_v2`，包括16条记录、原生输入、映射mask、映射有效性图、拟合点、来源哈希和拒绝原因。

首次v1已完成相同16条推理，但汇总器没有处理“拒绝且无候选”的缺失字段；修复后完整重跑v2，数值结果相同。v1保留为中间记录，v2为本报告依据。

新增6项CPU测试全部通过：同轴恒等映射/只读原输入、偏置相机的解析投影和原生反投影、无效/背面/画外点、同RGB像素z-buffer、尺寸一致性，以及无候选汇总。未修改此前冻结的 `stereo_robust.py` 或旧几何模块。

## 已验证的MuJoCo查看入口

科学结果v2保持原样，另导出查看bundle：`experiments/neck_localization/runs/native_stereo_viewer_20261025_v2`。记录副本把查看器分类 `evidence` 规范为 `rgb`，并保留 `source_evidence=original_RGB_mapped_mask`；显式引用同一实际映射mask。没有重新估计或修改圆环数值。

场景中的相机已改为**真实原生IR位置、朝向、848×480分辨率以及IR内参**。XML使用 `focalpixel/principalpixel/sensorsize`，不是只换分辨率后沿用RGB视角；MuJoCo3.8不允许同时填写fovy和sensorsize，故由完整IR内参导出fovy=58.319372°。48个相机检查均通过，MuJoCo有效投影与IR K/T反投影的最大差异约 **4.19×10⁻⁵像素**。这验证的是查看器坐标一致性，不是传感器精度。

```bash
conda run --no-capture-output -n winejar-stereo \
python scripts/export_native_stereo_viewer.py \
  --source experiments/neck_localization/runs/native_stereo_development_20261025_v2 \
  --output experiments/neck_localization/runs/native_stereo_viewer_NEW

conda run --no-capture-output -n winejar-stereo \
python scripts/view_neck_localization.py \
  --scenes experiments/neck_localization/runs/native_stereo_viewer_20261025_v2/native_ideal_z16/scenes \
  --records experiments/neck_localization/runs/native_stereo_viewer_20261025_v2/native_ideal_z16/records.json.gz \
  --case development_nominal_000 --method native_ir_rgb_mapped \
  --point-cloud --large-neck --overlay --ground-truth --band
```

将路径中的 `native_ideal_z16` 换成 `native_raw` 或 `native_spatial` 可查看拒绝输入；无圆候选时不画估计圆。左右箭头切换4个开发坛型；D切换原生IR深度伪彩，L切换保存的实际拟合点。右侧明确写为“RGB→IR取色／非相机原始RGB”，不会将重投影颜色误标为相机原始画面。

已经实际打开GLFW窗口并从呈现后的缓冲区捕获以下快照，包含原生点云、原生IR相机位姿、估计/真值圆及3 mm带。相邻JSON记录了对应输入和记录哈希。

![原生IR理想量化诊断：点云及重投影RGB取色](/home/wunicheng/Git/WineJar/experiments/neck_localization/results/native_stereo_viewer_20261025_v2/ideal_z16_pointcloud.png)
