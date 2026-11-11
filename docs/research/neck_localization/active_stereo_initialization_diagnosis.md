# 主动双目开发集：初始化失败诊断

日期：2026-10-24；RGB适配更新至2026-10-25。这里只分析已经打开的 **4 个开发坛型**。没有查看最终 8 个测试坛型，没有调整旧定位器阈值，也没有运行新 GPU 渲染。

**下面先保留 v2 灰背景故障的原始诊断。文末“蓝背景与暗颈的RGB适配”记录后续 v3 蓝背景数据的实际修复和完整对照。**

## 结论

当前 `active_stereo_development_20261024_v2` 的首个失败来自**环境迁移错误**：旧 RGB 提取器要求“灰色坛子、蓝色背景”，SAPIEN 实际背景是灰色。提取器把几乎全图作为前景，主动双目在背景生成的错误非零深度进入点云，再劫持依赖最高点的口沿初始化。

因此，这批 `stereo_raw/spatial` 的 0/4 不能作为“成熟主动双目模型或定位方法不适用”的证据。应先让渲染背景满足既定实验条件，保留本批记录，再重新运行开发集。

此外，真正口沿像素的深度也有毫米级误差；背景修复后仍应检查口沿平面初始化与旧质量阈值。当前证据支持把这两层问题分开处理。

## 旧初始化实际做了什么

`joint_profile.observed_initialization` 先用 `rgb_foreground` 提取图像，再以准确的合成 K/T 反投影，限制在固定工位体积（Z=400–760 mm，XY 半径280 mm）。

随后 `_mouth_plane`：

1. 取点云**最大世界 Z 下方35 mm**的点。
2. 从最大 Z 下方 **6 mm** 取种子。
3. RANSAC 450 次，要求法向距竖直不超过30°，点到平面距离小于 **0.18 mm**，至少 **60 票**。
4. 找到后才初始化颈轴和局部半径曲线。

因此，只要背景飞点比坛口高，当前算法连真正口沿的点都不会优先检查。这不是后续联合优化陷入局部最优；失败发生在联合优化之前。

## 观测证据

输入：`experiments/neck_localization/runs/active_stereo_development_20261024_v2`。

### 1. RGB 提取失效，最高点来自背景

三相机图像为848×480；灰背景角点 RGB 为 `[147,147,147]`。旧规则接受低饱和度且亮度大于35的区域，各图前景覆盖率 **99.63%–100%**。

下表的“理想最高Z”来自对应 `ideal_rgb` 的同一 RGB 提取流程，只作为诊断参照。“背景种子”指该像素在理想 RGB 深度中没有几何命中；**没有用这些标签过滤或修复估计器输入**。

| 开发坛型 | 理想最高Z/mm | raw最高Z/mm | raw顶6mm点数/最佳票数 | spatial顶6mm点数/最佳票数 | 两者种子中背景占比 |
|---|---:|---:|---:|---:|---:|
| nominal_000 | 616.196 | 759.512 | 80 / 19 | 80 / 22 | 100% |
| nominal_001 | 573.280 | 759.960 | 98 / 32 | 100 / 32 | 100% |
| stress_000 | 618.912 | 759.881 | 147 / 40 | 149 / 41 | 100% |
| stress_001 | 641.548 | 759.536 | 143 / 32 | 143 / 32 | 100% |

所有最高点都接近固定工位760 mm上界。`no_supported_near_upward_mouth_plane` 的具体原因是：在背景飞点种子中找不到60票，而非真实坛口完全不可见。

以 nominal_000 为例，raw 中超过650 mm的点，A/B/C分别788/1054/1360个，全部位于理想深度为0的背景像素。修复应来自环境或实际 RGB 提取，不能使用理想深度掩码。

### 2. 有效深度并不等于正确口沿深度

在理想 RGB 观测拟合出的口沿平面上，选取距离小于0.18 mm的像素，检查同一像素在其他条件下的观测。这里只是误差诊断，不是提供真值点给算法。

| 开发坛型 | 三相机理想口沿像素总数 | raw有效率范围 | raw口沿点平面误差P95/mm（A/B/C） | spatial同项/mm（A/B/C） |
|---|---:|---:|---:|---:|
| nominal_000 | 1557 | 99.54%–100% | 6.745 / 4.699 / 4.753 | 5.492 / 4.821 / 4.375 |
| nominal_001 | 2126 | 100% | 5.670 / 12.130 / 6.562 | 5.729 / 13.579 / 6.154 |
| stress_000 | 1435 | 100% | 4.493 / 8.261 / 3.320 | 4.430 / 8.254 / 3.297 |
| stress_001 | 1720 | 95.44%–100% | 5.906 / 8.832 / 0.198* | 6.715 / 9.271 / 0.147* |

`*` stress_001 的 C 相机仅有15个理想口沿像素，该相机的小误差不能代表充足观测支持。

raw 全部三相机中，仍落在理想平面±0.18 mm内的点，总计只有34/67/53/46个。即使排除了背景，不能假设原来的0.18 mm平面共识必然恢复。

`ideal_ir_registered` 没有双目匹配噪声，但包含左红外深度采样、毫米量化与 RGB 注册。多数相机的同项P95为 **1.1–1.4 mm**；原算法能够初始化，却在后续得到3例 `core_model_mismatch/target_readout_mismatch`（旧门槛0.4 mm），另1例区间触边。

这说明：需要将低分辨率采样/深度量化/跨视点注册，与双目匹配错误分开检查。当前数据还不足以把这1.1–1.4 mm逐一归因；也不能根据这4例直接放宽门槛。

### 3. 把最大值改成分位数也可能选择错误平面

表格单元为 raw 的“截断上界Z/mm；顶6mm RANSAC最佳票数”。这些是描述性探针，**不是新预处理方案，没有据此产生定位成功率**。

| Z上界分位 | nominal_000 | nominal_001 | stress_000 | stress_001 |
|---|---:|---:|---:|---:|
| P95 | 616.012；274 | 580.320；230 | 616.300；182 | 630.489；184 |
| P98 | 640.838；81 | 645.863；70 | 671.806；30 | 647.046；190 |
| P99 | 683.646；70 | 702.158；32 | 707.610；55 | 677.733；34 |
| P99.5 | 716.451；19 | 730.688；45 | 732.755；59 | 717.095；57 |
| P99.9 | 746.590；30 | 752.511；60 | 752.896；61 | 750.628；20 |
| 最大值 | 759.512；19 | 759.960；32 | 759.881；40 | 759.536；32 |

部分明显高于坛口的错误高度也能超过60票。通过初始化门槛不等于找对口沿；不能仅凭“拒绝减少”判断分位截断有效。

## 下一步与替代初始化

1. **先修正背景。** 保留旧 RGB 提取器，重新生成一致的蓝背景开发数据。继续报告全部4例，不能仅重跑容易成功的坛型。
2. 背景修正后仍有异常点，再考虑有出处的点云离群过滤或深度连通组件对照。参数需要在开发集冻结，最终测试才能使用；不得把理想掩码当实际提取。
3. 若真正口沿的深度平面持续不可靠，可以取消“口沿平面提供轴初值”这一硬依赖。一个可解释的候选是：从 RGB 外轮廓提供宽松颈部图像区域，利用已知 K/T 和实际深度构建上部外壁点云，以近竖直轴为多初值，联合拟合局部轴与自由半径曲线，并在观测曲线中选择颈部近最小区间。

第三项只是一项候选，尚未实现。它使用已知工位范围和轴距竖直不超过30°的先验，不预设坛高、颈高或目标半径，不要求口沿点共面。需要验证：宽松区域是否覆盖歪颈；轴/曲线是否可辨识；过渡段是否主导拟合；是否因多初值带来耗时或错误极小值。应与冻结口沿初值版本配对比较，而不是直接替换后声称改进。

## 复现及完整统计

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
conda run --no-capture-output -n winejar-stereo \
python scripts/diagnose_active_stereo_initialization.py \
  --run experiments/neck_localization/runs/active_stereo_development_20261024_v2 \
  --output experiments/neck_localization/results/active_stereo_initialization_diagnosis_NEW
```

新增脚本只读保存的观测，不开启 GPU、不读取设备，不修改旧算法或本批数据。

结果目录：`experiments/neck_localization/results/active_stereo_initialization_diagnosis_20261024_v2`。

- `diagnosis.json`：16条件的种子支持、每相机来源、背景种子分类、冻结算法原结果、源码哈希及诊断约束。
- `cameras.csv`：48条每相机记录，包括全部Z分位、口沿有效率/误差和背景飞点数量。
- `quantile_probes.csv`：48条 raw/spatial 分位截断探针，含种子数量和固定RANSAC支持。

这些结果解释的是**本批环境和观测为何使冻结算法失败**，不是实机精度、完整不确定性模型或最终方法性能。

## 蓝背景与暗颈的RGB适配（2026-10-25）

### 修复的具体内容

根代理已将 SAPIEN 背景修复为蓝色，保存为 `active_stereo_development_20261024_v3`，角点RGB为 `[19,33,57]`。但旧提取器还有第二个问题：`V>35` 会切断很暗的颈部，最大连通区域只剩坛身，口沿可能单独成为另一个小区域而被丢弃。因此蓝背景下，旧流程连 `ideal_rgb` 都是0/4，不能用它判断几何定位能力。

新增 `neck_localization/stereo_robust.py`，只替换RGB提取：

1. 使用明确的蓝幕色键：OpenCV H∈[95,125]，S≥65，B−R≥12，B−G≥4。
2. 对非蓝区域取最大的外轮廓并填充内部孔洞，不设置前景亮度下限，保留暗颈、暗口沿及口腔。
3. 将**原始RGB、原始深度与显式mask**传入适配的初始化；不生成伪RGB，不读取理想深度、模型尺寸或目标标签。
4. 圆环求解仍直接复用冻结的 `_mouth_plane/_mouth_center`、`WeightedAxisSplineProblem`、`_attempt/_candidate` 和 `_validate_task`，接受门槛未改。

色键参数根据本工位蓝背景与灰黑陶瓷的图像颜色选择，未使用定位真值调参。这是有声明环境假设的RGB提取，并不是一般场景分割模型。填孔会保留口腔像素，后续相机朝向筛选仍负责区分可用外壁点。

### 全部4例、16条件的配对结果

输入始终是同一批 v3 RGB-D，仅提取方法不同。成功要求圆环通过既定几何误差门槛且估计器接受；没有通过放宽拒绝条件提高成功率。

| 观测条件 | 旧RGB提取正确数 | 新RGB提取正确数 | 新提取后的状态 |
|---|---:|---:|---|
| ideal_rgb | 0/4 | **4/4** | 圆心最大0.620 mm，法向最大0.030°，直径最大0.036 mm |
| ideal_ir_registered | 0/4 | 0/4 | 4例都有圆数值，均因core/target残差门槛拒绝 |
| stereo_raw | 0/4 | 0/4 | 4例均通过口沿平面初值，随后core范围不足或近最小区间触边 |
| stereo_spatial | 0/4 | 0/4 | 4例均通过口沿平面初值，随后core范围不足/触边或近最小区间触边 |

`ideal_ir_registered` 的圆心误差依次0.110/0.163/0.674/1.117 mm，法向0.129/0.138/0.434/0.628°，直径0.778/0.825/0.733/0.982 mm，但目标区训练/留出RMS分别达到约0.43–0.65 mm，高于旧0.4 mm门槛。**这些是被拒候选的诊断数值，不计入成功。**

新RGB提取后，旧的“背景飞点占据最高种子”问题显著减小，但实际双目的口沿初始化仍有偏差。以下比较使用同坛型 `ideal_rgb` 的观测初值作诊断参照，不把它提供给求解器：

| 坛型 | raw口沿初始法向差/° | raw口沿初始中心差/mm | spatial法向差/° | spatial中心差/mm |
|---|---:|---:|---:|---:|
| nominal_000 | 0.455 | 7.101 | 0.104 | 11.314 |
| nominal_001 | 1.369 | 7.326 | 1.339 | 3.290 |
| stress_000 | 16.365 | 16.996 | 16.042 | 18.905 |
| stress_001 | 15.592 | 16.465 | 16.376 | 17.156 |

stress两例的raw初始轴偏差约16°，随后粗半径曲线把最小位置选到口沿下5.75 mm附近。说明现在可以开始研究真正的“有限且有噪声的口沿平面能否可靠初始化局部轴”问题。当前改动修复了RGB输入链路；它没有解决深度噪声和弱颈部范围的联合估计。

### 代码、实际mask和拟合点

- API：`estimate_stereo_robust(frames)`；`extract_stereo_masks(frames)` 返回估计器实际使用的mask及参数/哈希。
- 结果：`experiments/neck_localization/runs/active_stereo_rgb_adapted_development_20261025_v2`。此前 v1 保留，v2仅补充了拒绝后的真实拟合点诊断；16条结果数值相同。
- 每条件的 `scenes/CASE/foreground_masks/CAMERA_foreground.png` 保存实际mask，sha256与估计器diagnostics一致。
- 16条件均保留 `fit_point_positions_m`、相机ID、权重和初始口沿中心/法向。raw/spatial没有可靠圆候选时，明确标记这些只是最后真正准备并调用求解器的训练点，**不伪造圆心或法向输出**。
- `paired_records.json.gz` 同时保留原RGB和新RGB方法；`summary.json` 分别汇总。
- 更详细的RGB掩码下RANSAC/分位与每相机统计：`experiments/neck_localization/results/active_stereo_initialization_diagnosis_20261025_v3_blue_v2`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
conda run --no-capture-output -n winejar-stereo \
python scripts/evaluate_stereo_rgb_adaptation.py \
  --run experiments/neck_localization/runs/active_stereo_development_20261024_v3 \
  --output experiments/neck_localization/runs/active_stereo_rgb_adapted_NEW

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
conda run --no-capture-output -n winejar-stereo \
python scripts/diagnose_active_stereo_initialization.py \
  --run experiments/neck_localization/runs/active_stereo_development_20261024_v3 \
  --foreground blue \
  --output experiments/neck_localization/results/active_stereo_initialization_diagnosis_NEW
```

新增6项RGB适配测试通过，包含暗颈连接、空蓝背景、孔洞填充、小干扰区域、只读原图不变，以及**相同mask时新旧初始化的点集、轴初值、拟合索引和粗区间数值完全一致**。与原主动双目CPU测试合计13项通过，未占GPU。

当前仍只有4个开发坛型；最终8个测试坛型保持未打开。下一步应针对真实双目初值和局部几何误差研究，而不是继续调整RGB色键或直接放宽0.4 mm门槛。
