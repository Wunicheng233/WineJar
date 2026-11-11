# 现成深度与正确几何读出的能力复核

2026-10-27。用户要求继续检验成熟方法能否解决实际坛颈定位，并恢复原 MuJoCo 展示。本文记录实际运行结果，不把能力审查当成论文贡献。全程离线，没有连接相机或机器人，没有训练或微调网络。

**目前不支持把“圆心定位到 4 mm”作为现有方法解决不了的科研缺口。** 修正错误验证关系后，官方预训练 FoundationStereo 与传统 SimSense 在近距圆心上都已很强；独立常规曲面拟合也能取得相近的中心精度。仍存在法向误差、错误接受、图像误拒绝及有限带宽越界，但这些只是进一步检验的对象，不能自动成为 Contribution。

机器可读汇总：`experiments/neck_localization/results/geometry_capability_summary_20261027_v2/`，含首次冻结与后续SAM2配对统计、逐例记录、输入/结果散列、源码快照及评价口径侧车说明。v1汇总保留，不覆盖。

## 1. 本轮确实改变了什么

### 几何修正版 visible_profile

原 portfolio 要求“目标截面圆的切点贴合整坛 RGB 外轮廓”。口沿下的内部颈截面通常不满足该关系。原关系还参与候选评分，因此不能只删最终拒绝原因。

新模块 `neck_localization/visible_profile_baseline.py` 保留原拟合、多个观测初值、区域反馈、所有非接触质量门槛。对每个候选重新构造有限回转曲面，沿轴按原曲线采样、周向每 1° 采样，检验其投影是否包含在整坛前景内部。内部投影不要求贴边；最大外越界仍用原 2 px 数值，候选排序重新用留出径向残差与单侧越界 RMS。全部候选重算后再选，不按真值挑圆。该修正是纠正建模错误，不主张新颖性或绝对精度保证。

### 常规几何 standard_sor

新模块 `neck_localization/sor_baseline.py` 独立联合优化 4 个轴参数与三次 B 样条母线，采用有界 Huber 最小二乘、等视图基础权重、二阶平滑及空间块留出。支持区域只按观测粗曲线选一次，不随最终目标反馈；多初值来自观测口沿轴及世界竖直轴。最终按连续半径曲线的近最细区间中点读圆。

这是一套常规几何控制实现，**不是对某篇论文完整算法的忠实复现**。回转体重建及姿态/形状联合拟合已有成熟先例，包括 [Pottmann 等的特殊曲面重建工作](https://dmg.tuwien.ac.at/geom/ig/papers/pcv02b.pdf)及 [Liu、Pottmann、Wang 的约束曲面拟合](https://www.microsoft.com/en-us/research/publication/constrained-3d-shape-reconstruction-using-combination-surface-fitting-registration/)；后者已讨论陶器碎片回转曲面。因此，“联合拟合轴和曲线”本身不能作为我们的创新点。

两方法共用相同 RGB、相同注册深度、工位过滤与观测口沿初值来源，不输入 CAD、真实坛径、目标区间或评价真值。它们的原生接受合同不同：修正版保留 3 mm 可观测区间门槛，常规 SOR 原生接受不含该门槛。因此另报共同的名义 3 mm 带宽门槛，不用原生接受率直接排优劣。

## 2. 数据与评价边界

- 原 12 个已看坛型，在两个固定布局中观察；不是 24 个独立物体，也不是本轮未见测试。
- 另在方法及配置冻结后生成 8 个新坛型，仅近距，沿用同一名义范围与圆截面模型族。8 个都保留，没有按误差筛选。新 seed 同时产生新的纹理实现，因此不是仅改变几何的因果消融。
- 每例三相机、clean/shared_noise 两条件。FS 和 SimSense 逐文件使用相同 IR；噪声只生成保存一次。FS 沿用官方预训练权重、32 次细化，没有 WineJar 微调。
- 两个几何后端全部返回后才读取该例目标/轮廓真值。所有拒绝、错误输出和异常均保留分母。
- 主指标：中心最大 4 mm、优选 3 mm。法向 2°、直径 2 mm 只是研究分析阈值，未被酒厂确认为验收要求。
- 准确 K/T、合成纹理、圆截面假设仍在；没有实物精度、非圆截面泛化或产线成功率证据。

新例协议：`configs/geometry_confirmation_close_20261027_v1_protocol.json`（上述相对实验目录），锁定 seed `2026102707`；观测与前端验证：`runs/geometry_confirmation_20261027_v1/frontends_completed_validation.json`。48 对 FS 与 48 对 SimSense 全部完成、散列/形状/单位核验通过。原型定位器和实机配置均未修改。

## 3. 冻结几何修正版的结果

以下是 **visible_profile 的首次冻结评价**。候选与真正接受输出分开，避免用一个被拒绝的好圆冒充已完成输出。

| 数据/前端 | IR 条件 | 候选中心≤3 mm | 候选中心≤4 mm | 实际输出且中心≤4 mm | 候选法向≤2° | 完整圆参数正确输出 |
|---|---|---:|---:|---:|---:|---:|
| 已看12 / FS | clean | 12/12 | 12/12 | 12/12 | 12/12 | 11/12 |
| 已看12 / FS | shared_noise | 12/12 | 12/12 | 12/12 | 12/12 | 11/12 |
| 新8 / FS | clean | 7/8 | 8/8 | 8/8 | 7/8 | 7/8 |
| 新8 / FS | shared_noise | 7/8 | 8/8 | 7/8 | 7/8 | 6/8 |
| 已看12 / SimSense | clean | 12/12 | 12/12 | 12/12 | 12/12 | 5/12 |
| 已看12 / SimSense | shared_noise | 12/12 | 12/12 | 12/12 | 12/12 | 5/12 |
| 新8 / SimSense | clean | 7/8 | 8/8 | 8/8 | 8/8 | 1/8 |
| 新8 / SimSense | shared_noise | 7/8 | 8/8 | 8/8 | 8/8 | 1/8 |

“完整圆参数”同时要求中心 4 mm、法向 2°、直径 2 mm，不能用其较低通过率否定主中心需求。FS 新8最大中心为 clean 3.089 mm、shared 3.048 mm；已看12分别 2.369、2.128 mm。新8的 `confirmation_nominal_001` 中心略超优选 3 mm，但满足最大 4 mm。

FS 新8的 `_004/shared_noise` 候选中心 0.667 mm、法向 0.710°、直径 0.427 mm，却因曲面投影越界被拒绝。`_005` 两条件都被接受，但法向为 2.543°/2.512°，超过研究分析阈值。这说明必要几何一致性与留出残差尚不能作为经过校准的绝对精度置信度。

FS 对直径的改善明显，但本轮不能据暂设 2 mm 直径阈值宣称传统方法未解决厂方要求。特别是 **传统 SimSense 配合正确读出，也在这两批近距圆心上全部满足 4 mm**。这比单纯“FS 能做”更不利于把当前中心问题包装成独立创新。

## 4. 常规几何控制的结果

| 数据/前端 | 条件 | 候选中心≤3 mm | 候选中心≤4 mm | 候选法向≤2° | 完整圆参数正确输出 |
|---|---|---:|---:|---:|---:|
| 已看12 / FS | clean | 12/12 | 12/12 | 10/12 | 9/12 |
| 已看12 / FS | shared_noise | 11/12 | 11/12 | 9/12 | 8/12 |
| 新8 / FS | clean | 8/8 | 8/8 | 6/8 | 6/8 |
| 新8 / FS | shared_noise | 7/8 | 8/8 | 7/8 | 7/8 |
| 已看12 / SimSense | clean | 12/12 | 12/12 | 11/12 | 5/12 |
| 已看12 / SimSense | shared_noise | 12/12 | 12/12 | 12/12 | 5/12 |
| 新8 / SimSense | clean | 7/8 | 8/8 | 8/8 | 1/8 |
| 新8 / SimSense | shared_noise | 8/8 | 8/8 | 7/8 | 1/8 |

本表各组原生接受均为全部案例；这不能视为可靠性优点。FS 已看12/shared 中存在中心 15.815 mm、法向 11.763°的错误接受，必须保留。常规控制证明简单方法已有较强中心能力，也暴露个别坏分支；它不证明更复杂的区域反馈已形成通用科研优势。小样本、同模型族与不同原生接受合同限制了排名解释。

远距及开发/理想控制亦已跑完，完整记录在汇总中。远距不是用户确认必须达到的额外工况，不能用其失败挽救当前近距中心问题的创新主张。理想深度是后端正控而非成绩上界，存在范围边界拒绝，不能把噪声输出优于理想接受数解释成模型超过真值。

## 5. 分割正控：先排除输入错误

开发集检查发现，原蓝幕 HSV key 将陶坛上的蓝色反光误当背景，使正确曲面也越出前景。四开发例的真实颈曲面在几何真轮廓上 4/4 通过；在实际 RGB 的旧掩膜上 0/4 通过，最大外越界 6.460 px。故这些拒绝不是定位科学问题已成立的证据。

复用本机现成 [SAM2 tiny 官方模型](https://github.com/facebookresearch/sam2)，整坛粗 RGB bbox 固定外扩 8 px，无点提示、无颈部真值 ROI。仅按 SAM 自己预测的最高 score 选 mask，再取最大外轮廓填孔。配置在开发 RGB 上固定，之后才做真值正控。相同原 2 px 门槛下四例 4/4 通过，最大外越界 0.591 px；负控 4/4 拒绝。该结果说明现成分割适配可以修复这一工程错误，不是一个新分割方法。

缓存与协议：`runs/geometry_foreground_sam2_development_20261027_v1/manifest.json`；正控和图：`results/geometry_capability_20261027_reference/containment_controls_sam2_development.json`、`SAM2_and_geometric_silhouette.png`。近距两批的固定配置配对复核另存，不覆盖上表首次冻结结果，也不把已经查看的8例再次称为独立未见测试。

**配对复核现已完成。** 旧12近距36张、新8近距24张全部分割成功。加开发共72张的来源、模型、配置、固定protocol、RGB/mask文件和像素散列均核对通过。显式mask适配仅替换前景，两个几何后端的拟合/门槛/排序不改。理想原生深度开发正控两法均4/4接受且完整圆正确；随后正式160条定位记录全部执行、输入/源码验证通过、异常0。

下表均用相同SAM2配置；斜杠对表示 **clean / shared_noise**，每项仍以单独n为分母。所有组候选中心均全部满足4 mm，所以另列实际输出，避免隐藏拒绝。

| 数据/前端/读出 | n | 实际输出且中心≤4mm：clean / noise | 候选法向≤2°：clean / noise | 候选3mm带轴向包含：clean / noise |
|---|---:|---:|---:|---:|
| 已看12 / FS / 修正版 | 12 | 12 / 12 | 12 / 12 | 9 / 10 |
| 已看12 / FS / 标准SOR | 12 | 12 / 12 | 9 / 11 | 9 / 10 |
| 已看12 / SimSense / 修正版 | 12 | 12 / 12 | 12 / 12 | 12 / 11 |
| 已看12 / SimSense / 标准SOR | 12 | 12 / 12 | 11 / 11 | 11 / 11 |
| 已看8 / FS / 修正版 | 8 | 7 / 8 | 7 / 7 | 6 / 7 |
| 已看8 / FS / 标准SOR | 8 | 8 / 8 | 6 / 7 | 7 / 7 |
| 已看8 / SimSense / 修正版 | 8 | 8 / 8 | 8 / 8 | 6 / 6 |
| 已看8 / SimSense / 标准SOR | 8 | 8 / 8 | 8 / 8 | 8 / 8 |

SAM2修复FS的`_004/shared`误拒绝，但FS修正版`_006/clean`出现新的拒绝：候选中心0.660 mm、法向0.348°，有限拟合曲面仍越出掩膜。故不能说“换SAM后所有门槛都正确”或继续按这8例反复调整2px门槛。FS标准SOR在已看12/shared的15.815mm中心坏分支，经固定SAM输入改为全部≤3mm；相同成熟适配也能解决一部分原反例。

FS修正版`_005`仍有约2.50°/2.43°法向误差和约1.54/1.53mm轴向越界；但**传统SimSense＋SAM2＋标准SOR在这8例两条件均8/8中心≤4mm、8/8法向≤2°、8/8带轴向包含**，直径研究阈值通过仍低。不能只展示FS失败而忽略其他成熟组合的成功，进而宣称有限带宽问题已证明现有方法解决不了。独立只读复核在`results/geometry_capability_20261027_reference/SAM2_paired_readonly_review.json`。

## 6. 圆心达标与 3 mm 整带约束

用相同 3 mm 带宽和已标注完整颈段，仅评价轴向几何包含：FS 修正版在已看12两条件均为 10/12；新8为 clean 7/8、shared 6/8（候选口径）。这些不是接触、碰撞、收紧、滑移或真实扎口成功率。

新8 `_005/clean`：中心 2.759 mm、法向 2.543°、半径约 54.94 mm。约 2.24 mm 轴向偏移，加上倾斜圆周约 2.44 mm 的轴向摆幅和半带宽，使 9.58 mm 完整颈段下的轴向包络仍越界 1.382 mm。**中心≤4 mm不蕴含有限宽带能放进槽里。** 这个耦合是普通几何即可推导的事实，本身也不是创新。

若后续实物证明这一失败反复出现且有实际工艺影响，可把任务从单个点/圆推广为：从有偏、部分可见的观测中估计满足工艺约束的可放置集合，并给出可检验的接受/拒绝依据。适用对象可包括瓶颈、管件环槽、卡箍或其他窄区间环带装配。但首先必须证明真实可用区间定义清楚，且普通曲面拟合、独立标定、现成分割/监督适配和已有不确定性方法仍不足。不能因本轮几条反例直接宣告新科学问题已被确立。

SAM2配对结果进一步表明：可用区间问题值得以真实工艺继续审查，但目前同样缺少“合理成熟组合反复做不到”的证据。现阶段应把上述问题作为待证伪假设，而不是既定论文题目。

已有[不确定性/残差方法调研](uncertainty_methods_review.md)覆盖鲁棒拟合、显式偏差、标定、学习修正等先例。加一个网络、RL、置信度或约束 loss，都不自动产生贡献。当前没有理由为了凑齐模型名单继续无界扩展能力审查；先验证强基线的实物表现及真实工艺误差边界更有价值。

## 7. 回到原 MuJoCo 查看器

原命令现在读取本轮保存结果，保留旧查看束显式选择能力：

```bash
conda run --no-capture-output -n winejar-stereo \
  python /home/wunicheng/Git/WineJar/scripts/view_structural_study.py --ground-truth
```

查看束：`results/capability_structural_viewer_20261027_v1/`。三维网格、原 RGB、两前端实际深度、实际拟合点/权重、候选圆与真值沿用原交互。导出只做数据路由和显示，不重新运行模型/定位器；拟合点逐点核对对应保存深度，不用真值替代输入点云。

SAM2配对结果单独使用同一查看器，避免把不同mask的科学记录混成同一输入条件：

```bash
conda run --no-capture-output -n winejar-stereo \
  python /home/wunicheng/Git/WineJar/scripts/view_structural_study.py --ground-truth \
  --viewer /home/wunicheng/Git/WineJar/experiments/neck_localization/results/capability_structural_viewer_sam2_20261027_v1
```

## 8. 复现与验证

```bash
conda run --no-capture-output -n winejar-stereo python scripts/run_geometry_capability_stage.py \
  --manifest experiments/neck_localization/runs/capability_inputs_20261027_v1/manifest.json \
  --predictions experiments/neck_localization/runs/foundation_stereo_capability_20261027_v1 \
  --output /tmp/geometry_fs_replay --layouts close far

conda run --no-capture-output -n winejar-stereo python scripts/run_geometry_capability_stage.py \
  --manifest experiments/neck_localization/runs/geometry_confirmation_20261027_v1/inputs/manifest_confirmation.json \
  --predictions experiments/neck_localization/runs/geometry_confirmation_20261027_v1/foundation_stereo \
  --output /tmp/geometry_fs_confirmation_replay \
  --scope 'eight new circular-section jars; frozen first evaluation; no physical or cross-family claim'

conda run -n winejar-stereo python -m unittest discover -s tests/neck_localization
```

两个后端共 344 条 v1 runner 记录（开发 FS16、主 FS96、SimSense112、理想56、新8两前端64）；其中正式汇总收录后五组 328 条，早期16条开发另存。正式五目录 `validation.json` 均通过，输入/后端异常0；源散列与当前文件亦逐项相同。开发后修正过 runner 的指标命名/oracle元数据，其开发散列对应当时版本，不与正式源码混称。结果分目录保留，不拼接异输入当统一排行榜。源码快照和 SHA 在汇总目录；耗时来自并行 CPU 竞争，不作为部署延迟结论。

一个元数据限制已明确记录：v1 runner 把非 oracle `evaluation_role`统一写为 capability audit；新8的原 manifest、split、scope 和 `evaluation_context.json`侧车保留首次冻结评价身份。没有改动原数值记录，避免中途改代码破坏来源散列。

固定SAM2配对正式160条另存四目录`geometry_{fs,simsense}_sam2_{audit,confirmation}_20261027_v1`；其独立runner明确标记已看对象的固定适配并保留原split/role。开发正式16条另存，早期FS纯元数据修正前的8条先导亦保留。最终128项离线检查126通过/2默认GL跳过；正式原MuJoCo另做12路由/40方法路径、959来源散列、问题例及显式旧查看束的实际GL核查，全部通过。SAM独立束另8路由/16方法路径、665来源散列及实际显式原命令核查通过；80场景/240帧/160结果的224979个实际拟合点逐点对齐。全部窗口已关闭。没有将默认GL跳过算作实测通过。

最终9个正式几何目录的488条记录均无输入/后端异常、输入及源码冻结验证通过，且源散列与当前文件逐项相同，见汇总v2的`final_source_validation.json`。主束和SAM束的说明、截图及实际GL验证均在各自`README.md`和`interaction/`中。只读能力审查已经完成；当前结论是先撤下“现有方法不能解决中心定位”的主张，保留真实工艺可用区间与可靠性作为待检验方向。
