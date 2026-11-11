# 坛颈定位工程推进任务书与交接

日期：2026-10-27。工作区：`/home/wunicheng/Git/WineJar`。

**2026-10-29后续停点：**[标定与跨工位位姿传递](uncertain_transfer_20261029.md)已加入实际偏差FK后的工具到位评价，比较原像素/不确定性标定/最小关节映射/整图鲁棒加权，冻结后新偏差批次有明显收益，但实际RGB改善小、带仍失败，不更换默认或授实机执行。新入口`view_calibration_arrival.py --ground-truth --large-neck`复用原MuJoCo，仅静态到位；完整复现与新失败命令见该报告。下一步独立跨工位基准及工具计量、修正整条路径/净空核验；下游视觉受限修正不默认加入。旧任务书范围保留，真实标定/采集/运动仍需当时授权。

**2026-10-28后续停点：**已按用户新尺寸约束完成[三工位离线实现与验收拆分](three_station_tying_20261028.md)。28模型按物理颈段6–10mm/总角≤15°保留11，不删原件；150mm可容纳内径枪头+60°张开、理想输送/遮盖/新TCP与裸臂下降IK、原MuJoCo回放完成。正式结果`three_station_tying_20261028_v5`，44主分支+88事后四组合；中心通过不能替代3mm带包含（FS6/11、SS/SOR10/11）或最终扎口。无实物/锁紧剪断/全臂避障成绩，不发设备执行许可。完整复现与下一步测量在该报告，旧工程任务书/研究报告继续保留。

本文件供新的工程会话接手。目标是把已有坛颈定位方案推进为可在真实工位验证、部署和维护的技术。工程改进同样需要调研、假设、对照和证据；可以使用传统方法、成熟AI、合理适配或有依据的新方法，以实际效果决定取舍。论文新颖性不是本会话选方案的前置条件，也不以工作量或模块数量代替效果。

## 1. 可直接复制到新会话的提示词

> 请接手 WineJar 项目的坛颈定位工程推进工作，把已有方案逐步推进到真实工况下准确、稳定、可部署。先读取工作区 AGENTS.md，并依次阅读 docs/HANDOFF.md、docs/CURRENT_STATE.md、docs/RUNBOOK.md，再完整阅读 docs/engineering/neck_localization_handoff_20261027.md 和其中的最新结果报告。你的职责是提高实际技术效果，同时保持调研精神：理解失败机制，查找最相关的成熟方法，优先复用并进行合理适配，通过公平对照决定是否保留改进；必要时可以提出新的方法设计。请先核实已有资产与当前停点，给出主要问题、证据和优先级，然后在当前授权范围内推进最有价值的一项可复现改进，完成验证和交接。重点区分估计误差、算法接受/拒绝、真实工艺可用性和最终扎口成功，保留失败案例，避免只调阈值或在已看案例上报高成功率。以实际需求验收，不为追求论文创新刻意增加难度，也不把工程优化机械理解为只拼接已有模块。沿用原 MuJoCo 查看器，不另建展示框架。保留已有未提交改动和原始数据；实机、采集、运动、训练及资源使用按新会话当时的明确授权和仓库约定执行。发现具有共性的科研线索时，整理问题、已有方法证据和最小验证方案，供用户带回科研会话讨论。

## 2. 开始前的阅读顺序

1. 根目录 [AGENTS.md](/home/wunicheng/Git/WineJar/AGENTS.md)。
2. 依次读 [HANDOFF.md](/home/wunicheng/Git/WineJar/docs/HANDOFF.md)、[CURRENT_STATE.md](/home/wunicheng/Git/WineJar/docs/CURRENT_STATE.md)、[RUNBOOK.md](/home/wunicheng/Git/WineJar/docs/RUNBOOK.md)。这些文件保留多个历史停点，以最新时间、当前用户指令和实际代码/数据为准。
3. [正确读出与成熟几何能力复核](/home/wunicheng/Git/WineJar/docs/research/neck_localization/geometry_capability_20261027.md)：当前主要事实来源。
4. [现有方法能力审查](/home/wunicheng/Git/WineJar/docs/research/neck_localization/capability_audit_20261027.md)与[实验入口](/home/wunicheng/Git/WineJar/experiments/neck_localization/README.md)：了解FS/P2P/L-NDF的适配及证据边界。
5. 按具体问题补读[传感器误差与方法调研](/home/wunicheng/Git/WineJar/docs/research/neck_localization/uncertainty_methods_review.md)、[固定输入失败归因](/home/wunicheng/Git/WineJar/docs/research/neck_localization/failure_attribution_20261026.md)。历史负结果能避免重复试错，但不构成禁止改进的结论。

开始时检查 `git status --short`、相关路径/模型/环境是否仍存在及当前资源状态。仓库有大量已有未提交改动，包括现场标定和补偿文件，不能重置、覆盖或擅自提交推送。

## 3. 任务与真实需求

目标是凸起口沿下实际用于扎带的短颈/窄槽，估计中心、局部方向和必要的尺寸/可用范围，为后续执行提供依据。

已知约束：

- 中心绝对定位误差目标最大4 mm、优选3 mm。需要明确测量坐标系及独立参考精度，不能以拟合残差或目测验收。
- 用户确认带宽3 mm，物理颈段至少约6 mm。物理颈段、半径近最小代理区间、真正工艺可用区间三者尚不能直接等同。
- 坛高、局部颈位置和方向有个体差异，含偏颈、歪颈及弱边界；整坛按固定摆放处理。
- 背景清楚、无外物遮挡；每坛经过输送带时一次识别机会。不默认可转坛、移相机补看或大量多帧平均；一次窗口允许的帧数和工位时限尚待确认。
- 现有强结果是三视图近距。单视图、原远距布局和特定速度尚不是已确认的强制验收要求。
- 法向2°和直径2 mm是研究分析阈值，尚非厂方正式验收指标。可以分析它们与实际工艺的关系，但不能直接升级为用户承诺。

优先从文档和已有数据找答案，再向用户一次性询问确实缺失且影响决策的信息，例如真实可用区间定义、节拍、可用观测窗口、计量条件和当前硬件布置。已经明确的授权与要求无需重复确认。

## 4. 接手时已有哪些成果

截至本任务书日期：

- FoundationStereo使用官方完整预训练权重、32次细化，没有WineJar训练或微调；与SimSense接收相同保存IR，clean/shared_noise逐文件配对。
- 修正版 `visible_profile`：保留原鲁棒拟合/区域反馈，修正了不成立的“内部颈截面切点贴整坛轮廓”关系。每个候选重新检查有限拟合曲面的单侧投影包含，再重新排序。
- `standard_sor`：独立常规对照，固定支持区域，以Huber损失联合拟合轴和B样条半径曲线，不使用区域反馈。它不是某篇历史论文的完整复现。
- SAM2 tiny：整坛粗RGB框固定外扩8 px、按模型最高预测分数选mask、最大外轮廓填孔，不使用深度、颈部真值ROI或CAD。它修复了蓝幕规则误删陶坛蓝色反光的问题，但不是完美分割或已校准置信度。
- 原12坛已经反复分析；另8坛首次评价在方法和配置冻结后生成，仍属同一圆截面族。之后的SAM2配对复核使用已看对象，不能再称第二次独立测试。
- FS修正版在原12近距两条件均全部接受、中心≤3 mm；首次新增8坛两条件候选中心全部≤4 mm，实际接受clean8/8、shared7/8。
- SimSense＋SAM2＋标准SOR在这8坛两条件均8/8中心≤4 mm、法向≤2°、轴向带包含。直径研究指标仍有偏差，不能据此宣布真实扎口全部成功。
- 正式几何汇总488条，输入/源码验证通过、无输入或后端异常；当时128项离线检查126通过/2默认GL跳过。两个MuJoCo查看束另有实际GL交互验证。这些是交接时历史结果，新修改需对应验证。

这表明成熟路线值得工程推进，尚未证明任何一套方案在现实中整体最优或满足产线可靠性。选默认方案需要结合任务误差、误拒、错误放行、真实工艺、耗时和维护成本；不能仅凭“更先进”或某组8/8选型。

## 5. 已知问题：避免重走的弯路

| 问题 | 已观察的证据 | 工程处理原则 |
|---|---|---|
| 好圆被拒绝 | FS新8/004/shared原mask中心0.667 mm，仍因曲面投影越界被拒绝；SAM后修复 | 区分目标精度、上下文拟合和分割误差，定位实际触发点 |
| 新mask仍会误拒 | FS＋SAM新8/006/clean中心0.660 mm、法向0.348°，曲面最大越界2.395 px触2 px门槛 | 不能声称SAM解决全部问题；分析整段曲面验证与目标任务的关系 |
| “错误接受”可能只超研究阈值 | 原12 stress_002/FS修正版clean中心1.318 mm、直径误差2.200 mm | 指明失败指标，避免把中心达标也叫定位失败 |
| 小拟合残差不保证绝对正确 | 标准SOR存在过中心15.815 mm的错误接受；固定SAM适配后该组改善 | 接受器依据观测工作，真值只用于事后检验，可靠性需要独立评价 |
| 中心达标可能仍带越界 | 新8/005部分FS分支有约2.5°倾斜和毫米级轴向越界 | 分开评中心、轴、带几何；实际带材变形、滑移和收紧尚未模拟 |
| 某个FS反例不等于普遍难题 | 同一例可被传统SimSense＋SAM＋SOR改善 | 各强基线公平比较，保留成熟简单方案 |
| 仿真精度不等于现实精度 | 当前准确K/T、圆截面、合成纹理及传感器假设 | 优先确认真实误差来源和模型失配，保留测量不确定性说明 |

直接放宽阈值可以作为明确标注的诊断对照，但不能自动叫算法提升。若重设计接受器，应说明它要保护哪项实际指标，并在独立数据上同时报告误拒、错误接受和有效输出率。算法运行时不能用测试真值作选择依据。

## 6. 工作方式：以调研与证据推进效果

每一项实质改动按以下闭环推进，规模与风险相称，不要求为小修复写长论文：

1. **描述具体失效。** 固定输入、版本和案例，报告连续误差与触发原因，先分辨输入接口/单位/标定问题、观测误差、几何失配、优化分支及接受器问题。
2. **查已有方法。** 先看项目历史和现有实现，再查最相关论文、官方代码和文档。记录输入假设、监督需求、资源、部署条件及与当前场景的差异，必要时浏览核实。不同失败机制对应不同检索，不无界罗列模型。
3. **提出机制假设。** 写明改什么、为什么可能有效、什么结果会否定该解释。仅有“加网络/换损失/多迭代可能更好”不足以指导大实验。
4. **最小可复现对照。** 使用相同观测、正确接口和合理预算；先做理想/已知几何正控，再做真实误差条件。保留原方法、失败、异常、全部分母和实际拟合支持点。
5. **决定保留、回退或继续。** 改善有证据才进入候选部署版本；负结果写清，不把每次改动都积累进默认流水线。
6. **检验泛化与重复性。** 开发数据可以诊断，独立数据用于检验固定方案。只有新问题需要时才新增合成例或扩大实验。对同一保存输入的确定性重跑，不等于对噪声、摆放、光照和相机状态变化稳定。

可以修改算法，也可以设计新方法。优先顺序是解决真实主要误差，评估成熟方案及合理适配，再判断额外复杂度是否必要。必要的学习/微调应写清训练标签、数据划分、参数、预算、对照和复现方式，并按当前会话授权执行；本文件本身不启动训练或硬件作业。

## 7. 建议的推进阶段与通过条件

### A. 离线接手与候选方案审计

先复核已有强基线，不重新运行全部历史实验。交付一张当前候选表：输入来源、前景方法、几何方法、接受规则、中心/轴/尺寸连续误差、误拒、错误接受、带几何和耗时。

据此选一个最值得解决的问题，完成一次有机制依据的改进和配对验证。接受器与点/圆估计器可以分别改进；不要默认目前所有拒绝都应该取消。

### B. 准备真实观测与独立计量方案

在实际采集授权前，可离线完成采集清单、记录格式、计量方案和可执行离线评估入口。已有真实数据先盘点复用。

进入真实采集时，根据当时明确授权执行：核实相机型号、原始左右IR/RGB/深度可用性、曝光/投影器、时间信息、内外参、单位、坐标系和多相机干扰。仿真使用的相机参数、渲染IR和误差分布不能直接当作实物事实。

独立参考测量应明确中心/轴/真实可用颈区间如何定义、测得多准。区分参考计量误差和算法误差；不使用算法自己的拟合结果作为唯一真值。训练/开发/验收数据应按实物实例分开，避免同一只坛不同帧跨集合造成过于乐观的结果。

### C. 真实输入下离线验证与部署适配

先用保存真实观测离线运行候选方案，排查输入域、分割、标定与深度几何问题，再做部署性能优化。记录完整链路耗时、内存/显存、版本和失败原因；工位节拍未知时先报告测量值，不虚设延迟验收目标。

部署接口需明确中心、法向、半径/区间的单位与参考坐标系、接受状态、原因、时间戳和标定/模型版本。不确定性未经校准时如实标注；拒绝和输入异常需有可观察的输出，不能静默沿用上一次坐标。

### D. 工位联调与工艺验收

先定位测量验收，再坐标链/执行精度验收，最后完整工艺验收，各阶段分别记录。若进入机器人控制，遵循AGENTS与RUNBOOK，读取当前实际TCP和控制器状态；历史成功、名义116 mm工具长度、旧坛口位置均不能替代当前事实。

本任务书不授权自动连接硬件、运动、清报警或清急停。现场动作依据新会话明确授权与当前状态执行。使用当前已授权的事项继续推进，不反复询问；真正缺少授权或关键条件时明确指出缺少什么。

## 8. 代码、结果和环境入口

| 用途 | 路径 |
|---|---|
| 原冻结多候选/区域反馈 | [structure_portfolio.py](/home/wunicheng/Git/WineJar/neck_localization/structure_portfolio.py) |
| 有限曲面验证修正版 | [visible_profile_baseline.py](/home/wunicheng/Git/WineJar/neck_localization/visible_profile_baseline.py) |
| 独立标准SOR | [sor_baseline.py](/home/wunicheng/Git/WineJar/neck_localization/sor_baseline.py) |
| 显式mask适配，无全局替换 | [masked_geometry_baseline.py](/home/wunicheng/Git/WineJar/neck_localization/masked_geometry_baseline.py) |
| SAM2缓存生成 | [prepare_geometry_foreground.py](/home/wunicheng/Git/WineJar/scripts/prepare_geometry_foreground.py) |
| 原掩膜配对评价 | [run_geometry_capability_stage.py](/home/wunicheng/Git/WineJar/scripts/run_geometry_capability_stage.py) |
| 显式mask配对评价 | [run_masked_geometry_capability_stage.py](/home/wunicheng/Git/WineJar/scripts/run_masked_geometry_capability_stage.py) |
| 带几何规划/评价 | [band_geometry.py](/home/wunicheng/Git/WineJar/neck_localization/band_geometry.py) |
| 正式统计与来源验证 | [summary v2](/home/wunicheng/Git/WineJar/experiments/neck_localization/results/geometry_capability_summary_20261027_v2/summary.json)、[source validation](/home/wunicheng/Git/WineJar/experiments/neck_localization/results/geometry_capability_summary_20261027_v2/final_source_validation.json) |
| 默认MuJoCo查看束说明 | [README](/home/wunicheng/Git/WineJar/experiments/neck_localization/results/capability_structural_viewer_20261027_v1/README.md) |
| SAM2独立查看束说明 | [README](/home/wunicheng/Git/WineJar/experiments/neck_localization/results/capability_structural_viewer_sam2_20261027_v1/README.md) |

原12及开发IR清单位于 `experiments/neck_localization/runs/capability_inputs_20261027_v1/manifest.json`；新8输入清单位于 `experiments/neck_localization/runs/geometry_confirmation_20261027_v1/inputs/manifest_confirmation.json`。预测、RGB、mask和来源路径以manifest/provenance为准，避免误用旧不完整渲染目录。

模型目录和部分runs被Git忽略，但原始观测、权重和缓存有价值，先盘点再决定清理或重新计算。需要交接的唯一源代码放入模块/CLI/实验目录，不只留在ignored数据目录。

环境分工：`winejar-stereo`用于当前离线仿真与几何验证；`winejar-vision`用于PyTorch/已有SAM2等视觉推理；`winejar-xarm-sdk`用于控制。先核实当前依赖，不把控制和视觉环境混装。

默认查看：

```bash
conda run --no-capture-output -n winejar-stereo \
  python /home/wunicheng/Git/WineJar/scripts/view_structural_study.py --ground-truth
```

SAM2配对查看：

```bash
conda run --no-capture-output -n winejar-stereo \
  python /home/wunicheng/Git/WineJar/scripts/view_structural_study.py --ground-truth \
  --viewer /home/wunicheng/Git/WineJar/experiments/neck_localization/results/capability_structural_viewer_sam2_20261027_v1
```

当前几何检查：

```bash
conda run -n winejar-stereo python -m unittest discover -s tests/neck_localization
```

更改vision/control时分别按AGENTS运行对应测试。复用用户认可的原MuJoCo查看器，画保存的真实输入与实际拟合点；真值仅明确标记为评价显示。历史冻结结果与新版本分开保存，不覆盖历史记录来更新“成功率”。

## 9. 每轮应交付什么

用简短、可核对的记录说明：解决什么失效；调研了什么及为何适用；改动和机制假设；固定输入/版本/划分；对照效果及全部失败；对真实精度/可靠性的结论边界；默认方案是否改变；下一步最有价值的工作。

持续更新本阶段工程进展文档及 CURRENT_STATE/HANDOFF 的必要摘要。实机实验按仓库约定记录输入、日志、实际结果、最终载荷/IO状态和下一步。离线运行不可写成实机验收。

如果发现值得交回科研会话的共性现象，整理独立短备忘：科学问题抽象、最接近工作、现成方法在合理适配后仍失败的证据、候选机制、最小证伪实验及可推广对象。由用户决定带回讨论；不以工程会话的观察直接宣称论文贡献，也不自行跨会话发送消息。
