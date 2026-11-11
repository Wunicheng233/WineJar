# 现有方法能力审查：坛颈定位

2026-10-27。第一轮现成权重及接口适配审查完成。仅使用既有仿真观测，不连接机器人或相机，不训练新模型，不提交或推送。

**后续已实际完成：[正确几何读出、标准SOR和冻结后新8例复核](geometry_capability_20261027.md)。**近距FS修正版原12两条件均12/12真正输出、中心≤3mm；新8两条件候选中心8/8≤4mm，接受clean8/shared7，另有轴/带宽问题。传统SimSense经正确读出在这两批也全部中心≤4mm。原MuJoCo `view_structural_study.py --ground-truth`已接入本轮，成为主展示；以下原冻结后端的0接受率及专用Matplotlib入口保留为第一阶段历史，不能当作最新修正版表现。

后续展示：[仿真、三个方法机理与交互回放入口](simulation_and_methods_20261027.md)，专用CLI为`scripts/view_capability_audit.py`，可逐坛查看本报告实际保存结果。

本轮落实[科学问题与AI基线审查计划](scientific_question_ai_baselines_20261026.md)：先测试成熟方法及必要适配，再决定剩余缺口是否值得方法研究。此前已查看的12坛型在远、近两个固定布局中重复观察；这是固定能力审查集，不是24个独立物体，也不是未见泛化测试。另4个开发坛型供接口验证及L-NDF监督示教/配置选择。

## 已确认的关键发现

**现成 FoundationStereo 已显著缓解近距问题。** 同一红外图像、同一几何后端中，12例候选截面同时满足圆心≤4 mm、法向≤2°、直径≤2 mm的数量，从传统SimSense的5例提高到11例；clean和显式共享噪声条件均如此。网络使用官方完整预训练权重，没有训练或微调。这不能支持“现成AI做不了坛颈定位”的笼统前提。

候选达标与实际接受输出分开报告。冻结portfolio仍含已证伪的截面切点/整坛轮廓关系，并影响候选评分；其近距接受为0，不能直接用作FoundationStereo的能力失败证据。单纯删掉拒绝原因也不能冒充修正后的新算法成绩。

理想原生IR深度经相同1 mm量化和注册后，近距portfolio候选10/12、正确接受0/12，远距候选11/12、正确接受8/12。它是**后端正对照**，不是严格的成绩上界：读出、非线性分支和选择规则会导致输入更准确而任务成绩不单调，不能由11对10声称网络优于真值。

## 三组方法与适配边界

| 分组 | 实际输入与监督 | 已完成的推理 |
|---|---|---|
| FoundationStereo | 保存的左右IR；官方ViT-large权重，32次细化；不读目标尺寸/测试标签 | 168对，包含4开发+同12坛型的两个布局、3相机、clean/shared_noise |
| P2P-Bridge | 历史普通RGB-D，三视观测上部70 mm，8192采样点；官方XYZ/EMA权重及5步OT-ODE | v2共48云：12坛型×2布局×raw/ideal |
| L-NDF | cam_a普通RGB-D；4开发坛型的监督功能坐标系；官方描述符与优化器 | 开发16次选择后冻结cosine/全前景配置，正式48次全部完成 |

FoundationStereo和SimSense接收逐文件哈希相同的左右IR；clean均关闭SimSense内部噪声，共享噪声先在CPU生成保存再同时输入两个方法。共享噪声遵循既有合成分布，但不是历史CUDA随机样本，也未拟合实物D435i。P2P/L-NDF的历史raw与该共享IR实验不同，三个分组不能直接拼成统一输入条件下的排行榜。

P2P保留相同裁剪/采样的未恢复输入，与恢复云使用同样来源相机z-buffer重投影和1 mm量化，再运行冻结后端。v1错误沿用V>35旧掩码，漏掉暗颈；该批保留为接口诊断。v2已统一正式蓝幕前景、记录掩码及设置哈希，并核查48份采样输入与输出。即使v2失败，也须先看理想深度采样对照是否正常，不能把稀疏重投影损失当作网络恢复能力。

L-NDF只输出中心/轴，不估计半径，因此不参加完整圆参数通过率比较。官方bottle cosine配置在开发上优于直接L1；全前景和上部70 mm的16次开发选择在审查前落盘。两候选开发raw均0/4达到4 mm/2°；按事先规定的归一化中位误差选择全前景。开发也参与示教，数字只用于适配，不能称泛化成绩。查询圆环由开发尺寸构造，在模型空间固定，测试还原后的物理尺寸会随观测缩放变化，并非每坛恒定3 mm带。详见[L-NDF专项记录](lndf_capability_20261027.md)及[P2P专项记录](p2p_capability_20261027.md)。

## 1. 同输入双目前端比较

下表均为三视输入与冻结portfolio后端。每格分母12，候选同时满足4 mm/2°/2 mm；后两列按原门槛的实际输出计算。法向2°、直径2 mm是研究分析阈值，尚非酒厂正式验收条件。圆参数达标不等于3 mm整带落槽，更不等于实机扎口成功。

| 布局/IR条件 | 前端 | 候选几何达标 | 正确接受 | 错误接受 |
|---|---|---:|---:|---:|
| 近/clean | SimSense | 5/12 | 0/12 | 0/12 |
| 近/clean | FoundationStereo | 11/12 | 0/12 | 0/12 |
| 近/shared_noise | SimSense | 5/12 | 0/12 | 0/12 |
| 近/shared_noise | FoundationStereo | 11/12 | 0/12 | 0/12 |
| 远/clean | SimSense | 0/12 | 0/12 | 0/12 |
| 远/clean | FoundationStereo | 1/12 | 0/12 | 1/12 |
| 远/shared_noise | SimSense | 0/12 | 0/12 | 0/12 |
| 远/shared_noise | FoundationStereo | 2/12 | 2/12 | 3/12 |

近距clean的FS候选圆心/法向/直径误差中位为1.808 mm/0.571°/0.370 mm；shared_noise为1.676 mm/0.561°/0.678 mm。全部候选误差均保留，没有只挑正确接受的例子计算中位。旧stereo_robust后端近距候选clean8/12、shared5/12，也全部拒绝；cam_a单视portfolio近距两条件各3/12。完整逐例、旧后端和单视结果见JSON，不能用三视数字代表单视能力。

**按实际主需求进一步拆分：**两近距条件的候选圆心均12/12≤3 mm，法向均12/12≤2°。clean最大圆心2.305 mm、法向1.446°；shared最大圆心2.128 mm、法向1.822°。完整圆参数11/12的唯一未达标者均为stress_002：clean圆心1.285 mm/法向0.310°/直径2.126 mm；shared为1.716 mm/0.671°/3.051 mm。其失败仅来自暂设2 mm直径阈值，不能据此断言厂方主要中心定位要求未满足。派生统计保存于汇总目录`near_pose_metric_breakdown.json`，未改阈值或重算推理。也不能将这些候选数字称为实际正确接受或实物精度。

**深度RMSE与任务几何并不单调对应。** 同一外侧颈段、同一共同有效像素中，每坛先合并三相机平方误差，再对12坛取中位：

| 条件 | SimSense颈区Z-RMSE mm | FoundationStereo颈区Z-RMSE mm |
|---|---:|---:|
| 近/clean | 3.001 | 3.790 |
| 近/shared_noise | 2.988 | 4.186 |
| 远/clean | 3.731 | 23.123 |
| 远/shared_noise | 3.715 | 29.472 |

近/clean的有符号均差中位为SimSense−1.652 mm、FS−0.876 mm，但均差变化并未独立干预，不能称已证实的唯一机制。FS近距候选更好，同时点级RMSE更大，说明误差分布、支持选择和几何读出共同影响任务；远距则有明显零样本退化。168对、504个区域的交集像素数逐项一致，未因方法漏点而换评价域。这个结果本身不是新的“任务指标优于RMSE”理论贡献。

![固定合成能力审查](/home/wunicheng/Git/WineJar/experiments/neck_localization/results/capability_audit_20261027_v1/depth_and_geometry.png)

图中左栏是按有效射线数汇总的原始Z-RMSE，右栏是候选计量通过数；与上表“每坛池化再取中位”的统计口径不同。中栏给固定参考域覆盖率。理想控制只评clean，绿色在noise处空缺，不表示零通过。

## 2. 通用点云恢复的配对收益与损伤

P2P v2已修正前景接口；下表同一8192点、同裁剪/来源相机重投影、同portfolio读出，比较未恢复与恢复。历史raw与FS共享IR条件不同，不能将7/12与11/12直接排名。

| 输入/布局 | 未恢复采样云候选达标 | 恢复后候选达标 |
|---|---:|---:|
| 历史raw/近 | 5/12 | 7/12 |
| 历史raw/远 | 0/12 | 0/12 |
| 理想RGB深度/近（控制） | 10/12 | 2/12 |
| 理想RGB深度/远（控制） | 12/12 | 12/12 |

近/raw直径误差中位2.335→1.182 mm，但圆心1.463→1.598 mm、法向0.511→0.859°；近/ideal直径0.158→2.906 mm，圆心1.640→1.082 mm。其收益与损伤必须同时记录。portfolio实际正确接受均为0；旧后端结果另存，远/ideal未恢复12/12正确接受、恢复0/12，其中恢复候选9/12达标，进一步说明不能只看一套拒绝门槛。

这支持“该预训练恢复及适配流水线可能改变细结构计量”的配对现象。恢复点横向移动还会改变像素碰撞、空洞和有效支持；当前未将其与曲面形变独立隔离，且未进行同数据普通域适配。因此尚不能把损伤归为P2P原理缺陷，更不能仅追加一个几何loss就主张新颖性。

## 3. 少示教功能帧直接迁移

L-NDF正式48次均成功执行。下表只评价中心与无向轴，不包含直径。

| 布局/输入 | 圆心≤4 mm且轴≤2° | 圆心误差中位 mm | 轴误差中位 ° |
|---|---:|---:|---:|
| 远/历史raw | 0/12 | 89.37 | 30.76 |
| 远/ideal RGB深度 | 0/12 | 56.77 | 18.94 |
| 近/历史raw | 0/12 | 127.05 | 50.06 |
| 近/ideal RGB深度 | 0/12 | 120.00 | 43.94 |

该配置不满足本任务精度。但四示教开发ideal也未通过基本毫米级控制，查询几何、跨示教描述符平均、观测范围归一化和预训练域均可能影响结果。描述符控制采样的局部诊断亦不能替代正式优化。可成立的结论是**这套明确适配的直接迁移不奏效**；它不足以排除更合理query、局部几何细化或普通功能监督微调。不能把0/48写成L-NDF类方法能力上限或论文缺口证明。

## 评价与复核

深度/点云评价对每个case×condition先完成全部指定后端推理，再读取目标/轮廓真值评分；L-NDF另用独立evaluate子命令在48次预测全部保存后读审查真值。两冻结后端为`estimate_portfolio`、`estimate_stereo_robust`；不修改阈值、拟合算法或拒绝门槛。计入错误接受、拒绝和输入/后端异常；另报候选误差，存在所需区间读出时计算工艺带余量。

测量误差使用固定理想IR射线定义的评价区域，单列普通RGB前景、固定工位上部和仅评价用的外侧颈段标签；同时报告有效覆盖率，以及两个匹配器共同有效像素上的误差，防止靠漏掉难点降低RMSE。

已核查SAPIEN角原点K与整数像素中心K的0.5 px主点差：在本轮rectified原生IR同K身份注册中完全抵消，两个版本的f/B相同，注册系数逐项相同；转世界坐标统一使用已减0.5的K，未重复减半像素。诊断保存在`results/capability_intrinsics_audit_20261027.json`。

本机RTX4050 Laptop 6GB，FoundationStereo168对全部成功；稳态前向及输出拷贝耗时中位2.388 s/对，峰值分配3.193 GiB。两个独立进程对同一开发pair的视差、深度、有效掩码逐项完全相同。P2P v2 48云均成功，恢复中位约1.21–1.23 s/云，峰值约246 MiB；这些耗时不含采集传输，不能据此判定未知工位时限。

本轮944条几何后端记录的输入/源码/设置校验均通过，其中192条是旧P2P v1接口诊断，不计方法证据。L-NDF48预测及其20初值坐标框架另经完整性审计。最终98项离线检查96通过、2默认GL跳过；无失败。共享IR与源配置在收尾再次核查哈希，168对未变化；来源快照及结果校验保存在汇总目录的`evidence_manifest.json`。

## 对科研选题的判断

**有值得继续证伪的共性问题候选，尚未证明论文所需的方法缺口。** 本轮最直接改变判断的证据是现成FS的近距5→11：单纯替换深度前端已经有效，应归入成熟方法的工程复用。后端已知错误关系、旧掩码或坐标错误的修正均是基础工作。

更具体的候选问题可以收紧为：

> 有限可见、无精确实例CAD、存在结构相关测量误差时，如何跨实例恢复由连续曲面局部低谷定义的功能截面，并保持其位置、朝向及尺寸？

适用类先限制于局部近旋转的短缩颈/环槽，仍需非同一参数化坛型的独立形状族验证。12坛型两布局不证明能解决“一类问题”。本轮中“点级误差与功能参数不同向”“恢复改善raw却损伤ideal”是现象线索；同数据微调、正确可见轮廓约束和常规几何读出能否消除它们，尚未检验。

按导师标准，下一步的顺序应为：

1. **先让成熟组合成为可用强基线。** 以现成FS和原生观测为输入，修正已证伪的轮廓包络/截面关系与必要读出问题，重新完整运行。验收仍分候选、正确接受、错误接受与整带余量；不能在旧预测上删除拒绝字段后报成绩。这一步不称创新。
2. **做最低限度的公平学习适配。** 使用同样开发标签预算，比较普通功能参数监督、合适的功能帧query/几何细化及通用恢复域适配；留出独立形状族与测量条件。若它们解决任务，就接受成熟方法可解，停止当前缺口主张。
3. **只研究残留的可重复缺陷。** 若正常适配后仍出现随局部尺度、曲率、可见支持变化的系统定位/尺寸偏差，再隔离曲面改变、偏差和支持变化，提出有约束的观测或恢复方法。必须与普通校准、鲁棒拟合、任务加权loss、直接回归及简单微调比较。追加neck head、RL动作、LLM模块或loss本身不构成贡献。

当前不建议据此开始新网络、RL或LLM方法论文。首先完成正确读出与普通适配的证伪，才有依据决定是否沿“测量/恢复与局部功能几何的一致性”继续研究。实物参考几何、D435i真实误差及跨形状族推广均尚未验证。

## 产物与复现

以下路径相对于`experiments/neck_localization/`。正式结果下含summary、逐例记录、输入/源码/设置散列及validation；旧P2P v1仅作接口诊断。

| 内容 | 路径 |
|---|---|
| 共享IR清单 | `runs/capability_inputs_20261027_v1/manifest.json` |
| SimSense/FS实际推理 | `runs/simsense_capability_20261027_v1/`、`runs/foundation_stereo_capability_20261027_v1/` |
| 同后端计量评价 | `results/simsense_capability_20261027_v1/`、`results/foundation_stereo_capability_eval_20261027_v1/` |
| 理想IR后端控制，仅clean | `results/ideal_native_capability_eval_20261027_v1/` |
| 共同像素核查与每坛统计 | `results/capability_common_depth_summary_20261027.json` |
| P2P v2成对评价 | `results/p2p_capability_far_eval_20261027_v2/`、`results/p2p_capability_close_eval_20261027_v2/` |
| L-NDF开发选择 | `runs/capability_lndf_development_protocol_20261027_v1/selection.json` |
| L-NDF正式评分 | `results/capability_lndf_frozen_cos_full_20261027_v1/` |
| 汇总表、图及模型来源快照 | `results/capability_audit_20261027_v1/` |

FS上游提交`6e8806816b533e4d13ddbb95ffa907b797060a62`；官方23-51-11 checkpoint SHA-256为`60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1`，严格载入全部张量。模型与运行环境在`external_baselines/`隔离保存，控制环境未安装PyTorch。其余模型提交、权重与兼容层详见专项记录及汇总来源快照。

主要CLI为`prepare_capability_inputs.py`、`rematch_capability_inputs.py`、`run_foundation_stereo_capability.py`、`prepare_capability_reference.py`、`evaluate_capability_depth.py`、`run_point_restoration_capability.py`、`evaluate_capability_cloud.py`、`run_lndf_capability.py`、`summarize_capability_common_depth.py`、`report_capability_audit.py`。均位于仓库`scripts/`；输入/预测目录不可覆盖。示例：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run --no-capture-output -n winejar-stereo \
  python scripts/evaluate_capability_depth.py --manifest MANIFEST.json \
  --predictions FS_PREDICTIONS --common-with SIMSENSE_PREDICTIONS --output NEW_EVAL_DIR

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run --no-capture-output -n winejar-stereo \
  python scripts/evaluate_capability_cloud.py --predictions P2P_PREDICTIONS --output NEW_EVAL_DIR

conda run -n winejar-stereo python -m unittest discover -s tests/neck_localization
```
