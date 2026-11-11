# 坛颈窄槽圆：离线研究入口

2026-10-29最新：[标定、结构FK偏差与跨工位传递](../../docs/engineering/uncertain_transfer_20261029.md)。正式`results/uncertain_transfer_20261029_v1`，新PoE关节映射＋整图权重仅为离线候选：强结构误差P95 1.698→0.457mm、含坏图2.129→0.892mm，实际RGB仅1.268→1.014mm且总P95稍差，整带仍有原/新失败，不改默认。新`view_calibration_arrival.py --ground-truth --large-neck`沿用原MuJoCo看实际枪头静态到位；`reproduce_uncertain_transfer.sh NEW_DIR`完整重建并已独立复跑。上游视觉到下游统一基准、实际误差幅度、整条修正运动/接触/锁紧未验证，最终扎口null；报告含全部分母、源文献、失败查看、环境分工及后续计量计划。无设备/采集/训练/改现场标定/提交。

2026-10-28最新：[跨工位误差、带落点和净空对照](../../docs/engineering/cross_station_progress_20261028.md)，[有来源的模型/适用条件](../../docs/engineering/cross_station_uncertainty_review_20261028.md)。正式`results/cross_station_20261028_v3`，其中`timed_stop_main`为用户最新预设启停、小随机净到位误差主场景，不用编码器观测；编码器另作技术对照，运行时二次观测未做。固定批次偏差与逐周期误差分层，标准LS/Huber在独立参考点验证，不据已看坛分数选标定。SOR姿态+FS半径同11坛带各10/11且保留较好直径尾部，仅可选对照，不换默认；原共同负例保留。精确网格净空证书在假设2mm贴合包覆下两方案22/22，仍不代表扎紧成功。配置/复现命令/原MuJoCo新组合入口见报告；161核心项159过/2原GL跳过、实际GL66路径66路由与原465输入不变。所有新误差幅度为假设，未验证工厂概率/机器人真实精度/锁紧剪断。

2026-10-28三工位工程：[筛选、输送/遮盖与150mm内径扎带枪](../../docs/engineering/three_station_tying_20261028.md)。28模型按物理颈段6–10mm/总倾角≤15°保留11，原件不删；44条实际预测传递+xArm下降IK、88条四组合事后带评价。新`scripts/view_tying_stage.py --ground-truth`沿用原MuJoCo界面，左下切识别/遮盖/下降/闭合/收带，右侧保留t0RGB/深度；正式束`results/three_station_tying_20261028_v5`。中心通过不等于整带可用（FS6/11、SS/SOR10/11），收带仅运动学、未验锁紧剪断与全臂避障，未授实机执行许可。原定位入口和默认束保留。

2026-10-28工程更新：[接手、作业区验证与效果](../../docs/engineering/neck_localization_progress_20261028.md)。448配对记录含新8锁定测试；作业区验证恢复开发一次误拒但有小幅带回退，新8未见额外增益，保留可选、不换默认。新独立查看束`results/task_surface_viewer_20261028_v1`用原`view_structural_study.py --viewer ... --method task_surface`选择，旧束保留。真实工艺与扎口未验收，不能据中心4mm通过发执行许可。

用户最新偏好：主展示沿用原MuJoCo `scripts/view_structural_study.py --ground-truth`，新Matplotlib回放保留为辅助。原入口现已接入本轮FS/SimSense、旧后端与两种新几何读出，以及冻结后新增8坛型；P2P/L-NDF仍在辅助回放，未假称全部移植。

2026-10-27最新：[正确读出与成熟几何能力复核](../../docs/research/neck_localization/geometry_capability_20261027.md)。修正旧切点/整坛轮廓错误关系，逐候选重新验证有限曲面并排序；另实现标准Huber SOR对照。FS近距原12两条件均12/12输出且中心≤3mm；新8同族候选8/8≤4mm、7/8≤3mm，接受clean8/shared7。传统SimSense经正确读出也在这两批全部中心≤4mm，因此当前中心定位科研缺口不成立。FS无WineJar微调。法向和3mm带轴向包含仍有失败，不等于已证明新颖性。机器统计`results/geometry_capability_summary_20261027_v1/`；原MuJoCo正式束`results/capability_structural_viewer_20261027_v1/`，默认FS clean/visible_profile。344条v1 runner记录保留，原定位器/标定/控制不改。后续SAM2掩膜配对复核不覆盖首次冻结结果。

固定SAM2配对已完成：160条正式记录/0异常、72张cache完整性通过；新汇总`results/geometry_capability_summary_20261027_v2/`含488条正式几何记录。传统SimSense+SAM2+标准SOR在已看新8两条件均中心4mm/法向2°/轴向带包含8/8，不能只展示FS残余反例主张成熟方法失效。SAM复核属于已看对象的固定适配，非第二次独立测试；单独查看束`results/capability_structural_viewer_sam2_20261027_v1`用原脚本`--viewer`选择，不覆盖默认v1。128离线检查126通过/2默认GL跳过；原MuJoCo12条件40方法路径及SAM束8条件16方法路径实际GL均通过。

本轮三方法可视化与机理：[说明](../../docs/research/neck_localization/simulation_and_methods_20261027.md)。
`conda run --no-capture-output -n winejar-stereo python scripts/view_capability_audit.py`打开保存结果的专用三维回放；切11分支/远近/12坛，可打开实际IR输入，L-NDF仅画中心和轴。该入口不重新推理、不连接设备。

2026-10-27最新：[现有方法能力审查](../../docs/research/neck_localization/capability_audit_20261027.md)。
官方FS/P2P/L-NDF均实际推理；同IR近距候选5/12→FS11/12，接受仍受冻结错误门槛影响，远距仍有错接受。
P2P有收益也有尺寸损伤；L-NDF当前适配连ideal控制也差，不能当作方法缺口。全部为已看合成能力集，不是新未见或实机成绩。
汇总图表/模型来源快照在`results/capability_audit_20261027_v1/`，入口和完整复现命令见报告；98检查96通过/2默认GL跳过。

2026-10-26最新：[失败原因干预实验](../../docs/research/neck_localization/failure_attribution_20261026.md)。
近距主要受深度形变影响，远距还受偏轴影响；统一读出权重未改善，降低其主因优先级。432链路记录、72传感器读出及求解器诊断保留，命令在报告。
全部是已看案例/oracle归因，不是新定位成绩；冻结算法、传感器、查看器和工位配置未改。

2026-10-25最新：[多候选与结构验证结果](../../docs/research/neck_localization/structural_results_20261025.md)。
12新坛型、2个预先固定布局、624主比较完成；近距候选圆心/法向改善，但直径偏差与验证仍未解决，未替换默认算法。
另288次原生诊断已明确标为开封后分析。新增圆轮廓验证不普适，窄窗/相位补偿也未形成可靠方案；负结果全部保留。

```bash
# 从任意目录看本轮12+新8坛型，鼠标切布局/输入/算法，橙虚线为被拒绝候选
conda run --no-capture-output -n winejar-stereo \
  python /home/wunicheng/Git/WineJar/scripts/view_structural_study.py --ground-truth

# 仓库根目录复现两布局和全部624主比较，输出目录须不存在
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
conda run --no-capture-output -n winejar-stereo python scripts/reproduce_structural_stage.py \
  --output experiments/neck_localization/runs/MY_NEW_structural_reproduction
```

统计、实际输入/点集审计与新查看束导出命令见报告。以下内容保留各历史阶段范围，不能覆盖最新负结果。

已按用户反馈修正[模型与静态场景](../../docs/research/neck_localization/simulation_preparation.md)，
随后获准自主仿真/调试，完成[首轮理想深度闭环](../../docs/research/neck_localization/short_neck_baseline.md)。
未运行完整噪声矩阵，不操作实机。
目标是凸起口沿下实际扎带窄槽；新模型使用圆截面，并以连续近最细范围的轴向中点代表目标。

详细的输入输出、每步算法和失败查看方式见[系统机理与操作](../../docs/research/neck_localization/system_walkthrough.md)。

2026-10-25最新[成熟传感器模型、标定与补偿结果](../../docs/research/neck_localization/active_stereo_results_20261025.md)：
独立环境`winejar-stereo`，SAPIEN/SimSense主动双目、librealsense单帧处理、独立平面GP、实际棋盘图像标定。
8新坛型准确K/T+理想RGB深度8/8；主动双目/滤波/GP均0/8，工厂成功率未达标，不能读作真实D435性能预测。
已完整一键复现112条记录，原生IR点小对照和不依赖深度的轮廓失败亦保留。以下10月24日及更早段落均为历史。

最新查看器已加入**左侧鼠标导航**，点击可切六种观测条件、RGB/深度、全坛/实际拟合点、权重和拟合像素投影，
同坛同视角直接对照；快捷键保留，画面不再铺键位提示。
新增[配对深度消融与错误选区报告](../../docs/research/neck_localization/viewer_and_support_audit_20261025.md)，默认不启用滤波/GP。

```bash
# 本轮实际输入、点云、细圆/法向；G真值，D深度，L实际拟合点，方向键换坛型
conda run --no-capture-output -n winejar-stereo python scripts/view_active_stereo.py \
  --condition stereo_raw --ground-truth
# 理想成功：将condition改ideal_rgb；GP为stereo_gp；棋盘参数另加--parameters board

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
conda run --no-capture-output -n winejar-stereo python scripts/reproduce_active_stereo.py \
  --output experiments/neck_localization/runs/MY_NEW_active_stereo_reproduction
```

输出目录必须不存在，重渲染8坛型，使用已锁定独立标定产物；依赖、单独重建标定和全部统计见报告。
旧`view_neck_localization.py`无参数默认仍为此前常数偏差研究，不能用来查看本轮成像结果。

2026-10-24最新[夜间调研与偏差补偿结果](../../docs/research/neck_localization/overnight_uncertainty_results_20261024.md)：
开发108/108；首次新16坛型128条件127/128；再新8坛型40条件40/40，原法分别35/108、16/128、8/40。
只支持圆截面、准确K/T和假设加性偏差条件；共享世界外参平移仍可错误接受，置信度未校准，未训练RL。
最新一键复现（已完整实际运行，输出目录须不存在）：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  conda run --no-capture-output -n embodied-sim \
  python scripts/reproduce_bias_compensation_stage2.py \
  --output experiments/neck_localization/runs/my_bias_compensation_reproduction
```

含新8坛型渲染、40条原法和80条补偿对照，120条记录除耗时完全重现；原始/补偿后点云查看及其他批次命令见报告。
新增[不确定性与残差学习专项调研](../../docs/research/neck_localization/uncertainty_methods_review.md)，明确单次采集也可做软件多步RL。
查看器无参数默认已切到最新200预算补偿结果；加`--point-cloud --large-neck --overlay --ground-truth --band`直观看最新环带与点云。
历史算法需显式指定`--scenes/--records/--case/--method`；下方旧阶段默认入口说明仅为历史。

最新[固定深度偏差敏感度](../../docs/research/neck_localization/depth_bias_stage1.md)：12个已有≥6 mm颈段案例×9档，
3 mm带、准确相机、冻结原算法，共108次。零12/12；共同+1为11/12；共同+2有11拒绝/1尺寸超差错误接受。
仅A+2的候选12/12满足当前几何阈值却全拒绝，质量判据与偏差模型均需改进。偏差值是假设，非设备实测。
新入口`run_depth_bias_stage1.py`/`report_depth_bias_stage1.py`，实际扰动点云/错误接受/拒绝查看命令见报告。

最新[现场尺寸更正](../../docs/research/neck_localization/factory_band_3mm_update.md)：带宽3 mm、物理颈段至少约6 mm。
3 mm审计记录位于`results/band_width_3mm_20261023/records.json.gz`；同一旧48例47/48整带容纳，未生成新工况测试。
唯一越界旧压力样本颈长4.517 mm，低于新现场尺度。CLI默认带宽改3 mm，复现历史5 mm请显式加`--width-mm 5`。

最新[正确输出率改进](../../docs/research/neck_localization/reliable_output_stage1.md)：两轮独立测试、不利结果保留。
第二批48例三相机：旧区域39/48、V1为44/48、V2目标范围验证备用解48/48，无错误接受；仍仅理想同族仿真。
查看用`--method task_validated_output`，新场景/记录路径、求解路径标记和复现命令见报告。
已完成第一项现实约束[5 mm环带审计](../../docs/research/neck_localization/finite_band_stage1.md)：
原48条圆预测不改，近最细代理整带42/48、完整颈段假设45/48；两者均不是实际扎带验收。
`audit_neck_band.py --output NEW_DIR`复现；查看器加`--band`，正式记录位于`results/band_width_20261023/verified/`。
[输送与TCP接口](../../docs/research/neck_localization/band_and_station_contract.md)仍仅契约，尚无跨工位端到端或噪声验证。

最新[相机系数/鲁棒/区域反馈对照](../../docs/research/neck_localization/support_feedback_stage1.md)：
14开发252条、6个另锁新测试108条；区域权重改善法向，圆心端点问题仍在。
支持一键重建，所有记录除耗时外已逐项复现：

```bash
bash scripts/run_support_feedback_stage1.sh experiments/neck_localization/runs/my_support_feedback
```

查看新结果和实际权重（W灰低黄高、M换方法、1/2/3换视图）：

```bash
conda run --no-capture-output -n embodied-sim python scripts/view_neck_localization.py \
  --scenes experiments/neck_localization/runs/support_feedback_test_scenes_v1 \
  --records experiments/neck_localization/runs/support_feedback_test_v1/records.json \
  --case eval_slender_flat --method region_equal_trust --large-neck --weights --overlay
```

加`--ground-truth`才读取真值叠加；查看器不参与估计。当前默认无参数入口仍保留旧结果。

另完成[36个随机坛型冻结审计](../../docs/research/neck_localization/random_geometry_audit.md)：
24个假设常规范围+12压力范围，216条两方法/三视图记录，算法和相机不变。
用报告中的新`--scenes`/`--records`路径可在窗口中←/→遍历全部36例，包括拒绝案例。
默认复现入口：`scripts/run_random_neck_audit.py --output NEW_DIR`，之后运行`report_random_neck_audit.py --audit NEW_DIR`。

## 直观看场景、相机与定位结果

```bash
cd /home/wunicheng/Git/WineJar
conda run --no-capture-output -n embodied-sim python scripts/view_neck_localization.py
```

桌面窗口左侧为MuJoCo酒坛/相机布局和颈部近景，红色细圆线/箭头为估计圆/法向；
右侧为实际保存的三台RGB输入，D看深度。1/2/3切换既有一/二/三视图结果，O在输入显示副本上叠线，
G显示评估真值（默认关闭），方向键换酒坛。鼠标可旋转、平移和缩放主景或近景。
N放大近景，也可启动时加`--large-neck`。已修复三维GL深度状态挡住主景/近景叠线的问题，
旧窗口需要关闭重启；截图现在从实际窗口回读，不只保存CPU合成图。
该入口需要已有场景与记录、桌面DISPLAY，当前默认读取本机首轮留出批次。
重新生成场景后用`--scenes NEW_DIR/heldout_scenes --records NEW_DIR/heldout/records.json`指定。
查看器不重算估计、不改变测量场景或输入；[预览](results/short_neck_20261022/inspection/scene_inputs.png)。
启动时加`--point-cloud`或按P查看普通RGB前景+保存深度反投影的实际点云；C切来源相机颜色，
V切全坛可见前景/定位器工位裁剪。没有网格采样补全；显示抽稀不影响算法。

## 局部联合估计首轮（2026-10-23）

[实现依据、变量投影机理与结果解释](../../docs/research/neck_localization/joint_profile_stage1.md)。
新增局部轴/样条联合估计，保留冻结旧基线及同曲线固定轴消融；8开发/6新测试，仍是固定布局、理想深度。
单视图有收益，三视图法向相对旧理想口沿代理退步；没有真实精度或全面优越性结论。

```bash
bash scripts/run_joint_stage1.sh experiments/neck_localization/runs/my_joint_stage1
```

输出目录须不存在；已验证20项核心检查和完整72+54条数值复现，另4项查看器检查通过。
查看新测试结果：

```bash
conda run --no-capture-output -n embodied-sim python scripts/view_neck_localization.py \
  --scenes experiments/neck_localization/runs/my_joint_stage1/test_scenes \
  --records experiments/neck_localization/runs/my_joint_stage1/test/records.json \
  --case test_oblique_flat --method joint_axis_spline --large-neck
```

M切三种已保存方法，L查看新方法实际送入拟合器的点；G真值默认关。
[小结果与源码快照](results/joint_profile_20261023/README.md)。

## 优化前分析（没有布局搜索）

[联合估计、相机位置理论及现实边界](../../docs/research/neck_localization/optimization_analysis.md)
给出相近论文、推荐目标函数、局部导数、范围中点敏感性及可辨识性限制。
有限差分与预设解析退化例可运行：

```bash
conda run --no-capture-output -n embodied-sim python scripts/analyze_neck_observability.py \
  --output experiments/neck_localization/runs/my_theory_checks.json
```

输出须不存在；这不是新定位精度实验、噪声扫描或相机布局搜索。[核查记录](results/optimization_theory_20261022/README.md)。

## 首轮闭环一键复现

```bash
bash scripts/run_short_neck_stage1.sh experiments/neck_localization/runs/my_stage1
```

输出目录必须不存在；默认`embodied-sim`，新环境可设置`WINEJAR_SIM_ENV=winejar-neck-sim`。
包含17项检查、4开发/4留出模型、一/二/三相机、竖直与口沿方向代理两个基线、
普通RGB提取及显式GT诊断、失败归因和结果图表。没有把真值圆/高度/尺寸提供给普通估计器。
口沿与颈轴一致是当前受限假设；单视图存在错误接受，未验证抗噪或实机有效性。
留出场景已分析，不能再用作下一版未见最终测试。冻结源码不匹配时入口会拒绝。

## 当前准备入口

```bash
MUJOCO_GL=egl conda run --no-capture-output -n embodied-sim python scripts/prepare_neck_scene.py \
  --config experiments/neck_localization/configs/preparation_v2.json \
  --output experiments/neck_localization/runs/my_preparation_v2
MUJOCO_GL=egl conda run --no-capture-output -n embodied-sim python -m pytest -q \
  tests/neck_localization/test_jar_model.py tests/neck_localization/test_scene_preparation.py
```

新环境依据本目录`environment.yml`建立，环境名改为`winejar-neck-sim`。
预览与检查结果见[2026-10-22准备记录](results/short_neck_20261022/README.md)。
模型以坛底为物体系原点；精确目标只在`evaluation_only/`，三台相机固定且不跟随各坛目标真值。
`review/`额外视图仅为人审。新schema明确10 mm颈长不包含另设的6 mm口沿（口沿高度仍是假设）。
旧schema 1保留供历史核对，新入口会拒绝，以免混用尺寸定义。

## 下面保留早期探索实现，尚未定案

以下的coupon、人工槽带、目标系原点、唯一最小值与简单提取器属于旧探索模型；
不能套用到当前整坛模型或解释为用户认可的路线。旧产物保留，不自动重跑。

## 旧探索入口

已在现有 `embodied-sim` 环境运行，无需修改控制/视觉环境：

```bash
MUJOCO_GL=egl conda run -n embodied-sim python -m pytest -q tests/neck_localization
MUJOCO_GL=egl conda run --no-capture-output -n embodied-sim python scripts/run_neck_simulation.py \
  --config experiments/neck_localization/configs/benchmark.json \
  --output experiments/neck_localization/runs/my_benchmark
```

输出目录必须不存在，避免覆盖旧证据。`--quick` 是冒烟子集；`--development-only` 只跑开发集。
新机器可 `conda env create -f experiments/neck_localization/environment.yml`，将上述环境名换为
`winejar-neck-sim`。环境文件给出可安装范围；本轮精确版本及源码/配置 SHA-256 在结果
`provenance.json`。算法结果固定种子，耗时和底层数值仍可能随硬件/库版本变化。

## 代码与数据边界

- `neck_localization/`：相机几何、渲染、估计器、标定诊断、扰动和评估模块；不导入机器人或设备接口。
- `scripts/run_neck_simulation.py`：薄批量实验入口；配置/开发与测试种子在本目录 `configs/`。
- `tests/neck_localization/`：离线实现检查；不以测试通过代替实物精度。
- `runs/`：可再生 RGB、float32 米制光轴深度、诊断掩码、MJCF、相机参数及完整报告；本地忽略。
- `results/20261021/`：随版本控制保存的小体积原始记录、汇总、配置、真值和源码散列。
- `mujoco_xarm6/`：独立仓库，只读核查，本轮没有修改、提交或复制其全部资产。

原始 RGB-D/标定数据和已有未提交改动均保留。所有新源码在非忽略目录。

## 坐标、单位与目标

采用右手世界 `W`，+Z向上，长度为米；报告圆心/直径误差为毫米、角度为度。
物体系 `O` 原点在目标截面中心，+Z为坛轴；`T_world_from_object` 把 O 坐标送到 W。
目标环系 `R` 在本模型中与 O 重合，`p_R=[r cosθ,r sinθ,0]`。
圆没有可观测的面内自转。法向一律选 `n_world.z>=0`，当前场景远离符号退化位置。

`Camera.T_world_from_camera` 为 OpenCV相机坐标 (+X右、+Y下、+Z前) 到 W 的变换：
`p_W=R_WC p_C+t_WC`。投影先取逆再乘 K，反投影 `p_C=Z K^-1[u,v,1]`。
MuJoCo相机朝 -Z、+Y向上；转换 `R_W_GL=R_W_C diag(1,-1,-1)`。
针孔无畸变，`fx=fy=H/(2 tan(fovy/2))`，像素中心为整数，
`cx=(W-1)/2, cy=(H-1)/2`；分辨率640×480。相机实际 XML 变换有独立测试。

`Renderer.enable_depth_rendering()` 输出**光轴 Z、米**，不是射线距离、SDK Z16或彩色深度预览。
RGB/depth共用同一相机与像素，未模拟 RealSense 的双视点 RGB/深度遮挡对齐。
设置 `offsamples=0`，保留 float32 深度，不量化为整数毫米。
倾斜平面解析检查覆盖离轴深度与半像素主点；默认多重采样不适合作为本轮理想采样假设。

## 诊断条件和普通输入

1. `exact_edge_diagnostic`：精确投影圆与精确3D点，包含不可见圆弧；只检查求解器，不能作为可见性或图像精度成绩。
2. `mask_diagnostic`：渲染可见 geom-ID 掩码；目标真值仅生成器和评估器持有。仍有光栅离散、网格和遮挡。
3. `rgb`：仅 RGB 中低饱和度亮区域提取，无真值 ROI。coupon外轮廓向外半像素校正；槽带用形态骨架。
   暗背景、人工高对比槽带都是显式先验，不能推广到裸陶坛。
4. `no_artificial_contrast`：删除槽带人工颜色，检验当前提取器依赖；不等于已经实现自然槽边检测。

两个几何分别是平面圆环计量 coupon、带凸起口唇/窄槽/坛肩的简化旋转曲面。
coupon的外上边界仅为独立回归诊断，不能冒充扎带目标。
槽带轴向全宽1.4mm，最小半径截面位于其中点；可见像素厚度/骨架会产生偏差。
256边网格近似圆截面，R=55mm时边中点弦高约0.00414mm；它不等于解析真圆。
造型不是计量扫描/CAD，不含真实釉面、主动双目相关器、光学模糊或真实圆度分布。

## 估计器与拒绝

`depth_ls`：目标证据点反投影，PCA平面＋平面内代数圆；保留最简单基线。
`depth_plane_rays`：用观测深度估平面，RGB边界射线与其相交后拟圆；仅适合有平面支持的诊断目标。
`mv_robust`：>=2视图共享3D圆，解析圆锥投影/Sampson残差、伪Huber、均衡每视图权重；深度只初始化。
`mv_depth_fused`：再加深度3D圆距离；2mm尺度只是与1px残差比较的假设权重，不是独立概率模型。
消融：`depth_robust`（RANSAC）、`mv_l2`、`mv_unbalanced`。

没有使用已知真实半径。宽范围半径先验20–120mm、倾角上限55°；最少12点，最小角支持120°，
深度残差3mm、像素内点2.5px等阈值在开发阶段固定，均非实物验收门槛。
少点、短弧、单RGB无尺度、退化相机、越界、残差过大及求解失败均有拒绝原因。
低残差仍会错误接受共同系统偏差；不能自动批准运动。

## 批次与指标

开发种子7721、每种几何3例，共6例；测试种子20261021、每种几何9例，共18例。
参数/阈值只看开发集后固定；测试集未经按结果挑选或删除。
位置、尺寸、倾斜/方位、光照、距离变化，三种布局以及部分遮挡；噪声/消融在每种几何的前三例运行。
样本量仅用于首轮诊断，不能支持可靠的罕见失败率/未来最大误差承诺。

全部尝试进入分母：成功=接受且满足分析阈值；拒绝=算法拒绝；错误接受=接受且任一指标超阈值。
分析阈值为中心4mm、法向2°、直径2mm，后两项仅建议。另有错误接受在接受结果中的占比。
分位数对**已求解案例（含被拒绝者）**统计，未求解不伪造误差，另报 `solved_n`。
完整记录包含世界XYZ、法向/切向误差、法向符号、运行时间。时间仅估计器，不含渲染/提取；
总批次耗时另见provenance。法向/半径也需与周向接触总误差合并预算。

协方差为固定K/T/标签下局部加权Jacobian近似，排除系统误差、结构选择和模型不匹配；
检查中心95%椭球经验覆盖（3自由度χ²阈值7.8147）。不能把它直接称为绝对精度置信度。

虚拟标定使用28个训练板姿态、10个独立留出姿态的**合成已识别角点**，
不是渲染棋盘角点检测。求解K和相对外参，camera0真值世界位姿仅锁坐标规范，
不代表完成工位坐标的独立标定。原始像素和Z按求得K/T重新反投影再定位。

噪声因子、依据、假设和待补机制见[误差预算](../../docs/research/neck_localization/error_budget.md)。
所有观察域扰动均非传感器仿真：像素扰动不重新采样遮挡边界深度；运动不重渲染可见性；
空洞包含随机和扇区删除；标定/深度固定偏差与随机误差分开。
`shape_mismatch_ellipticity_4pct`只有名义圆参考，没有真实精确圆，单列压力结果。
