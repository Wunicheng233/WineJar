# WineJar：酒坛坛颈定位与扎口仿真

当前主线是估计**凸起口沿下实际扎带窄槽的截面圆**，再研究输送后如何把位姿传给机械臂。输出包括圆心、法向、半径及质量信息，目标不是坛口外沿。研究结果更新至2026-10-29；2026-11-11整理并同步跨设备资源。尚无实机扎口验收结果。

## 给制作PPT的AI：先看这些资源

**先读[图片与数据使用说明](experiments/presentation_assets_20261111/README.md)。做PPT直接读PNG、CSV和报告即可，不必安装仿真环境或下载模型。**

| 想找什么 | 资源入口 |
|---|---|
| 同一酒坛的单／双／三机圆环、法向和点云 | [三种配置都成功](experiments/presentation_assets_20261111/camera_count/nominal_014_comparison.png)、[单机拒绝的例子](experiments/presentation_assets_20261111/camera_count/stress_010_comparison.png) |
| 相机数量总体对照 | [36例统计图](experiments/presentation_assets_20261111/camera_count/all_36_cases.png)、[汇总CSV](experiments/presentation_assets_20261111/data/camera_count_summary.csv)、[全部216条CSV](experiments/presentation_assets_20261111/data/camera_count_records.csv) |
| 给非技术听众的流程、坛型、相机输入和动作图 | [6张高清图及历史PPT](experiments/factory_briefing_20261028/output/)、[指标说明](experiments/factory_briefing_20261028/output/指标与讲解.md)、[指标JSON](experiments/factory_briefing_20261028/metrics.json)、[原画面素材](experiments/factory_briefing_20261028/assets/) |
| FoundationStereo / SimSense和几何方法比较 | [最新汇总TSV](experiments/neck_localization/results/geometry_capability_summary_20261027_v2/summary.tsv)、[方法与结果](docs/research/neck_localization/geometry_capability_20261027.md) |
| 三工位、有限带宽、输送误差和净空 | [三工位报告](docs/engineering/three_station_tying_20261028.md)、[跨工位报告](docs/engineering/cross_station_progress_20261028.md)、[定时停位汇总](experiments/neck_localization/results/cross_station_20261028_v3/timed_stop_main/summary.json) |
| 相机—机器人标定六方法对照 | [报告](docs/engineering/uncertain_transfer_20261029.md)、[四主方法图](experiments/neck_localization/results/uncertain_transfer_20261029_v1/analysis/calibration_execution_comparison.png)、[主结果JSON](experiments/neck_localization/results/uncertain_transfer_20261029_v1/analysis/confirmation_summary.json)、[补充简化对照](experiments/neck_localization/results/uncertain_transfer_20261029_v1/final_audit_v2/simple_control_summary.json) |
| 所有已上传结果的位置 | [109批次资源目录CSV](experiments/presentation_assets_20261111/data/result_catalog.csv)、[发布清单](experiments/presentation_assets_20261111/publication_manifest.json)、[全部文档索引](docs/README.md) |

单／双／三机图是从**10月23日保存结果重新导出**，不是11月新定位试验：早期区域反馈算法、理想深度、准确K/T，单机A、双机A+B、三机A+B+C，没有搜索最佳布置。不能与后来较真实传感器条件的成绩混成一张排行榜。

旧对照常规24例单/双/三机分别19/22/23成功，压力12例为3/8/11。成功要求接受且中心≤4 mm、法向≤2°、直径误差≤2 mm；后两项当时是研究分析阈值，**不是最终扎带成功率**。原36例包含后来工况以外的颈长。目前6–10 mm颈段、倾角≤15°和成熟算法条件下的完整相机数量对照尚未完成，不能声称三台必要或最优。

## 算法在哪里

主定位使用三台RGB-D相机观测融合及局部几何拟合。一台RealSense本身也能产生点云，三台不是获得三维坐标的理论必需条件。

| 环节 | 源码与说明 |
|---|---|
| 参数化酒坛、材质、相机场景 | [jar_model.py](neck_localization/jar_model.py)、[scene_preparation.py](neck_localization/scene_preparation.py)；[模型与理想闭环](docs/research/neck_localization/short_neck_baseline.md) |
| 轴/剖面联合拟合、区域反馈 | [joint_profile.py](neck_localization/joint_profile.py)、[support_feedback.py](neck_localization/support_feedback.py)；[联合估计](docs/research/neck_localization/joint_profile_stage1.md)、[区域反馈对照](docs/research/neck_localization/support_feedback_stage1.md) |
| 较真实深度、预训练前端和固定分割适配 | [active_stereo.py](neck_localization/active_stereo.py)、[foundation_stereo_baseline.py](neck_localization/foundation_stereo_baseline.py)、[masked_geometry_baseline.py](neck_localization/masked_geometry_baseline.py)；[能力审查](docs/research/neck_localization/capability_audit_20261027.md) |
| 修正几何读出与常规旋转曲面基线 | [visible_profile_baseline.py](neck_localization/visible_profile_baseline.py)、[sor_baseline.py](neck_localization/sor_baseline.py)；[成熟方法复核](docs/research/neck_localization/geometry_capability_20261027.md) |
| 3 mm带、工位变换、150 mm内径枪头及净空 | [band_geometry.py](neck_localization/band_geometry.py)、[station_tying.py](neck_localization/station_tying.py)、[tool_clearance.py](neck_localization/tool_clearance.py)、[tying_scene.py](neck_localization/tying_scene.py)；[工程报告](docs/engineering/three_station_tying_20261028.md) |
| 标定与结构运动学偏差校正 | [fixed_camera_calibration.py](neck_localization/fixed_camera_calibration.py)、[uncertain_handeye.py](neck_localization/uncertain_handeye.py)、[poe_camera_calibration.py](neck_localization/poe_camera_calibration.py)、[block_handeye.py](neck_localization/block_handeye.py)；[六方法与失败](docs/engineering/uncertain_transfer_20261029.md) |
| 原MuJoCo界面、实际观测点云 | [inspection_viewer.py](neck_localization/inspection_viewer.py)、[point_cloud_display.py](neck_localization/point_cloud_display.py)、[tying_viewer.py](neck_localization/tying_viewer.py) |

控制/上料和视觉服务仍在[xarm_sdk_control](xarm_sdk_control/README.md)、[vision](vision/README.md)。最近有证据的实机进展为2026-10-14三类物料分别吸起；不代表当前硬件状态、完整上料或扎口验收。

## 换设备读取与看三维

```bash
git clone https://github.com/Wunicheng233/WineJar.git
cd WineJar
# 已有克隆：先保留当地改动，再使用 git pull --ff-only
```

PNG、CSV、源码及报告可直接读。新两案例包包含真实文件与相对资产路径，不依赖原机`runs/`或独立机器人仓库。要开三维，再按已有环境或环境文件准备MuJoCo依赖：

```bash
# 可选；已有 winejar-stereo 时不用创建新环境
conda env create -f experiments/neck_localization/environment.yml
conda run --no-capture-output -n winejar-neck-sim \
  python scripts/view_camera_count_comparison.py --point-cloud
conda run --no-capture-output -n winejar-neck-sim \
  python scripts/view_camera_count_comparison.py --case stress_010 --camera-set cam_a --point-cloud
```

左侧“参与估计的视图”切A、A+B、三机；红线为接受的圆/法向，青线为评估真值。读取保存结果，不重新定位，不连接设备。本机已有`winejar-stereo`可替换环境名。

## 已上传什么，哪些仍需迁移

此次同步包含维护源码、配置、测试、文档、选定图表/汇总及两案例查看包。[发布清单](experiments/presentation_assets_20261111/publication_manifest.json)列出具体历史文件。大批原始RGB-D/IR、逐点诊断、重复重放目录、第三方权重、临时PPT构建缓存及本机符号链接仍本地保留，未删除、未全部上传。

完整实验入口见[研究README](experiments/neck_localization/README.md)、[RUNBOOK](docs/RUNBOOK.md)。除新两案例包外，历史查看束可能引用原机绝对路径或被忽略的`runs/`；**仅克隆不能运行所有旧查看命令**，需按报告重建或迁移输入。机器人资产另在[Winejar-mujoco](https://github.com/Wunicheng233/Winejar-mujoco)维护，本次没有修改或推送该独立仓库。

历史图、理想仿真、主动双目仿真和最终工艺须分开。实机绝对精度、全链置信度、完整碰撞/接触、锁紧和剪断均未验收；不要把动画、算法接受率、标定残差或仿真整带包含率标为实机扎口成功率。

## 接手与离线验证

先读[AGENTS](AGENTS.md)、[HANDOFF](docs/HANDOFF.md)、[CURRENT_STATE](docs/CURRENT_STATE.md)、[RUNBOOK](docs/RUNBOOK.md)，再读[工程任务书](docs/engineering/neck_localization_handoff_20261027.md)。历史文档在[docs/archive](docs/archive/)，不是当前运行参数来源。

```bash
conda run -n winejar-stereo python -m unittest discover -s tests/neck_localization
conda run -n winejar-xarm-sdk python -m pytest -q xarm_sdk_control/tests
conda run -n winejar-vision python -m unittest discover -s vision/tests
```

控制与视觉环境分开验证；普通开发/PPT准备只做离线操作。数据与迁移规则见[DATA_MANAGEMENT](docs/DATA_MANAGEMENT.md)。

<details>
<summary>历史README快照（截至2026-10-25，保留原工作记录；以上文最新入口为准）</summary>

# WineJar 酒坛视觉上料

xArm6 + RealSense RGB-D + GroundingDINO/SAM2 + 双吸盘真空末端。
当前主线是**扎带窄槽截面圆的方法调研与 MuJoCo 仿真**，本阶段不操作实机。
研究入口与复现命令见[离线实验](experiments/neck_localization/README.md)，
方法对比见[调研报告](docs/research/neck_localization/method_review.md)，
模型修正与首轮理想深度结果见[短颈闭环](docs/research/neck_localization/short_neck_baseline.md)。
实机上料记录保留，恢复工作暂停。

2026-10-25最新研究见[成熟传感器模型、标定与补偿](docs/research/neck_localization/active_stereo_results_20261025.md)：
主动双目成像、官方滤波、独立平面GP、实际棋盘标定和8新坛型对照已运行并完整复现。
理想RGB深度8/8；主动双目条件尚未可靠输出，未宣称实机有效。新查看入口为`scripts/view_active_stereo.py`。

此前[夜间不确定性调研与偏差补偿](docs/research/neck_localization/overnight_uncertainty_results_20261024.md)：
完成显式偏差联合估计、两批独立仿真和模型失配反例；提供已验证的一键重建与MuJoCo点云查看。
仿真中的固定偏差补偿不等于实机精度或通用标定误差处理，3 mm扎带与约6 mm颈段约束已记录。

## 接手入口

1. [Agent 交接](docs/HANDOFF.md)：阅读顺序、已完成事项、下一步和证据。
2. [当前状态](docs/CURRENT_STATE.md)：2026-10-21 整理时核查的最后实机记录。
3. [现场运行手册](docs/RUNBOOK.md)：环境、只读诊断、采图、标定和恢复流程。
4. [代码结构](docs/ARCHITECTURE.md)、[数据管理](docs/DATA_MANAGEMENT.md)、[全部文档索引](docs/README.md)。

**最后有证据的实机进展在 2026-10-14：白纸、陶瓷片、竹壳分别吸起并原位放回成功。
搬动底座后的完整四步上料尚未执行；3–4 mm 绝对精度尚未验收。**
2026-10-21 进行了仓库整理、文献调研及离线仿真，没有连接或移动机械臂。

## 当前参数与边界

| 项目 | 最后核实值 |
|---|---|
| 机械臂 | xArm6，控制箱 `192.168.1.212` |
| 全局 / 腕部相机 | `013222073103` / `011422070172` |
| 末端名义 TCP | `[0, 0, 116, 0, 0, 0]` mm/deg，运行前读取控制器核对 |
| 真空 IO | CO0 泵，CO1 吸取阀；1 启动/吸取，0 关闭/泄压 |
| CI0 | 0 表示达到真空阈值；去抖 50 ms、等待最多 2 s |
| 最近成功吸取 | 白纸/陶瓷额外下压 0 mm；叠放竹壳总共额外下压 8 mm |
| 当前全局标定 | 10 月 14 日实验候选，尚未通过独立绝对精度验收 |

旧坛口教点、旧 home、旧 `(-5,+44,0)` 补偿和旧工位高度避让条件均不能直接用于新工位。
已有配置中的新 `+10 mm Z` 补偿是待复核工作副本，最近三类成功试验没有应用它。

## 目录

```text
AGENTS.md                       Agent 工作约定
docs/                           当前文档、关键日志快照、历史归档
tools/project_maintenance.py     离线盘点与保守清理工具
neck_localization/              独立离线几何/渲染/扰动/评估模块
scripts/run_short_neck_stage1.sh 当前短颈首轮闭环入口
scripts/run_neck_simulation.py   保留的早期探索批次
experiments/neck_localization/   配置、复现说明及小体积研究结果
tests/neck_localization/         无设备的研究实现检查
xarm_sdk_control/
  scripts/                      通用控制、标定、诊断和历史任务入口
  experiments/                  版本管理中的工位专用恢复代码
  calibration/experimental/     明确标记的未验收标定候选
  calibration/                  标定与历史配置，不能按日期自动视为已验收
  config/                       机器人、末端、补偿和历史教点
  tests/                        无实机离线测试
  data/                         本地原始样本及任务产物，Git 忽略
vision/                         独立视觉环境、推理脚本和服务
xarm6/                          厂商手册
```

历史机器人仿真在独立仓库维护；本机 `mujoco_xarm6/` 是独立 Git 工作区，
本轮只读核查，没有修改。新的圆环计量实验在本仓库受版本管理。

## 安装与离线检查

```bash
conda env create -f xarm_sdk_control/environment.yml
conda env create -f vision/environment.yml
conda run -n winejar-vision pip install -r vision/requirements-sam2.txt

conda run -n winejar-xarm-sdk python -m pytest -q xarm_sdk_control/tests
conda run -n winejar-vision python -m unittest discover -s vision/tests
python tools/project_maintenance.py audit
```

模型权重与原始实验数据不在 Git 中。换机器须额外迁移它们；仅克隆仓库可以阅读
关键实验快照、运行控制层测试，但不能重放所有视觉实验。

完整上料旧入口 `21_complete_loading.py` 仍保留，其默认 home、料区和补偿与旧工位绑定。
当前恢复代码及使用限制见 [恢复实验说明](xarm_sdk_control/experiments/recommissioning_20261014/README.md)。

</details>
