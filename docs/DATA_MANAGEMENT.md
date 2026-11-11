# 文件、数据与迁移规则

**2026-11-11跨设备发布：**维护源码、配置、报告、选定图表/统计及自包含两案例查看包按用户授权整理同步，见[根README](../README.md)和[PPT资源](../experiments/presentation_assets_20261111/README.md)。约2.5 GB结果不全入Git：`results/.gitignore`默认保留大结果在本地，小报告/图表/冻结源由`tools/prepare_research_publication.py`列清单后显式加入；已跟踪文件仍正常显示修改。精确文件在资源包`publication_manifest.json`，109批次目录在`data/result_catalog.csv`。下文“results进入Git”指审计选集，不代表所有点云、JSONL及重放输入。没有删除原始数据、标定或失败。

新两案例`experiments/presentation_assets_20261111/viewer`使用真实文件和相对路径，可在新克隆中查看。旧JSON的`/home/...`及历史软链接是来源记录，不是跨设备运行配置；只用已提交图表无需迁移大输入，完整重放仍需迁移或重建。历史散列针对当时版本，新发布不会回写旧冻结记录。第三方模型和独立仿真仓库保持原维护范围。

10-28工程配对结果在`experiments/neck_localization/results/task_surface_{development_*,locked_*,engineering_20261028_v2,viewer_20261028_v1}`，完整448记录/变更案例/负控/源码和配置审计保留。新输入/预测在`runs/engineering_target_surface_20261028_v1`、渲染在`runs/engineering_target_surface_render_20261028_v1`；查看束链接这些大输入，迁移需一起携带或按冻结配置重建。`task_surface_engineering_20261028_v1`是过严数值复核容差触发的中止目录，不作正式结果；原算法/研究阈值没有改变。已有旧488条正式结果与模型/标定不覆盖。

2026-10-26失败归因保留在`experiments/neck_localization/results/structural_failure_chain_20261026_v3/`、`structural_sensor_transfer_20261026_v1/`、`structural_clean_objective_20261026_v1/`与`structural_failure_report_20261026_v1/`。包含完整oracle干预记录、冻结源散列、图表和脚本审计副本；明确不属于上线估计器输入。复现依赖原结构批次runs，勿仅迁移小结果后假定图像仍可用；链路v1/v2为前置检查中止目录，保留记录不当正式结果。

## 什么进入Git

2026-10-21新增离线研究：`neck_localization/`、批次CLI、
`experiments/neck_localization/configs/`、`results/`与研究文档进入Git；
可再生渲染与完整批次在`experiments/neck_localization/runs/`本地忽略。
小结果保留完整压缩记录、配置、开发/测试场景、评估真值、环境与源码散列，
渲染可按固定种子重建，不依赖独立仿真仓库或历史实机数据。

主动双目批次另保留左右IR、原生浮点/Z16/滤波深度、RGB注册深度、模型/标定产物、
输入mask、实际拟合点与拒绝原因。理想深度/几何真值明确标诊断或放evaluation_only，不进入实际条件估计器。
`runs/`仍本地忽略；锁定GP/棋盘参数和80条主比较压缩记录已原样复制到
`results/active_stereo_20261025/artifacts/`，一键复现优先用这些小产物，附SHA256清单。
完整标定图像、原生深度和展示输入仍需迁移runs或按各自脚本重建。
批次中有指向源观察/geometry/assets的符号链接，迁移要保留被引用目录或显式解引用，不能只复制展示子目录。
小汇总、CSV、源码hash及输入审计在`results/active_stereo*`；实验失败版本也保留，不擅自删除腾空间。

2026-10-25结构对照在`runs/structural_stage_20261025_v1/`，另有原生诊断与两例重渲染；中断的旧近距渲染和配置序列化原件均保留。
`results/structural_stage_20261025/artifacts/`归档624主记录、288追加记录、锁定模型、冻结算法/最终脚本及SHA256清单。
这些是审计副本，活动代码仍在正式模块和scripts中，不能直接覆盖当前工作区。
`results/structural_stage_20261025_viewer/`含指向大观察/网格的符号链接及绝对路径dataset清单，迁移须携带源runs或重跑实验/导出。
不要只复制查看目录后把失效链接解释为数据缺失；小报告/配置/冻结源自身可独立保留。

代码、配置、标定结果、文档、单元测试、体积小且能解释结论的实验报告/日志快照进入Git。
原始RGB-D、模型、连续录制、大量掩码和临时输出留在本地被忽略的数据目录。
所有仍需维护的脚本都应放正式源码目录，不能只留在data中。

`docs/evidence/2026-10-14/`保留不可改写的执行快照；index记录本地原件路径和SHA-256。
日志内的绝对路径描述当时现场，不等于换机后必定存在。

## 数据分级

| 类别 | 保留策略 |
|---|---|
| 原始color、Z16 depth、IR、frame.json | 保留；外参/内参回放、诊断和误差验证需要 |
| 标定样本、手眼原始位姿、求解候选与独立验证 | 保留，包括被拒绝样本与失败原因 |
| 成功/失败执行日志、路线检查、压力与时序 | 保留；没有日志不能声称实验已执行 |
| 当前关键抬升/最终照片 | 保留；压力确认不能代替实物验证 |
| 模型权重、HF缓存 | 保留或明确下载来源，不在清理时顺手删除 |
| 预览/掩码/角点覆盖图 | 依赖具体实验；不是看到PNG就删，部分被验收和基准引用 |
| Python字节码、pytest/ruff缓存 | 可直接再生，可用维护工具清理 |
| 废弃视觉外壳PLY导出 | 确认旁边NPZ是同一组点后，可删除重复ASCII导出；保留NPZ与报告 |
| 历史一次性脚本 | 归档原件与散列；仍需执行维护的代码迁到experiments |

本次保留全部原始标定图像、失败记录、深度诊断数组和重要照片，没有为减少体积而随机抽样。
9月的大量多相机数据仍有回放价值，不能因当前暂停就整目录删除。

## 本地目录索引

| 本地目录（相对xarm_sdk_control/data） | 用途 |
|---|---|
| `recommissioning/2026-10-14/` | 当前恢复的标定、RGB-D、路线和三材料吸起记录 |
| `recommissioning/2026-10-12/`及其它复工子目录 | 初始接线、气路、手眼/相机核查 |
| `hand_eye/` | 历史手眼采样 |
| `complete_loading/` | 旧工位完整实验，含8月30日历史成功案例 |
| `multiview_neck/` | 多相机内外参、独立验证、原始场景、深度尺度与坛口测量 |
| `wrist_guided_pick_test/`、`suction_surface/` | 腕部吸取与表面候选回放 |
| `optimization/` | 性能优化回放 |
| `archive/` | 一次性脚本归档、清理清单等本地产物 |

精确目录大小和文件数用维护工具查询，避免文档中的统计长期失效。
本次统计快照在`docs/maintenance/`，它只记录当时状态。

## 盘点与清理

工具仅访问文件，不导入相机或机器人控制：

```bash
python tools/project_maintenance.py audit
python tools/project_maintenance.py clean
python tools/project_maintenance.py clean --apply \
  --output xarm_sdk_control/data/archive/cache_cleanup.json
```

`audit`检查目录大小、Markdown本地链接和证据散列；发现断链/快照改动返回非0。
`clean`默认只列出候选，`--apply`只清理明确列出的Python和开发工具缓存。
它排除Git、模型和独立仿真工作区，不自动删除任何实验数据。
旧PLY和一次性脚本的本次清理是单次审查操作，细目见[清理报告](CLEANUP_2026-10-21.md)。

## 换机器/换Agent

同一机器上的新Agent先读HANDOFF，运行audit，检查关键数据存在，再做现场只读检查。
换机器还要迁移以下内容：

1. 仓库改动。当前工作尚未自动commit/push，单纯clone远端不包含未提交整理成果。
2. 所需`xarm_sdk_control/data/`，至少完整复工目录；要复现多视图再迁移multiview_neck和其标定。
3. `vision/models/`以及GroundingDINO需要的Hugging Face模型缓存，或具备联网下载条件。
4. 根据environment.yml重新建环境；不复制整个Conda目录作为唯一环境说明。
5. 相机序列号、机器人网段与实机安装状态，现场重新核对。

只读盘点示例：

```bash
du -sh xarm_sdk_control/data vision/models
python tools/project_maintenance.py audit --output xarm_sdk_control/data/archive/inventory.json
```

迁移可使用rsync，但不要使用`--delete`清空接收端。先校验清单/散列，再移除旧副本。
本地`data/archive/recommissioning_20261014_scripts.zip`保留整理前一次性脚本原件；
恢复时先查看压缩包目录，按需解到新的临时目录，不覆盖当前正式源码。

## 更新约定

实验后更新CURRENT_STATE；重大成功/失败导出简短证据并更新index散列。
数据清理前确保备份、依赖和重建路径明确；不要批量改写历史JSON中的路径或判定。
