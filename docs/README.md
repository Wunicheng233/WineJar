# 文档索引

| 文档 | 用途 |
|---|---|
| [HANDOFF](HANDOFF.md) | 新 Agent 快速接手、优先任务、证据索引 |
| [CURRENT_STATE](CURRENT_STATE.md) | 最后核实的工位、实验结果、未解决问题 |
| [RUNBOOK](RUNBOOK.md) | 环境安装、现场诊断、恢复与记录步骤 |
| [ARCHITECTURE](ARCHITECTURE.md) | 模块分工、数据流、测试入口 |
| [DATA_MANAGEMENT](DATA_MANAGEMENT.md) | 保留/清理规则、数据迁移、工具用法 |
| [圆环离线研究](../experiments/neck_localization/README.md) | 当前主线、MuJoCo批次、坐标与复现命令 |
| [10-28工程接手结果](engineering/neck_localization_progress_20261028.md) | 作业区验证/误拒、448条配对、新8锁定测试、带回退、原MuJoCo对照与部署边界 |
| [10-26失败归因](research/neck_localization/failure_attribution_20261026.md) | 固定输入单项干预、深度形变与偏轴、局部最优/目标偏差、同像素IR注册诊断 |
| [多候选与结构验证新结果](research/neck_localization/structural_results_20261025.md) | 12新坛型/2固定布局/624主比较、失败原因、原生追加诊断及新查看入口 |
| [主动双目与成熟方法结果](research/neck_localization/active_stereo_results_20261025.md) | 最新成像、官方滤波、GP、棋盘标定、负结果及一键复现 |
| [拟合区域与鼠标导航](research/neck_localization/viewer_and_support_audit_20261025.md) | 后处理配对消融、口沿混入证据、左侧可点击导航及验证 |
| [方法调研](research/neck_localization/method_review.md) | 六类方法证据、选择与适用边界 |
| [研究方向与阶段约定](research/neck_localization/research_direction.md) | RealSense视觉优先、科学问题候选、逐阶段讨论 |
| [视觉方案提案](research/neck_localization/visual_proposal.md) | 局部颈部形状、候选视图、深度处理与首阶段验证 |
| [最小模型草案](research/neck_localization/minimal_model.md) | 几何、观测、参数规范、最细截面敏感度与仿真前提 |
| [模型与静态场景](research/neck_localization/simulation_preparation.md) | 短颈、上宽下窄、局部歪颈、材质与RGB-D检查 |
| [当前系统机理与操作](research/neck_localization/system_walkthrough.md) | 输入输出、固定轴/半径剖面算法、失败过程图、复现与查看命令 |
| [首轮短颈闭环](research/neck_localization/short_neck_baseline.md) | 开发/留出、受限基线、错误接受与范围定义诊断 |
| [非视觉与工艺备选](research/neck_localization/alternative_routes.md) | 接触测量、机械定位、导带与专用轮廓测量来源 |
| [误差预算](research/neck_localization/error_budget.md) | 噪声机制、扫描依据和后续独立验收 |
| [CLEANUP_2026-10-21](CLEANUP_2026-10-21.md) | 此次整理范围、清理结果和验证 |
| [关键证据](evidence/2026-10-14/index.json) | 随 Git 保存的成功/失败日志及未执行计划快照 |
| [三相机技术说明](multiview-neck-localization.md) | 历史算法、复标和独立验收方法，供研究对照 |

历史资料保留其原始参数和结论，不能直接用于当前工位：

- [10 月复工全过程](archive/recommissioning/复工检查_2026-10-12.md)。
- [9 月标定交接](archive/legacy/标定工作交接_2026-09-10.md)。
- [旧工位实施指南](archive/legacy/WineJar项目实施与复现指南.md)，[Word 快照](archive/legacy/WineJar项目实施与复现指南.docx)。Word 不作为当前维护版本。
- [旧 README](archive/legacy/README_202609.md)、[原始策划](archive/legacy/项目策划.docx)。
- [历史优化总记录](archive/optimization/optimization-log.md)，原始报告仍在 `benchmarks/`。

冲突处理：当前现场实测和执行日志优先，当前状态文档优先于历史说明。
文件日期较新或训练误差较小均不等于当前场景标定已验收。
