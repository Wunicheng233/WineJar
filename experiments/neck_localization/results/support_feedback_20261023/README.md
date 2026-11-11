# 支持区域/鲁棒权重首轮证据

日期2026-10-23。仅离线理想仿真；未搜索相机、添加噪声、操作实机或改原冻结算法。

- 方法、结果、边界、查看命令：[研究报告](../../../../docs/research/neck_localization/support_feedback_stage1.md)。
- `plan.json`：新6例规格在实现前锁定；配置散列及开发/测试约定。
- `freeze.json`：开发完成后、生成新测试前冻结的9个源码、设置、配置及开发记录散列。
- `source_snapshot/`：冻结定位/生成代码；`review_sources/`为报告、查看、测试及复现脚本。
- `report/tables.md`、`report/test_comparison.png`：完整六组统计与新测试误差图。
- `report/development_records.json.gz`、`report/test_records.json.gz`：全部252+108条记录，含每点权重、每轮反馈、分轴误差、拒绝原因。
- `report/development_weight_profiles.png`：不读取真值范围的实际权重/剖面；红线为统一完整上下文读出。
- `test_weights.png`：实际GL窗口回读，包含参与相机、原始RGB、实际权重及可选评估真值。
- `checks.log`：24项核心检查；`viewer_checks.log`：4项及3个投影子检查。
- `development_validation.json`：42组固定平衡几何匹配旧冻结结果，Huber目标逐轮下降。
- `weight_view_validation.json`：252条开发记录的点坐标、顺序和实际权重颜色逐项一致，无GT。
- `iteration_diagnostic.json`：开发阶段单次区域更新与稳定反馈的机理消融，不能用其未收敛候选比较成功率。
- `reproduction_check.json`：一键完整重建已执行，252+108全部字段除耗时外逐项完全相同。
- `environment.json`：运行环境版本；日志中保留开发及完整重建输出。

正式主批次：`runs/support_feedback_development_v3`、`runs/support_feedback_test_scenes_v1`、`runs/support_feedback_test_v1`。
完整重建：`runs/support_feedback_reproduction_v1`。
开发pilot_v1/v2保留为探索记录；不得将其当作最终新测试。本文新测试也已查看，未来改方法需另锁未见案例。

复现命令（项目根目录，输出必须不存在）：

```bash
bash scripts/run_support_feedback_stage1.sh experiments/neck_localization/runs/my_support_feedback
```

结果仅支持同一圆截面模型家族下的局部轴改进，未证明抗真实深度偏差或实机有效性。
