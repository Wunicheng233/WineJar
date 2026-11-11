# 局部联合估计首轮，2026-10-23

[实现、文献依据与结果解释](../../../../docs/research/neck_localization/joint_profile_stage1.md)；
[全部统计](report/tables.md)；[误差图](report/test_errors.png)。

- plan.json：开发前锁定新测试规格与阶段范围。
- freeze.json：开发后、测试前冻结求解器/运行器/设置；旧基线未改。
- test_config.json：6个新测试规格，同一固定相机布局、理想RGB-D。
- report/development/、report/test/：压缩逐次记录、汇总、设置和来源；包括失败，不只成功案例。
- locality_ablation.json：开发数据的局部/宽带拟合诊断，不是新测试。
- review/：旧错误接受、联合结果和实际拟合点的窗口回读图。
- source_snapshot/：本轮冻结维护代码与运行/报告入口快照；常规维护仍在项目模块和scripts中。
- environment.json、checks.log、reproduction_check.json：实际环境、20项检查与72+54条数值复现核对。

新测试联合方法18/18通过本轮分析阈值，但仅有6个坛型和理想传感器。
三视图法向相对近乎理想的旧口沿基线退步，报告明确保留该不利结果，不据此替换所有旧方法。
没有抗噪、实机有效性或论文新颖性结论。测试已经查看，后续版本需新的未见数据。
