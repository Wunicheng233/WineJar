# 正确输出率改进，2026-10-23

本轮仅理想RGB-D、固定相机/准确K/T，不操作实机，不模拟带宽/输送/机器人。

- [方法与两轮结果说明](../../../../docs/research/neck_localization/reliable_output_stage1.md)。
- [5 mm环带与跨工位/TCP契约](../../../../docs/research/neck_localization/band_and_station_contract.md)。
- plan.json/freeze.json：V1实现前锁规格、开发后冻结；v2_plan.json/v2_freeze.json为第二轮。
- V1首批48新例三视图46/48，旧法47/48。此不利结果保留在report/和v1_test/。
- 首批48仅作V2开发，另锁新48例三相机：旧法39/48、V1 44/48、V2 48/48。完整表在v2_report/。
- v1_development/v1_test/v2_development/v2_test：全部压缩逐条记录、每点权重、留出指标、恢复路径、分轴误差与耗时。
- final_checks.log：31项不同核心检查；viewer_checks.log：4项及3个投影子检查。
- viewer_subset_validation.json/v2_viewer_validation.json：193条开发/48条V2输出的训练点与原始观测、权重显示一致。
- limited_reproduction_check.json/v2_limited_reproduction.json：指定开发案例重跑一致，未完整再跑一键脚本。
- fallback_budget_diagnostic.json：开发案例强制仅1轮，验证备用路径；不是正式测试成绩。
- v2_recovered_test.png：第二批压力案例实际窗口回读，显示所选备用路径与真值。
- source_snapshot/：本轮新增源码、测试、配置、查看与复现脚本。

完整场景/原始JSONL保留在runs/，不删原始证据。旧support_feedback冻结源码及V1冻结源码均保持不变。
第二批仅三相机，不外推单/双视图；48例零失败也不是产线成功率证明。
复制报告中的命令可查看和重建，输出必须为新目录。
