# 随机坛型冻结审计，2026-10-23

24个假设常规范围、12个压力范围，固定算法/相机、理想深度。范围不是工厂实测分布。
两方法×三视图共216次；所有采样案例保留，没有按结果重抽或调参。

- 研究说明和查看/复现命令：[报告](../../../../docs/research/neck_localization/random_geometry_audit.md)。
- config.json/plan.json：生成前固定的随机规格、种子、假设和源码散列。
- records.json.gz/summary.json：全部尝试、逐点权重和反馈历史；拒绝不填零误差。
- report/tables.md：分组完整统计、拒绝和错误接受。
- report/gallery.png：全部36例的三视图候选/真值及接受状态。
- report/review_index.json：最坏圆心/法向和拒绝案例索引。
- report/validation.json：种子及规格验证；独立重建一例输入/六条结果完全匹配，未重跑整批36例。
- source_snapshot/：新增采样/审计/报告和查看源码；原估计器9个冻结源码散列保持不变。

完整场景和原始日志在runs/random_geometry_audit_20261023_v1/。
当前36例已查看，不能用于未来调参后声称未见评价。未验证抗真实噪声或实机精度。
