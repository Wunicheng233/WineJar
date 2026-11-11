# 现场尺寸更正与“颈部范围”的定义

2026-10-23。用户与酒厂确认：扎带宽3 mm，坛颈范围至少约6 mm。此前5 mm只是暂定值，保留历史结果。
本轮只更新工艺先验、默认带宽及冻结预测上的几何审计；不重新拟合、不扩展噪声或机器人阶段。
来源记录：`experiments/neck_localization/configs/factory_process_constraints_v1.json`。
约6 mm是用户转述的现场信息，暂无本项目逐坛独立测量或尺寸公差，不当成精确测量数据。

## 1. 三种范围必须分开

| 名称 | 如何获得 | 用途与局限 |
|---|---|---|
| 候选/拟合点范围 | 观测口沿初始位置/方向下，取下方局部外表面；粗半径曲线找局部最低区，再保留附近上下文与抽样训练点 | 用于估计轴和r(s)，允许包含过渡段；不等于物理颈部或可放带区 |
| 输出的近最细范围 | 在最终轴下读出样条半径曲线r(s)，从最小值处向两边找到r(s)≤r_min+1 mm的连通区间，并检查观测支持 | 取区间中点为目标圆位置，给带宽规划器当观测代理；1 mm是人为的**半径容差**，不是轴向颈高 |
| 模型完整颈段 | 建模参数中口沿下边与肩部过渡上边之间的轴向区间；保存为evaluation_only标注 | 上轮完整颈段列使用它评分；估计器没有读取它，也没有独立识别真实物理上下边界 |

例如r_min=50 mm，则算法把r≤51 mm的连续区域称为近最细范围；若肩部过渡在一段距离内仅缓慢变粗，
它也可能被算进去。即使物理颈段6 mm，阈值区间也可能更宽或更窄，不能互换。

当前候选深度默认口沿下3–28 mm，必要时同帧扩至42/56 mm，随后再按粗最小区域选局部点；
初始局部半宽至少8.5 mm，并保留2 mm上下文。不是直接把整段28 mm都当颈部。
实际训练点及权重保存在记录，L/W可见；区域反馈用于估计轴的高权重核心也不是最终目标区间。
当前V2连续主路径固定最终轴后，用同一训练点的全上下文重新拟合读出曲线，再按+1 mm阈值取范围。
因此“实际点云在哪里”“权重重点在哪里”“输出近最细范围在哪里”应分别显示、分别解释。
相关源码：`joint_profile.observed_initialization/supported_interval`、`reliable_output._candidate`。

## 2. 3 mm带与至少约6 mm颈段的含义

若这6 mm全是允许的轴向放带区域，且目标圆恰在其中点、法向正确，3 mm带的每侧名义余量为

$$(6-3)/2=1.5\text{ mm}.$$

比原暂定5 mm带下的0.5 mm余量更宽松。1.5 mm是轴向几何预算，不是允许任意方向三维误差1.5 mm，
也不是相机单独可用的误差预算。法向误差引起的周向摆动约R sinθ，后续还需计入输送/机器人轴向误差，
避免重复计算同一误差。真实完整颈段是否都可接触带材仍与口沿/肩部、带材贴合有关。
原4 mm圆心阈值仍保留作历史定位指标，不独自保证有限带宽任务。

已将`plan_band`和`audit_neck_band.py`的默认带宽改为3 mm；6 mm不注入估计器当真值边界，
不把检测到的区间强行撑到6 mm。未来按现场约束设计新的场景批次；旧模型/冻结算法/原预测全部保留。

## 3. 同一48例改用3 mm后的审计

正式结果：`experiments/neck_localization/results/band_width_3mm_20261023/`。
旧48例全部保留，不按新增约6 mm先验删样本：

| 指标 | 近最细代理 | 完整颈段假设 |
|---|---:|---:|
| 原圆正确输出 | 48/48 | 48/48 |
| 规划器名义可容纳 | 48/48 | 48/48 |
| 实际整带轴向容纳 | 47/48 | 47/48 |
| 常规 / 压力容纳 | 32/32、15/16 | 32/32、15/16 |
| 错误兼容 | 1/48 | 1/48 |
| 整带最小余量：中位 / 最小，mm | 2.594 / −0.155 | 3.428 / −0.155 |

唯一越界为旧`stress_010`：完整颈段4.517 mm，圆心轴向偏差约−0.871 mm、法向周向摆动0.043 mm，
3 mm带越界0.155 mm。它低于新确认的约6 mm现场尺度，是旧假设压力范围中的例子，不能直接代表新工况。
同时不能删掉它后把剩余结果宣称为新工况独立验证100%；本轮没有生成新测试案例。
在两种边界下，3 mm带的理想真值几何均可容纳48/48，剩余1例是位置/范围估计问题。
近最细算法阈值与物理颈段的语义差别仍存在，只是3 mm带余量更大，旧5 mm结论不能原样套用。

复现（输出目录需不存在）：

```bash
cd /home/wunicheng/Git/WineJar
env OPENBLAS_NUM_THREADS=1 conda run --no-capture-output -n embodied-sim \
  python scripts/audit_neck_band.py --width-mm 3 \
  --output experiments/neck_localization/runs/my_band_3mm_audit
```

看3 mm环带（带宽来自新记录，不由查看器改写旧记录）：

```bash
conda run --no-capture-output -n embodied-sim python scripts/view_neck_localization.py \
  --scenes experiments/neck_localization/runs/reliable_output_v2_test_scenes \
  --records experiments/neck_localization/results/band_width_3mm_20261023/records.json.gz \
  --case stress_002 --method task_validated_output \
  --large-neck --overlay --ground-truth --band
```

6项原带宽解析测试通过；新3 mm记录与上轮预先保存的3 mm宽度扫描统计一致；原48条预测、13个冻结源码不变。
本阶段仍仅为轴向圆柱条带几何检查，无接触/厚度/扣头模型、无传感器偏差、无跨工位或实机验收。
