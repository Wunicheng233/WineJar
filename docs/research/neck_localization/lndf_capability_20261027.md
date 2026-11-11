# L-NDF 预训练功能坐标迁移能力审查

本记录属于已经查看过的固定能力审查，不是新的未见测试，也不是新方法论文结果。只使用保存的 RGB、深度、相机参数和四个开发坛型的监督功能框架；不连接机器人或相机。

## 被审查的能力

[L-NDF 官方实现](https://github.com/elchun/lndf_robot)以局部神经描述符匹配迁移示教中的查询点坐标系。
本任务将“扎带目标截面圆心与局部轴”转成一个功能坐标系，以固定环形查询点表示。
四个开发坛型的目标圆心/轴是**监督示教标签**，不是无监督或零示教；网络权重不再训练。
该方法没有截面半径输出，所以只能评价圆心/轴，不能把它的局部框架转移叫作完整的坛颈计量通过。

网络使用官方 `lndf_weights.pth`，1,588,289 个参数，严格匹配全部权重。
官方 `optimizer_lite.py` 中的 `Demo`、`process_demos` 和 `optimize_transform_implicit` 方法体保持不变。
依赖兼容层使用 PyTorch `scatter_add/scatter_reduce` 实现原 `torch_scatter` 前向归约，绕过未调用的机器人、ROS、绘图模块；不修改描述符或优化目标。
坐标及归约的五项检查通过。依赖放在独立 `.venv`，继承现有视觉环境；控制环境不变。

## 任务适配与固定协议

- `cam_a` 单视角；原始主动双目深度与理想 RGB 深度配对。理想深度是输入控制，不是部署结果。
- RGB 蓝背景分割提取观测点；不从网格采样，不读取审查目标或真值掩码。
- 根据观测点范围作可逆相似缩放，使点云最大中心偏差为 0.04 m；网络内部仍按官方 `scaling=10`。
- 查询圆环半径取四个监督开发标签半径乘各自观测缩放的中位数；半高取四开发例 `1.5 mm×各自缩放` 的中位数。查询几何随后固定在 model-space；还原到审查对象后物理半径、半高会随其观测缩放变化，**不代表每坛始终是3 mm宽的查询带**。审查对象的半径不进入推理。
- 每次使用 1000 查询点、500 步、20 个初始化；同坛 raw/ideal、far/close 使用同一随机种子。
- 仅比较两个预先写下的开发适配：官方 bottle 配置的 cosine 损失配全前景，及同损失配观测上包络 `z99.5−70 mm` 与现有固定工位范围。后者与点恢复基线使用同等区域先验。
- 只按四个开发 raw 输入的圆心≤4 mm且无向轴≤2°数量选择，再按中位 `center_mm/4+axis_deg/2` 打破平局；理想输入仅作控制。选择后冻结，再审查两布局×同12坛型×两深度输入。

开发选择前的协议为 `experiments/neck_localization/runs/capability_lndf_development_protocol_20261027_v1/protocol.json`。
此处开发例也参与示教，其表现只能用于检查与适配，不能当作独立泛化精度。

## 接口和资源检查

CPU 已跑通完整读入、描述符示教、优化、保存及独立评分。
GPU 按完整 500×20 预算在一个开发例测得 7.69 s、峰值 1639 MiB，四示教描述符约 0.45 s。
原默认 L1/全前景在该开发例的圆心误差 162.06 mm、轴 51.98°；记录保留在
`runs/capability_lndf_gpu_development_probe_20261027_v1/`。
这暴露的是直接迁移适配尚未通过基本控制，不能据此声称现有方法存在未解决的科学问题。

## 执行入口

```bash
experiments/neck_localization/external_baselines/lndf/.venv/bin/python \
  scripts/run_lndf_capability.py infer \
  --source experiments/neck_localization/runs/structural_stage_20261025_v1 \
  --development experiments/neck_localization/runs/active_stereo_development_20261024_v3 \
  --output NEW_DIR --layouts development --conditions stereo_raw ideal_rgb \
  --cos-loss --upper-window-mm 70

experiments/neck_localization/external_baselines/lndf/.venv/bin/python \
  scripts/run_lndf_capability.py evaluate --predictions NEW_DIR --output NEW_RESULT_DIR
```

推理先保存全部预测和散列，独立 `evaluate` 子命令在检查 `completed.json` 和预测散列后才读取审查 truth。
每次输出记录输入文件、权重、上游源码、适配器与脚本散列、观测区域、归一化、随机种子和实际资源耗时；执行异常计入失败。

## 开发适配选择

16 次完整开发推理已完成，均正常输出。下表为圆心/无向轴的中位误差，四开发例也参与示教，因此不能作为独立泛化成绩。

| 固定候选 | 深度输入 | 4 mm且2°达标 | 圆心中位 mm | 轴中位 ° |
| --- | --- | ---: | ---: | ---: |
| 官方 cosine，全前景 | raw | 0/4 | 19.38 | 12.10 |
| 官方 cosine，全前景 | ideal | 0/4 | 17.14 | 10.67 |
| 官方 cosine，上部70 mm | raw | 0/4 | 80.87 | 29.53 |
| 官方 cosine，上部70 mm | ideal | 0/4 | 53.57 | 30.64 |

按事先协议选中**官方 cosine / 全前景**：raw 中位归一化误差10.48，小于 upper70 的34.98。
选择在12坛型推理前写入 `runs/capability_lndf_development_protocol_20261027_v1/selection.json`，并保存开发记录散列。
没有因上部区域使分辨率更高就假定它一定更好；负面结果完整保留。

## 固定48次审查结果

48次均执行完成且输出有限的刚体坐标系，没有依赖或运行异常。圆心/轴分别达标也均为0/12。

| 固定布局 | 深度输入 | 圆心≤4 mm且轴≤2° | 圆心中位/最大 mm | 轴中位/最大 ° |
| --- | --- | ---: | ---: | ---: |
| far | raw | 0/12 | 89.37 / 215.32 | 30.76 / 67.92 |
| far | ideal | 0/12 | 56.77 / 155.01 | 18.94 / 65.55 |
| close | raw | 0/12 | 127.05 / 1134.98 | 50.06 / 73.99 |
| close | ideal | 0/12 | 120.00 / 192.84 | 43.94 / 62.21 |

推理为 `runs/capability_lndf_frozen_cos_full_20261027_v1/`，独立评分为
`results/capability_lndf_frozen_cos_full_20261027_v1/`。四组耗时中位约6.74–7.02 s；不作为工位节拍验收。
无向轴允许法向翻转，朝口沿的有向法向误差也逐条保留；没有半径输出，不评价完整圆计量成功率。

产物审计核对了48份预测、每份20个初始化框架、全部输入文件与源码快照散列；4,064,369个保存观测点均有限，配对raw/ideal的RGB散列一致，同坛四条件初值种子一致。
旋转正交误差最大5.52e−6。还原后查询带高度实际范围2.71–14.57 mm，进一步说明这不是固定3 mm物理带材的计量模型。

## 可以与不可以得出的结论

**可以确认：这套官方预训练权重、四监督示教、当前查询与尺度适配没有完成本任务。**
理想深度仍表现差，不能把失败全归给主动双目噪声。
开发 ideal 也0/4，尚缺一个可靠的基本功能迁移正控；原版L1、官方cosine和上部ROI控制的完成，不等于已经排除了合理的查询构造、稳健尺度处理、已知工位姿态、几何细化或简单微调。
因此这些失败不能证明L-NDF这类方法原理上做不到，不能直接据此宣称论文创新点。

开发局部目标诊断位于 `runs/capability_lndf_development_protocol_20261027_v1/development_local_descriptor_sensitivity.json`。
它用四开发理想点云、固定真轴、2000点控制采样，沿真轴扫±10 mm；全前景cosine在四例离散最小分别位于−2/+4/+2/−4 mm，而非统一0 mm标签。
这说明当前描述符目标与毫米级物理定义没有严格一致；该控制的latent采样不同于正式每初始化1500点，且只扫了一维，不能用来证明全局最优、不可辨识或通用结构缺陷。
