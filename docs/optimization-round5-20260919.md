# 第五轮：修复上料计时并定位长等待

日期：2026-09-19。范围：成熟上料流程的计时与离线分析；三相机定位和避障继续暂停。

## 环节、问题与方法

**环节：任务性能测量。** `TaskTimeline.mark()` 原先先生成 `elapsed_seconds`，
再合并调用方传入的详情。吸附确认、失败、搬运中失压、透气材料告警，以及单物料测试的
吸附失败事件，都把局部压力等待时长以同名字段传入，覆盖任务累计时间。

历史完整上料日志中有 4 处实际受影响。例如白纸吸附确认发生在任务开始后约 **16.747 s**，
原事件却记为 **0.052865 s**。相邻间隔 `delta_seconds` 没有被覆盖。
该次任务最后一个事件正常，所以原总耗时 81.266 s 和生产周期 77.442 s 仍然有效；
不能据此推断每个事件时间也正常。

本轮修改：

1. 所有这类调用统一改用 `suction_wait_seconds` 存储压力等待时长；累计时间仍由单调时钟计算。
2. `TaskTimeline` 拒绝详情覆盖累计时间和相邻间隔，且拒绝时不消耗时钟、不追加事件。
3. 新增纯离线汇总器。按事件结束边界分配互不重叠的耗时区间；细分的 IK、规划尝试和
   模型计时保留在 JSON 中，不与父区间重复相加。
4. 对旧日志用 `delta_seconds` 累加重建事件累计时间，标明每处差异并核对总和。
   缺失、负数、非有限间隔或无法对齐总时长时拒绝出具报告，不猜测补齐。原日志不改写。
5. 多次输入逐条保留成功、失败、未知状态、执行/预演标记、配置和尝试次数；不会只筛最快成功样本。
   输入文件 SHA-256 随报告保存，重复输入路径被拒绝，输出必须使用新目录。

相关代码：

- `xarm_sdk_control/task_timing.py`
- `xarm_sdk_control/cross_place_motion.py`
- `xarm_sdk_control/scripts/20_visual_pick_test.py`
- `xarm_sdk_control/loading_performance.py`
- `xarm_sdk_control/scripts/summarize_loading_timing.py`

事件字段兼容性：局部压力等待从事件详情的 `elapsed_seconds` 迁移为 `suction_wait_seconds`；
异常对象和其他非时间线记录中的局部 `elapsed_seconds` 不受影响。

## 历史实机记录分析

输入：`xarm_sdk_control/data/complete_loading/20260830_175757_904/result.json`。
荷叶已预铺，执行白纸、陶瓷、两片竹壳，4 个物料均首次尝试成功。
这是旧版本的一次历史记录，**不是本轮修改后的实机测量**。

| 区间 | 耗时 | 占总耗时 |
|---|---:|---:|
| 启动与复位 | 3.824 s | 4.7% |
| 规划 | 12.740 s | 15.7% |
| 采图 | 5.067 s | 6.2% |
| 视觉处理 | 2.288 s | 2.8% |
| 运动指令与等待 | 51.741 s | 63.7% |
| 真空开关与稳定等待 | 4.809 s | 5.9% |
| 显式吸附确认 | 0.212 s | 0.3% |
| 其他与未细分 | 0.585 s | 0.7% |

区间总和为 **81.266 s**，扣除启动与复位为 **77.442 s**。
运动区间包括主机通信、队列等待和部分未单独打点的压力检查，不能当成纯机械运动时间；
异步排队的多段运动可能到后一个同步等待事件才体现耗时，不能由此断言某段速度偏慢。
失败过程中尚未完成的采图、规划等操作可能落入“其他”，需同时读错误信息和前序事件。

两个优先调查点：

- 白纸与第一片竹壳内部规划计时分别 **7.376 s、5.002 s**，另两次约 **0.170 s、0.187 s**。
  旧日志没有第二轮新增的 IK/回退细分，无法判断是求解、重试还是控制器检查导致。
- 陶瓷放置后的竹壳扫描采图区间 **4.901 s**，另三次为 **0.073 s、0.047 s、0.046 s**。
  该区间含采集和文件保存，现有日志不足以确认是相机等待、重启、文件写入还是其他原因。

不能把上述耗时全算成可节省时间，也不能用单次记录估计稳定均值或 P95。
本轮成果是计时正确性和可复用分析入口，**没有宣称实机节拍已经缩短**。

完整产物：[Markdown 报告](benchmarks/loading-round5-20260919/report.md)、
[JSON 明细](benchmarks/loading-round5-20260919/report.json)。

## 验证

控制环境相关 **101 项 pytest 全部通过**，包括时间线、旧日志恢复与总和核验、
成功/失败保留、吸附确认/失败/中途失压/告警的真实时间线集成、重试、上料入口和规划诊断。
所有运动测试使用假硬件；未启动机械臂或相机。

```bash
conda activate winejar-xarm-sdk
python -m pytest -q \
  xarm_sdk_control/tests/test_task_timing.py \
  xarm_sdk_control/tests/test_loading_performance.py \
  xarm_sdk_control/tests/test_visual_cross_place.py \
  xarm_sdk_control/tests/test_cross_place_retry.py \
  xarm_sdk_control/tests/test_complete_loading.py \
  xarm_sdk_control/tests/test_visual_pick_test.py \
  xarm_sdk_control/tests/test_planning_diagnostics.py \
  xarm_sdk_control/tests/test_suction_sensor.py
```

汇总入口只依赖 Python 标准库，不需要连接硬件：

```bash
python xarm_sdk_control/scripts/summarize_loading_timing.py \
  xarm_sdk_control/data/complete_loading/20260830_175757_904/result.json \
  --output-dir /tmp/winejar-loading-timing-review
```

可以传入多个 `result.json`，也可传入只有失败报告的 `failure.json`。
同一次运行若同时产生两者，选其中一份，避免重复计数。脚本只按路径检查重复输入，
不能自动识别改名副本或属于同一次运行的两种报告。

## 下一轮实验记录方式

现场确认与第四轮相同：固定全局相机、吸盘/TCP、料台相对上次上料是否变动尚未得到回复。
确认后按原速度、原 FP32 模式做 3～5 次完整流程，保存失败和重试；用新代码采集规划细分。
这是初步定位实验，不能据此宣称量产成功率或稳定 P95。

每次附记：场景/摆放、硬件或标定改动、是否首次启动、是否开视频录制、服务精度模式，
并保留运行目录下所有 `result.json`、`failure.json`、RGB-D 和视觉结果。
用上述命令一次传入所选报告，即可逐次比较规划、采图和总节拍。
若采图长等待重现，再按采集、对齐、落盘阶段加细分测量；若规划长等待重现，
先根据已经加入的 IK 次数、IK 耗时和回退原因决定算法优化位置。
