# 第二轮优化：批量吸点搜索与规划耗时诊断

日期：2026-09-19。基线是第一轮优化完成后的代码，不是最初版本。

## 优化了哪个环节

优化 RGB-D 表面分析中的双吸盘候选搜索。它负责根据边缘距离、有效深度、裂缝/缺陷、
粗糙度和倾斜程度，选择双吸盘的中心及角度。

第一轮已经提取角度相关重复计算。本轮排查发现：每个网格中心仍逐角度执行 Python
评分调用，并为所有可行候选构造对象、计算三维点，最后却通常只返回前 20 个。
历史荷叶图有 8493 个可行候选，这些中间对象和逐项计算占用了大量时间。

## 用了什么方法

修改 [suction_surface.py](../xarm_sdk_control/suction_surface.py)：

1. **像素评分表。** 把五项表面指标转为 float64，批量应用原来的门槛与权重。
   每个像素的吸盘评分只算一次，不合格位置标为负无穷。
2. **按角度批量查询。** 一次处理该角度的全部网格中心，根据双吸盘间距得到两端坐标，
   批量检查是否越界，再查两端评分。取整继续使用与 Python round 一致的半偶舍入。
3. **排序后构造结果。** 按原来的“行、列、角度”顺序展开，使用稳定降序排序保留同分顺序，
   只为需要返回的候选构造吸盘诊断对象并计算三维中心。

搜索密度、角度步长、双吸盘尺寸、各项阈值和权重均沿用原设置。
没有引入新模型，也没有通过少搜索一部分位置来换速度。

## 性能结果

预热后，每组输入执行 7 次，交替先运行新旧版本；OpenCV 固定单线程。
下表为完整 `analyze_suction_surface` 调用的中位耗时，包含表面图生成和默认 top-20 输出，
不含图像读盘。环境为 Python 3.11.15、NumPy 2.4.6、OpenCV 4.13.0。

| 输入 | 第一轮版本 | 本轮版本 | 耗时减少 | 全部可行候选 |
|---|---:|---:|---:|---:|
| 合成平面 | 429.170 ms | 14.240 ms | 96.7% | 9712 |
| 合成裂缝 | 391.772 ms | 14.152 ms | 96.4% | 8115 |
| 合成深度孔洞 | 347.951 ms | 13.621 ms | 96.1% | 6657 |
| 无可行吸点（超大间距） | 59.771 ms | 13.090 ms | 78.1% | 0 |
| 非默认步长和朝向 | 78.883 ms | 12.623 ms | 84.0% | 1586 |
| 物体覆盖图像边缘 | 982.267 ms | 15.686 ms | 98.4% | 25388 |
| 历史荷叶 RGB-D | 493.401 ms | 26.228 ms | **94.7%** | 8493 |
| 历史陶瓷片 RGB-D | 98.253 ms | 33.898 ms | **65.5%** | 1254 |

真实输入均来自 `xarm_sdk_control/data/suction_surface/evaluation/20260903_150240_907/`。
不同轮次机器运行状态略有变化，因此基于本轮同场测得的新旧耗时比较，不能把两轮百分比相加。
逐次时间、输入数组哈希、代码哈希和参数见
[可随 Git 保存的原始报告](benchmarks/suction-search-round2-20260919.json)。

## 如何确认没有改变选择结果

- 对 8 组输入，比较**全部可行候选**的数量、顺序、像素中心、角度、三维点、分数和每个吸盘诊断值。
  本环境下严格逐项相等，不是只比较“最佳点差不多”。
- 物体掩码、缺陷掩码和粗糙度图逐元素相等；默认 top-20 输出与旧版全排序前 20 项严格相等。
- 新增门槛边界、越界吸盘、非默认参数和 top-k 排序测试。
- 吸点、规划、上料重试等相关测试共 **69 项通过**；`git diff --check` 通过。

批量计算使用更多临时数组。单次独立 tracemalloc 诊断中，荷叶峰值追踪分配约从 24.46 增至
28.89 MiB，陶瓷片从 13.25 增至 24.77 MiB。它不是进程总 RSS，也不完整代表 OpenCV 原生分配，
仅用于提示速度与内存的取舍。更高分辨率或更密网格仍需重新评测。

本轮没有运行机械臂，不能推导抓取成功率提高，也不能把吸点提速比例套用到 81 秒的完整上料节拍。
两张真实图不能覆盖全部材质和现场变化，后续应补充竹壳、白纸及更多深度异常样本。

## 轨迹规划为什么先补测量

历史完整任务 `data/complete_loading/20260830_175757_904/result.json` 中已有独立的
`planning_seconds`：白纸 7.376 s、陶瓷片 0.170 s、竹壳一 5.002 s、竹壳二 0.187 s。
这比相邻事件间隔更直接，但仍未区分规划重试、IK 等待和控制器预检查。

原流程在 IK 发生关节分支跳变时最多执行 3 次稠密规划，随后可能回退到端点规划。
旧记录不足以证明历史长尾由哪一步造成。本轮保留原来的调用次序和重试策略，补充测量：

| 新记录 | 意义 |
|---|---|
| `estimation_attempts[].method` | 每次采用 dense 还是 endpoint 方法 |
| `estimation_attempts[].status/error` | 该次成功或失败，以及失败原因 |
| `estimation_attempts[].seconds` | 本次规划尝试耗时 |
| `estimation_attempts[].ik_calls` | 本次实际 IK 调用次数，包含失败调用 |
| `estimation_attempts[].ik_seconds` | 这些 IK 调用的累计等待/执行时间 |
| `controller_preflight_seconds` | 控制器不运动预检查耗时 |

通过 [cross_place_motion.py](../xarm_sdk_control/cross_place_motion.py) 记录，
由 [cross_place_retry.py](../xarm_sdk_control/cross_place_retry.py) 和完整上料 dry-run 分支写入
`planning_diagnostics`。成功返回的尝试记录会同时进入 timeline、pick_attempts 和 cycle 汇总。
程序异常导致整个规划未返回时，不能依赖当前任务日志自动保存这份局部诊断。

用模拟时钟验证了“直接成功”和“三次失败后端点回退”两条路径：IK 参数、调用链和预检查对象保持原样，
每次调用 3 ms 与预检查 20 ms 被分别统计，无需 sleep 或连接设备。
这部分属于诊断能力建设，本轮没有声称轨迹规划已提速。

## 复现与文件

可复用脚本：[benchmark_suction_surface.py](../xarm_sdk_control/scripts/benchmark_suction_surface.py)。
默认使用合成输入；加 `--frame-dir` 可回放本次固定的荷叶/陶瓷片样本。
`--baseline` 会导入指定的本地 Python 文件，应使用保存的可信源码快照。

```bash
conda activate winejar-xarm-sdk
python xarm_sdk_control/scripts/benchmark_suction_surface.py \
  --baseline xarm_sdk_control/data/optimization/20260919-round2/suction_surface_before.py \
  --frame-dir xarm_sdk_control/data/suction_surface/evaluation/20260903_150240_907 \
  --output /tmp/winejar-suction-round2-replay.json

python -m pytest -q \
  xarm_sdk_control/tests/test_suction_surface.py \
  xarm_sdk_control/tests/test_planning_diagnostics.py \
  xarm_sdk_control/tests/test_trajectory_speed_planner.py \
  xarm_sdk_control/tests/test_visual_cross_place.py \
  xarm_sdk_control/tests/test_cross_place_retry.py
```

输出路径必须尚不存在，脚本不会覆盖旧报告。基线快照和原始 RGB-D 位于被 Git 忽略的 `data/`，
跨机器复现须另行携带；基线 SHA256 为
`70324638a9cb80de893f85700190c629ab2a4c1dd7f2ad7018997057783660b4`。

后续优化继续按[总记录](optimization-log.md)中的同一格式整理。
