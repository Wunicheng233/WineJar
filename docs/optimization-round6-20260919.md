# 第六轮：深度预览计算优化与采图长等待诊断

日期：2026-09-19。继续聚焦已跑通的上料流程，不开展三相机定位算法或避障优化。

## 优化对象与方法

**环节：固定相机 RGB-D 保存时的深度预览生成。** 原函数先对整幅浮点深度图做裁剪，
再重复提取有效像素。本轮只提取一次有效深度，在该临时数组上原地裁剪，减少数组创建和
重复索引。保留原来的有效像素判断、2%/98% 分位数、浮点运算次序、取整方式和色图。
不改输入 RGB、原始 Z16 深度、定位算法或相机参数。

相关代码：`xarm_sdk_control/realsense_capture.py::make_depth_preview`。
该公共函数的其他调用者也使用相同的等价实现；本轮未修改三相机定位代码。

## 离线性能结果

复用第四轮四帧真实 RGB-D。每帧、每种实现先预热一次，再进行 **31 次交错顺序测量**。
对比本轮开始时保存的源码快照与修改后的正式函数。完整保存测试使用假的相机配置对象和
临时目录，执行实际 PNG 编码、文件写入和元数据保存，没有连接相机或机械臂。

| 历史帧 | 预览计算：修改前 → 后 | 预览减少 | 完整保存：修改前 → 后 | 保存减少 |
|---|---:|---:|---:|---:|
| 白纸 scan_3 | 4.989 → 3.883 ms | 22.2% | 17.556 → 16.083 ms | 8.4% |
| 陶瓷 scan_4 | 4.583 → 2.973 ms | 35.1% | 17.467 → 15.969 ms | 8.6% |
| 第二次竹壳 scan_5 | 4.998 → 3.960 ms | 20.8% | 17.731 → 17.068 ms | 3.7% |
| 陶瓷后竹壳扫描 | 4.705 → 3.852 ms | 18.2% | 18.543 → 17.323 ms | 6.6% |

以上均为中位数；逐次测量和环境、源码/输入哈希见
[基准 JSON](benchmarks/capture-round6-20260919.json)。

每次回放均核验：预览像素完全相同；`color.png`、`depth_raw.png`、`depth_preview.png`、
`frame.json` 四个保存文件逐字节相同；输入深度未改变。原图未覆盖。

**收益边界：** 完整保存每帧减少约 0.66～1.50 ms，四帧中位数差合计约 4.86 ms。
这不等于实际周期已经缩短，也无法解决历史 4.90 s 的采图长等待。
保存测量使用操作系统文件缓存，没有 `fsync`，不能作为磁盘持久化延迟或相机采集耗时。
输出在临时目录，文件系统和现场机器状态会影响结果。

另做过 PNG 压缩参数试探：关闭彩色图压缩仅节省少量毫秒而文件明显变大；其他参数也没有
显示足以替换当前默认值的优势。因此本轮保留原 PNG 参数。

## 为长等待补充的测量

代码中确实存在取帧失败后启动相机恢复流程的分支，但旧日志没有记录是否走过该分支。
**不能认定重启就是那次 4.90 s 的原因。**

本轮新增 `RealSenseCaptureSession.last_capture_diagnostics`，每次调用新建记录，包含：

- 模式：直接采图或录像线程提供帧；状态：成功/失败；读取尝试次数和恢复流程次数。
- 首次取帧错误、最终错误、成功输出路径，以及该次 `capture()` 总耗时。
- 直接采图的取帧等待、RGB-D 对齐、数组复制与内参读取。
- 相机恢复流程总耗时；包含原有重新启动、预热、内部重试等待，不进一步细分。
- 录像模式等待下一帧及复制的耗时；不重复统计后台线程的采集/对齐时间。
- 彩色 PNG 保存、深度 PNG 保存、预览计算、预览 PNG 保存、元数据序列化与写入。

每个细分阶段在异常时也记录已消耗时间。阶段时长互不重叠；总时长另列，包含未单独
测量的目录创建、质量统计、元数据组装、打印等开销，不能把总时长与细分相加。
恢复流程次数表示 `_restart_pipeline()` 调用次数，不表示其内部 `start()` 尝试次数。

上料 `run_scan()` 将诊断放进 `capture_complete` 或新增的 `capture_failed` 时间线事件；
失败后继续抛出原异常，不进入视觉推理。离线汇总器保留诊断并在 Markdown 中显示细分，
失败采图归入采图区间，旧日志缺少诊断时明确提示。
独立 `frame.json` 的原有格式不变；完整细分在上料任务日志中。
没有复用会话而走外部采图脚本时，事件仅有外围耗时，细分标为缺失。

取帧超时、恢复策略、录像线程的新帧等待和数据复制语义保持原样。
另修正一个保存检查遗漏：预览 PNG 写入返回失败时现在会报告采图失败，避免日志声称
文件已保存但实际缺失；这比旧代码忽略该失败更严格，测试已覆盖。

## 验证与复现

**123 项相关 pytest 全部通过**。覆盖：预览与旧实现逐像素一致（全无效、等深度、
极值、深度空洞和不同缩放值）、输入不变、无损保存、直接取帧/录像模式、读取超时、
恢复成功/失败、只恢复一次、文件保存失败、采图错误向任务日志传递、报告不重复计时，
以及现有上料、吸附、规划与重试测试。全程使用假硬件。

```bash
conda activate winejar-xarm-sdk
python -m pytest -q \
  xarm_sdk_control/tests/test_realsense_capture.py \
  xarm_sdk_control/tests/test_visual_targeting.py \
  xarm_sdk_control/tests/test_loading_performance.py \
  xarm_sdk_control/tests/test_complete_loading.py \
  xarm_sdk_control/tests/test_cross_place_retry.py \
  xarm_sdk_control/tests/test_task_timing.py \
  xarm_sdk_control/tests/test_visual_cross_place.py \
  xarm_sdk_control/tests/test_visual_pick_test.py \
  xarm_sdk_control/tests/test_planning_diagnostics.py

python xarm_sdk_control/scripts/benchmark_capture_processing.py \
  --baseline xarm_sdk_control/data/optimization/20260919-round6/realsense_capture_before.py \
  --manifest docs/benchmarks/vision-round4-manifest.json \
  --output /tmp/winejar-capture-round6.json
```

基准会导入指定的可信本地源码快照；输出必须不存在。快照和 RGB-D 在本机被忽略的
`data/` 中，跨机器复现需一并转移；JSON 保留对应哈希。

下一次原参数完整上料即可自动采集这些诊断。仍需先明确现场硬件是否移动过，
再沿用第五轮的 3～5 次初步实验方案。重点核对长采图时 `frame_wait`、`stream_recovery`
与各保存阶段，依据实际占比决定下一项改进。
