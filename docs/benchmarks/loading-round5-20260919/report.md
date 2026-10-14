# 上料节拍离线报告

按事件结束边界划分互不重叠的耗时区间；细分诊断包含在对应区间内，不重复相加。
运动区间可能包含排队、通信和传感器检查，不能作为单段实际运动时长。
失败、未知状态和预演记录逐条保留；不同配置与现场条件应分别对照。

输入记录：success 1 条。

## /home/wunicheng/Git/WineJar/xarm_sdk_control/data/complete_loading/20260830_175757_904/result.json

状态：success；run_mode：True；已完成物料周期：4。
总耗时：81.266 s。
生产周期：77.442 s。
各物料尝试次数：[1, 1, 1, 1]（缺失值不当作首次成功）。

| 区间 | 耗时 / s | 占总耗时 |
|---|---:|---:|
| 启动与复位 | 3.824 | 4.7% |
| 规划区间 | 12.740 | 15.7% |
| 采图区间 | 5.067 | 6.2% |
| 视觉处理区间 | 2.288 | 2.8% |
| 运动指令与等待区间 | 51.741 | 63.7% |
| 真空开关与稳定等待区间 | 4.809 | 5.9% |
| 显式吸附确认区间 | 0.212 | 0.3% |
| 其他与未细分区间 | 0.585 | 0.7% |

规划、采图与视觉事件（子项不再计入总和）：

| 事件 | 区间 / s | 内部规划计时 / s |
|---|---:|---:|
| scan_3.capture_complete | 0.073 | — |
| scan_3.vision_complete | 0.542 | — |
| material_01_paper.attempt_01.route_planned | 7.377 | 7.376 |
| scan_4.capture_complete | 0.047 | — |
| scan_4.vision_complete | 0.549 | — |
| material_02_ceramic.attempt_01.route_planned | 0.171 | 0.170 |
| scan_shell_after_ceramic.capture_complete | 4.901 | — |
| scan_shell_after_ceramic.vision_complete | 0.605 | — |
| material_03_shell_1.attempt_01.route_planned | 5.004 | 5.002 |
| scan_5.capture_complete | 0.046 | — |
| scan_5.vision_complete | 0.592 | — |
| material_04_shell_2.attempt_01.route_planned | 0.188 | 0.187 |

- 4 个事件累计时间不一致；报告使用 delta_seconds 累加，原日志未修改。
- 部分规划事件缺少 IK/重试诊断，不能据此判定规划长等待的原因。

该报告是测量与诊断结果，不代表已经缩短了实机节拍。
