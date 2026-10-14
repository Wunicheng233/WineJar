# xArm SDK Control

本目录是 WineJar 项目的实机控制层。根目录 [README](../README.md) 是当前
硬件状态和现场运行命令的权威说明；本文只记录代码结构和维护边界。

## 设计原则

- `scripts/` 只负责命令行参数、任务编排和结果落盘。
- 可复用的视觉、运动、真空和重试逻辑放在顶层 Python 模块中。
- dry-run 与单元测试不得发送机械臂运动或 GPIO 指令。
- 实机入口必须显式给出工作空间边界，并在运动前执行相机避让检查。
- 当前已经实机跑通的 `21_complete_loading.py` 不做无收益的大规模改写。

## 模块职责

| 模块 | 职责 |
|---|---|
| `robot_motion.py` | 连接、状态检查、准备、相机安全复位 |
| `visual_targeting.py` | 标定加载、RGB-D 扫描、检测分类和目标选择 |
| `cross_place_route.py` | 吸取、搬运、放置和撤离的路径几何 |
| `cross_place_motion.py` | 运动剖面、关节感知规划、路由执行和压力持续检查 |
| `cross_place_retry.py` | 吸附失败后的恢复、重扫描、重规划和有限重试 |
| `trajectory_speed_planner.py` | IK 采样、速度缩放、端点回退和控制器预检 |
| `pick_test_workflow.py` | 单物吸取与源到目标搬运测试 |
| `suction_sensor.py` | CI0 压力开关读取、超时和去抖 |
| `vacuum_control.py` | CO0/CO1 真空状态机和安全关断 |
| `end_effector_config.py` | 执行器 YAML 解析与参数校验 |
| `realsense_capture.py` | 对齐 RGB-D 会话、录制和断线重试 |
| `positioning_bias.py` | 人工测量的基坐标补偿 |
| `cli_utils.py` | 公共 CLI 参数和工作空间检查 |
| `run_output.py` | `result.json` / `failure.json` |
| `task_timing.py` | 单调时钟事件时间线 |
| `geometry.py` | 位姿和角度辅助函数 |
| `hand_eye_calibration.py` | 腕部相机到法兰的手眼标定与固定棋盘一致性验证 |
| `suction_surface.py` | 通用表面缺陷评估和双吸盘候选姿态搜索 |
| `calibration_data.py` | 内参加载、数值快照、全画面模型检查及不可覆盖的结果保存 |
| `calibration_target.py` | 普通棋盘/ChArUco 统一角点观测，不与内参采集板绑定 |
| `calibration_sampling.py` | 采样静止、分布、重复姿态及双路帧时序检查 |
| `pair_calibration_session.py` | `24/35` 共用的采集窗口和求解流程 |
| `calibration_validation.py` | 独立双向投影检查和多姿态验收，独立于拟合 |
| `multiview_rig.py` | 阵列几何、内外参绑定及当前场景采集准入 |

## 任务入口

- `19_visual_cross_place_two_leaves.py`：只验证两片竹壳交叉放置。
- `20_visual_pick_test.py`：按物料类型执行吸起、悬停、放回；适合硬件调试。
- `21_complete_loading.py`：完整上料生产入口，当前实机主流程。
- `23_calibrate_camera_intrinsics.py`：三相机坛颈定位的第一步；逐台采集普通棋盘并保存内参与重投影误差，不连接机械臂。
- `24_calibrate_camera_pair_extrinsics.py`：三相机坛颈定位的第二步；用同时可见的普通棋盘标定俯视相机与一台侧视相机的相对外参，输出 `T_俯视←侧视`，不连接机械臂。
- `25_create_multiview_camera_rig.py`：默认只生成候选阵列；显式 `--accept-for-capture` 在独立验证合格后更新现场采集状态，不授权机器人运动。
- `26_capture_three_camera_frame.py`：按阵列配置采集一组近同步 RGB-D 帧及采集偏差；`--lock-color` 在预热后冻结 RGB 曝光、增益和白平衡，适用于固定背景差分。
- `30_calibrate_workcell_plane.py`：从空工位 RGB-D 帧拟合传送面，建立以盒面为 `Z=0`、朝向俯视相机为 `+Z` 的工位坐标系；不连接机械臂。
- `31_estimate_mouth_height.py`：离线联合拟合三视图坛口中心、平面和外径；坛肩转折与可捆扎窄槽分开报告，不连接机械臂。
- `32_calibrate_wrist_hand_eye.py`：棋盘固定、机械臂带腕部相机移动的手眼标定；同步保存 TCP、活动 TCP 偏置、法兰位姿和关节角，不发送运动命令。
- `33_inspect_suction_surface.py`：在腕部 RGB-D 帧中对荷叶、竹壳、陶瓷片或白纸搜索双吸盘姿态并输出缺陷图；不连接机械臂。
- `34_wrist_guided_pick_test.py`：先复位机械臂，再同步采集腕部 RGB-D 与法兰位姿，筛选吸附面并执行带 CI0 确认和换点重试的单物料吸取测试。
- `35_calibrate_camera_pair_charuco.py`：ChArUco 入口，与 `24` 共用采集与质检。
- `36_validate_charuco_extrinsics.py`：两种板的离线外参五折对照；历史无绑定文件需显式 `--allow-legacy-input`。
- `37_preview_three_cameras.py`：不依赖标定的三相机实时预览。
- `38_check_stationary_charuco_pair.py`：两种板的独立新姿态检查，保存双向投影覆盖图，不重新拟合。
- `39_reprocess_chessboard_pair.py`：重检测历史普通棋盘/ChArUco 原图，输出新候选；默认不重复生成覆盖图。
- `40_evaluate_camera_intrinsics.py`：内参 Brown4/Brown5 五折比较及全画面数值检查，只生成候选。

当前坛颈任务已经收缩为只测量坛口相对于传送面的高度。早期视觉外壳实验链
（原脚本 `27`/`28`/`29` 及 `visual_hull.py`、`neck_localization.py`）已删除，
需要时可从 git 历史恢复。

## 腕部相机与吸取表面诊断

腕部相机序列号为 `011422070172`。手眼标定使用 `11x8` 内角点、边长
`20 mm` 的固定棋盘：

```bash
python xarm_sdk_control/scripts/32_calibrate_wrist_hand_eye.py
```

该脚本只读取机械臂。控制器当前报告的活动 TCP 偏置与执行器配置不同时，求解
始终采用控制器实际值反算法兰位姿，同时在结果中保留两者供审计。

启动常驻 Grounded-SAM2 服务后，对当前腕部画面执行一次诊断：

```bash
python xarm_sdk_control/scripts/33_inspect_suction_surface.py --target lotus
python xarm_sdk_control/scripts/33_inspect_suction_surface.py --target bamboo
python xarm_sdk_control/scripts/33_inspect_suction_surface.py --target ceramic
python xarm_sdk_control/scripts/33_inspect_suction_surface.py --target paper
```

`suction_surface.py` 不负责识别物料类别，只接收目标掩码、对齐深度和内参。它按
双吸盘直径 `8 mm`、中心距 `48 mm` 联合搜索中点与角度，并同时检查两个吸盘
覆盖区的边缘余量、暗裂纹比例、深度有效率、局部起伏和倾角。材料差异只存放在
`config/suction_surface_profiles.yaml`。当前阈值是保守初值，必须先根据叠加图和
CI0 实验记录校正，不能直接用于无人值守抓取。

荷叶实机吸取测试（当前验证参数：弹簧压缩 `13 mm`、绝对抬升高度
`Z=280 mm`、最多三个空间分离候选点）使用：

```bash
python xarm_sdk_control/scripts/34_wrist_guided_pick_test.py \
  --run --target lotus --compression-mm 13 --lift-z-mm 280 \
  --inspection-hold-seconds 5 --transit-speed 60 --precision-speed 15 --acc 120 \
  --min-x-mm 180 --max-x-mm 520 --min-y-mm -250 --max-y-mm 180 \
  --min-z-mm -85 --max-z-mm 300
```

实机验证中，第一个候选点在抬升后丢失 CI0 真空确认，程序自动放气、垂直撤离并
改用第二个候选点；第二次吸取成功抬升至 `Z=280 mm`，保持 5 秒后放回原处。
这证明重试链路可用，但单次实验不代表成功率已经满足现场要求。

将荷叶转运到同一腕部画面中的陶瓷片：

```bash
python xarm_sdk_control/scripts/34_wrist_guided_pick_test.py \
  --run --target lotus --place-on-ceramic \
  --compression-mm 13 --lift-z-mm 280 --placement-clearance-mm 3 \
  --transit-speed 60 --precision-speed 15 --acc 120 \
  --min-x-mm 180 --max-x-mm 520 --min-y-mm -250 --max-y-mm 180 \
  --min-z-mm -85 --max-z-mm 300
```

由于左侧吸盘弹簧在压缩较大时可能卡滞，通用单物料测试的推荐压缩量统一下调
`2 mm`：陶瓷片、荷叶和白纸为 `13 mm`，竹壳为 `11 mm`。`20 mm` 仅表示弹簧
物理最大行程，不是推荐工作值。

## 工位高度基准标定

**2026-09-08：相机移动后必须先重新标定相机间外参，再采集新的工位平面。**
相机位置改变通常不改变内参，但本次发现原三台 RGB 内参的全画面数值检查失败，
不能直接继续推荐它们。新模型候选须验证，外参必须用对应内参重求，不能只换路径。
完整命令、独立验收步骤和参数依据见
[三相机坛口定位与复标说明](../docs/multiview-neck-localization.md)。

固定最终承载面，采集空场 RGB-D，用承载面内部 ROI 标定。以下尖括号需要替换：

```bash
python xarm_sdk_control/scripts/30_calibrate_workcell_plane.py \
  --rig <新阵列JSON> \
  --capture <新空场采集目录> \
  --roi <平面内部的x,y,width,height>
```

ROI 必须按新画面选择。输出 `workcell_frame.json` 包含平面方程、
双向齐次变换、深度有效率、RANSAC 内点率以及 RMS/P95 残差。当前阶段只使用
`Z=0` 和 `+Z`；X 轴暂由俯视相机图像右方向投影到盒面，后续确定真实传送方向后
再固定 X/Y 朝向。相机或塑料盒移动后必须重新执行本标定。

新 `31` 面向可见暗色开口的裸口酒坛，不再兼用封盖瓶的旧深度半径规则。
保留了相机移动前的离线回放配置，可运行：

```bash
python xarm_sdk_control/scripts/31_estimate_mouth_height.py \
  --config xarm_sdk_control/config/neck_localization_20260908.json
```

需要本机保留该配置引用的历史 RGB-D 与掩码。输出包含三路覆盖图、4 倍局部放大图、
拟合残差、拒绝的观测和条件不确定度。琥珀色为拟合外沿，俯视绿色为内沿，蓝色为
候选坛肩平面，紫色十字为坛口中心；蓝线不是已验证的扎带槽边界。

返回码 `0` 表示几何检查通过，`3` 表示已保存结果但需要复核，`2` 表示输入或拟合
失败。即使返回 `0`，仍不代表绝对精度或机械臂运动已验收。
旧 `--planar-radius-mm`、`--silhouette-neck-width-ratio` 等参数不再支持。
旧瓶盖实验的 `0.92 mm` 极差仅属于当时的固定摆放重复性，不能作为新算法精度。

## 当前完整流程参数

`21_complete_loading.py` 的稳定优先默认值：

| 参数 | 值 |
|---|---:|
| 空载水平速度 | `250 mm/s` |
| 载荷水平速度 | `180 mm/s` |
| 空载竖直速度 | `170 mm/s` |
| 载荷竖直速度 | `100 mm/s` |
| 精确运动速度 | `50 mm/s` |
| 接触速度 | `15 mm/s` |
| 空载/载荷/精确加速度 | `400/200/160 mm/s²` |
| 关节速度与加速度利用率 | `0.35/0.35` |
| 搬运净空 | `100 mm` |
| 白纸/陶瓷/竹壳下压 | `17/7/13 mm` |
| 竹壳目标角 | `+45/-60 deg` |

路径规划会先尝试密集连续 IK；靠近腕部奇异位形时，xArm 求解器可能切换
J4/J6 分支，此时自动退回端点插值估算。控制器 `only_check` 的 C24 结果当前
作为警告记录，因为现场已经多次出现“预检 C24、实际路径正常完成”的情况；
真正的运动指令返回非零仍会立即中止任务。

## 压力策略

- 所有物料在接触后都必须通过 CI0 初次吸附确认。
- 陶瓷片和竹壳在载荷段结束后继续强制确认。
- 白纸因材料透气和边缘漏气，第一次载荷压力丢失记为警告，后续载荷段不再
  重复轮询；这一例外只适用于当前白纸和当前压力阈值。
- 吸附失败的偏移序列为 `(0,0) -> (0,-5) -> (5,0) mm`，最多三次。

## 测试

```bash
conda activate winejar-xarm-sdk
python -m pytest -q xarm_sdk_control/tests
python -m compileall -q xarm_sdk_control
```

测试使用假机械臂、假真空控制器和临时目录，不连接实机。修改安全边界、真空
状态机、路径顺序、重试恢复或配置解析时，必须同步增加测试。

## 配置与产物

上料节拍可以从已保存的结果离线汇总，无需连接相机或机械臂：

```bash
python xarm_sdk_control/scripts/summarize_loading_timing.py \
  xarm_sdk_control/data/complete_loading/20260830_175757_904/result.json \
  --output-dir /tmp/winejar-loading-timing-review
```

可传入多个结果路径，输出目录必须尚不存在。保留失败样本，但同一次运行的
`result.json` 与 `failure.json` 只选一份。生成 JSON 明细与 Markdown 报告；
区间计时不等于纯硬件执行时间，细分诊断不重复相加。方法和历史瓶颈见
[第五轮优化记录](../docs/optimization-round5-20260919.md)。

复用相机会话的上料日志还会保存 `capture_diagnostics`，拆分取帧、对齐、恢复、
预览计算与文件保存；采图失败也保留诊断。历史日志缺少细分时会明确提示。
深度预览的等价计算优化、基准数据及诊断字段说明见
[第六轮优化记录](../docs/optimization-round6-20260919.md)。

- `config/end_effector.yaml`：TCP、弹簧行程、CO0/CO1、CI0。
- `config/jar_position.yaml`：固定坛口教点。
- `config/visual_position_bias.json`：当前相机与双吸盘的人工补偿。
- `calibration/global_camera_*.json`：固定相机外参。
- `data/`：可清理运行产物，代码不读取历史任务目录。

不要把标定文件、当前配置或模型权重误当成普通实验数据删除。


## 单手眼相机实时预览

在项目根目录运行 `./预览手眼相机.sh`，默认使用腕部相机011422070172，640×480 RGB-D。
只连接相机，不连接机器人。点击窗口后：S保存当前原始彩色、对齐深度和元数据，D切换深度并排显示，G切换网格，Q或Esc退出。
快照保存至 `xarm_sdk_control/data/wrist_live_preview/`；网格只用于显示，不写入原图，不代表机器人可达区域。
关闭预览后再运行其他采图/上料程序，避免独占相机冲突。
可传 `--serial <序列号>`，或用 `--duration 3 --save-first` 做短时采图检查。
脚本入口为 `scripts/41_preview_wrist_camera.py`，启动器支持WINEJAR_PYTHON指定Python环境。
