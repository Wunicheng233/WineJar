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

## 任务入口

- `19_visual_cross_place_two_leaves.py`：只验证两片竹壳交叉放置。
- `20_visual_pick_test.py`：按物料类型执行吸起、悬停、放回；适合硬件调试。
- `21_complete_loading.py`：完整上料生产入口，当前实机主流程。
- `23_calibrate_camera_intrinsics.py`：三相机坛颈定位的第一步；逐台采集普通棋盘并保存内参与重投影误差，不连接机械臂。
- `24_calibrate_camera_pair_extrinsics.py`：三相机坛颈定位的第二步；用同时可见的普通棋盘标定俯视相机与一台侧视相机的相对外参，输出 `T_俯视←侧视`，不连接机械臂。
- `25_create_multiview_camera_rig.py`：校验一组内参和两组俯视-侧视外参后，生成唯一的三相机阵列配置。
- `26_capture_three_camera_frame.py`：按阵列配置采集一组近同步 RGB-D 帧及采集偏差；`--lock-color` 在预热后冻结 RGB 曝光、增益和白平衡，适用于固定背景差分。
- `27_segment_multiview_foreground.py`：使用固定空背景差分、工作区 ROI 和外轮廓填充生成三路物体轮廓，供视觉外壳重建使用。
- `28_reconstruct_visual_hull.py`：在俯视相机坐标系中对三路轮廓求交，输出局部体素视觉外壳点云和可追溯的搜索范围。
- `29_fit_neck_from_visual_hull.py`：从视觉外壳估计旋转体轴线、坛口平面、局部外径曲线和候选捆扎平面，并显式报告置信度。
- `30_calibrate_workcell_plane.py`：从空工位 RGB-D 帧拟合传送面，建立以盒面为 `Z=0`、朝向俯视相机为 `+Z` 的工位坐标系；不连接机械臂。
- `31_estimate_mouth_height.py`：由三路 RGB-D 独立估计最高稳定表面，并由两台侧视相机估计瓶肩以上的稳定窄颈高度范围；不连接机械臂。
- `32_calibrate_wrist_hand_eye.py`：棋盘固定、机械臂带腕部相机移动的手眼标定；同步保存 TCP、活动 TCP 偏置、法兰位姿和关节角，不发送运动命令。
- `33_inspect_suction_surface.py`：在腕部 RGB-D 帧中对荷叶、竹壳、陶瓷片或白纸搜索双吸盘姿态并输出缺陷图；不连接机械臂。
- `34_wrist_guided_pick_test.py`：先复位机械臂，再同步采集腕部 RGB-D 与法兰位姿，筛选吸附面并执行带 CI0 确认和换点重试的单物料吸取测试。

当前坛颈任务已经收缩为只测量坛口相对于传送面的高度。脚本 `28`、`29` 保留为
视觉外壳实验记录，不再是正式高度定位链路。

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

将空的黑蓝色塑料盒固定在最终工位，采集 RGB-D 后，用盒面内部 ROI 标定：

```bash
python xarm_sdk_control/scripts/30_calibrate_workcell_plane.py \
  --rig xarm_sdk_control/calibration/multiview_neck/three_camera_rig_20260901.json \
  --capture xarm_sdk_control/data/multiview_neck/workcell_frame_candidates/<capture-run> \
  --roi 300,250,135,90
```

当前 ROI 仅适用于现有相机和塑料盒位置。输出 `workcell_frame.json` 包含平面方程、
双向齐次变换、深度有效率、RANSAC 内点率以及 RMS/P95 残差。当前阶段只使用
`Z=0` 和 `+Z`；X 轴暂由俯视相机图像右方向投影到盒面，后续确定真实传送方向后
再固定 X/Y 朝向。相机或塑料盒移动后必须重新执行本标定。

放入裸口酒坛或测试瓶后，使用同一阵列配置采集一组 RGB-D，再执行：

```bash
python xarm_sdk_control/scripts/31_estimate_mouth_height.py \
  --rig xarm_sdk_control/calibration/multiview_neck/three_camera_rig_20260901.json \
  --workcell-frame xarm_sdk_control/calibration/multiview_neck/workcell_frames/<workcell-run>/workcell_frame.json \
  --capture xarm_sdk_control/data/multiview_neck/mouth_height_candidates/<capture-run> \
  --roi overhead:320,230,130,150 \
  --roi side_a:250,40,150,440 \
  --roi side_b:380,110,170,370
```

绿色点是工位中心半径 `75 mm` 内且高于盒面 `30 mm` 的候选物体点，橙色点是
用于估计顶面高度的最高稳定带，蓝色点是两台侧视相机确认的窄颈区间。三台相机
先独立测量顶面，默认至少两台结果要在 `5 mm` 内一致，再取中位数。颈部下边界
由侧视深度的高度-可见半径曲线确定：当半径持续低于下方瓶身参考半径的 `75%`
时，判定瓶肩已经收敛到窄颈。封盖测试瓶测量的是瓶盖顶面和包括瓶盖在内的窄段；
裸口酒坛测量的才是坛口最高边缘和真实坛颈。

当前固定测试瓶四次顶面结果为 `207.32/207.83/207.94/208.24 mm`，总体极差
`0.92 mm`。当前首组颈部范围为 `183.32..207.32 mm`，两台侧视相机的下边界
差为 `0 mm`。这些数字说明固定摆放重复性和多视图一致性，绝对误差仍应与人工
实测值比较。

## 历史视觉外壳实验流程

这套流程目前只使用三台固定 RealSense，不连接或驱动机械臂。坐标输出在俯视
相机光学坐标系中；与机械臂基坐标系的关系要在下一阶段通过独立的手眼/工位标定建立，
不能把这里的坐标直接发送给机器人。

1. 相机、支架或曝光参数改变后，先采一组空场背景；相机位置不变时可复用该背景。
2. 将酒坛放入已限定的工位，使用**相同的手动 RGB 成像参数**采一组前景帧。
3. 以工作区 ROI 做背景差分。每个 ROI 和阈值缩放都是工位参数，应随相机安装和背景
   变化重新检查，不能泛化到另一台设备。
4. 对三路实心轮廓求视觉外壳，再拟合旋转体轴线与半径曲线。仅当输出 JSON 中
   `status` 为 `ok` 且 `failure_reason` 为 `null` 时，才允许使用候选平面。

当前台架已验证的一组采集参数为 RGB 曝光 `166`、增益 `64`、白平衡 `4600`。它们
不是通用相机参数，只用于让一对前景/背景图具有可比性。下面的命令用环境变量避免把
一次运行目录写死：

```bash
conda activate winejar-xarm-sdk

RIG=xarm_sdk_control/calibration/multiview_neck/three_camera_rig_20260901.json
BG=xarm_sdk_control/data/multiview_neck/locked_backgrounds/<empty-scene-run>
FG=xarm_sdk_control/data/multiview_neck/locked_capture_candidates/<jar-run>

# 空场和有坛前景均必须使用完全相同的三项颜色控制。
python xarm_sdk_control/scripts/26_capture_three_camera_frame.py \
  --rig "$RIG" \
  --output-root xarm_sdk_control/data/multiview_neck/locked_backgrounds \
  --color-exposure 166 --color-gain 64 --color-white-balance 4600

python xarm_sdk_control/scripts/26_capture_three_camera_frame.py \
  --rig "$RIG" \
  --output-root xarm_sdk_control/data/multiview_neck/locked_capture_candidates \
  --color-exposure 166 --color-gain 64 --color-white-balance 4600

python xarm_sdk_control/scripts/27_segment_multiview_foreground.py \
  --rig "$RIG" --foreground "$FG" --background "$BG" \
  --roi overhead:250,260,160,190 \
  --roi side_a:300,40,190,440 \
  --roi side_b:420,70,220,410 \
  --camera-threshold-scale side_b:0.62

# 将上一条命令打印出的目录填入 SEG。
SEG=xarm_sdk_control/data/multiview_neck/segmentations/<segmentation-run>
python xarm_sdk_control/scripts/28_reconstruct_visual_hull.py \
  --rig "$RIG" --segmentation "$SEG" \
  --voxel-size-mm 2 --max-auto-depth-span-mm 180

# 将上一条命令打印出的目录填入 HULL。
HULL=xarm_sdk_control/data/multiview_neck/visual_hulls/<visual-hull-run>
python xarm_sdk_control/scripts/29_fit_neck_from_visual_hull.py \
  --visual-hull "$HULL" --slice-width-mm 8 --min-circularity 0.55
```

`27` 的 `--camera-threshold-scale` 是对该相机 Otsu 阈值的乘数，小于 `1` 会保留
更多弱对比轮廓。`0.62` 仅针对当前侧视 B 的白色瓶身；应优先通过补光、遮光和背景
稳定性改善成像，再少量调整该参数。`29` 的 `--min-circularity` 是半径包络
`min/max` 的最低比例；当前三相机视觉外壳有轻微椭圆化，因此取 `0.55`，并会相应
降低输出置信度。若无法找到稳定截面，脚本仍会生成 `status: failed` 的结果 JSON，
其中包含失败原因与本次阈值参数。

早期的非接触接近脚本和失败的荷叶竖直运输原型已经删除。Git 历史仍保留这些
实验，不应把它们重新复制回生产代码。

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

- `config/end_effector.yaml`：TCP、弹簧行程、CO0/CO1、CI0。
- `config/jar_position.yaml`：固定坛口教点。
- `config/visual_position_bias.json`：当前相机与双吸盘的人工补偿。
- `calibration/global_camera_*.json`：固定相机外参。
- `data/`：可清理运行产物，代码不读取历史任务目录。

不要把标定文件、当前配置或模型权重误当成普通实验数据删除。
