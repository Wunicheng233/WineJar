# 代码结构与维护边界

## 当前离线圆环研究

用户2026-10-21调整主线为文献与仿真。`neck_localization/`负责纯离线相机几何、
MuJoCo参数场景、估计器、虚拟板标定诊断、观察域扰动、真值评估；
`scripts/run_neck_simulation.py`负责批次，`experiments/neck_localization/`保留配置/小结果。
研究模块不导入机器人、RealSense设备或模型服务；测试在`tests/neck_localization/`。
当前首轮入口`scripts/run_short_neck_stage1.sh`，仅生成静态资产并调用受限深度剖面基线。
`jar_model.py`/`scene_preparation.py`维护完整短颈模型和固定相机，`depth_profile.py`只接收观测；
`evaluation_only/`由评估器持有，`diagnose_depth_profile_geometry.py`明确是允许真值干预的诊断。
旧`scripts/run_neck_simulation.py`仍为早期人工槽带探索，不是当前默认研究流程。
历史`xarm_sdk_control/rim_geometry.py`保持原样，仅作为思路与回放对照。
独立`mujoco_xarm6/`只读核查，维护边界不变。

2026-10-25扩展：`active_stereo.py`用SAPIEN/SimSense生成主动双目测量，
通过librealsense **software_device离线帧**做官方空间滤波/对齐，不扫描或连接实机。
`depth_correction.py`维护论文GP，`rendered_calibration.py`从实际棋盘RGB求RGB参数，
`stereo_robust.py`只适配RGB前景，原冻结几何保持不变。
`native_stereo.py`为原生IR点表示的小对照，`silhouette_profile.py`为未成功的RGB轮廓原型，均非默认替换。
MuJoCo场景/模型/查看器保留；新环境`winejar-stereo`及`requirements-active-stereo.txt`与旧环境隔离。
新入口`reproduce_active_stereo.py`/`view_active_stereo.py`和边界见[完成报告](research/neck_localization/active_stereo_results_20261025.md)。

## 数据与控制流

```text
RealSense RGB + aligned Z16 + frame.json
    → GroundingDINO检测 + SAM2掩码（独立视觉环境/Unix socket）
    → 类别与轮廓复核 + 原始深度 + RGB内参
    → 相机点（米）→ base_T_camera → 基坐标（毫米）
    → 明确是否应用位置补偿、工具转换、材料下压
    → 路线几何 → 工作空间 / IK / TCP与关节限位 / 速度规划
    → xArm运动 + CO0/CO1 → CI0吸附确认与有限重试
    → 原始输入、执行日志、图像、计时与最终状态
```

## 模块分工

| 模块/目录 | 职责 |
|---|---|
| `robot_motion.py`、`cli_utils.py` | 控制器连接、状态、复位与工作空间参数 |
| `visual_targeting.py` | 标定加载、采图/视觉服务、目标选择 |
| `cross_place_route.py` | 取放路线几何与分层高度 |
| `cross_place_motion.py`、`trajectory_speed_planner.py` | 路线执行、速度/逆解预查、载荷检查 |
| `cross_place_retry.py`、`pick_test_workflow.py` | 单物料、失败撤离、重扫和有限重试 |
| `vacuum_control.py`、`suction_sensor.py` | 真空IO状态机、压力等待和去抖 |
| `realsense_capture.py` | 对齐采集、保存、预览、会话重启和诊断 |
| `hand_eye_calibration.py`、`calibration_utils.py` | 手眼/固定相机求解与位姿转换 |
| `calibration_*.py`、`pair_calibration_session.py` | 板检测、采样质检、内参、独立验收 |
| `multiview_rig.py`、`rim_geometry.py`、`mouth_height*.py` | 暂停的三相机坛口测量 |
| `suction_surface.py` | 双吸盘候选姿态和表面风险评估 |
| `run_output.py`、`task_timing.py`、`loading_performance.py` | 结果、失败、事件和节拍报告 |
| `scripts/` | 通用CLI和历史任务入口，编号不代表当前推荐顺序 |
| `experiments/recommissioning_20261014/` | 特定工位恢复器；明确区分实物验证与未执行草稿 |
| `vision/scripts/` | 离线/常驻推理与三视图掩码生成，无机器人控制 |

## 代码入口的状态

- `01`、`22`：状态/压力读取；`13`：仅相机采图；`41`：腕部预览。
- `15`：会切换气路，不是只读；`02`：会准备机器人，不是状态检查。
- `19/20/21/34`：有实机能力，先读副作用、默认home/补偿/料区，不能把dry-run理解为完全不接设备。
- `23–26/30–32/35–40`：采集/求解/离线检查入口，按各自`--help`分辨设备访问。
- `experiments`中的单物料恢复器基于10月14日固定场景；完整恢复器仅生成过clear计划。

## 测试策略

控制测试使用模拟控制器、合成几何、临时文件；不要求接相机、机器人或下载模型。
视觉测试在视觉环境用unittest和模拟predictor，验证默认精度、可选BF16和输出。
 benchmark脚本是有输入要求的实验，不包含在日常无硬件测试命令里。

```bash
conda run -n winejar-xarm-sdk python -m pytest -q xarm_sdk_control/tests
conda run -n winejar-vision python -m unittest discover -s vision/tests
```

修改逻辑时优先复用模块，不复制整段抓取状态机到数据目录。
当前恢复草稿保留了历史实现，尚未抽象为正式通用入口；不要把目录迁移称为实机功能验收。
环境依赖由两个environment.yml及SAM2 requirements维护；本次不升级SDK或视觉库。
