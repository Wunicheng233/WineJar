# WineJar

基于 xArm6、RealSense D435i、GroundingDINO 和 SAM2 的酒坛柔性物料视觉上料项目。

当前实机流程能够完成：

- 固定全局相机 RGB-D 采集与坐标标定
- GroundingDINO 开放词汇检测
- SAM2 像素级分割
- 粽叶与陶瓷片三维定位
- xArm6 视觉引导运动
- 自制双吸盘的泵与放气阀控制
- 单片粽叶吸放
- 两片叠放粽叶二次视觉定位与交叉放置

MuJoCo 仿真与演示动画维护在独立仓库：
[Wunicheng233/Winejar-mujoco](https://github.com/Wunicheng233/Winejar-mujoco)。

## 仓库结构

```text
.
├── docs/                  项目设计、标定与实施文档
├── vision/                GroundingDINO + SAM2 独立视觉环境
├── xarm6/                 xArm 用户与开发者手册
├── xarm_sdk_control/      实机控制、标定、视觉抓取与测试
└── 项目策划.docx          原始项目策划
```

采集图像、深度图、分割结果、模型权重和运行日志不会提交到 Git。

## 环境配置

控制与 RealSense 环境：

```bash
conda env create -f xarm_sdk_control/environment.yml
conda activate winejar-xarm-sdk
```

视觉环境：

```bash
conda env create -f vision/environment.yml
conda activate winejar-vision
pip install -r vision/requirements-sam2.txt
```

视觉模型首次运行时由 Hugging Face 下载。不要把模型权重提交到仓库。

## 硬件约定

- xArm6 控制器地址默认存放在 `xarm_sdk_control/config/robot.yaml`
- RealSense 型号：Intel RealSense D435i
- `CO0`：真空泵继电器
- `CO1`：放气阀继电器
- 当前没有真空压力传感器

GPIO 真值表：

| 状态 | CO0 气泵 | CO1 放气阀 |
|---|---:|---:|
| 待机 | 0 | 0 |
| 吸取与保持 | 1 | 0 |
| 主动释放 | 0 | 1 |
| 释放完成 | 0 | 0 |

## 常用流程

只读检查机械臂：

```bash
python xarm_sdk_control/scripts/01_read_status.py
```

检查真空 I/O，不启用机械臂运动：

```bash
python xarm_sdk_control/scripts/15_vacuum_io_check.py --test status
python xarm_sdk_control/scripts/15_vacuum_io_check.py --test cycle --hold-seconds 2
```

单片粽叶完整吸放：

```bash
python xarm_sdk_control/scripts/16_visual_vacuum_pick.py \
  --run \
  --stage full \
  --confirm-no-pressure-sensor \
  --allow-validated-clipped-leaf-for-full \
  --compression-mm 15 \
  --min-x-mm 180 --max-x-mm 520 \
  --min-y-mm -180 --max-y-mm 180 \
  --min-z-mm -75 --max-z-mm 430 \
  --transit-speed 70 \
  --descent-speed 20 \
  --acc 240
```

两片粽叶交叉放置：

```bash
python xarm_sdk_control/scripts/19_visual_cross_place_two_leaves.py \
  --run \
  --confirm-no-pressure-sensor \
  --allow-validated-clipped-leaf \
  --min-x-mm 180 --max-x-mm 520 \
  --min-y-mm -180 --max-y-mm 180 \
  --min-z-mm -75 --max-z-mm 430 \
  --transit-speed 70 \
  --vertical-speed 35 \
  --contact-speed 15 \
  --acc 240
```

## 安全规则

- 实机命令必须明确提供工作空间边界。
- 第一次运行新路径时，操作者必须位于急停按钮可及位置。
- 控制器错误 `C1` 表示控制箱急停按钮被按下，必须先物理释放再重新使能。
- 没有压力传感器时，程序无法确认物料是否吸牢。
- 吸附状态下发生异常时，程序会优先保持真空，避免物料突然掉落。
- 标定文件和 `visual_position_bias.json` 与当前实验台布局绑定；移动相机、机械臂或执行器后必须重新验证。

## 测试

在仓库根目录运行：

```bash
PYTHONPATH=. pytest -q xarm_sdk_control/tests
```
