# 独立仿真仓库与运行环境核查

核查日期：2026-10-21。全程只读取本地文件并执行无设备的 MuJoCo 渲染；未修改独立仿真仓库、环境或安装依赖。

## 维护边界与复用结论

- `mujoco_xarm6/` 是独立 Git 仓库，远端 `git@github.com:Wunicheng233/Winejar-mujoco.git`，当前分支 `main`，HEAD `8c84712`（`fix: adopt raised right-shifted side camera`）。核查前后工作树均干净。
- 父仓库 `.gitignore:41` 忽略整个 `mujoco_xarm6/`，父仓库没有跟踪其代码。不能把本次唯一研究代码留在那里，否则父仓库交接无法复现。
- 独立仓库无额外 `AGENTS.md`。其 README 的主线是上料/扎口动画、xArm6 与柔性物料模拟，不是相机计量或坛颈研究。
- 本次应在父仓库 `experiments/neck_localization/` 建立独立、受版本管理的研究模块；现有仓库保留，只读参考资产与相机转换。研究模块不需要引入机器人、IO 或实机 SDK。

## 可复用接口与限制

| 路径 | 可参考内容 | 不能直接采用的原因 |
|---|---|---|
| `mujoco_xarm6/scripts/04_capture_sim_camera_frame.py` | `opencv_camera_to_world()`：`R_W_C = R_W_MJ @ diag(1,-1,-1)`；同一 Renderer 生成 RGB 与 depth | `render_color_and_depth()` 将深度量化为毫米 `uint16`，破坏高精度无噪声诊断；应保留原始 `float32 m` |
| 同上 `intrinsics_from_fovy()` | `fx=fy=H/(2*tan(fovy/2))` | 旧 `cx=W/2,cy=H/2` 与整数像素索引采样相差半像素；本次采用 `(W-1)/2,(H-1)/2` 并用解析场景核验 |
| `tools/generate_scene_assets.py:generate_wine_jar()` | 可参数化旋转体的简单实现、内外沿分别建模 | 这是手工设计的展示模型，无真实酒坛测量来源，不能作为真实 CAD 或目标定义依据 |
| `scene/scene_winejar*.xml` | 材质、光照、相机朝向、近似酒坛外观 | 生产动画包含叶片、纸、碰撞代理和可见 target sites；研究场景应显式关闭真值 sites，避免目标泄露 |

现有酒坛模型的物体坐标系以底部中心为原点、Z 向上、单位米：总高 `0.46 m`；外坛口半径 `0.06 m`、内坛口半径 `0.0525 m`；上部坛颈是 `z=0.445..0.46 m` 的半径 60 mm 圆柱。生产场景另有 `neck_tie_target_site` 位于 `z=0.452 m`，而 `jar_mouth_center` 位于 `z=0.46 m`。这些分别是颈部中间作业位置与上沿位置，不能混为同一圆环；其数值也不是实机测量。

外观网格只有 96 个周向分段，截面是多边形近似圆。半径 60 mm 时边中点与解析圆的最大径向差约 `60*(1-cos(pi/96))=0.0321 mm`。更大的问题是分段直线轴向轮廓与真实酒坛未知，而非该数值本身。

## 可用环境

无需安装即可用于本次研究的环境为：

```bash
MUJOCO_GL=egl conda run -n embodied-sim python <研究脚本>
```

| 环境 | Python | MuJoCo | NumPy | OpenCV | SciPy |
|---|---:|---:|---:|---:|---:|
| `embodied-sim` | 3.12.14 | 3.8.1 | 2.2.6 | 5.0.0 | 1.18.1 |
| `embodied-course061` | 3.12.14 | 3.8.1 | 2.2.6 | 5.0.0 | 1.18.1 |
| `winejar-mujoco` | 3.11.15 | 3.10.0 | 2.4.6 | 4.13.0.92 | **缺失** |

`winejar-vision` 与 `winejar-xarm-sdk` 没有 MuJoCo，不要为了此研究修改其职责或安装依赖。`winejar-mujoco/environment.yml` 也未声明 SciPy。现有 `embodied-sim` 仅作为当前执行环境；研究模块应提供自己的依赖文件，不能把另一个项目的环境名称作为唯一复现说明。GPU 为 RTX 4050 Laptop，驱动 580.126.09；EGL 离屏渲染通过，不需要窗口。

## 已执行的相机/深度诊断

1. 相机 `[0,0,1] m` 朝本地 `-Z`，水平平面 `z=0`，320×240、fovy=60°。`winejar-mujoco` 的 EGL RGB shape `(240,320,3)`、depth dtype `float32`。深度中心、离轴像素和全图均为 `1.000000119 m`。这验证 `Renderer.enable_depth_rendering()` 返回沿光轴的 **Z 深度（米）**，不是相机至点的欧氏距离，也不是原始 OpenGL 非线性 z-buffer。
2. 相同相机观察绕 Y 倾斜 20° 且经过世界原点的平面，在 `embodied-sim` 比较解析交点深度与渲染值。整数像素中心以 `(W-1)/2` 为主点时：

| `visual.quality.offsamples` | 主点 cx | 平均绝对深度差 |
|---:|---:|---:|
| 0 | 159.5 | 0.00128 mm |
| 4（默认） | 159.5 | 0.23703 mm |
| 0 | 160.0（旧脚本） | 0.94972 mm |

该诊断揭示了两个不同来源：半像素主点约定与多重采样的深度取样行为。无噪声基线应显式使用 `<quality offsamples="0"/>`；后续若研究多重采样，必须另报配置。这里的数值不是坛颈定位误差，也不构成任何实机精度证据。

解析深度可按下式复核（本诊断的特定相机/平面）：

```python
fx = height / (2 * np.tan(np.deg2rad(fovy_deg) / 2))
cx = (width - 1) / 2
theta = np.deg2rad(20)
u = np.arange(width)
expected_z = np.cos(theta) / (
    np.cos(theta) - np.sin(theta) * (u - cx) / fx
)
```

实现时仍需将深度背景/远裁剪面判为无效；不能把所有有限正值都当作物体表面。还需在新模块针对像素 Y 方向、相机位姿往返、目标投影与 RGB/depth/segmentation 对齐做独立检查。真值 segmentation 可用于理想诊断，但不得默认进入实际 RGB 提取算法。
