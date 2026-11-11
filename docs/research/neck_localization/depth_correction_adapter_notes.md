# 成熟 GP 深度校正适配（2026-10-24）

本步采用 Heindl 等人的**已有高斯过程深度校正模型**，用独立标定平面训练位置/距离相关的深度修正；
不是继续给每台相机拟合一个常数，也不是 Basso 完整联合标定的复现。
本文记录接口和方法边界；实际主动双目平面/酒坛对照结果在主实验报告中汇总。

## 选型与原实现核查

| 方法 | 已核查的原始资料 | 本步决定 |
|---|---|---|
| Basso、Menegatti、Pretto，TRO 2018 | [论文原文](https://arxiv.org/pdf/1701.05748)、[项目](https://iaslab-unipd.github.io/rgbd_calibration/)、[indigo 源码](https://github.com/iaslab-unipd/rgbd_calibration/tree/1839fce1a00d2013c0920a6cc52e2857648514fe) | 保留为完整标定参考，不把自己写的深度回归称为它的复现。 |
| Heindl 等，ICMV 2017，Spatio-thermal depth correction of RGB-D sensors based on Gaussian Processes in real-time | [作者代码、实验说明和数据链接](https://github.com/cheind/rgbd-correction/tree/3b2eb5d3dea0e34a3e37a15cea376e1e061c3bc4)、[原 GP 核/后验实现](https://github.com/cheind/rgbd-correction/blob/3b2eb5d3dea0e34a3e37a15cea376e1e061c3bc4/sensor_correction/gp_cpu.py)、[原特征构造及训练](https://github.com/cheind/rgbd-correction/blob/3b2eb5d3dea0e34a3e37a15cea376e1e061c3bc4/sensor_correction/apps/train.py) | 移植其 CPU 精确 GP 方程与特征形式，运行在现有 Python 3.12 / NumPy 2 / SciPy 环境。 |

Basso 实际代码在 `indigo` 分支，`master` 只有空 README；源码中的 local/global 校正和外参求解依赖
ROS、PCL、Ceres 与 `calibration_toolkit`。源文件含 BSD 三条款许可头，尽管 `package.xml` 的许可字段还是 TODO。
本文没有复制 Basso 代码，也没有完成虚拟棋盘角点标定、RGB–深度位姿优化或相机内参估计。

Heindl 仓库的 BSD-3-Clause LICENSE 和相关原文件已阅读。许可全文、作者和版本已保留在新模块。
原 NumPy 独立分支可以参考，但文件顶层依赖旧 `sklearn.externals.joblib`，独立分支漏导入三角求解函数；
独立分支和 sklearn 分支对 `noise_std` 是否平方也不一致。
本适配明确采用作者 sklearn 分支的标准差约定 `noise_std**2`，移除过时依赖并采用 SciPy Cholesky 三角求解。
未下载/重跑作者约 80 MB 数据集，不声称重现原论文数值。

## 实际模型

对测量像素 \((u,v)\)，以标定内参反投影测量深度 \(z\)，生成 \(q=(x,y,z,T)\)。
训练目标为独立平面参考的光轴深度减去测量值：\(\Delta z=z_{ref}-z\)。
模型为作者使用的各向异性平方指数 GP：

\[
k(q_i,q_j)=\sigma_f^2\exp[-\tfrac12\sum_d(q_{id}-q_{jd})^2/\ell_d^2],
\quad C=K+\sigma_n^2I.
\]

训练求 GP 后验，并通过边缘似然选择长度尺度、信号和噪声参数；推理输出
\(z_{corrected}=z+\widehat{\Delta z}(x,y,z,T)\)。
它可以表达平滑的空间和距离相关偏差，不要求事先为偏差指定多项式或幅值。
目标均值先扣除再加回，与作者 `normalize=True` 的意图一致。

适配与原训练脚本的差异是：按独立帧划分训练与测试；最多 512 训练点；批量推理和无 pickle 保存；
通过 SciPy 的带梯度边缘似然优化替代旧 sklearn；单次确定性初始化，无作者脚本的随机重启。
超参数初值/界限属于工程配置，不能当作 D435i 实测噪声；优化不收敛会原样记录。

## 标定数据与目标测试隔离

每个模型只训练一台相机。如果多个虚拟相机完全共享一个传感器预设，本轮可共享
`camera_id=sim_d435_preset` 模型；这不能推导出现实中三台相机可共用实测校正。

输入 JSON 的路径相对于 manifest 文件，也可以使用绝对路径：

```json
{
  "schema_version": "winejar-depth-calibration-v1",
  "reference_source": "known_virtual_planar_fixture",
  "frames": [{
    "frame_id": "plane_z0p7_tilt_train",
    "camera_id": "sim_d435_preset",
    "split": "train",
    "measured_depth": "plane_z0p7/native_float_z.npy",
    "reference_depth": "plane_z0p7/reference_z.npy",
    "valid_mask": "plane_z0p7/board_mask.npy",
    "K": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
    "temperature_c": 25.0
  }]
}
```

深度为米、光轴 Z。像素遵循传感器 K 的整数像素中心约定，不额外加 0.5。
只有 `train` 平面参与模型训练，`validation`、`test` 只报告误差；重复帧 ID 或重复测量文件拒绝加载。
参考平面距离/姿态从独立虚拟夹具解析求交取得，不能拿酒坛真值表面训练；没有“从同一测量点云拟合的平面”冒充绝对参考。
这仍是理想已知参考的标定实验，与通过虚拟图像求解标定板姿态的完整标定实验分开。

训练每帧取近似 8×8 格点，仅使用独立平面掩码内有效测量，超出总上限后固定种子选点。
测试报告原始/校正深度对独立平面的偏差、RMSE、中位数、P95、最大值和有效率。
误差采用配对有效像素计算，另外明确报告无效化及覆盖率，不能通过丢点制造改善。

## 运行与集成

```bash
cd /home/wunicheng/Git/WineJar
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  conda run --no-capture-output -n winejar-stereo python scripts/fit_depth_correction.py \
  --manifest /absolute/path/to/plane_manifest.json \
  --output /absolute/path/to/NEW_calibration_result --evaluation-splits validation
```

输出 `models/<camera_id>.npz` 和 `calibration_report.json`；输出目录必须不存在。
模型保留训练帧 ID、输入哈希、相机 K、配置和优化器状态。
调参时只报告 validation；锁定模型后，单独评价测试平面，不重新拟合：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  conda run --no-capture-output -n winejar-stereo python scripts/fit_depth_correction.py \
  --manifest /absolute/path/to/plane_manifest.json \
  --evaluate-models /absolute/path/to/NEW_calibration_result/models \
  --evaluation-splits test --output /absolute/path/to/NEW_locked_test_result
```

```python
from neck_localization.depth_correction import GaussianProcessDepthCorrection
model = GaussianProcessDepthCorrection.load('/absolute/path/to/model.npz')
corrected_z, diagnostics = model.correct_depth(
    native_float_z, native_K, temperature_c=25.0, return_diagnostics=True)
```

推理没有真值参数；在原生左 IR 浮点深度上校正，再交给下游量化、滤波及 RGB 注册。
校正时 K 必须与标定相同，禁止悄悄用模型补偿改变后的相机内参。
深度 0、负值、NaN、Inf 保持无效，模型不会生成新表面或填孔。

## 必须保留的局限

- 恒温 25°C 时，温度只是固定输入，不能声称已验证温漂补偿；渲染传感器的“thermal noise”也不等于真实温度漂移模型。
- GP 的零均值/常均值外推可能失效；逐帧报告是否超出训练特征包围盒，包围盒内也不保证可靠。
- 训练用平面、应用于曲面时，入射角、材料响应、遮挡边界、局部纹理造成的错误未必是位置/距离的单值函数。
  这些正是平面残差改善后仍可能无法改善坛颈的原因，必须看独立酒坛任务指标。
- 本模型不修复无效深度，不估计相机外参，不解决共享世界坐标误差，也不保证去除像素关联或匹配错误。
- GP 后验标准差是在 K、参考和模型假设固定条件下的统计量；本步不把它作为完整传感器或圆环置信度。
- 精确 GP 训练复杂度三次方、逐像素均值预测随训练点数线性增长；训练点上限与分块是实现取舍，运行耗时需随实际分辨率报告。

## 已完成的代码验证

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  conda run --no-capture-output -n winejar-stereo \
  python -m unittest discover -s tests/neck_localization -p test_depth_correction.py -v
```

6 项通过：独立稠密 GP 方程对照、光轴/像素约定、无效点/模型保存往返、掩码采样、训练测试输入隔离、
假设平滑空间/距离误差的留点插值。最后一项只是实现诊断，不能作为真实相机或主动双目有效性结论。

## 已完成的主动双目平面基线

输入为 `experiments/neck_localization/runs/active_depth_calibration_20261024_v1/manifest.json`：
93 训练平面（0.50–1.10 m、0.02 m 步长、三种倾角），14 个独立验证平面，12 个独立测试平面。
均为同一虚拟 D435 预设和恒温输入；没有坛模型或坛真值进入训练。
原生深度来自主动双目匹配，参考为已知平面的解析交点，不是从测量深度拟合的平面。

事先固定默认 512 点、单次 GP 超参数优化；只查看 validation 后锁定模型，随后单独运行 test。
没有为了取得改善调整长度尺度、采样点数或偏差，也没有在 test 上重训。

| 全部配对有效像素合并 | 原始深度 RMSE / P95绝对误差 | GP校正 RMSE / P95绝对误差 | 原始 → 校正平均有符号误差 |
|---|---|---|---|
| 验证 14 帧，3,189,392 点 | 4.987 / 10.700 mm | 5.002 / 10.728 mm | +0.011 → +0.380 mm |
| 测试 12 帧，2,631,724 点 | 4.285 / 8.926 mm | 4.300 / 8.944 mm | −0.0166 → +0.352 mm |

标定mask构造时已经要求原生测量深度>0，所以其中原始/校正有效率均为100%是**条件比例**，
不能用来证明平面本身或整个传感器没有孔洞。校正没有进一步删点或填孔，平面边缘和全视野覆盖率不在该数值中。
测试绝对中位数 2.456 → 2.474 mm，最大值 22.797 → 23.166 mm。
各距离/正负倾角的原始 RMSE 约 1.66–7.25 mm，校正约 1.69–7.28 mm；12 测试帧只有 2 帧 RMSE 略降，
不能用总体不变掩盖个别退步，也不能将数百万相关像素看作数百万独立样本。

GP 29 次迭代收敛，选择的信号标准差压到本适配配置下界 0.001 mm，观测噪声尺度约 5.776 mm，
空间长度尺度变得很长，最终近似统一加上 0.369 mm。
**本次成熟模型适配没有获得改善，因此不应作为默认必加的补偿。**
这也不是“GP 方法一般无效”的结论：当前传感器仿真使用准确 K/T，没有单独注入真实单机的空间标定畸变；
稀疏训练、单优化初值、固定温度和缺少材料/表面方向特征均可能影响迁移。
在这批数据上，平面整体平均误差已经小于散布误差，给它再加一项均值修正没有充分收益。

进一步观测诊断显示误差具有强空间相关性：26 个验证/测试平面的相邻像素误差 Pearson 相关系数，
横向 0.947–0.992、纵向 0.990–0.996。
这支持后续按匹配空间结构处理误差，而不支持把每个深度点当独立同方差高斯样本。

## 从头重建这批平面与GP

下面从仓库根目录运行，每个NEW目录必须不存在。先渲染119帧，训练只看train并报告validation，
然后以同一锁定模型单独评价test，不根据test重训。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
conda run --no-capture-output -n winejar-stereo \
python scripts/prepare_active_depth_calibration.py \
  --output experiments/neck_localization/runs/NEW_active_planes

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
conda run --no-capture-output -n winejar-stereo \
python scripts/fit_depth_correction.py \
  --manifest experiments/neck_localization/runs/NEW_active_planes/manifest.json \
  --output experiments/neck_localization/runs/NEW_gp_fit --evaluation-splits validation

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
conda run --no-capture-output -n winejar-stereo \
python scripts/fit_depth_correction.py \
  --manifest experiments/neck_localization/runs/NEW_active_planes/manifest.json \
  --evaluate-models experiments/neck_localization/runs/NEW_gp_fit/models \
  --evaluation-splits test --output experiments/neck_localization/runs/NEW_gp_test
```

独立标定从头生成仅完整做过一遍；8坛型基准与一键重跑已进行完整输入/输出对照。
该一键入口直接读取results中原样保留的小型锁定模型，不会悄悄换成新训练模型。
只观察单次平面仍不能将可重复系统图案和随帧变化的相关噪声完全分开；上述相关系数没有给出这种因果分解。
验证全幅像素中约 1%–41% 超出稀疏训练 XYZ 包围盒，距离越远/倾斜时通常越多，亦须记录为训练支持域局限。

全幅测试校正耗时中位 2.114 s、P95 2.519 s，训练/预测 BLAS 均为单线程；运行时有其他实验，
仅作为实现开销记录，不是受控实时性结论。

复现与成果路径（均相对于项目根目录）：

- `runs/` 以下均位于 `experiments/neck_localization/`。
- 验证及锁定模型：`runs/gp_depth_calibration_20261024_v1/`。
- 单独锁测：`runs/gp_depth_calibration_locked_test_20261024_v1/calibration_report.json`。
- 全像素合并统计：同目录 `pooled_holdout_summary.json`。
- 每个距离/倾角统计及空间相关性：同目录 `plane_pose_metrics.csv`。
- 模型：`runs/gp_depth_calibration_20261024_v1/models/sim_d435_preset.npz`；SHA256：
  `bf0f1699578616f153d6d00c038df21ae47680544d074565505d1c44caf7cf3a`。

实际执行命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
conda run --no-capture-output -n winejar-stereo python scripts/fit_depth_correction.py \
  --manifest experiments/neck_localization/runs/active_depth_calibration_20261024_v1/manifest.json \
  --evaluation-splits validation \
  --output experiments/neck_localization/runs/gp_depth_calibration_20261024_v1

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
conda run --no-capture-output -n winejar-stereo python scripts/fit_depth_correction.py \
  --manifest experiments/neck_localization/runs/active_depth_calibration_20261024_v1/manifest.json \
  --evaluate-models experiments/neck_localization/runs/gp_depth_calibration_20261024_v1/models \
  --evaluation-splits test \
  --output experiments/neck_localization/runs/gp_depth_calibration_locked_test_20261024_v1
```

复跑时换新的输出目录，原始记录保留。推理只有测量深度、K、固定温度；比较指标在推理完成后才读取独立平面参考。
