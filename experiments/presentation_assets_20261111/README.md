# PPT图片与数据入口

2026-11-11导出与整理。这是**保存结果的可视化与跨设备发布**，没有新增定位算法实验。做PPT直接使用PNG、CSV和下方报告，不必运行仿真。

## 单／双／三机直接比较

| 文件 | 内容 |
|---|---|
| [nominal_014_comparison.png](camera_count/nominal_014_comparison.png) | 三种配置都有输出。上排实体+圆/轴，下排真实观测点云；三机不是每项误差都最小 |
| [stress_010_comparison.png](camera_count/stress_010_comparison.png) | 单机被旧质量规则拒绝，双/三机有输出；单机无输出时不伪造圆 |
| [all_36_cases.png](camera_count/all_36_cases.png) | 全部常规24例、压力12例的统计，不只展示两个例子 |
| [汇总CSV](data/camera_count_summary.csv) | 两算法×两组×三种相机组合，包含全部分母和拒绝 |
| [逐次CSV](data/camera_count_records.csv)、[JSON](data/camera_count_records.json) | 全部216条连续误差、接受/拒绝、错误接受及耗时，不含巨大的逐点诊断 |
| [来源与散列](manifest.json) | 原结果散列、案例选择理由及导出文件散列 |

两张对比图3200×1800，统计图3080×1540，可直接插PPT。红细线为已接受的估计圆/法向，青色虚线为评价真值；点云A蓝、B绿、C金。上排网格只作显示，下排仅保存RGB前景与有效深度，没有网格补点；三列使用同一观察角。

数据来自10-23冻结的`region_equal_trust`：理想深度、准确K/T、固定相机布局。两个示例颈段约8.04/6.41 mm，分别展示“都成功”和“单机拒绝”；不能以示例代替总体统计。全部36例有后来6–10 mm工况外的颈长，同属圆截面模型族。

“成功”=算法接受且圆心≤4 mm、法向≤2°、直径误差≤2 mm；后两项为当时分析阈值，不是厂方工艺验收。**不能标成真实相机实验、当前成熟方法的相机数对照或扎带成功率。** 单机失败的直接原因是旧核心范围规则，不证明所有单机方法都不能完成任务。

原报告：[36坛随机审计](../../docs/research/neck_localization/random_geometry_audit.md)、[区域反馈对照](../../docs/research/neck_localization/support_feedback_stage1.md)；[完整216条含诊断记录](../neck_localization/results/random_geometry_audit_20261023/records.json.gz)也已纳入发布选集。

## 其余素材与最新算法

- [10-28酒厂汇报输出](../factory_briefing_20261028/output/)：6张高清图、指标说明、旧PPT；[原画面素材](../factory_briefing_20261028/assets/)包括三工位、11坛、近景、RGB/深度、点云和枪头，便于重排版。
- [最新几何汇总](../neck_localization/results/geometry_capability_summary_20261027_v2/summary.tsv)、[对应报告](../../docs/research/neck_localization/geometry_capability_20261027.md)：比早期理想相机数实验更晚，输入条件不同，不能混用分母。
- [跨工位误差](../../docs/engineering/cross_station_progress_20261028.md)、[标定六方法](../../docs/engineering/uncertain_transfer_20261029.md)：区分视觉误差、传递/执行新增误差、整带几何和未验证工艺。
- [109批次资源目录](data/result_catalog.csv)列出已上传图/表/汇总入口；[发布清单](publication_manifest.json)列具体文件和未上传类别。大量逐点输入、重放缓存仍保留在原机。

## 另一设备上的三维查看

`viewer/`含两案例场景、纹理/网格、三机RGB/深度、保存结果和仅供评价的真值，都是实际文件和相对路径，无原机软链接。

在仓库根目录、已有MuJoCo环境运行：

```bash
conda run --no-capture-output -n winejar-stereo python scripts/view_camera_count_comparison.py --point-cloud
conda run --no-capture-output -n winejar-stereo python scripts/view_camera_count_comparison.py --case stress_010 --camera-set cam_a --point-cloud
```

无该环境时按根README创建`winejar-neck-sim`并替换环境名。左侧A/A+B/三机切保存结果，不重新估计，不连接硬件。

## 重新导出

在仍有原36坛`runs/`的机器上：

```bash
conda run --no-capture-output -n winejar-stereo python scripts/export_camera_count_assets.py --output /tmp/winejar_camera_count_NEW
```

输出目录须不存在。导出器不改算法/记录，不重推理；复用原MuJoCo界面渲染器。真值只用于评价叠线及公平的共同显示视角，不反馈给估计器。
