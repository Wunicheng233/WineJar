# xArm SDK Control

本目录维护实机控制、标定、取放规划及离线测试。

- [当前工位状态](../docs/CURRENT_STATE.md)：硬件、已验证结果与配置冲突。
- [运行手册](../docs/RUNBOOK.md)：安装、只读诊断、采图、现场恢复。
- [代码结构](../docs/ARCHITECTURE.md)：模块职责与测试策略。
- [恢复实验代码](experiments/recommissioning_20261014/README.md)：单物料验证器和未执行的完整取放草稿。
- [数据管理](../docs/DATA_MANAGEMENT.md)：数据保留与迁移。

`scripts/21_complete_loading.py`是旧工位完整任务实现，默认值不能直接用于搬动底座后的新工位。
最近验证的名义TCP为116 mm；纸/陶瓷额外下压0、叠放竹壳8 mm。
这些值只描述最近有限实验，运行前核对实际控制器和当前现场。

通用逻辑放顶层模块，CLI放scripts，工位专用代码放experiments。
需要维护的唯一源码不放Git忽略的data中。配置/标定要保留来源和验收状态。

```bash
conda run -n winejar-xarm-sdk python -m pytest -q xarm_sdk_control/tests
```

历史模块详解、旧任务参数和三相机命令保存在
[旧控制层说明](../docs/archive/legacy/xarm_sdk_control_README_202609.md)，仅供回放和开发参考。
