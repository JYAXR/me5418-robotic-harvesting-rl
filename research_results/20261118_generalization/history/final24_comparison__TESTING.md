# 核对记录

冻结协议与模型名单后、正式评估前：

`python -m pytest -q experiments/final_comparison_20261118/test_comparison.py tests/test_scene_cache.py`

结果：16 passed in 13.17s。覆盖全部24个原生模型加载器、相同环境、固定场景划分、
原034三个场景40/47/43步复现、缓存不改变完整轨迹，以及既有场景缓存测试。

评估期间检查统计逻辑：

`python -m pytest -q experiments/final_comparison_20261118/test_summary.py`

结果：3 passed in 1.92s。覆盖成功/损伤/超时计数、无成功时不虚构平均步数、
新增/丢失成功的配对计数，以及只对共同成功场景比较步数与距离。

三份测试文件合并再次执行：19 passed in 25.61s。

最终完整审计已通过，记录见`results/run01/audit.json`。
核对24模型×133场景的3192条记录、所有冻结哈希、相同初始状态、100个不同新几何布局，
并对4种代表策略分别重新加载、无缓存复跑1个旧场景与1个新场景，共8次；完整轨迹一致。
这不是把3192回合全部独立重跑一遍。
正式评估脚本不会训练、更新权重或更改模型选择。
