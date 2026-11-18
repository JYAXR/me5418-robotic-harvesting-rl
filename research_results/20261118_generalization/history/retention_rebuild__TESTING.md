# 检查记录

启动前执行：

```bash
python -m pytest -q experiments/retention_rebuild_20261118/test_rebuild.py tests/test_scene_cache.py
```

结果：18 passed in 3.73s（本轮6项，原场景缓存12项）。
训练期间再次执行同一命令：18 passed in 3.96s。
覆盖阶段边界、旧新权重、原定向动作权重、Adam切换、目标网络行为及保存后续跑的一致性。

训练/测试完成后另执行：

```bash
python experiments/retention_rebuild_20261118/audit.py
```

最终已执行上述审计，结果为`AUDIT PASSED`，详见`results/run01/audit.json`。
核对了52000次准备更新、12000次续训更新、195次最终运行及两个独立重加载场景。
两次重加载的逐步轨迹与原记录完全一致；并未再次独立运行全部训练。
模型与旧记录是否逐字节一致，以本次计算的SHA256与历史记录对照为准；不靠成功计数相同来判断。
本次三个文件均未匹配历史SHA256，审计通过不代表精确恢复了旧模型。
