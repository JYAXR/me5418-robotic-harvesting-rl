# 先读这里：恢复后的模型与实验记录

本次是对丢失实验的重建与重跑，不是新的优化方案。测试场景编号沿用之前的版本，
因此不能将相同场景上的复现当作新的独立泛化证据。

## 两种压缩包

- `retention_rebuilt_models_20261118.zip`：较小，包含三个选定模型、推理代码、环境源码、实际测试轨迹与结果。适合先下载保存，或交给队友运行模型。**不包含完整训练数据和优化器检查点，不能用于恢复训练。**
- `retention_rebuilt_full_20261118.zip`：包含本轮全部训练脚本、训练数据、检查点、验证记录、测试记录和审计结果，可继续训练或检查复现过程。它不是整个GitHub项目的副本；无关旧实验没有重复打包。

两个包有重叠，不必同时解压到同一目录。建议各自解压到新文件夹，避免覆盖原项目。
压缩包根目录的`REBUILD_MANIFEST.json`列出了包含文件的SHA256。

## 环境与推理

在解压后的根目录执行。若已有项目依赖，只需将当前解压副本安装为可编辑包：

```bash
python -m pip install --no-deps -e .
```

若没有依赖，可按照包内`environment.yml`创建独立环境：

```bash
conda env create -n retention-rebuild -f environment.yml
conda activate retention-rebuild
```

本次验证组选中的是分阶段组3000次更新的检查点，而不是该组最后的6000次检查点。
下面是这个选中模型的单场景运行示例，输出目录必须尚不存在。
这只是加载/运行示例，不保证该场景一定成功：

```bash
python experiments/fresh_coordinate_dqn/evaluate.py \
  --checkpoint experiments/retention_rebuild_20261118/results/run01/full_staged/checkpoint_003000/model.pt \
  --seed 1182000 --episodes 1 --output rerun_staged_example
```

其他模型的准确路径见
`experiments/retention_rebuild_20261118/results/run01/selection_lock.json`中的`models`。
把命令中的`--checkpoint`换成相应`path`即可；每次使用不同的`--output`。
必须使用上述相对坐标加载器，不能直接用原项目普通DQN加载器代替。

## 如何看结果

先阅读`experiments/retention_rebuild_20261118/RESULTS.md`。
`results/run01/summary.json`保留每个场景的成功、损伤、步数和回报；
`results/run01/evaluation/`保留逐步动作轨迹。
`audit.json`记录检查范围，`historical_hash_comparison.json`记录模型是否与历史哈希一致。

## 仅完整包：继续或重新运行

按实验目录`README.md`中的顺序执行三个阶段。默认输出已有完整结果时，脚本核验后跳过已完成部分；
它不会为了展示进度而重复训练。要另开一次实验，请在三个命令中都添加同一个新路径，
例如`--output experiments/retention_rebuild_20261118/results/run02`。
现有`audit.py`固定检查本次`run01`，不应将其通过结果当成`run02`的检查结果。

原`src/`、环境配置和原默认模型均未修改；本轮修改仅在新增实验目录中。
没有推送或覆盖GitHub上的项目。
