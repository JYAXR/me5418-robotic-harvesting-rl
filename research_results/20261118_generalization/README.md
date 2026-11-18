# 泛化实验：代码、训练记录、测试数据与模型

本目录对标主项目 `docs/experiments/034_scene10002_supervision/` 的证据类型，展开保存实际文件，不只提供结论。仅添加到独立成果分支，不修改主项目默认代码或模型。

## 实际文件

- [models/](models/)：7份推理权重，分别是准备模型、joint/staged保持能力候选，以及原奖励、停滞惩罚、损伤惩罚、噪声四组续训模型。来源和哈希见 [models.json](models.json)。
- [history/](history/)：89份历史代码、训练更新日志、实验协议、逐场景CSV、结果与审计记录，涉及五组研究；包含18个Python源码文件和6份CSV。
- [evaluate_model.py](evaluate_model.py)、[coordinate_policy.py](coordinate_policy.py)：归档模型的评估入口和相对位置模型加载代码。
- [FILE_INDEX.json](FILE_INDEX.json)：100份原始文件与压缩包内路径的对应关系、大小及SHA256。复制文件内容未修改。
- [verify_materials.py](verify_materials.py)：核验这100份展开文件；这是本次新增工具，不是旧实验代码。

## 优先查看

| 内容 | 直接打开 |
|---|---|
| 之前24模型的逐场景成绩（3192行） | [per_episode.csv](history/final24_comparison__results__run01__per_episode.csv) |
| 最新奖励/噪声四组训练代码 | [study.py](history/four_arm_rl__study.py) |
| 最新一轮逐场景成绩（含失败） | [per_episode.csv](history/four_arm_rl__results__evaluation__per_episode.csv) |
| 停滞惩罚训练过程的更新日志 | [updates.json](history/four_arm_rl__results__stagnation__updates.json) |
| 最新四组对照结果 | [comparison.csv](history/four_arm_rl__comparison.csv) |
| 监督准备、校准与保持能力实验代码 | [run.py](history/retention_rebuild__run.py) |
| 准备模型第二批100场景复测 | [per_episode.csv](history/prepared_confirmation__results__per_episode.csv) |
| 同学009模型的同场景对照 | [paired_comparison.csv](history/teammate009_comparison__results__paired_comparison.csv) |

## 为什么history文件名很长

`history/` 是便于网页浏览的原始快照：用双下划线 `__` 表示原路径层级，避免同名 `run.py`、`summary.json` 混淆。文件内容保持原样，但这些历史脚本并不是直接运行入口，原来的相对导入、绝对路径与部分中间依赖没有迁移。原始目录结构和更多逐步轨迹仍在仓库根目录 `ME5418_Results_20261118.zip` 中。请勿直接运行history中的训练/打包脚本。

## 运行已归档模型

在仓库根目录、安装好项目依赖后：

```bash
python research_results/20261118_generalization/verify_materials.py
python research_results/20261118_generalization/evaluate_model.py --model prepared --seed 10000 --episodes 3
```

评估入口和模型直接保留归档中的字节，不训练、不覆盖权重。归档时已做过7份模型加载与单场景验证，原记录见 `ARCHIVE_VALIDATION.json`。本次只是上传文件，并未重跑全部实验；完整性检查不等于重新验证训练效果。

## 与同学034归档的差别

同学034还保存了包含Adam等信息的完整续训检查点；本目录的7份权重仅用于推理，不包含完整优化器、经验回放及随机状态，不能等价地接着原训练继续跑。部分更早的候选模型和中间文件未保留，不能声称全部历史实验可从这里一键重现。不会用新训练结果替代遗失记录。

结果解释：准备模型旧场景33/33，在三批不同100陌生场景分别25/100、24/100、22/100。最新停滞惩罚候选陌生31/100，但旧场景降为26/33，尚未同时解决泛化和遗忘。BC历史成绩不可当作DQN成绩。详见仓库根目录 `GENERALIZATION_RESULTS_20261118.md`。
