# 统一对比包使用说明

这是独立评估副本，不是整个GitHub仓库。请解压到新文件夹，不要覆盖你正在使用的原项目。
原代码与模型在本轮没有被修改；包中带有其冻结副本，是为了独立复查结果。

## 先看哪里

- `experiments/final_comparison_20261118/SIMPLE_RESULTS.md`：简明结论。
- 同目录`RESULTS.md`：全部24个版本的统一对比。
- `results/run01/comparison.csv`：可用Excel打开的简表。
- `results/run01/per_episode.csv`：3192次运行的逐场景结果。
- `results/run01/episodes/`：原33场景和新100场景的全部轨迹，包括失败。
- `results/run01/protocol.json`：测试前冻结的名单、场景、设置和模型哈希。
- `results/run01/audit.json`：最终核对结果与范围。

24个模型分别跑同样133个场景，不是每个模型挑不同的考题。纯模仿与DQN结果分开列。
同一批100个测试场景用在多个模型上，不能将它们当成2400个不同测试场景。
100是本次计算预算，不是课程规定；统一测试没有增加训练或让规划器替模型决定动作。

## 运行与核对

在解压目录根部，使用已安装项目依赖的Python环境：

```bash
python -m pip install --no-deps -e .
python -m pytest -q experiments/final_comparison_20261118/test_comparison.py experiments/final_comparison_20261118/test_summary.py tests/test_scene_cache.py
python experiments/final_comparison_20261118/audit_and_summarize.py
```

没有依赖时，从解压根目录使用本轮独立环境文件创建新环境，勿覆盖原环境：

```bash
conda env create -n harvesting-comparison -f experiments/final_comparison_20261118/environment-evaluation.yml
conda activate harvesting-comparison
```

本轮实际运行Python 3.12.14；原项目根目录的历史`environment.yml`声明3.13.13，保持未修改，
不作为本次环境复现的首选。独立环境文件根据实际已安装版本记录，未在另一台机器重新安装验证。

`audit_and_summarize.py`核对所有已保存结果，并实际复跑8个代表回合，不是把3192回合重跑一次。
`run.py --phase run --workers 8`默认逐场景续跑，遇到已完成且校验一致的场景会跳过。
模型使用各自原生加载器，不能把纯模仿模型或相对坐标模型直接交给普通DQN加载器。

所有用于本次评估的模型都已包含；不保证包含恢复各历史实验训练所需的完整状态和训练数据。
此包用于复查统一测试，不是复现所有历史训练过程。更完整的最近一次训练记录另见之前的重跑完整包。
