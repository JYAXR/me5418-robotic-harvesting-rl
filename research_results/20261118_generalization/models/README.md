# 7份推理模型

这些文件来自之前已上传的成果压缩包，逐字节复制。不是新训练结果，也不会替换仓库根目录 models 中同学的模型。

- prepared.pt：相对位置准备模型（不是同学034）。
- retention_joint.pt、retention_staged.pt：早期联合/分阶段保持能力候选。staged选中的是监督阶段检查点，不应把成绩归功于后续TD。
- control.pt、stagnation.pt、damage.pt、noise.pt：最新等预算四组续训终点。

来源、哈希及加载参数见上层 models.json。使用上层 evaluate_model.py 评估。仅用于推理，没有完整续训所需的优化器与经验回放状态。
