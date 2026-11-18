"""Predeclared existing endpoints, not selected on the new comparison bank."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
NEW = list(range(2026111800, 2026111900))
OLD = [10000, 10001, 10002] + list(range(290000, 290030))

# id, family, human-readable change, repository-relative inference checkpoint
MODELS = [
    ('original034','dqn','最初项目034模型','models/supervision_034_targeted.pt'),
    ('pilot_control','dqn','早期短期在线续训，无新增惩罚','experiments/online_pilot/results/pilot01/online_control/final.pt'),
    ('pilot_penalty','dqn','早期短期续训，加每步/受阻惩罚','experiments/online_pilot/results/pilot01/online_penalties/final.pt'),
    ('lr_1e4','dqn','原学习率0.0001续训；也是恢复实验的对照','experiments/parameter_sweep/results/run01/lr_1e-04/final.pt'),
    ('lr_3e5','dqn','降低学习率至0.00003续训','experiments/parameter_sweep/results/run01/lr_3e-05/final.pt'),
    ('lr_1e5','dqn','降低学习率至0.00001续训','experiments/parameter_sweep/results/run01/lr_1e-05/final.pt'),
    ('stagnation','dqn','原模型续训，加物理停滞惩罚','experiments/recovery_comparison/results/run01/stagnation_cost/final.pt'),
    ('recovery','dqn','原模型续训，补充成功脱困经验','experiments/recovery_comparison/results/run01/recovery_data/final.pt'),
    ('fresh_abs','dqn','从零重训，绝对位置，无停滞惩罚（匹配V2）','experiments/fresh_penalty_coordinates/results/run02/absolute_no_penalty/final.pt'),
    ('fresh_abs_penalty','dqn','从零重训，绝对位置，加停滞惩罚（匹配V2）','experiments/fresh_penalty_coordinates/results/run02/absolute_penalty/final.pt'),
    ('fresh_rel','dqn','从零重训，相对位置，无停滞惩罚（匹配V2）','experiments/fresh_penalty_coordinates/results/run02/relative_no_penalty/final.pt'),
    ('fresh_rel_penalty','dqn','从零重训，相对位置，加停滞惩罚（匹配V2）','experiments/fresh_penalty_coordinates/results/run02/relative_penalty/final.pt'),
    ('rebuilt_prepared','dqn','本次重建准备模型，相对位置，原33场景','experiments/retention_rebuild_20261118/results/run01/preparation/prepared.pt'),
    ('rebuilt_joint','dqn','增加67场景并复习旧场景，联合训练，已选6000次','experiments/retention_rebuild_20261118/results/run01/full_joint/checkpoint_006000/model.pt'),
    ('rebuilt_staged','dqn','增加67场景并复习旧场景，分阶段组，已选3000次','experiments/retention_rebuild_20261118/results/run01/full_staged/checkpoint_003000/model.pt'),
    ('bc33','imitation','纯模仿，绝对位置，33场景，4000次','experiments/imitation_routes/results/run01/ce33/model.pt'),
    ('bc97','imitation','纯模仿，绝对位置，扩至97场景，4000次','experiments/imitation_routes/results/run01/ce_expanded/model.pt'),
    ('bc_rel4k','imitation','纯模仿，相对位置加镜像，97场景，4000次','experiments/imitation_routes/results/run01/ce_relative_mirror/model.pt'),
    ('bc_rel20k','imitation','上述模仿模型继续训练至20000次','experiments/imitation_routes/results/relative_extended/model.pt'),
    ('bc_repeat_A','imitation','相对位置加镜像，独立训练A，20000次','experiments/generalization_repeats/results/run01/relative_mirror_17/model.pt'),
    ('bc_repeat_B','imitation','相对位置加镜像，独立训练B，20000次','experiments/generalization_repeats/results/run01/relative_mirror_18/model.pt'),
    ('bc_repeat_C','imitation','相对位置加镜像，独立训练C，20000次','experiments/generalization_repeats/results/run01/relative_mirror_19/model.pt'),
    ('bc_no_mirror','imitation','匹配模仿对照A：相对位置，不镜像','experiments/generalization_repeats/results/run01/relative_only_17/model.pt'),
    ('bc_no_relative','imitation','匹配模仿对照A：绝对位置，加镜像','experiments/generalization_repeats/results/run01/absolute_mirror_17/model.pt'),
]

EXCLUDED = {
    'recovery_comparison/control': 'Equivalent network to lr_1e4; verify equivalence before reusing its row.',
    'online_pilot/best.pt': 'Both selected the unchanged original baseline; show trained final endpoints instead and mark them as previously rejected.',
    'fresh_coordinate_dqn original two endpoints': 'Earlier separate run of the same absolute/relative recipe; use the four-arm, shared-pretraining V2 for matched penalty/coordinate comparisons. Not an exhaustive list of every training attempt.',
    'fresh_penalty_coordinates/run01': 'Aborted before full trial; superseded by matched V2.',
    'reward_diagnosis': 'Read-only diagnostic, no newly trained policy.',
    'lost restored_stages/generalization_routes/retention_followup': 'Original files not present; rebuilt models are explicitly labeled, not claimed identical.',
    'intermediate training checkpoints': 'No re-selection using the new test results; retain previously selected checkpoints or fixed final endpoints.',
}
