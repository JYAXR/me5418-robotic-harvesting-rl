# ME5418 Group 18 — Robotic Harvesting

From this directory, create and activate the CPU environment, then run:

```text
conda env create -f environment.yml
conda activate me5418-harvesting
python scripts/validate.py
```

**Validation loads the bundled, fully trained compact DQN.** Seed `5420000`
should succeed in 46 actions with zero damage. No training is required.
For the full 100-layout and 33-familiar-layout evaluation:

```text
python scripts/validate.py --seed 5420000 --episodes 100 --known --workers 6 --output runs/published-evaluation.json
```

**Train the compact DQN from the bundled teacher and data:**

```text
python scripts/reproduce_compact.py --stage train --out runs/compact
python scripts/validate.py --checkpoint runs/compact/training/update_06000.pt
```

The training command runs 6,000 updates and records a comparison with the
published tensors. To regenerate the demonstration files before training,
add `--regenerate-data --workers 6` and use a new output directory.

| Entry | Purpose |
| --- | --- |
| `models/compact_guided_generalization.pt` | Published, trained DQN used for the baseline comparisons |
| `src/me5418_harvesting/` | Environment, contact dynamics, learning utilities and planners |
| `scripts/tune_coordinate_dqn.py` | Compact-agent training loop |
| `data/` | Hashed regular and recovery demonstrations for training |
| [Experiments](experiments/README.md) | Results, protocols and commands for control comparisons |

The published DQN reached the target in **81/100** evaluation layouts and
**31/33** familiar layouts, with zero damage. The documented local
retraining run reached **82/100** and **27/33**, also with zero damage.
The two regenerated data files match the original file hashes. Local
retraining was repeatable; its weights differ from the published weights.

The policy takes 89 compact geometry features and returns one Q value for
each of eight discrete actions. Two ReLU hidden layers let joint geometry
signals such as gate direction and action clearance interact. The retained
width of 256 gives each layer room to represent combinations of these
signals while keeping the full policy to 90,888 trainable parameters for
CPU inference. This is a design choice inherited from the earlier agent.
The project did not measure an optimal width or depth. The TD term uses
Smooth L1 (Huber) loss: small
Bellman errors get a quadratic penalty and large errors get a linear
penalty. This limits the influence of unusually large TD errors, which can
arise around terminal outcomes. Clipping the combined global gradient
norm to 10 limits the gradient magnitude passed to Adam on a high-gradient batch.
These choices explain the design intent; the TD ablation varies only the TD
loss weight.

The tested runtime is Python 3.13.13 with CPU PyTorch 2.15.0, Gymnasium
1.4.0, NumPy 2.5.3, Matplotlib 3.11.2, Pillow 12.3.0 and PyYAML 6.0.3.
No separate robotics simulator is required. See [code provenance and AI
assistance](PROVENANCE.md).
