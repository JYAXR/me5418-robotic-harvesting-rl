# ME5418 Group 18 final code and experiment evidence

[Download the verified code ZIP](ME5418_Group18_Code.zip) or
[browse the bundled source and data](ME5418_Group18_Code/).
The archive was tested after extraction in an isolated CPU environment.
After creating the environment described in the [code README](ME5418_Group18_Code/README.md),
run `python scripts/validate.py` from the code directory. The command loads
the published checkpoint and reaches the target in 46 actions with zero
damage on seed 5420000.

## Changes

- **Submission code and training.** Added a direct validation entry point,
  the published 6,000-update compact DQN, the exact dependency specification,
  the two recovered demonstration files, and a complete training and data
  regeneration driver. The regenerated files match the published SHA-256
  hashes and contain 13,502 regular and 1,494 recovery transitions.
- **Reproduction.** Two independent local retrains produced identical
  parameters and final losses. The published and retrained checkpoints
  have different parameters. They reached 81/100 and 82/100 on the previous
  evaluation bank, and 31/33 and 27/33 on familiar layouts, respectively.
  The validation entry point defaults to the published checkpoint.
- **Waypoint ablation.** Added paired results on the ten original clean
  layouts. Adding the same local waypoint rule raises H5 from 0/10 to
  6/10, with zero damage in both settings.
- **Repeated execution noise.** Added three matched perturbations at 10%
  of action step size for each of six further layouts. In the 18 paired
  noisy trials, original H5 succeeds 0/18, waypoint H5 18/18, and the
  published DQN 14/18. All methods record zero damage.
- **TD ablation.** Added three matched training-seed pairs with TD weights
  0 and 0.03. On 100 shared new layouts, mean success changes from 81.7%
  to 84.7%, with seed-level changes of +8, +8 and -7 percentage points.
  On the 33 familiar layouts, the mean change is -7.1 percentage points.
- **Documentation.** Put evaluation and training commands first, documented
  the architecture, Huber loss, gradient clipping and AI assistance, and
  verified that the submitted code comments and generated outputs are English.

The [experiment index](ME5418_Group18_Code/experiments/README.md) links to
the raw or paired evidence, protocols and reproduction commands. The
[acceptance record](acceptance.json) gives the archive hash, runtime
versions and the outcome of validation and full retraining from the ZIP.
