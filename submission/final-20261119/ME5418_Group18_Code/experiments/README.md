# ME5418 Group 18 experiment evidence

Run the commands below from the code archive root after creating the
environment in the [main README](../README.md).

| Experiment | Evidence | Scope |
| --- | --- | --- |
| Planner depth | [Original results](../research_results/paired_planner_evaluation/RESULTS.md), [rollout tables](../research_results/paired_planner_evaluation/tables/) | Ten matched clean layouts, DQN and five planner settings including the beam-width control |
| Waypoint ablation | [Summary](waypoint-comparison/results/summary.csv), [paired results](waypoint-comparison/results/paired.csv) | Same ten layouts. H5 changes from 0/10 to 6/10 when waypoint progress is added to the search score |
| Repeated noise | [Summary](waypoint-comparison/runs/repeated-noise-v1/summary.csv), [paired results](waypoint-comparison/runs/repeated-noise-v1/paired_noise.csv), [protocol](waypoint-comparison/runs/repeated-noise-v1/protocol.json) | Six new layouts and three matched 10% noise draws per layout. H5 0/18, waypoint H5 18/18, published DQN 14/18, zero damage in all arms |
| TD ablation | [Summary](td-ablation/results/summary.csv), [paired results](td-ablation/results/paired.csv), [protocol](td-ablation/results/protocol.json) | Three paired training seeds, 100 shared new layouts and 33 familiar layouts. Average TD change: +3.0 and −7.1 percentage points respectively |
| Training reproduction | [Data checks](training-reproduction/data-comparison.json), [weight comparison](training-reproduction/weight-comparison.json), [evaluation comparison](training-reproduction/evaluation-comparison.json) | The two new data files exactly match the released hashes. A repeatable local retrain produced different tensors and outcomes |

```text
python research_results/paired_planner_evaluation/scripts/replay.py 5420000
python experiments/waypoint-comparison/compare_waypoints.py --out runs/waypoint-new --workers 4 --memoize-branches
python experiments/waypoint-comparison/repeat_noise.py --out runs/noise-new --workers 6
python experiments/td-ablation/compare_td.py --out runs/td-new --train-workers 3 --eval-workers 6
```

The noise run reuses the initial clean and first noisy trials under
`experiments/waypoint-comparison/runs/fresh-noise-v1` before adding two
repeats. The TD experiment trains its own seed-matched weights. The six
noisy layouts are the units of layout variation; three perturbations per
layout measure repeat variability at one noise level.
