# Does looking further ahead help? A paired planning-depth evaluation

This folder evaluates a frozen compact, geometry-guided DQN and the project's receding-horizon planner. It contains the experiment, **not a new training run**. The question is whether the implemented planner matches the policy as lookahead increases, and how much online decision time it uses. Start with [RESULTS.md](RESULTS.md). The coverage matrix and episode traces are the source of truth; the larger DQN-only evaluation is not a planner benchmark.

## Protocol

- **Version:** [project snapshot e532751](https://github.com/JYAXR/me5418-robotic-harvesting-rl/tree/e532751939fc5e6745700f426b79cdf6f7ed5f9c). Checkpoint: `models/compact_guided_generalization.pt`, SHA-256 `75d7d7ca728b1c85c8ba6fa226076f85f4f72bdce3cfe4c6c0177946e5fe51e6`. Not attempt 034. No optimization, exploration, external action override, or checkpoint selection during this run.
- **Layouts:** first ten consecutive seeds 5420000–5420009 of the existing 100-layout evaluation range. No filtering by observed outcome within that prefix. This is an already examined evaluation bank and a small diagnostic subset, not a new blind sample. One deterministic rollout per method per seed: rerunning the same seed is a reproducibility check, not another independent trial.
- **Environment:** three two-lane regions in a unit square, up to 14 branches, five nodes per branch; eight cardinal moves (0.025 or 0.060), contact radius 0.035, goal radius 0.05, 200-action limit. Use the model's original clean environment: zero execution noise. Each accepted reset must contain a collision-free A* route verified in the environment. Hash the initial observation to verify pairing across methods.
- **Unmodified reward:** 10 times reduction in Euclidean distance to target, minus total normalized branch deformation, minus 0.2 times actual travel, minus 50 for damage, plus 100 for safe success. End on safe arrival, peak deformation above 1, or budget exhaustion. A contact attempt need not cause damage, nonzero deformation, or forward motion.
- **DQN:** greedy inference, including online computation of the compact geometric input and local waypoint derived from observed branch geometry. It exploits the known gate structure. It does not receive the A* route. Comparing this system with an unguided planner does not isolate the benefit of RL alone.
- **Primary planners:** H1 and H3 exhaustive; H5 and H7 beam search (keep 32 sequences per layer). Simulate the real, deterministic step model, sum the environment rewards without discount to the search horizon, choose the native planner's first action, then replan. Horizon is clipped by the remaining episode budget. No learned terminal value, waypoint bonus, search over the full 200 steps, or new tie-breaking rule. The physical reward is shared, but a finite undiscounted search score is not the DQN's learned long-term value.
- **Additional depth control:** repeat H3 with beam width 32, separately labelled H3-B32. H1 considers eight actions, so width 32 would not prune it. This *post hoc* control lets H1/H3-B32/H5/H7 be compared without changing the beam cap. Deeper beam search is not exhaustive. Duplicate search paths are not eliminated before beam pruning.
- **A*:** retain the reset verifier's executed collision-free witness for each accepted layout. This is a useful feasible-route and path-length reference, not an independent success-rate estimate or a receding-horizon competitor with matched timing.

## Metrics and measurement

Success, damage and timeout are reported separately. All completed episodes count, including failures. Per-episode peak is the maximum normalized deformation over the trajectory; travel is cumulative actual distance; contacts counts environment steps with contact, not individual branches. Return is the undiscounted sum used for evaluation. Per-method means average episodes; decision times weight individual executed actions. Do not interpret low travel on a timeout as a better route.

Decision times include observation encoding and inference for DQN or planning for the baseline, but exclude checkpoint load, environment reset/certification, actual execution of the chosen action and writing the trace. Identical deterministic transitions and decisions can be reused **within** one method and layout. The log distinguishes candidate transition requests on newly computed plans (`expansions`), physical simulator cache misses (`model_calls`), and whole-plan reuse. Zero DQN lookahead calls does not imply free inference. The post-action audit can touch the transition cache outside the decision timer. Trials ran on CPU, some concurrently; neither compute budget nor wall-clock scheduling was matched. Timings are descriptive, not a general speedup or deployment latency guarantee. Offline DQN training cost is excluded.

## Files

| Path | Purpose |
|---|---|
| `data/episodes/*.json` | Seed, method, reset fingerprint, reference witness, every executed action, reward, xy, cumulative travel/peak/contact count, decision timings and termination |
| `tables/coverage.csv`, `tables/primary_paired.csv` | Coverage per seed/method; primary comparison on common completed seeds only |
| `tables/fixed_beam_paired.csv`, `tables/all_completed.csv`, `tables/episodes.csv` | Supplement, all available outcomes, individual episodes |
| `data/dqn_133.json`, `data/astar.json`, `tables/astar_pairs.csv` | Repository evaluator output on 100 range + 33 known layouts; A* witnesses; paired path metrics |
| `audits/`, `scripts/`, `SHA256SUMS` | Check outputs, runnable source, file integrity manifest |

## Reproduction

Use the pinned repository snapshot above, not an arbitrary version of main. Create the repository environment with `conda env create -f environment.yml` and activate `me5418-harvesting`. The source environment requests Python 3.13.13; the recorded run used Linux x86-64, Python 3.12.14, torch 2.15.0+cpu, numpy 2.5.3, gymnasium 1.4.0 and PyYAML 6.0.3. Do not run Python with -O (checks use assertions). Keep these scripts and the supplied data together. Commands below assume this folder was extracted at `research_results/paired_planner_evaluation` inside the pinned checkout. Run from the **repository root**:

```sh
E=research_results/paired_planner_evaluation
(cd "$E" && sha256sum -c SHA256SUMS)
sha256sum -c "$E/audits/source-sha256.txt"
python "$E/scripts/tables.py" --output /tmp/me5418-reviewed-tables
python "$E/scripts/replay.py" 5420000
python "$E/scripts/audit_h1.py"
python "$E/scripts/audit_h3.py"
python "$E/scripts/plot_example.py" --out /tmp/me5418-first-layout.png
```

To rerun just one comparison, use a new output path. Depth 0 = DQN. Drop `--beam 32` at depth 3 to request exhaustive search; at depths 5 and 7 the default is beam 32.

```sh
python "$E/scripts/run_episode.py" --seed 5420000 --depth 3 --beam 32 --out /tmp/me5418-5420000-h3b32.json
```

To regenerate the full matrix, both references, summary tables and uncached action replays in a **new** directory:

```sh
python "$E/scripts/rerun.py" --repo . --out /tmp/me5418-new-run --workers 1
```

Exact H3 can be slow on scenes where geometry keeps changing (~584 candidate edges per full nonterminal search before reuse); a wall-time interruption is not an episode failure. Expect physical outcomes, not raw timing fields or file digests of newly timed runs, to reproduce. For a clean latency study rerun serially on an otherwise idle machine, repeat trials, and report hardware and timing distributions separately.

## What was verified

1. Initial observation hash and A* witness agree for every method on the same seed.
2. Each executed planner action's simulated next physical state, step reward and terminal flag agree with the actual environment. For the clean environment, the transition cache stores robot position, branch shapes and contact flags; each decision also keys on effective horizon. The cache is not valid for noisy rollouts.
3. Separately replay stored actions in a normal uncached environment, checking positions, rewards, travel, deformation, contact counts and termination step by step. This verifies the trajectory, **not** independently recomputing every candidate action.
4. On seed 5420000, an independent no-cache H1 calculation checks every decision; exhaustive brute-force H3 checks all eight root-action scores at two distinct states. Separate end-to-end environment evaluation cross-checks all ten DQN episode outcomes, returns and path metrics.
5. No subgroup or noise results are inferred from these runs. A stronger follow-up would pair both methods on a fresh larger bank, test execution noise explicitly, and compare a planner supplied with the same geometric guidance or a terminal value.

AI use: Codex assisted with experiment scripting, checks, execution and documentation. Team members should review this package before citing or submitting it.
