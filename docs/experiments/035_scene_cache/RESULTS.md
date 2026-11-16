# Attempt 035: exact trajectories with faster repeated evaluation

## Measured result

The opt-in certified scene cache preserves all tested behavior. Eighteen complete
episodes form nine paired comparisons across seeds 10000/10001/10002 and three
passes. Every paired raw observation, successor, eight-action Q vector, chosen
action, reward, end flag, info dictionary and environment RNG state matches
exactly. Reset observations, accepted geometry, RNG seed/state and original reset
information also match. All traces and metrics reproduce the final targeted
attempt-034 historical trajectories, in 40/47/43 steps with success.

| Matched work | Uncached time | Cached time | Speedup |
| --- | ---: | ---: | ---: |
| First pass, three episodes (cache population included) | 7.7654 s | 7.7578 s | Essentially equal |
| Passes 1/2, six repeated episodes | 16.7873 s | 5.9164 s | 2.8374x |
| All three passes, nine episodes per arm | 24.5527 s | 13.6742 s | 1.7955x |

Warm repeated evaluation uses approximately 64.8% less episode time; the complete
three-pass workload uses 44.3% less, including cache population. Individual warm
paired speedups range from 2.5060x to 3.1122x. Three seeds and two warm passes are
a small local benchmark, not a universal performance guarantee. New seeds still
require generation/certification; caching does not accelerate first-time validation.

## Where the time goes

| Segment, nine episodes per arm | Uncached | Cached, including first pass |
| --- | ---: | ---: |
| Reset total | 15.7153 s | 5.0761 s |
| Scene generation within reset | 0.1029 s | 0.0312 s |
| Route certification within reset | 15.6100 s | 5.0399 s |
| Policy decision | 0.2404 s | 0.2284 s |
| Environment step / physics | 8.5940 s | 8.3669 s |

Reset consumes approximately 64% of the uncached episode total, primarily route
certification. On the six warm episodes, reset time falls from 10.7962 s to
0.003478 s total. Physics then accounts for about 97% of warm episode time.
The small physics/policy timing differences do not represent algorithm changes;
both arms perform identical work in those stages.

These are timed normal evaluation loops with identical transition-audit hooks
in both arms. Audit instrumentation contributes about 0.1594/0.1534 s across the
uncached/cached episodes and is included. Checkpoint loading and trace export
are measured separately and excluded from episode speedups. Initial framework
startup makes total model-load time asymmetric (2.6016 s uncached versus 0.0367 s
cached); it is not attributed to caching. Subsequent historical/pair comparisons
are also outside the episode timer. The complete benchmark process takes about
42.56 s. This is not a direct comparison with attempt 034's 19-minute training
run and does not establish end-to-end training acceleration.

## Implementation and invariants

- `CertifiedSceneCache` is a bounded, process-local LRU cache, disabled by default.
- Keys include explicit integer seed, resolved generation/physics/reward settings,
  workspace, action directions, observation size and exact environment class.
- Entries contain the successfully accepted scene, post-generation RNG and seed,
  and original reset info, including rejected-candidate certificate history.
- Every restoration deep-copies cached state and reinitializes episode counters.
  Branch deformation, trajectories and returned info cannot contaminate the cache.
- Unseeded resets, non-None options, subclasses and uncertified environments bypass
  the cache. Failed certifications are not cached. Configuration changes miss.
- The cache is not serialized and does not cross processes/software versions.
  Use a cache per worker and clear it after runtime code monkey-patching.
- `last_reset_profile` holds cache status, generation/certification/reset duration
  and counts of **actual** certification work. Timing/cache data never enters
  observations, rewards or returned reset info.
- On hits, `reference_route_checks` remains the original certificate history;
  it must not be counted as newly executed simulation. Actual certification
  calls/actions are zero on each of the six hits.
- `evaluate(..., timings=...)` adds separate reset/policy/physics/episode telemetry
  without changing its result schema. Normal untimed calls retain their interface.

The original model hash remains
`202fd8d4f6929da61e991962c6f667dd80a9cbcc330b1b744434230f140f8b8b`.
Every episode preserves the frozen model, target, optimizer and agent/Torch RNG.
Training settings, environment parameters, action selection and seeds are unchanged.
There are no benchmark optimizer updates or forced actions.

## Protocol, checks and cost

The [protocol](../035_scene_cache.md) and implementation were committed/pushed as
`c8012e8` before benchmarking. There are three passes; pass 0 runs uncached then
cold cached, pass 1 cached then uncached, pass 2 uncached then cached. The cached
arm shares one capacity-3 cache. Every episode loads a fresh model/environment.
The schedule and seeds were not changed after observing times.

All 49 relevant tests and Ruff passed before the benchmark. Coverage includes
nonzero execution-noise RNG parity, state mutation isolation, nested configuration
changes, rejected candidates, total certification failure, LRU eviction, cache
sharing/bypass and timing/result separation, plus environment/training regressions.
An initial lint import-order issue was corrected; no benchmark rerun was needed.
Test execution costs are separate from the frozen benchmark.

The single benchmark completed without errors from 2026-11-16 07:29:57.264 UTC
to 07:30:39.828 UTC (15:29--15:30 Asia/Shanghai). It executes 18 policy episodes,
780 episode actions and 468 actual certification actions, totaling 1248 simulation
actions. The uncached arm performs nine certifications (351 actions); the cached
arm performs three (117 actions), with six hits. All 18 episodes succeed.

## Usage

Reuse one cache across repeated evaluations or checkpoint environments:

```python
from me5418_harvesting.envs.scene_cache import CertifiedSceneCache
from me5418_harvesting.agents.training import evaluate

cache = CertifiedSceneCache(capacity=128)
env.scene_cache = cache
env.profile_reset = True
timings = []
first = evaluate(agent, env, episodes=3, seed=10000, timings=timings)
repeat = evaluate(agent, env, episodes=3, seed=10000, timings=timings)
assert first == repeat
```

Alternatively construct `HarvestingEnv(config, scene_cache=cache, profile_reset=True)`.
For ordinary CLI timing:

```bash
python scripts/evaluate_dqn.py --checkpoint models/supervision_034_targeted.pt --episodes 3 --seed 10000 --timings results/evaluation_timing.json
```

The CLI also accepts `--scene-cache-capacity 128`. A single CLI invocation uses
distinct seeds, so it normally only populates entries; separate invocations do
not share an in-memory cache. Actual reuse requires repeated seeds in the same
process. Historical experiment helpers that assume fresh certification on every
reset should remain uncached; use the normal evaluator and actual-work telemetry
for cached runs. Historical source-hash-pinned experiments should be reproduced
at their original Git commits.

## Evidence

- [Segmented measurements and metrics](evidence/episodes.json)
- [Summary](evidence/summary.json), [nine exact paired audits](evidence/pair_verification.json)
- [Source/model provenance](evidence/provenance.json), [raw archive manifest](evidence/archive_manifest.json)
- [Artifact verification](artifact_verification.json): 23 raw files (10814164 bytes)
  match the run; 34 source/input hashes are unchanged and 24 staged archive files
  match local bytes exactly.
- Each episode's complete trace is retained alongside these files.

The measured benefit is repeated fixed-scene evaluation. No training parameter,
network operation, simulator dynamics or evaluation budget was reduced. Further
optimization should profile physics before changing it, with the same trajectory
and numerical-consistency checks.
