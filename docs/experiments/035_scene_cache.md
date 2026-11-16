# Attempt 035: certified reset caching and segmented evaluation timing

Freeze models/supervision_034_targeted.pt (SHA-256
202fd8d4f6929da61e991962c6f667dd80a9cbcc330b1b744434230f140f8b8b).
No training, optimizer updates, action overrides or hyperparameter changes.

Add opt-in bounded process-local caching of accepted certified reset states.
Key entries by explicit integer seed, resolved simulator/generation/reward
configuration and environment class. Restore a fresh copy of the accepted scene,
post-generation RNG and original reset information; initialize episode counters.
Preserve rejected-candidate history. Cache only successful certificates; bypass
unseeded resets, options, subclasses and uncertified environments. Code changes
require a fresh process/cache; runtime monkey-patching requires clear().

Keep timing/cache telemetry outside observations, rewards and reset info.
Original reference_route_checks on a hit are historical certificates, not newly
executed work. last_reset_profile must report zero new certification calls and
actions on hits. Record reset generation/certification/total, policy, physics and
episode time. Model load, detailed audit and disk export are timed separately.

Predeclare three passes over seeds 10000, 10001, 10002 for both arms: 18 complete
episodes. Pass 0 runs uncached then cached (cold); pass 1 cached then uncached;
pass 2 uncached then cached. Each seed uses a fresh environment/model load;
the cached arm shares one capacity-3 cache. Compare complete transition records,
all eight Q values, raw observations, RNG states, reset info and episode metrics
against the uncached counterpart and historical attempt-034 final traces.
Use the unchanged normal evaluate loop and one Torch thread. No seed replacement
or extra run selected according to speed/outcome.

Report cold cache cost, warm-hit speedup using matched passes 1/2, and the total
three-pass workload including cache population. Use measured wall times, not
simulation-step counts as a timing proxy. This measures evaluation only, not
overall training acceleration. Timing includes identical transition-audit
instrumentation in both arms. Do not claim speedup for previously unseen seeds.

Unit tests cover RNG parity with execution noise, mutation isolation, configuration
changes, failed/rejected candidates, LRU eviction and cache bypass. Run relevant
environment/training regression tests. Commit/push implementation/protocol before
benchmarking, then archive every outcome and measurement and sync results.

```bash
python -m pytest tests/test_scene_cache.py tests/test_scene_feasibility.py tests/test_environment_contract.py tests/test_training.py -q
python docs/experiments/035_scene_cache/run_benchmark.py
```
