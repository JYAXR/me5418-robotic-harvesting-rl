# Verified inference policy

`step6_repair_027.pt` is the final TD-10000 inference checkpoint from
[attempt 027](../docs/experiments/027_step6_supervision/RESULTS.md). It includes
network weights, encoder version and the environment configuration. It can be
loaded without the local historical experiment outputs.

From the repository root in the configured project environment:

```bash
python scripts/evaluate_dqn.py --checkpoint models/step6_repair_027.pt --episodes 3 --seed 10000
```

This evaluates known training scenes 10000, 10001 and 10002. The verified
results are success in 35, 41 and 41 steps, respectively. No inference-time
correction or scripted prefix is used by this command. Unseen-scene performance
has not been measured for this candidate.

`step6_repair_027.json` records the SHA-256, source update, independent normal
evaluation and scope. The checkpoint is for inference, not resuming Adam state.
Full training-state snapshots remain in the local experiment output. Running
the generic trainer uses its existing defaults; it does not automatically
reproduce the staged calibration and per-observation supervision of this repair.

SHA-256: `06733fccb5422d10ad7246e5368bbd5ed9cc3367f0b413847544ec26a96b0c88`.
