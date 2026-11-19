"""Write the Markdown report from validated CSVs, without specifying outcomes by hand."""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--folder", type=Path, default=Path(__file__).resolve().parents[1])
folder = p.parse_args().folder
def rows(f):
    with f.open(newline="") as stream:
        return list(csv.DictReader(stream))
def table(name):
    return rows(folder/"tables"/(name+".csv"))
def dig(name):
    return json.loads((folder/"data"/name).read_text())
allrows=table("all_completed")
eps=table("episodes")
by={(int(r["seed"]),r["method"]):r for r in eps}
old=rows(folder/"audits/earlier_pilot.csv")
for r in old:
    new=by[int(r["seed"]),r["method"]]
    for k in ("steps","success","damage","timeout","travel","peak","reward","contacts"):
        assert r[k]==new[k],(r["seed"],r["method"],k,r[k],new[k])
primary=table("primary_paired")
fixed=table("fixed_beam_paired")
ref=table("astar_pairs")
dqn=dig("dqn_133.json")["summary"]["compact"]
contact=sum(int(r["dqn_contact_steps"])>0 for r in ref)
shorter=sum(float(r["dqn_minus_astar"])<0 for r in ref)
equal=sum(float(r["dqn_minus_astar"])==0 for r in ref)
def results(rs):
    header = ["| Method | Layouts | Success | Damage | Timeout | Steps | Travel | Peak deform. | Return | CPU ms/action | Wall ms/action |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in rs:
        label={"H1":"H1 exact","H3":"H3 exact","H5":"H5 beam 32","H7":"H7 beam 32","H3-B32":"H3 beam 32"}.get(v["method"],v["method"])
        header.append("| "+" | ".join([label]+[v[k] for k in ("episodes","success","damage","timeout")]+[f"{float(v[k]):.{n}f}" for k,n in (("mean_steps",1),("mean_travel",3),("mean_peak",3),("mean_reward",2),("per_action_cpu_ms",2),("per_action_wall_ms",2))])+" |")
    return "\n".join(header)
def avg(k):
    return st.mean(float(r[k]) for r in ref)
def stats(key):
    d=dqn[key]
    return f'{d["success"]}/{d["episodes"]} successful, {d["damage"]} damage, {d["timeout"]} timeouts'
paired_n=int(primary[0]["episodes"])
text=f"""# Results: matched clean layouts

## Main comparison

The table below uses identical layouts for every row, not a separate denominator for each method. {paired_n}/10 planned layouts have all five primary methods available. The archive contains {len(eps)}/60 complete method–layout rollouts across the main experiment and supplement. There are {sum(int(r['steps']) for r in eps):,} executed actions in these traces. Each rollout was evaluated once for comparison; validation repeats are not extra trials.

{results(primary)}

Steps, travel, peak and return are episode means, including timeouts. Peak is normalized to the damage limit. CPU/wall time is per executed decision, including repeated states; neither reset time nor offline learning time is included.

Exhaustive H3 on seed 5420006 is especially expensive: {int(by[5420006,"H3"]["model_calls"]):,} physical transition cache misses, {float(by[5420006,"H3"]["wall_ms"]):.1f} ms per decision averaged over that episode. This is one reason not to read the time column as a monotonic depth curve. Primary H5/H7 prune the search, and repeated identical decisions are cheap.

## Does beam width change the interpretation?

The original H3 is exhaustive; H5/H7 are not. This additional analysis holds the effective cap at 32 across depths. At H1 all eight alternatives fit. H3 beam 32 is an extra run, not a relabelling of the exhaustive H3. DQN, H1, H5 and H7 rows are reused, not new independent observations.

{results(fixed)}

On these ten layouts raising the beam planner's depth to seven did not recover a successful approach. This does **not** exclude success at another depth, beam width, terminal heuristic or compute budget. The engineered waypoint used by the DQN is not supplied to the planner; the outcome is a system-level comparison, not proof that learning beats search under identical guidance.

## Independent context: the 100-layout DQN evaluation

The repository's own evaluator gave **{stats('range')}** on seeds 5420000–5420099, and **{stats('known')}** on the 33 known layouts. Its ten overlapping episode outcomes and physical metrics match the detailed DQN traces exactly. The ten-layout prefix should not replace the 100-layout number. There are no matched planner results for the other 90 layouts.

## What comparison with a feasible global route shows

On the ten accepted scenes the collision-free A* witness reached the goal each time, with zero contact. DQN mean travel was {avg("dqn_travel"):.3f} versus A* {avg("astar_travel"):.3f}; mean steps were {avg("dqn_steps"):.1f} versus {avg("astar_steps"):.1f}. The DQN made contact on {contact}/10 layouts. Its path was strictly shorter on {shorter}/10, tied on {equal}/10 and longer on {10-shorter-equal}/10. Distances are in normalized workspace units. These data show **no average path-length advantage from the learned policy** in this subset. Without a no-contact ablation, an individual shorter path is not evidence that contact caused the improvement. A* success is conditional on scene admission, not an unbiased global-planner success rate.

## Why might a deeper search still time out?

![Actual DQN and H7 trajectories on the first layout](example.png)

The figure uses recorded end-effector positions over the original resting branch shapes. Repeated actions at the same position are drawn on top of one another; they still count toward the 200-step limit.

On seed 5420000 near the first gate, at (0.23803, 0.50155), independently enumerating all depth-three continuations yields a best score of zero for both rightward first actions. The best score starting with either upward move is about −0.000000238; the downward alternatives score lower still. Thus not advancing can be locally competitive under this finite-horizon reward. This is one diagnosed state, not a proof that all failures have the same cause. We did not independently brute-force H5 or H7, nor optimize planner-specific terminal values.

## Audit and limits

- All {len(old)} available outcomes of the earlier pilot reproduce on steps, success, damage, timeout, travel, peak deformation, cumulative reward and contact count. Timing is intentionally not required to be identical.
- See [coverage](tables/coverage.csv), [per-layout metrics](tables/episodes.csv), [per-seed A* comparison](tables/astar_pairs.csv) and [audits](audits/). Missing cells, if any, are not scored as timeouts.
- This is a deterministic, previously used evaluation range, a single checkpoint and one layout family. It is not a blind robustness benchmark and does not test execution noise. There is no matched time-budget, equal-feature ablation, training-cost comparison, real-robot claim or state-of-the-art optimal-planning claim.
"""
(folder/"RESULTS.md").write_text(text,encoding="utf-8")
print(f"PASS: {len(old)} historical pilot rows match on eight physical/outcome fields; wrote results from CSV.")
