"""Check trace integrity and aggregate a complete paired waypoint run."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "research_results/paired_planner_evaluation/data/episodes"
SEEDS = range(5420000, 5420010)
METHODS = {"1": "H1", "3": "H3", "3b32": "H3-B32", "5": "H5", "7": "H7"}
OUTCOMES = ("success", "damage", "timeout")
METRICS = ("steps", "travel", "peak", "reward", "contacts")


def load_row(path, seed, method):
    row = json.loads(path.read_text(encoding="utf-8"))
    trace = row["trace"]
    assert (row["seed"], row["method"]) == (seed, method)
    assert 0 < len(trace) == row["steps"] <= 200
    assert sum(row[k] for k in OUTCOMES) == 1
    assert not row["timeout"] or row["steps"] == 200
    assert [step["step"] for step in trace] == list(range(1, row["steps"] + 1))
    assert all(0 <= step["action"] < 8 for step in trace)
    assert trace[-1]["terminal"] and not any(step["terminal"] for step in trace[:-1])
    cumulative = 0.0
    for step in trace:
        cumulative += step["reward"]
    assert row["reward"] == cumulative
    assert all(row[k] == trace[-1][k] for k in ("travel", "peak"))
    assert row["contacts"] == trace[-1]["contact_steps"]
    assert row["damage"] == (row["peak"] > 1)
    return row


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--supplement", type=Path, action="append", default=[],
                        help="Additional run directory with identical configuration.")
    parser.add_argument("--out", type=Path, help="Directory for summary tables; defaults to run.")
    parser.add_argument("--settings", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    paired = []
    summary = []
    weights = set()
    for suffix in args.settings:
        method = METHODS[suffix]
        original, guided = [], []
        for seed in SEEDS:
            name = f"{seed}-{suffix}.json"
            a = load_row(ARCHIVE / name, seed, method)
            copies = [path / name for path in [args.run, *args.supplement]
                      if (path / name).is_file()]
            if not copies:
                raise FileNotFoundError(name)
            assert all(path.read_bytes() == copies[0].read_bytes() for path in copies[1:]), name
            b = load_row(copies[0], seed, method)
            assert a["initial_observation_sha256"] == b["initial_observation_sha256"]
            assert a["beam"] == b["beam"]
            weights.add(b["waypoint_weight"])
            original.append(a)
            guided.append(b)
            result = {"seed": seed, "method": method, "waypoint_weight": b["waypoint_weight"]}
            for label, row in (("original", a), ("guided", b)):
                result.update({f"{label}_{k}": row[k] for k in (*OUTCOMES, *METRICS)})
                trace = row["trace"]
                result[f"{label}_final_x"], result[f"{label}_final_y"] = trace[-1]["xy"]
                result[f"{label}_unchanged_positions"] = sum(
                    x["xy"] == y["xy"] for x, y in zip(trace, trace[1:])
                )
            paired.append(result)
        for label, rows in (("original", original), ("guided", guided)):
            successful = [r for r in rows if r["success"]]
            summary.append({
                "method": method, "variant": label, "episodes": len(rows),
                **{k: sum(r[k] for r in rows) for k in OUTCOMES},
                "successful_mean_steps": mean(r["steps"] for r in successful) if successful else "",
                "successful_mean_travel": mean(r["travel"] for r in successful) if successful else "",
                "mean_final_x": mean(r["trace"][-1]["xy"][0] for r in rows),
            })
    assert len(weights) == 1 and next(iter(weights)) > 0
    output = args.out or args.run
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "paired.csv", paired)
    write_csv(output / "summary.csv", summary)
    print(f"{len(paired)} pairs checked at waypoint weight {next(iter(weights)):g}.")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
