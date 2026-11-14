"""Evaluate frozen models selected from four step-22 reference reversal intervals."""
import importlib.util
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results/training/half_td_local_supervision_20261114"
OUTPUT = ROOT / "results/training/neighbor_rollouts_20261114"
SEGMENTS = ((2000, 2200), (2200, 2400), (9800, 10000))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


r = module("neighbor023_training", ROOT / "docs/experiments/022_half_td_local_supervision/run_local_supervision.py")
t = module("neighbor023_trace", ROOT / "docs/experiments/020_late_rollouts/run_late_rollouts.py")
v = r.v
read, require, equal = v.read_json, v.require, v.require_equal


def select_windows(dense):
    index = dense["reference_steps"].index(22)
    values = [(row[0], row[2][index]) for row in dense["rows"] if row[0] > 2000]
    require(bool(values) and all(math.isfinite(g) for _, g in values), "finite reference gaps")
    require([u for u, _ in values] == list(range(values[0][0], values[-1][0] + 1)), "continuous reference clock")
    negative = [u for u, gap in values if gap < 0]
    require(bool(negative), "at least one negative reference gap")
    groups = []
    for update in negative:
        if not groups or update != groups[-1][-1] + 1:
            groups.append([update])
        else:
            groups[-1].append(update)
    longest = max(groups, key=len)
    worst = min(values, key=lambda item: item[1])
    anchors = sorted({negative[0], worst[0], longest[(len(longest) - 1) // 2], negative[-1]})
    chosen = [group for group in groups if any(anchor in group for anchor in anchors)]
    available = {row[0] for row in dense["rows"]}
    window = sorted({u for group in chosen for u in range(group[0] - 1, group[-1] + 2) if u in available})
    return {"anchors": anchors, "window": window, "negative_intervals": chosen,
            "all_negative_count": len(negative), "all_interval_count": len(groups),
            "selected_negative_count": sum(u in negative for u in window), "minimum": worst}


def save(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run():
    OUTPUT.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    save("status.json", {"status": "running", "started_at_utc": started})
    torch.set_num_threads(1)
    archive = ROOT / "docs/experiments/022_half_td_local_supervision/evidence"
    archive_manifest = read(archive / "archive_manifest.json")
    committed = subprocess.check_output(["git", "show", "HEAD:" + (archive / "archive_manifest.json").relative_to(ROOT).as_posix()], cwd=ROOT)
    require(json.loads(committed) == archive_manifest, "committed source archive")
    bound = {}
    for name, entry in archive_manifest["files"].items():
        for path in (archive / name, *((SOURCE / name,) if not name.endswith("_samples.jsonl") else ())):
            require(v.file_hash(path) == entry["sha256"], "immutable source evidence")
            bound[str(path)] = entry["sha256"]
    for name, entry in archive_manifest["raw_log_coverage"].items():
        require(v.file_hash(SOURCE / name) == entry["sha256"], "immutable source raw log")
        bound[str(SOURCE / name)] = entry["sha256"]
    require(read(SOURCE / "status.json")["status"] == "completed" and read(SOURCE / "verification.json")["passed"], "verified source run")
    require(v.file_hash(Path(r.__file__)) == read(SOURCE / "provenance.json")["script_sha256"], "original update implementation")
    trace_provenance = read(ROOT / "docs/experiments/020_late_rollouts/evidence/provenance.json")
    require(v.file_hash(Path(t.__file__)) == trace_provenance["script_sha256"], "unchanged decision observer")
    settings, manifest, batch, _, episodes, _, _ = r.inputs()
    env_config = settings["environment"]
    for name, digest in {**manifest["production_source_sha256"], **manifest["helper_sha256"]}.items():
        require(v.file_hash(ROOT / name) == digest, "source dependency hash")
        bound[str(ROOT / name)] = digest
    bound[str(Path(r.__file__))] = v.file_hash(Path(r.__file__))
    bound[str(Path(t.__file__))] = v.file_hash(Path(t.__file__))
    selection = select_windows(read(SOURCE / "dense.json"))
    require(selection["anchors"] == [2031, 2227, 2345, 9847] and len(selection["window"]) == 19, "predeclared selection")
    save("selection.json", selection)
    save("provenance.json", {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
         "script_sha256": v.file_hash(Path(__file__)), "source_protocol": read(SOURCE / "protocol.json"),
         "source_hashes": bound, "segments": SEGMENTS, "selected_models": selection})
    model_hashes = read(SOURCE / "model_hashes.json")
    updates = {row["update"]: row for row in v.read_log(SOURCE / "updates.jsonl")}
    probes = {row["update"]: row for row in v.read_log(SOURCE / "sentinels.jsonl")}
    directory = OUTPUT / "models"
    directory.mkdir()
    models, reconstruction = {}, []
    for start, stop in SEGMENTS:
        paths = [SOURCE / "models" / f"training_state_{step}.pt" for step in (start, stop)]
        for path in paths:
            require(v.file_hash(path) == model_hashes[path.name], "verified full boundary snapshot")
            bound[str(path)] = model_hashes[path.name]
        source, endpoint = [torch.load(path, weights_only=True) for path in paths]
        for independent in (False, True):
            agent, env = v.prior.independent_restore(source)
            try:
                equal(r.snapshot(agent, env_config), source, "unchanged replay restore")
                for step in range(start + 1, stop + 1):
                    stats = r.independent_update(agent.learner, batch) if independent else r.weighted_td_update(agent.learner, batch)
                    require({"update": step, **stats} == updates[step], "exact original update log")
                    v.prior.verify_sentinel_network(agent, probes[step], batch)
                    if step in selection["window"]:
                        current = r.snapshot(agent, env_config)
                        if independent:
                            equal(current, models[step], "independent selected full state")
                        else:
                            models[step] = current
                            torch.save(current, directory / f"training_state_{step}.pt")
                equal(r.snapshot(agent, env_config), endpoint, "exact full segment endpoint")
                reconstruction.append({"start": start, "end": stop, "independent": independent, "updates": stop - start, "passed": True})
                print(f"Exact replay verified: {start}--{stop}, independent={independent}.", flush=True)
            finally:
                env.close()
    require(sorted(models) == selection["window"], "all selected states recovered")
    save("reconstruction.json", reconstruction)
    save("model_hashes.json", {p.name: v.file_hash(p) for p in directory.glob("*.pt")})
    rows, cert, reloads = [], [], []
    for verify in (False, True):
        for step in (selection["anchors"] if verify else selection["window"]):
            payload = torch.load(directory / f"training_state_{step}.pt", weights_only=True)
            agent, env = v.prior.independent_restore(payload)
            records, incomplete, scenes = [], [], []
            try:
                with v.capture_resets(env, step, records, incomplete):
                    for episode in episodes:
                        part = v.multi.episode_batch(batch, episode)
                        if verify:
                            outcome = v.multi.diagnostic.fit.fixed_scene_rollout(agent, env, episode["seed"], part)
                            expected = next(s["rollout"] for row in rows if row["update"] == step for s in row["scenes"] if s["seed"] == episode["seed"])
                            require(outcome == expected, "exact uninstrumented complete trajectory")
                            scenes.append({"seed": episode["seed"], "rollout": outcome})
                        else:
                            scenes.append(t.trace_rollout(agent, env, episode["seed"], part))
                equal(r.snapshot(agent, env_config), payload, "evaluation preserves full training state")
                require(not incomplete and len(records) == 3, "complete reset accounting")
                if verify:
                    require(records == [entry for entry in cert if entry["update"] == step], "exact reloaded reset certification")
                    reloads.append({"update": step, "scenes": scenes, "reset_certification": records})
                    save("reloads.json", reloads)
                else:
                    rows.append({"update": step, "scenes": scenes})
                    cert.extend(records)
                    save("observations.json", rows)
                    save("reset_certification.json", cert)
                print(f"{'Reload' if verify else 'Evaluate'} TD {step}: " + ", ".join(f"{s['seed']}:{s['rollout']['success']}/{s['rollout']['steps']}" for s in scenes), flush=True)
            finally:
                env.close()
    for path, digest in bound.items():
        require(v.file_hash(Path(path)) == digest, "source artifacts unchanged")
    save("verified_source_hashes.json", bound)
    accounting = {"reconstruction_updates": sum(r["updates"] for r in reconstruction), "new_remedy_training_updates": 0,
                  "main_episodes": len(rows) * 3, "reload_episodes": len(reloads) * 3,
                  "main_policy_actions": sum(s["rollout"]["steps"] for row in rows for s in row["scenes"]),
                  "main_certification_actions": sum(r["certification_environment_steps"] for r in cert),
                  "reload_policy_actions": sum(s["rollout"]["steps"] for row in reloads for s in row["scenes"]),
                  "reload_certification_actions": sum(r["certification_environment_steps"] for row in reloads for r in row["reset_certification"]),
                  "forced_actions": 0, "held_out_scenes": 0}
    save("accounting.json", accounting)
    save("verification.json", {"passed": True, "independent_reconstruction_updates": 600, "main_reconstruction_updates": 600,
         "matched_selected_full_states": 19, "exact_reloaded_episodes": 12, "all_evaluations_preserve_full_state": True})
    save("status.json", {"status": "completed", "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    try:
        run()
    except BaseException as error:
        if OUTPUT.exists():
            save("failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
