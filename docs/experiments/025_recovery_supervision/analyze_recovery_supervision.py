"""Summarize every declared recovery-training check and archive compact evidence."""
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/training/recovery_supervision_20261115"
ARCHIVE = Path(__file__).parent / "evidence"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (RUN / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def outcomes(rows):
    failures = [{"update": row["update"], "seed": s["seed"], "rollout": s["rollout"]}
                for row in rows for s in row["scenes"] if not s["rollout"]["success"]]
    return {"checkpoints": len(rows), "episodes": len(rows) * 3,
            "all_scene_success": sum(all(s["rollout"]["success"] for s in row["scenes"]) for row in rows),
            "successful_episodes": len(rows)*3-len(failures), "failures": failures,
            "policy_actions": sum(s["rollout"]["steps"] for row in rows for s in row["scenes"])}


def ranking(entries):
    negative = [u for u,gap,_ in entries if gap < 0]
    wrong = [u for u,_,correct in entries if not correct]
    return {"updates": len(entries), "negative_count": len(negative), "negative_updates": negative,
            "wrong_greedy_count": len(wrong), "wrong_greedy_updates": wrong,
            "first_negative": min(negative,default=None), "last_negative": max(negative,default=None),
            "minimum_gap": min(gap for _,gap,_ in entries), "final_gap": entries[-1][1]}


def main():
    assert read(RUN / "status.json")["status"] == "completed"
    check = read(RUN / "verification.json")
    assert check["passed"] and check["independent_updates"] == 401
    assert not ARCHIVE.exists()
    sample=read(RUN / "recovery_sample.json")
    encoded=np.asarray(sample["observation"],dtype=np.float32)
    assert sample["action"]==4 and sample["td_transitions_added"]==0
    assert hashlib.sha256(encoded.tobytes()).hexdigest()==sample["encoded_observation_sha256"]
    new, old = read(RUN / "observations.json"), read(RUN / "historical_controls.json")
    assert [r["update"] for r in new] == [r["update"] for r in old]
    assert len(new) == 69
    tails, old_tails = read(RUN / "conditional.json"), read(RUN / "baseline_conditional.json")
    assert len(tails) == 69 and len(old_tails) == 7
    assert [t["update"] for t in tails] == [r["update"] for r in new]
    for t in tails + old_tails:
        assert t["scripted_prefix_actions"] == 24
        assert all(d["scripted_prefix"] == (d["step"] <= 24) for d in t["decisions"])
        assert all(d["policy_action"] == d["executed_action"] for d in t["decisions"][24:])
        assert t["autonomous_tail_actions"] == t["rollout"]["steps"] - 24
    prefix=read(ROOT / "results/training/step25_rescue_20261115/td_2031_greedy.json")["rollout"]["actions"][:24]
    actual_visits=[]
    by_step={row["update"]:row for row in tails}
    for row in new:
        scene=next(s["rollout"] for s in row["scenes"] if s["seed"]==10001)
        if scene["actions"][:24]==prefix:
            conditional_path=by_step[row["update"]]["rollout"]["robot_path"]
            assert scene["robot_path"][:25]==conditional_path[:25]
            actual_visits.append({"update":row["update"],"action25":scene["actions"][24],"success":scene["success"],"steps":scene["steps"],"exact_24_action_prefix":True})
    dense = read(RUN / "dense.json")
    assert [r["update"] for r in dense] == list(range(2000,10001))
    entries = dense[1:]
    reference_steps = [s["reference_step"] for s in entries[0]["reference"]["states"]]
    reference = {str(step): ranking([(row["update"], row["reference"]["states"][i]["signed_demonstration_gap"],
                  row["reference"]["states"][i]["greedy_action"] == row["reference"]["states"][i]["demonstrated_action"]) for row in entries])
                 for i,step in enumerate(reference_steps)}
    recovery = ranking([(row["update"],row["recovery"]["signed_gap"],row["recovery"]["greedy_action"]==4) for row in entries])
    # Preserve every update's decisions/gaps and all recovery Q values compactly.
    compact = {"columns":["update","original_correct","reference_gaps","reference_actions","recovery_q","recovery_gap","recovery_action"],
        "reference_steps": reference_steps, "rows":[[row["update"],row["original_correct"],
         [s["signed_demonstration_gap"] for s in row["reference"]["states"]],
         [s["greedy_action"] for s in row["reference"]["states"]],row["recovery"]["q_values"],row["recovery"]["signed_gap"],row["recovery"]["greedy_action"]] for row in dense]}
    (RUN / "dense_compact.json").write_text(json.dumps(compact,separators=(",",":"),allow_nan=False)+"\n",encoding="utf-8")
    failure_ids = [r["update"] for r in new if not all(s["rollout"]["success"] for s in r["scenes"])]
    last_failure = max(failure_ids,default=None)
    comparison = {"historical_control":outcomes(old),"treatment":outcomes(new),
        "known_five_failures":{arm:outcomes([r for r in rows if r["update"] in (2031,2226,2344,2345,2346)]) for arm,rows in (("historical_control",old),("treatment",new))},
        "late_window":{arm:outcomes([r for r in rows if 9946<=r["update"]<=9966]) for arm,rows in (("historical_control",old),("treatment",new))}}
    old_path=ROOT / "results/training/half_td_local_supervision_20261114/dense.json"
    old_manifest=read(ROOT / "docs/experiments/022_half_td_local_supervision/evidence/archive_manifest.json")
    assert digest(old_path)==old_manifest["files"]["dense.json"]["sha256"]
    old_dense=read(old_path)
    assert old_dense["reference_steps"]==reference_steps
    baseline_reference={str(step):ranking([(row[0],row[2][i],row[3][i]==int(entries[0]["reference"]["states"][i]["demonstrated_action"])) for row in old_dense["rows"][1:]]) for i,step in enumerate(old_dense["reference_steps"])}
    conditional={arm:{"episodes":len(rows),"successes":sum(r["rollout"]["success"] for r in rows),
        "north_large_at_recovery":sum(r["recovery_action"]==4 for r in rows),
        "failures":[{"update":r["update"],"recovery_action":r["recovery_action"],"rollout":r["rollout"]} for r in rows if not r["rollout"]["success"]]}
        for arm,rows in (("treatment_all",tails),("baseline_seven",old_tails),("treatment_matched_seven",[r for r in tails if r["update"] in {t["update"] for t in old_tails}]))}
    cert=read(RUN / "reset_certification.json")
    reset_counts=Counter()
    cert_steps=Counter()
    for group in cert:
        reset_counts[group["phase"]]+=len(group["records"])
        cert_steps[group["phase"]]+=sum(c["certification_environment_steps"] for c in group["records"])
    assert dict(reset_counts)=={"collection":1,"baseline_conditional":7,"treatment_full":207,"treatment_conditional":69,"verification_full":21,"verification_conditional":7}
    extra=read(RUN/"failure_reload_verification.json")
    assert extra["passed"] and extra["optimizer_updates"]==0
    assert [x["update"] for x in extra["results"]]==failure_ids
    reset_counts["posthoc_verification_full"]=3*len(extra["results"])
    reset_counts["posthoc_verification_conditional"]=len(extra["results"])
    cert_steps["posthoc_verification_full"]=sum(c["certification_environment_steps"] for row in extra["results"] for c in row["reset_certification"][:3])
    cert_steps["posthoc_verification_conditional"]=sum(row["reset_certification"][3]["certification_environment_steps"] for row in extra["results"])
    policy={"collection":24,"baseline_conditional":sum(t["rollout"]["steps"] for t in old_tails),
            "treatment_full":comparison["treatment"]["policy_actions"],"treatment_conditional":sum(t["rollout"]["steps"] for t in tails),
            "verification_full":sum(s["steps"] for r in check["reloads"] for s in r["scenes"]),
            "verification_conditional":sum(r["conditional"]["steps"] for r in check["reloads"])}
    policy["posthoc_verification_full"]=sum(s["steps"] for row in extra["results"] for s in row["scenes"])
    policy["posthoc_verification_conditional"]=sum(row["conditional"]["rollout"]["steps"] for row in extra["results"])
    accounting={"resets_by_phase":dict(reset_counts),"actions_by_phase":policy,"certification_actions_by_phase":dict(cert_steps),
        "total_resets":sum(reset_counts.values()),"total_executed_actions":sum(policy.values()),
        "total_certification_actions":sum(cert_steps.values()),"total_simulator_actions":sum(policy.values())+sum(cert_steps.values()),
        "main_optimizer_updates":8000,"gate_optimizer_updates":4,"independent_optimizer_updates":401,
        "total_new_optimizer_updates":8405,"scripted_prefix_actions_included":24*(69+7+7+len(extra["results"])),
        "forced_recovery_actions":0,"held_out_scenes":0,
        "historical_baseline_episodes_not_new_work":207,
        "note":"Collection uses the frozen policy for 24 actions; conditional probes have 24 scripted prefix actions each. Unit tests are outside formal counts."}
    result={"comparison":comparison,"conditional":conditional,"recovery_ranking":recovery,
        "autonomous_full_rollout_recovery_visits":actual_visits,
        "reference_ranking":reference,"historical_reference_ranking":baseline_reference,
        "final_fit":new[-1]["fit"],"final_return_fit":new[-1]["return_fit"],
        "original_perfect_update_count":sum(r["original_correct"]==117 for r in entries),
        "original_worst_correct":min(r["original_correct"] for r in entries),
        "last_sampled_full_failure":last_failure,
        "subsequent_successful_checkpoints":[r["update"] for r in new if last_failure is not None and r["update"]>last_failure],
        "scope":"One added supervision state and inherited initialization. Known training scenes only; checkpoint success and conditional recovery are separate. No claim about unevaluated updates or generalization.",
        "analysis_script_sha256":digest(Path(__file__))}
    inference_hashes={}
    for row in new:
        path=RUN / "models" / f"update_{row['update']}.pt"
        inference=torch.load(path,weights_only=True)
        full=torch.load(RUN / "models" / f"training_state_{row['update']}.pt",weights_only=True)
        assert all(torch.equal(value,full["online"][name]) for name,value in inference["online"].items())
        inference_hashes[path.name]=digest(path)
    save("inference_model_hashes.json",inference_hashes)
    save("analysis.json",result)
    save("accounting.json",accounting)
    ARCHIVE.mkdir()
    files={}
    for source in sorted(RUN.glob("*.json")):
        if source.name=="dense.json":
            continue
        target=ARCHIVE/source.name
        shutil.copy2(source,target)
        assert target.read_bytes()==source.read_bytes()
        files[target.name]={"sha256":digest(target),"bytes":target.stat().st_size}
    raw=RUN/"updates.jsonl"
    lines=raw.read_text().splitlines()
    assert len(lines)==8000 and [json.loads(line)["update"] for line in lines]==list(range(2001,10001))
    selected=[line for i,line in enumerate(lines) if i<11 or json.loads(line)["update"] in {r["update"] for r in new} or json.loads(line)["update"]%50==0]
    target=ARCHIVE/"updates_samples.jsonl"
    target.write_text("\n".join(selected)+"\n",encoding="utf-8")
    files[target.name]={"sha256":digest(target),"bytes":target.stat().st_size}
    manifest={"source_run":str(RUN),"files":files,
        "raw_log_coverage":{"updates.jsonl":{"sha256":digest(raw),"records":8000,"sampled_records":len(selected)},
          "dense.json":{"sha256":digest(RUN/"dense.json"),"records":8001,"coverage":"dense_compact.json retains every update, reference gap/action and recovery Q value"}},
        "excluded":"Model binaries and full raw logs remain local; paths and hashes are recorded."}
    (ARCHIVE/"archive_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"full_success":{a:comparison[a]["successful_episodes"] for a in ("historical_control","treatment")},
        "all_scene_success":{a:comparison[a]["all_scene_success"] for a in ("historical_control","treatment")},
        "conditional":{a:{k:v for k,v in x.items() if k!="failures"} for a,x in conditional.items()},
        "recovery_negative_updates":recovery["negative_count"],"accounting":accounting}))


if __name__=="__main__":
    main()
