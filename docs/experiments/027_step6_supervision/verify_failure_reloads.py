"""Post-hoc frozen reloads of every new treatment failure, without training."""
import importlib.util
from pathlib import Path

import torch

SPEC=importlib.util.spec_from_file_location("step6_failure_reload",Path(__file__).with_name("run_step6_supervision.py"))
r=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def main():
    torch.set_num_threads(1)
    r.require(r.read(r.OUTPUT/"status.json")["status"]=="completed", "completed primary run")
    destination=r.OUTPUT/"failure_reload_verification.json"
    r.require(not destination.exists(), "no overwritten supplemental verification")
    rows=r.read(r.OUTPUT/"observations.json")
    selected=[row for row in rows if row["aggregate"]["success_count"]<3]
    if not selected:
        r.save("failure_reload_verification.json", {"passed":True,"selection":"All autonomous treatment failures after the primary run; none were found.",
            "optimizer_updates":0,"episodes":0,"results":[],"script_sha256":r.v.file_hash(Path(__file__)),
            "scope":"No failed-model replay was needed; no additional simulation was performed."})
        print("No autonomous failures required post-hoc reloads.",flush=True)
        return
    _,_,batch,_,episodes,_,_=r.b.inputs()
    sample=r.read(r.OUTPUT/"recovery_sample.json")
    recovery=torch.tensor([sample["observation"]],dtype=torch.float32)
    prefix=r.read(r.RESCUE/"td_2031_greedy.json")["rollout"]["actions"][:24]
    expected_tails={x["update"]:x for x in r.read(r.OUTPUT/"conditional.json")}
    part=r.v.multi.episode_batch(batch,next(e for e in episodes if e["seed"]==10001))
    results=[]
    for row in selected:
        step=row["update"]
        path=r.OUTPUT/"models"/f"training_state_{step}.pt"
        r.require(r.v.file_hash(path)==r.read(r.OUTPUT/"model_hashes.json")[path.name],"unchanged failed model")
        payload=torch.load(path,weights_only=True)
        agent,env=r.v.prior.independent_restore(payload)
        records,incomplete,scenes=[],[],[]
        try:
            with r.v.capture_resets(env,step,records,incomplete):
                for episode,expected in zip(episodes,row["scenes"],strict=True):
                    actual=r.v.multi.diagnostic.fit.fixed_scene_rollout(agent,env,episode["seed"],r.v.multi.episode_batch(batch,episode))
                    r.require(actual==expected["rollout"],"exact failed-model full episode")
                    scenes.append(actual)
                tail=r.conditional_rollout(agent,env,part,recovery,prefix)
                r.require(tail=={k:value for k,value in expected_tails[step].items() if k!="update"},"exact failed-model conditional replay")
            r.require(not incomplete and len(records)==4,"complete supplemental reset accounting")
            original_cert=r.read(r.OUTPUT/"reset_certification.json")
            expected_records=[c for group in original_cert if group.get("update")==step and group["phase"] in ("treatment_full","treatment_conditional") for c in group["records"]]
            r.require(records==expected_records,"exact supplemental reset certification")
            r.equal(r.b.snapshot(agent,payload["environment_config"]),payload,"unchanged failed-model full state")
            results.append({"update":step,"scenes":scenes,"conditional":tail,"reset_certification":records})
            r.save("failure_reload_verification.json",{"passed":False,"selection":"All treatment checkpoints with fewer than three autonomous successes, selected after the primary run.","results":results})
            print(f"Reproduced all four frozen evaluations at failed TD {step}.",flush=True)
        finally:
            env.close()
    r.save("failure_reload_verification.json",{"passed":True,"selection":"All treatment checkpoints with fewer than three autonomous successes, selected after the primary run.",
        "optimizer_updates":0,"episodes":4*len(selected),"results":results,
        "script_sha256":r.v.file_hash(Path(__file__)),"scope":"Post-hoc verification only; declared 105-checkpoint statistics and training are unchanged."})


if __name__=="__main__":
    main()
