"""Freeze unfinished LSE/C4 seed-2 jobs from W&B, without filtering by accuracy."""
import base64,json,netrc,os,shutil,urllib.request
from datetime import datetime,timezone
from pathlib import Path

def select_jobs(task_ids,runs,variant):
    done=set();active=set();evidence=[]
    for r in runs:
        name=r["displayName"];task=name.split("-training-",1)[0]
        if task not in task_ids or not name.endswith("-seed2"):continue
        if any(s in name for s in ("profile","_clean","latency","benchmark")):continue
        matched=("projected_lse" in name) if variant=="lse" else ("projected_conv" in name and "projected_conv_d4" not in name)
        if not matched:continue
        summary=json.loads(r["summaryMetrics"] or "{}")
        step=summary.get("train_step",-1)
        if r["state"]=="finished" and step is not None and step>=1499:done.add(task)
        elif r["state"] in ("running","pending","preempting"):active.add(task)
        evidence.append(dict(id=r["name"],task=task,state=r["state"],step=step))
    return [t for t in task_ids if t not in done|active],dict(completed=sorted(done),active=sorted(active),runs=evidence)

def fetch_runs():
    key=os.environ.get("WANDB_API_KEY")
    if not key:
        auth=netrc.netrc(os.environ.get("NETRC")).authenticators("api.wandb.ai")
        if not auth:raise RuntimeError("Log in to W&B before refreshing the remaining tasks")
        key=auth[2]
    headers={"Content-Type":"application/json","Authorization":"Basic "+base64.b64encode(("api:"+key).encode()).decode()}
    query='query($cursor:String){project(name:"compressarc-final-ablation",entityName:"arc_agi"){runs(first:100,after:$cursor){pageInfo{endCursor hasNextPage}edges{node{name displayName state summaryMetrics}}}}}'
    runs=[];cursor=None
    while True:
        req=urllib.request.Request("https://api.wandb.ai/graphql",data=json.dumps(dict(query=query,variables=dict(cursor=cursor))).encode(),headers=headers)
        with urllib.request.urlopen(req,timeout=90) as response:payload=json.load(response)
        if payload.get("errors"):raise RuntimeError(payload["errors"])
        batch=payload["data"]["project"]["runs"]
        runs.extend(e["node"] for e in batch["edges"])
        if not batch["pageInfo"]["hasNextPage"]:return runs
        cursor=batch["pageInfo"]["endCursor"]

def main():
    repo=Path(__file__).resolve().parents[1]
    frozen=json.loads((repo/"experiments/seed2_40/manifest.json").read_text())
    tasks=frozen["task_ids"]
    runs=fetch_runs();counts={}
    for variant,source,config in [
        ("lse","seed2_finish_optimised","projected_lse_optimised.json"),
        ("c4","seed2_finish_conv_c4","projected_conv_c4_triton.json")]:
        remaining,audit=select_jobs(tasks,runs,variant)
        target=repo/"experiments"/f"seed2_final_{variant}_remaining"
        (target/"configs").mkdir(parents=True,exist_ok=True)
        cfg=json.loads((repo/"experiments"/source/"configs"/config).read_text())
        # Preserve the existing arm's architecture, parameters and eager recipe.
        cfg["training"].update(cuda_graph=False,step_timing="wall",operator_profile=False)
        cfg["logging"].update(save_model=False,log_config=True)
        cfg["logging"]["wandb_tags"].append("seed2-final-"+variant+"-remaining")
        cfg["experiment"]["name"]+="_eager_retry"
        (target/"configs"/config).write_text(json.dumps(cfg,indent=2)+"\n")
        jobs=[dict(task=t,variant=variant,seed=2,config="configs/"+config) for t in remaining]
        manifest=dict(training_seed=2,iterations=1500,split="training",task_ids=tasks,jobs=jobs,
                      selection_audit=dict(audit,collected_at=datetime.now(timezone.utc).isoformat()))
        (target/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        shutil.copyfile(repo/"experiments/seed0_combined_d4/run_job.py",target/"run_job.py")
        submit=(repo/"experiments/seed0_combined_d4/submit.slurm").read_text()
        submit=submit.replace("arc-combined-d4-s0","arc-final-"+variant+"-s2")
        submit=submit.replace("--array=0-39%2",f"--array=0-{max(0,len(jobs)-1)}%4")
        submit=submit.replace("experiments/seed0_combined_d4/run_job.py",f"experiments/seed2_final_{variant}_remaining/run_job.py")
        submit=submit.replace("set -euo pipefail","set -euo pipefail\nunset ARC_CUDA_GRAPH  # Worker uses explicit cuda_graph=false config.")
        (target/"submit.slurm").write_text(submit,newline="\n")
        counts[variant]=len(jobs)
        print(f"{variant}: {len(audit['completed'])} complete, {len(audit['active'])} active excluded, {len(jobs)} remaining")
    (repo/"experiments/seed2_final_remaining_counts.json").write_text(json.dumps(counts,indent=2)+"\n")
if __name__=="__main__":main()
