"""Freeze unfinished combined-model jobs from W&B; exclude active runs."""
import json, shutil
from datetime import datetime, timezone
from pathlib import Path

def select_jobs(manifest, runs):
    completed, active = set(), set()
    evidence = []
    for run in runs:
        name = run.name
        if "-training-combined_d4_conv_sharing_seed0_sample40" not in name:
            continue
        if "packed_" in name or "profile" in name:
            continue
        task = name.split("-training-", 1)[0]
        if task not in manifest["task_ids"]:
            continue
        step = run.summary.get("train_step", -1)
        if run.state == "finished" and step is not None and step >= 1499:
            completed.add(task)
        elif run.state in ("running", "pending", "preempting"):
            active.add(task)
        evidence.append(dict(id=run.id, task=task, state=run.state, step=step))
    jobs = [job for job in manifest["jobs"] if job["task"] not in completed | active]
    return jobs, dict(completed=sorted(completed), active=sorted(active), runs=evidence)

def main():
    import wandb
    root = Path(__file__).resolve().parents[1]
    source = root / "experiments/seed0_combined_d4"
    target = root / "experiments/seed0_combined_d4_packed_remaining"
    manifest = json.loads((source / "manifest.json").read_text())
    runs = wandb.Api(timeout=60).runs("arc_agi/compressarc-final-ablation",
        filters={"tags": {"$in": ["seed0-combined-d4"]}}, lazy=False)
    jobs, audit = select_jobs(manifest, runs)
    if not jobs:
        raise SystemExit("No unstarted/failed tasks remain; do not submit an old manifest.")
    target.mkdir(exist_ok=True); (target / "configs").mkdir(exist_ok=True)
    cfg = json.loads((source / "configs/combined_d4.json").read_text())
    cfg["model"]["direction_share_packed"] = True
    cfg["logging"]["wandb_tags"].append("combined-d4-packed-remaining")
    cfg["experiment"]["name"] += "_packed"
    assert cfg["model"]["shift_implementation"] == "d4_tied_conv"
    assert cfg["model"]["direction_share_implementation"] == "d4"
    assert cfg["training"]["seed"] == 0 and cfg["training"]["iterations"] == 1500
    assert cfg["logging"]["save_model"] is False
    (target / "configs/combined_d4.json").write_text(json.dumps(cfg, indent=2)+"\n")
    manifest["jobs"] = jobs
    manifest["selection_audit"] = dict(audit, collected_at=datetime.now(timezone.utc).isoformat())
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    shutil.copyfile(source / "run_job.py", target / "run_job.py")
    script = (source / "submit.slurm").read_text().replace("arc-combined-d4-s0", "arc-comb-pack-s0")
    script = script.replace("--array=0-39%2", f"--array=0-{len(jobs)-1}%2")
    script = script.replace("experiments/seed0_combined_d4/run_job.py", "experiments/seed0_combined_d4_packed_remaining/run_job.py")
    (target / "submit.slurm").write_text(script)
    print(f"Completed: {len(audit['completed'])}; active excluded: {len(audit['active'])}; remaining: {len(jobs)}")
    print("sbatch experiments/seed0_combined_d4_packed_remaining/submit.slurm")

if __name__ == "__main__": main()
