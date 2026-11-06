import importlib.util,json
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[4]
def test_remaining_selection_excludes_complete_and_active_not_unsolved():
    spec=importlib.util.spec_from_file_location("prepare_packed",ROOT/"experiments/prepare_combined_d4_packed_remaining.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    manifest={"task_ids":["a","b","c","d"],"jobs":[{"task":t} for t in "abcd"]}
    def run(t,state,step):
        return SimpleNamespace(id=t,name=f"{t}-training-combined_d4_conv_sharing_seed0_sample40-seed0",state=state,summary={"train_step":step})
    jobs,audit=mod.select_jobs(manifest,[run("a","finished",1499),run("b","running",700),run("c","crashed",800),run("d","finished",39)])
    assert [j["task"] for j in jobs]==["c","d"]
    assert audit["completed"]==["a"] and audit["active"]==["b"]

def test_packed_package_architecture_and_submission():
    folder=ROOT/"experiments/seed0_combined_d4_packed_remaining"
    cfg=json.loads((folder/"configs/combined_d4.json").read_text())
    model=cfg["model"]
    assert model["shift_implementation"]=="d4_tied_conv"
    assert model["direction_share_implementation"]=="d4" and model["direction_share_packed"] is True
    assert model["cummax_implementation"]=="triton_hard_grouped"
    assert cfg["training"]["seed"]==0 and cfg["training"]["iterations"]==1500
    assert cfg["logging"]["save_model"] is False
    manifest=json.loads((folder/"manifest.json").read_text())
    excluded=set(manifest["selection_audit"]["completed"]+manifest["selection_audit"]["active"])
    assert {j["task"] for j in manifest["jobs"]}==set(manifest["task_ids"])-excluded
    script=(folder/"submit.slurm").read_text()
    assert f"--array=0-{len(manifest['jobs'])-1}%2" in script
    assert "--gres=gpu:1" in script and "seed0_combined_d4_packed_remaining/run_job.py" in script
