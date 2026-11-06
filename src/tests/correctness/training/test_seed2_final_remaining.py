import importlib.util,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]

def loader():
    spec=importlib.util.spec_from_file_location("final_remaining",ROOT/"experiments/prepare_seed2_final_remaining.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def test_selection_uses_completion_not_accuracy_and_excludes_active():
    def run(task,state,step,name="projected_lse"):
        return dict(name=task,displayName=f"{task}-training-{name}-seed2",state=state,summaryMetrics=json.dumps(dict(train_step=step,pass_2_correct=False)))
    runs=[run("a","finished",1499),run("b","running",99),run("c","failed",100),run("d","finished",39),run("e","finished",1499,"projected_conv_c4")]
    tasks,audit=loader().select_jobs(list("abcde"),runs,"lse")
    assert tasks==list("cde") and audit["completed"]==["a"] and audit["active"]==["b"]

def test_retry_architectures_and_controls_match_existing_arms():
    for variant,source,config in [("lse","seed2_finish_optimised","projected_lse_optimised.json"),("c4","seed2_finish_conv_c4","projected_conv_c4_triton.json")]:
        folder=ROOT/"experiments"/f"seed2_final_{variant}_remaining"
        cfg=json.loads((folder/"configs"/config).read_text())
        original=json.loads((ROOT/"experiments"/source/"configs"/config).read_text())
        assert cfg["model"]==original["model"]
        assert cfg["training"]["seed"]==2 and cfg["training"]["iterations"]==1500
        assert cfg["training"]["split"]=="training" and cfg["training"]["cuda_graph"] is False
        assert cfg["logging"]["save_model"] is False and cfg["logging"]["log_config"] is True
        if variant=="lse":assert cfg["model"]["cummax_implementation"]=="optimised" and cfg["model"]["shift_implementation"]=="primitives"
        else:assert cfg["model"]["shift_implementation"]=="triton" and cfg["model"]["cummax_implementation"]=="primitives"
        manifest=json.loads((folder/"manifest.json").read_text())
        excluded=set(manifest["selection_audit"]["completed"]+manifest["selection_audit"]["active"])
        assert {j["task"] for j in manifest["jobs"]}==set(manifest["task_ids"])-excluded
        spec=importlib.util.spec_from_file_location("retry_worker",folder/"run_job.py")
        worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
        for i in range(len(manifest["jobs"])):
            cmd,job,out=worker.command(folder,i)
            assert job["seed"]==2 and job["variant"]==variant
            assert str(folder/job["config"]) in cmd and "--save-checkpoints" not in cmd
            assert "logging.save_model=false" in cmd
        script=(folder/"submit.slurm").read_text()
        assert f"--array=0-{len(manifest['jobs'])-1}%4" in script and "--gres=gpu:1" in script
        assert f"experiments/seed2_final_{variant}_remaining/run_job.py" in script

def test_c4_old_script_cannot_launch_lse_worker():
    script=(ROOT/"experiments/seed2_finish_conv_c4/submit.slurm").read_text()
    assert "experiments/seed2_finish_conv_c4/run_job.py" in script
    assert "experiments/seed2_finish_optimised/run_job.py" not in script

def test_lse_first_is_a_scheduler_dependency():
    s=(ROOT/"experiments/submit_seed2_final_lse_first.sh").read_text()
    assert "--dependency=\"afterany:$LSE_JOB\"" in s
    assert s.index("prepare_seed2_final_remaining.py")<s.index("sbatch --parsable")
