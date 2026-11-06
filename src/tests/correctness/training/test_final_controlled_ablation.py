from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf
from config import load_config
import run_training_timing as launcher
from preprocessing import Task
from compressarc.model.model import ARCCompressor
from compressarc.train import take_step

SRC=Path(__file__).resolve().parents[3]
NAMES=("final_original_20","final_projected_conv_20","final_projected_d4_20","final_projected_lse_20")
CHANGES=({}, {"shift_implementation":"tied_conv"}, {"direction_share_implementation":"d4"},
         {"cummax_implementation":"optimised"})


def test_final_configs_change_only_one_model_setting_and_share_all_controls():
    configs=[load_config(SRC/"config/models"/(name+".yaml")) for name in NAMES]
    baseline=OmegaConf.to_container(configs[0],resolve=True)
    assert len(set(baseline["experiment"]["task_ids"]))==20
    assert baseline["logging"]["wandb_project"]=="compressarc-final-ablation"
    assert baseline["logging"]["wandb_entity"]=="arc_agi"
    assert baseline["training"]["iterations"]==1500
    assert baseline["training"]["seed"]==0
    assert baseline["experiment"]["task_sample_seed"]==42
    assert baseline["experiment"]["seeds"]==[0,1,2]
    assert baseline["model"]["n_layers"]==4
    assert baseline["model"]["multitensor_constraints"]=="strict"
    for config,changes in zip(configs,CHANGES):
        current=OmegaConf.to_container(config,resolve=True)
        assert current["training"]==baseline["training"]
        assert current["logging"]==baseline["logging"]
        assert current["experiment"]["task_ids"]==baseline["experiment"]["task_ids"]
        assert current["experiment"]["seeds"]==[0,1,2]
        wanted=dict(baseline["model"],**changes)
        assert current["model"]==wanted
    assert [path.stem for path in launcher.DEFAULT_CONFIGS]==list(NAMES)


def jobs(calls):
    result=[]
    for command in calls:
        task=command[command.index("--task")+1]
        config=Path(command[command.index("--config")+1])
        seed=int([item for item in command if item.startswith("training.seed=")][-1].split("=")[1])
        directory=Path(command[command.index("--output-dir")+1])
        assert directory.name==f"seed_{seed}"
        result.append((task,config.stem,seed))
    return result


def test_default_final_launcher_schedules_240_runs_and_separates_outputs(monkeypatch):
    calls=[]
    monkeypatch.setattr(launcher.train,"main",lambda command:calls.append(command))
    monkeypatch.setattr(launcher.torch.cuda,"is_available",lambda:False)
    monkeypatch.setattr(launcher.gc,"collect",lambda:None)
    launcher.main(["--mode","train"])
    assert len(calls)==240
    triples=jobs(calls)
    assert len(set(triples))==240
    assert len(set(task for task,_,_ in triples))==20
    assert set(seed for _,_,seed in triples)=={0,1,2}
    for command in calls:
        config=Path(command[command.index("--config")+1])
        directory=Path(command[command.index("--output-dir")+1])
        assert directory.parent.name==config.stem


def test_seed_sweep_shards_cover_all_jobs_once(monkeypatch):
    calls=[]
    monkeypatch.setattr(launcher.train,"main",lambda command:calls.append(command))
    monkeypatch.setattr(launcher.torch.cuda,"is_available",lambda:False)
    monkeypatch.setattr(launcher.gc,"collect",lambda:None)
    for shard in range(3):
        before=len(calls)
        launcher.main(["--mode","train","--shard-count","3","--shard-index",str(shard)])
        assert len(calls)-before==80
    assert len(set(jobs(calls)))==240


@pytest.mark.parametrize("flags,seeds",[(["--seed","7","--seed","11"],{7,11}),
                                       (["--set","training.seed=9"],{9})])
def test_explicit_seed_selection_overrides_experiment_sweep(monkeypatch,flags,seeds):
    calls=[]
    monkeypatch.setattr(launcher.train,"main",lambda command:calls.append(command))
    monkeypatch.setattr(launcher.torch.cuda,"is_available",lambda:False)
    monkeypatch.setattr(launcher.gc,"collect",lambda:None)
    launcher.main(["--mode","train","--config",str(launcher.DEFAULT_CONFIGS[0]),
                   "--task","50cb2852",*flags])
    assert len(calls)==len(seeds)
    assert set(seed for _,_,seed in jobs(calls))==seeds


@pytest.mark.parametrize("name",NAMES)
def test_each_final_variant_can_take_a_finite_training_step(name):
    overrides=["model.n_layers=1","model.channel_dim_shared=2","model.channel_dim_grid=2",
               "model.share_up_dim=2","model.share_down_dim=2","model.decoding_dim=2",
               "model.softmax_dim=2","model.cummax_dim=2","model.shift_dim=2",
               "model.nonlinear_dim=2","model.direction_share_dim=2"]
    config=load_config(SRC/"config/models"/(name+".yaml"),overrides)
    task=Task("smoke",{"train":[{"input":[[0,1],[1,0]],"output":[[1,0],[0,1]]}],
                       "test":[{"input":[[1,0],[0,1]]}]},None)
    torch.manual_seed(0)
    model=ARCCompressor(task,config.model)
    optimizer=torch.optim.Adam(model.weights_list,lr=config.training.learning_rate,
                               betas=tuple(config.training.optimizer_betas))
    result=take_step(task,model,optimizer,0,config.training)
    assert torch.isfinite(torch.tensor(result["loss"]))
    for kind in ("shift","direction_share","cummax"):
        for core in getattr(getattr(model,kind),"models",()):
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in core.parameters())


@pytest.mark.parametrize("name,variant",list(zip(NAMES,["original","projected-conv","projected-d4","projected-lse"])))
def test_wandb_tags_identify_experiment_variant_and_seed(monkeypatch,name,variant):
    import sys
    from types import SimpleNamespace
    from compressarc.train.logging import initialize_wandb
    config=load_config(SRC/"config/models"/(name+".yaml"))
    calls=[]
    run=SimpleNamespace(define_metric=lambda *a,**k:None)
    def initialize(**kwargs):
        calls.append(kwargs)
        return run
    monkeypatch.setitem(sys.modules,"wandb",SimpleNamespace(init=initialize,Settings=lambda **k:k))
    for seed in (0,1,2):
        config.training.seed=seed
        initialize_wandb(config,SimpleNamespace(task_name="74dd1130"))
        expected={"arc-agi","training","final-ablation","sample20","steps1500","strict",f"seed{seed}",variant}
        if variant=="projected-conv":
            expected.add("C4")  # Existing control is the rotation-only conv backend.
        assert set(calls[-1]["tags"])==expected
        assert calls[-1]["name"].endswith(f"-seed{seed}")
