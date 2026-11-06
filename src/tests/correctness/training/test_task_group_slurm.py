from pathlib import Path
import re
import shlex
from config import load_config
import run_training_timing as launcher

SRC=Path(__file__).resolve().parents[3]


def test_task_arrays_cover_each_task_variant_seed_once(monkeypatch):
    calls=[]
    monkeypatch.setattr(launcher.train,"main",lambda command:calls.append(command))
    monkeypatch.setattr(launcher.gc,"collect",lambda:None)
    monkeypatch.setattr(launcher.torch.cuda,"is_available",lambda:False)
    planned=set()
    all_tasks=[]
    for group in range(1,5):
        path=SRC/"slurm"/f"final_tasks_{group:02d}.sbatch"
        raw=path.read_bytes()
        assert b"\r" not in raw
        text=raw.decode()
        assert "#SBATCH --array=0-59%1" in text
        assert "#SBATCH --gres=gpu:A40:1" in text
        assert "#SBATCH --partition=gpu" in text
        assert "#SBATCH --qos=normal" in text
        assert "#SBATCH --account=vf38" in text
        assert '--shard-count 60 --shard-index "${SLURM_ARRAY_TASK_ID}"' in text
        tasks=shlex.split(re.search(r"^TASKS=\((.*?)\)$",text,re.MULTILINE).group(1))
        assert len(tasks)==len(set(tasks))==5
        all_tasks.extend(tasks)
        configs=[SRC.parent/config for config in re.findall(r"--config (\S+)",text)]
        assert configs==launcher.DEFAULT_CONFIGS
        for index in range(60):
            before=len(calls)
            command=["--mode","train","--shard-count","60","--shard-index",str(index)]
            for config in configs:command.extend(["--config",str(config)])
            for task in tasks:command.extend(["--task",task])
            launcher.main(command)
            assert len(calls)==before+1
            actual=calls[-1]
            task=actual[actual.index("--task")+1]
            config=Path(actual[actual.index("--config")+1])
            seed=int([x for x in actual if x.startswith("training.seed=")][-1].split("=")[1])
            assert task==tasks[index//12]
            assert config==configs[(index%12)//3]
            assert seed==index%3
            triple=(task,config.stem,seed)
            assert triple not in planned
            planned.add(triple)
    reference=load_config(launcher.DEFAULT_CONFIGS[0]).experiment.task_ids
    assert all_tasks==list(reference)
    assert len(planned)==240
