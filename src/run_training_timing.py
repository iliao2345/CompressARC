"""Shared launcher: call train.main for full training or timing runs."""
import argparse
import gc
import shlex
from pathlib import Path

import torch
import train
from config import load_config

MODEL_CONFIG_DIR = Path(__file__).resolve().parent / "config" / "models"
DEFAULT_CONFIGS = [MODEL_CONFIG_DIR / filename for filename in (
    "final_original_20.yaml", "final_projected_conv_20.yaml",
    "final_projected_d4_20.yaml", "final_projected_lse_20.yaml",
)]

DEFAULT_TASKS = ["694f12f3", "760b3cac", "94f9d214", "3428a4f5", "dae9d2b5"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "benchmark"), default="benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Print planned calls without executing training")
    parser.add_argument("--config", type=Path, action="append",
                        help="Model YAML; repeat to run several. Defaults to the four final controlled 20-task configs.")
    parser.add_argument("--task", action="append", help="Repeat for multiple real ARC task IDs")
    parser.add_argument("--seed", type=int, action="append", help="Training seed; repeat for a seed sweep")
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--measured-iterations", type=int, default=30)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--clean-only", action="store_true", help="Skip the separate operation-profile pass")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Split task/config/pass jobs across this many independent workers")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Zero-based worker index; each job is assigned to exactly one worker")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save final .pt weights and upload them to W&B in train mode")
    parser.add_argument("--output-dir", type=Path,
                        help="Checkpoint directory; defaults to outputs/training in the trainer")
    args = parser.parse_args(argv)
    if args.save_checkpoints and args.mode != "train":
        parser.error("--save-checkpoints requires --mode train")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Require shard-count >= 1 and 0 <= shard-index < shard-count")
    configs = args.config or DEFAULT_CONFIGS
    config_tasks = {}
    config_seeds = {}
    seed_directories = {}
    for config_path in configs:
        if not config_path.is_file():
            parser.error(f"Configuration does not exist: {config_path}")
        config = load_config(config_path, args.overrides)
        selected = args.task or config.get("experiment", {}).get("task_ids", DEFAULT_TASKS)
        if isinstance(selected, str) or not selected:
            parser.error("experiment.task_ids must be a nonempty list")
        config_tasks[config_path] = [str(task) for task in selected]
        seed_override = any(item.split("=", 1)[0] == "training.seed" for item in args.overrides)
        seeds = args.seed if args.seed is not None else (
            [config.training.seed] if seed_override else
            config.get("experiment", {}).get("seeds", [config.training.seed]))
        if isinstance(seeds, (str, int)) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
            parser.error("Seeds must be a nonempty list of nonnegative integers")
        if len(set(seeds)) != len(seeds):
            parser.error("Training seeds must be distinct")
        config_seeds[config_path] = list(seeds)
        seed_directories[config_path] = args.seed is not None or "seeds" in config.get("experiment", {})
    # Preserve explicit duplicate --task flags for cold/warm repeat experiments.
    tasks = args.task or list(dict.fromkeys(
        task for config_path in configs for task in config_tasks[config_path]
    ))
    passes = ["train"] if args.mode == "train" else (
        ["clean"] if args.clean_only else ["clean", "operations"]
    )
    job_index = 0
    for task in tasks:
        for config_path in configs:
            if task not in config_tasks[config_path]:
                continue
            for seed in config_seeds[config_path]:
                for pass_name in passes:
                    assigned_index = job_index
                    job_index += 1
                    if assigned_index % args.shard_count != args.shard_index:
                        continue
                    print(f"{task} / {config_path.stem} / seed {seed} / {pass_name}", flush=True)
                    command = ["--mode", args.mode, "--task", task]
                    if args.mode == "benchmark":
                        command.extend(["--benchmark-pass", pass_name,
                                        "--warmup-iterations", str(args.warmup_iterations),
                                        "--measured-iterations", str(args.measured_iterations)])
                    command.extend(["--config", str(config_path)])
                    if args.save_checkpoints:
                        command.append("--save-checkpoints")
                    output_dir = args.output_dir
                    if args.mode == "train" and (len(configs) > 1 or args.shard_count > 1):
                        base = output_dir or Path(__file__).resolve().parents[1] / "outputs/training"
                        output_dir = base / config_path.stem
                    if args.mode == "train" and seed_directories[config_path]:
                        base = output_dir or Path(__file__).resolve().parents[1] / "outputs/training"
                        output_dir = base / f"seed_{seed}"
                    if output_dir is not None:
                        command.extend(["--output-dir", str(output_dir)])
                    for override in args.overrides:
                        command.extend(["--set", override])
                    command.extend(["--set", f"training.seed={seed}"])
                    if args.dry_run:
                        print("python " + shlex.join([str(Path(train.__file__)), *command]))
                        continue
                    # Every task enters the same trainer with its configured seed.
                    train.main(command)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()



if __name__ == "__main__":
    main()
