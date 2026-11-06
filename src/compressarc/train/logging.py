import time

import torch

"""Optional per-step training metrics and per-run summaries."""


def initialize_wandb(config, task):
    options = config.logging
    if not options.wandb:
        return None
    import wandb
    variant = config.get("experiment", {}).get("name")
    run_name = f"{task.task_name}-{config.training.split}" + (f"-{variant}" if variant else "")
    run_name += f"-seed{config.training.seed}"
    tags = ["arc-agi", config.training.split,
            *list(options.get("wandb_tags", [])), f"seed{config.training.seed}"]
    if config.get("model", {}).get("shift_implementation") in ("tied_conv", "c4_tied_conv", "unfold", "triton"):
        tags.append("C4")
    if config.get("model", {}).get("shift_implementation") == "d4_tied_conv":
        if config.get("model", {}).get("direction_share_implementation") == "d4":
            tags.append("combined-d4-conv-sharing")
        else:
            tags.append("D4-conv")
    if config.get("model", {}).get("direction_share_implementation") == "vectorized":
        tags.append("direction-share-vectorized")
    if config.get("model", {}).get("direction_share_implementation") == "vectorized_packed":
        tags.append("direction-share-vectorized-packed")
    if config.get("model", {}).get("cummax_implementation") == "triton_hard":
        tags.append("cummax-triton-hard")
    if config.get("model", {}).get("cummax_implementation") == "triton":
        tags.append("cummax-triton-lse")
    if config.get("model", {}).get("softmax_implementation") == "fused":
        tags.append("softmax-fused")
    if config.get("training", {}).get("optimizer_implementation", "default") == "fused":
        tags.append("adam-fused")
    if config.get("training", {}).get("diagnostics_implementation", "reference") == "triton":
        tags.append("numerics-triton")
    sharing = config.get("model", {}).get("share_implementation", "primitives")
    if sharing != "primitives":
        tags.append("share-"+sharing)
    if config.get("training", {}).get("runtime_device_dispatch", "global") == "explicit":
        tags.append("dispatch-explicit")
    if config.get("model", {}).get("cummax_implementation") == "triton_hard_grouped":
        tags.append("cummax-triton-hard-grouped8")
    step_timing = config.get("training", {}).get("step_timing", "off")
    if config.get("training", {}).get("operator_profile", False):
        tags.append("operator-profile")
    elif step_timing != "off":
        tags.append(f"timing-{step_timing}")
    if config.training.get("cuda_graph", False):
        tags.append("cuda-graph-forward-backward")
    variant_tag = config.get("experiment", {}).get("variant")
    if variant_tag:
        tags.append(variant_tag)
    init_options = {}
    if options.get("log_config", False):
        from omegaconf import OmegaConf
        effective = OmegaConf.to_container(config, resolve=True)
        effective["task_id"] = task.task_name
        init_options["config"] = effective
    run = wandb.init(
        project=options.wandb_project, entity=options.wandb_entity,
        mode=options.wandb_mode, name=run_name, tags=list(dict.fromkeys(tags)),
        settings=wandb.Settings(disable_git=True, disable_code=True, x_disable_stats=True, console="off"),
        **init_options,
    )
    run.define_metric("train_step")
    run.define_metric("predictions/*", step_metric="train_step")
    run.define_metric("train/*", step_metric="train_step")
    run.define_metric("candidates/*", step_metric="train_step")
    run.define_metric("kernels/*", step_metric="train_step")
    run.define_metric("numerics/*", step_metric="train_step")
    run.define_metric("timing/*", step_metric="train_step")
    return run


def log_training_metrics(run, task, tracker, train_step, metrics):
    if run is None:
        return
    payload = {"train_step": train_step,
               "train/reconstruction_error": metrics["reconstruction_error"],
               "train/total_KL": metrics["kl"]}
    if task.solution_hash is not None:
        top1 = hash(tracker.solution_most_frequent) == task.solution_hash
        second = hash(tracker.solution_second_most_frequent) == task.solution_hash
        payload["predictions/top_1_correct"] = int(top1)
        payload["predictions/pass_2_correct"] = int(top1 or second)
        run.summary["top_1_correct"] = bool(top1)
        run.summary["pass_2_correct"] = bool(top1 or second)
    if tracker.first_correct_generation_step is not None:
        run.summary["candidates/first_correct_generation_step"] = tracker.first_correct_generation_step
    if (train_step + 1) % 50 == 0:
        payload.update(tracker.candidate_metrics())
    run.log(payload)


def log_final_candidates(run, tracker):
    if run is None:
        return
    import wandb
    table = wandb.Table(columns=["hash", "first_step", "occurrences", "is_correct"])
    for grid in tracker.ranked_candidates():
        record = tracker.candidate_records[grid]
        table.add_data(record["hash"], record["first_generation_step"],
                       record["occurrences"], record["correct"])
    run.log({"candidates/all_unique": table})
    run.summary.update(tracker.candidate_metrics())
    if tracker.first_correct_generation_step is not None:
        run.summary["candidates/first_correct_generation_step"] = tracker.first_correct_generation_step


def log_model_checkpoint(run, checkpoint_path):
    """Upload the saved final model before closing its W&B run."""
    if run is None:
        return
    import wandb
    artifact = wandb.Artifact(name=f"model-{run.id}", type="model")
    artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
    return run.log_artifact(artifact, aliases=["final", "latest"])


class RunSummary:
    """Measure one task from model construction through training/checkpoint saving."""

    def __init__(self, run, device):
        self.run = run
        self.device = device

    def __enter__(self):
        if self.run is None:
            return self
        self.started = time.perf_counter()
        self.run.summary["run/status"] = "running"
        self.cuda = self.device.type == "cuda"
        try:
            if self.cuda:
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise
        return self

    def __exit__(self, error_type, error, traceback):
        if self.run is None:
            return False
        status = "finished" if error_type is None else (
            "interrupted" if issubclass(error_type, (KeyboardInterrupt, SystemExit)) else "failed")
        sync_error = None
        try:
            if self.cuda:
                try:
                    torch.cuda.synchronize(self.device)
                except RuntimeError as failure:
                    if error_type is None:
                        status = "failed"
                        sync_error = failure
                self.run.summary["run/peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated(self.device) / 2**20
                self.run.summary["run/peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved(self.device) / 2**20
            self.run.summary["run/wall_clock_seconds"] = time.perf_counter() - self.started
            self.run.summary["run/status"] = status
        finally:
            self.run.finish(exit_code=0 if status == "finished" else 1)
        if sync_error is not None:
            raise sync_error
        return False
