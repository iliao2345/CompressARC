"""Modal entrypoint for running the normal src/train.py workflow."""

import base64
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

import modal


SRC_ROOT = Path(__file__).resolve().parent

REMOTE_SRC_ROOT = PurePosixPath("/root/compressarc/src")
REMOTE_OUTPUT_ROOT = PurePosixPath("/tmp/compressarc-checkpoints")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(SRC_ROOT / "requirements.txt"))
    .env({"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        SRC_ROOT,
        remote_path=str(REMOTE_SRC_ROOT),
        ignore=["**/__pycache__/**", "*.pyc"],
    )
)

app = modal.App("compressarc-training")


@app.function(image=image, timeout=24 * 60 * 60, max_containers=1)
def run_training(config_yaml_b64: str, task: str, save_checkpoints: bool):
    """Run the standard trainer with the supplied resolved config."""
    config_yaml = base64.b64decode(config_yaml_b64).decode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        encoding="utf-8",
        delete=False,
    ) as config_file:
        config_file.write(config_yaml)
        config_path = Path(config_file.name)

    command = [
        sys.executable,
        str(REMOTE_SRC_ROOT / "train.py"),
        "--config",
        str(config_path),
    ]
    if task:
        command.extend(["--task", task])
    if save_checkpoints:
        command.extend(
            [
                "--save-checkpoints",
                "--output-dir",
                str(REMOTE_OUTPUT_ROOT),
            ]
        )

    try:
        subprocess.run(command, cwd=REMOTE_SRC_ROOT, check=True)
    finally:
        config_path.unlink(missing_ok=True)

    if not save_checkpoints:
        return []

    return [
        (path.name, path.read_bytes())
        for path in sorted(Path(REMOTE_OUTPUT_ROOT).glob("*.pt"))
    ]


@app.local_entrypoint()
def main(
    config_yaml_b64: str,
    output_dir: str,
    gpu: str = "T4",
    task: str = "",
    save_checkpoints: bool = False,
):
    """Submit one configured training run and copy checkpoints back locally."""
    import yaml

    config = yaml.safe_load(base64.b64decode(config_yaml_b64))
    secrets = []
    if config.get("logging", {}).get("wandb", False):
        secrets.append(modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"]))
    checkpoint_files = run_training.with_options(
        gpu=gpu,
        secrets=secrets,
    ).remote(config_yaml_b64, task, save_checkpoints)

    if checkpoint_files:
        local_output_dir = Path(output_dir)
        local_output_dir.mkdir(parents=True, exist_ok=True)
        for filename, contents in checkpoint_files:
            safe_name = Path(filename).name
            if safe_name != filename:
                raise ValueError(f"Unexpected checkpoint filename: {filename!r}")
            (local_output_dir / safe_name).write_bytes(contents)
        print(f"Saved {len(checkpoint_files)} checkpoint(s) to {local_output_dir}")