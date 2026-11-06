"""Launch the normal training script on Modal."""

import base64
import shutil
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


def launch_modal(config, args):
    modal_cli = shutil.which("modal")
    if modal_cli is None:
        raise RuntimeError(
            "Modal backend selected, but the Modal CLI is not installed or not on PATH. "
            "Install it with pip install modal and run modal setup first."
        )

    src_root = Path(__file__).resolve().parents[2]
    config_yaml = OmegaConf.to_yaml(config, resolve=True)
    config_yaml_b64 = base64.b64encode(config_yaml.encode("utf-8")).decode("ascii")

    command = [
        modal_cli,
        "run",
        str(src_root / "modal_train.py"),
        "--config-yaml-b64",
        config_yaml_b64,
        "--output-dir",
        str(args.output_dir.expanduser().resolve()),
        "--gpu",
        args.modal_gpu,
    ]
    if args.task:
        command.extend(["--task", args.task])
    if args.save_checkpoints:
        command.append("--save-checkpoints")

    subprocess.run(command, cwd=src_root, check=True)