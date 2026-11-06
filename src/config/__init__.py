"""Load the default and user-supplied OmegaConf configuration."""

from pathlib import Path

from omegaconf import OmegaConf


_DEFAULT_CONFIG = Path(__file__).with_name("default.yaml")


def load_config(config_path=None, overrides=None):
    """Merge defaults, an optional YAML file, and OmegaConf dot-list overrides."""
    config = OmegaConf.load(_DEFAULT_CONFIG)
    if config_path is not None:
        config = OmegaConf.merge(config, OmegaConf.load(Path(config_path)))
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    return config


__all__ = ["load_config"]
