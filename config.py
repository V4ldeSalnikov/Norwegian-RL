"""Training configuration schema and loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    reference_model_name: str | None = None
    dataset_path: str | None = None
    dataset_name: str = "V4ldeLund/no_robots_da_translated"
    dataset_split: str = "train"
    dataset_max_samples: int | None = 128
    output_dir: str = "checkpoints"

    steps: int = 500
    prompts_per_step: int = 16
    group_size: int = 8

    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0

    learning_rate: float = 1e-6
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_eps: float = 1e-8
    warmup_ratio: float = 0.1
    beta: float = 1e-2
    train_batch_size: int = 128
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0

    seed: int = 42

    save_every: int = 25
    print_every: int = 1
    parse_fail_score: float = 3.0
    wandb_project: str = "DanishRL"
    wandb_run_name: str = "run-1 test"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> TrainConfig:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Create this file before running training."
        )
    raw = yaml.safe_load(config_path.read_text()) or {}
    return TrainConfig(**raw)


__all__ = ["TrainConfig", "load_config", "DEFAULT_CONFIG_PATH"]
