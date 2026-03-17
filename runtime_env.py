"""Runtime environment loading helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _set_if_missing(target_key: str, source_key: str) -> None:
    value = os.getenv(source_key)
    if value and not os.getenv(target_key):
        os.environ[target_key] = value


def load_runtime_env(dotenv_path: str | None = None) -> None:
    """Load .env and normalize common secret variable names."""
    path = Path(dotenv_path) if dotenv_path else Path(".env")
    load_dotenv(dotenv_path=path, override=False)

    # Hugging Face aliases
    _set_if_missing("HF_TOKEN", "HUGGINGFACE_TOKEN")
    _set_if_missing("HF_TOKEN", "HUGGINGFACE_API_KEY")
    _set_if_missing("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN")
    _set_if_missing("HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")

    # Weights & Biases aliases
    _set_if_missing("WANDB_API_KEY", "WANDB_KEY")
