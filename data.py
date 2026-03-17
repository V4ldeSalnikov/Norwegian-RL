"""Dataset loading utilities for prompt/gold-response records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset


@dataclass
class PromptRecord:
    prompt: str
    gold_response: str | None = None


def _extract_gold_from_messages(messages: list[dict[str, Any]] | None) -> str | None:
    if not messages:
        return None
    assistants = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
    if not assistants:
        return None
    return str(assistants[-1])


def _parse_local_row(row: dict[str, Any]) -> PromptRecord:
    prompt = (
        row.get("prompt")
        or row.get("question")
        or row.get("instruction")
        or row.get("user")
        or ""
    )
    gold = row.get("gold_response") or row.get("answer") or row.get("reference")
    if gold is None and isinstance(row.get("messages"), list):
        gold = _extract_gold_from_messages(row.get("messages"))
    return PromptRecord(prompt=str(prompt), gold_response=None if gold is None else str(gold))


def _load_local_prompt_records(path: str) -> list[PromptRecord]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}. "
            "Run `python translate.py` to build the translated dataset first."
        )

    rows: list[dict[str, Any]] = []
    if file_path.suffix.lower() == ".jsonl":
        for line in file_path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    else:
        content = json.loads(file_path.read_text())
        if isinstance(content, list):
            rows = content
        else:
            raise ValueError("Expected a JSON list for local dataset file.")

    records = [_parse_local_row(row) for row in rows]
    records = [r for r in records if r.prompt.strip()]
    if not records:
        raise ValueError("Local dataset produced zero valid prompt records.")
    return records


def _load_no_robots_records(
    dataset_name: str,
    dataset_split: str,
    max_samples: int | None,
) -> list[PromptRecord]:
    ds = load_dataset(dataset_name, split=dataset_split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    records: list[PromptRecord] = []
    for row in ds:
        prompt = str(row.get("prompt", "")).strip()
        messages = row.get("messages")
        gold = _extract_gold_from_messages(messages if isinstance(messages, list) else None)
        if prompt:
            records.append(PromptRecord(prompt=prompt, gold_response=gold))

    if not records:
        raise ValueError(
            f"No valid records loaded from dataset={dataset_name}, split={dataset_split}."
        )
    return records


def load_prompt_records(
    *,
    dataset_path: str | None,
    dataset_name: str,
    dataset_split: str,
    dataset_max_samples: int | None,
) -> list[PromptRecord]:

    if dataset_path:
        records = _load_local_prompt_records(dataset_path)
    else:
        records = _load_no_robots_records(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            max_samples=dataset_max_samples,
        )

    return records
