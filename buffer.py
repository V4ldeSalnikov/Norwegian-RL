"""Experience buffer utilities for on-policy language-model RL.

This module is intentionally lightweight and self-contained so it can be used
by both `train.py` and `loss.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class Experience:
    """One rollout used for policy-gradient optimization.
    """

    sequence_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    token_type_ids: torch.Tensor | None = None


    reward: torch.Tensor | None = None
    advantage: torch.Tensor | None = None

    prompt_id: torch.Tensor | None = None
    group_size: torch.Tensor | None = None
    response_len: torch.Tensor | None = None
    group_total_response_len: torch.Tensor | None = None
    policy_version: torch.Tensor | None = None

    log_probs_old: torch.Tensor | None = None
    log_probs_ref: torch.Tensor | None = None
    values_old: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "Experience":
        """Move tensor fields to a device and return a new Experience."""
        moved: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            moved[f.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return Experience(**moved)


def _pad_1d_tensors(tensor_list: list[torch.Tensor], how: str = "start") -> torch.Tensor:
    """Pad a list of 1D tensors to equal length and stack.
    """
    if not tensor_list:
        raise ValueError("Cannot pad an empty tensor list.")
    if how not in {"start", "end"}:
        raise ValueError("`how` must be 'start' or 'end'.")

    max_len = max(t.size(0) for t in tensor_list)
    padded: list[torch.Tensor] = []
    for tensor in tensor_list:
        if tensor.ndim != 1:
            raise ValueError(f"Expected 1D tensors for padding, got shape={tuple(tensor.shape)}")
        pad_len = max_len - tensor.size(0)
        if pad_len < 0:
            raise ValueError("Negative padding encountered.")
        pad_cfg = (pad_len, 0) if how == "start" else (0, pad_len)
        padded.append(F.pad(tensor, pad_cfg))
    return torch.stack(padded, dim=0)


def _stack_or_pad_tensors(values: list[torch.Tensor]) -> torch.Tensor:
    """Stack tensors if shapes match, otherwise pad variable-length 1D tensors."""
    if all(v.shape == values[0].shape for v in values):
        return torch.stack(values, dim=0)

    if all(v.ndim == 1 for v in values):
        return _pad_1d_tensors(values, how="start")

    shapes = [tuple(v.shape) for v in values]
    raise ValueError(f"Cannot collate tensors with incompatible shapes: {shapes}")


def split_experience_batch(experience: Experience) -> list[Experience]:
    """Split a batched Experience into per-sample Experience objects.
    """
    if experience.sequence_ids.ndim == 1:
        return [experience]

    batch_size = experience.sequence_ids.size(0)
    per_item: list[dict[str, Any]] = [{} for _ in range(batch_size)]

    for f in fields(experience):
        value = getattr(experience, f.name)

        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                for i in range(batch_size):
                    per_item[i][f.name] = value
            else:
                if value.size(0) != batch_size:
                    raise ValueError(
                        f"Field '{f.name}' has batch dimension {value.size(0)} but expected {batch_size}."
                    )
                slices = torch.unbind(value, dim=0)
                for i in range(batch_size):
                    per_item[i][f.name] = slices[i]
        else:
            for i in range(batch_size):
                per_item[i][f.name] = value

    return [Experience(**item) for item in per_item]


def join_experiences_batch(experiences: list[Experience]) -> Experience:
    """Collate a list of per-sample Experience objects into one batched Experience.

    Rules:
    - All `None` -> `None`
    - All tensors -> stack if same shape, else pad variable 1D tensors
    - Numeric python scalars -> tensorized
    - Other values -> kept as python list
    """
    if not experiences:
        raise ValueError("Cannot join an empty experience list.")

    batch: dict[str, Any] = {}

    for f in fields(Experience):
        values = [getattr(exp, f.name) for exp in experiences]

        if all(v is None for v in values):
            batch[f.name] = None
            continue

        if any(v is None for v in values):
            raise ValueError(f"Mixed None/non-None values for field '{f.name}'.")

        if all(isinstance(v, torch.Tensor) for v in values):
            batch[f.name] = _stack_or_pad_tensors(values)  # type: ignore[arg-type]
            continue

        if all(isinstance(v, bool) for v in values):
            batch[f.name] = torch.tensor(values, dtype=torch.bool)
            continue

        if all(isinstance(v, int) for v in values):
            batch[f.name] = torch.tensor(values, dtype=torch.long)
            continue

        if all(isinstance(v, (int, float)) for v in values):
            batch[f.name] = torch.tensor(values, dtype=torch.float32)
            continue

        batch[f.name] = values

    return Experience(**batch)


class ReplayBuffer:
    """On policy buffer
    """

    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit
        self.buffer: list[Experience] = []

    def add(self, experience: Experience) -> None:
        """Add one experience to the buffer."""
        items = split_experience_batch(experience)
        self.buffer.extend(items)

        if self.limit is not None and len(self.buffer) > self.limit:
            overflow = len(self.buffer) - self.limit
            self.buffer = self.buffer[overflow:]

    def clear(self) -> None:
        """Remove all stored experiences."""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(self, idx: int) -> Experience:
        return self.buffer[idx]

    def state_dict(self) -> dict[str, Any]:
        return {"limit": self.limit, "buffer": self.buffer}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the buffer from `state_dict()` output."""
        self.limit = state.get("limit")
        self.buffer = state.get("buffer", [])
