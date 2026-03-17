

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from buffer import Experience


def policy_loss(token_log_probs: torch.Tensor, experience: Experience) -> torch.Tensor:
    """Policy loss term .

    Args:
        token_log_probs:
            Log-probabilities of sampled tokens.
        experience:
            Batch of rollout samples. Uses:
            - action_mask 
            - advantage 
            - group_total_response_len 
    """
    mask = experience.action_mask.to(token_log_probs.dtype)
    seq_logprob_sum = (token_log_probs * mask).sum(dim=-1) 

   
    advantages = experience.advantage.to(token_log_probs.dtype).squeeze(-1)
    normalizer = experience.group_total_response_len.to(token_log_probs.dtype).squeeze(-1)

    loss_per_sample = -(advantages * seq_logprob_sum) / normalizer
    return loss_per_sample.mean()


def kl_loss(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """KL loss term .

    Args:
        policy_logits: Current-policy logits.
        ref_logits: Reference-policy logits.
        action_mask: Mask of generated tokens.
    """
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    policy_probs = policy_log_probs.exp()

    token_kl = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)
    token_kl = token_kl * action_mask.to(token_kl.dtype)

    return token_kl.sum(dim=-1).mean()


@dataclass
class LossOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor


def compute_total_loss(
    token_log_probs: torch.Tensor,
    experience: Experience,
    beta: float = 1e-2,
    policy_logits: torch.Tensor | None = None,
    ref_logits: torch.Tensor | None = None,
) -> LossOutput:
    """Compute total loss = policy_loss + beta * kl_loss."""
    policy = policy_loss(token_log_probs=token_log_probs, experience=experience)

    kl = kl_loss(
            policy_logits=policy_logits,
            ref_logits=ref_logits,
            action_mask=experience.action_mask,
        )
    
    total = policy + beta * kl
    return LossOutput(loss=total, policy_loss=policy, kl_loss=kl)


__all__ = [
    "LossOutput",
    "policy_loss",
    "kl_loss",
    "compute_total_loss",
]
