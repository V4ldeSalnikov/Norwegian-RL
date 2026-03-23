"""Backward-compatible re-export of prompt templates.

Canonical location is `prompt.py`.
"""

from prompt import DANISH_REWARD_JUDGE_PROMPT_TEMPLATE, build_danish_reward_judge_prompt

__all__ = ["DANISH_REWARD_JUDGE_PROMPT_TEMPLATE", "build_danish_reward_judge_prompt"]
