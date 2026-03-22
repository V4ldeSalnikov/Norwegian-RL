"""Judge-based reward scoring using google/gemma-3-12b-it.

This module always uses the Gemma judge model and returns one score (1-10)
per candidate response.
"""

from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt import build_danish_reward_judge_prompt

JUDGE_MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"
JUDGE_TEMPERATURE = 0.2
JUDGE_MAX_NEW_TOKENS = 2048

_JUDGE_MODEL = None
_JUDGE_TOKENIZER = None


def _load_judge():
    global _JUDGE_MODEL, _JUDGE_TOKENIZER

    if _JUDGE_MODEL is not None and _JUDGE_TOKENIZER is not None:
        return _JUDGE_MODEL, _JUDGE_TOKENIZER

    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=False,
    )

    model.eval()
    _JUDGE_MODEL = model
    _JUDGE_TOKENIZER = tokenizer
    return model, tokenizer


def get_judge_model_and_tokenizer():
    """Public accessor so other modules can reuse the same singleton judge model."""
    return _load_judge()


def _build_judge_prompt(prompt: str, response: str, gold_response: str | None) -> str:
    payload = {
        "conversation_history": [
            {"role": "user", "content": prompt},
        ],
        "gold_response": gold_response or "",
        "ai_response": response,
    }
    return build_danish_reward_judge_prompt(payload)


def _extract_score(text: str, default_score: float) -> tuple[float, bool]:
    matches = list(re.finditer(r"(?:Score\s*:\s*)?(10|[1-9])\s*/\s*10", text))
    if matches:
        return float(matches[-1].group(1)), False
    return float(default_score), True


@torch.no_grad()
def _judge_one(
    prompt: str,
    response: str,
    gold_response: str | None,
    default_score: float,
) -> tuple[float, bool, str]:
    model, tokenizer = get_judge_model_and_tokenizer()

    user_prompt = _build_judge_prompt(prompt=prompt, response=response, gold_response=gold_response)

    chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    model_inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    prompt_len = model_inputs["input_ids"].shape[1]

    generated = model.generate(
        **model_inputs,
        do_sample=True,
        temperature=JUDGE_TEMPERATURE,
        top_p=1.0,
        max_new_tokens=JUDGE_MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id,
    )

    completion_ids = generated[0][prompt_len:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    score, used_default = _extract_score(completion_text, default_score=default_score)
    return score, used_default, completion_text


def score_group(
    prompt: str,
    responses: list[str],
    gold_response: str | None,
    default_score: float,
) -> tuple[list[float], int, str]:
    """Score a group of candidate responses with Gemma judge."""
    scores: list[float] = []
    parse_fail_count = 0
    first_judge_response_text = ""
    for response in responses:
        score, used_default, judge_response_text = _judge_one(
            prompt,
            response,
            gold_response,
            default_score,
        )
        scores.append(score)
        if not first_judge_response_text:
            first_judge_response_text = judge_response_text
        if used_default:
            parse_fail_count += 1
    return scores, parse_fail_count, first_judge_response_text
