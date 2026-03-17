"""
Rougly the logic is :
1) sample G responses per prompt from the policy,
2) score them with a judge,
3) compute group-normalized advantages,
4) optimize policy loss + beta * KL loss .5
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from buffer import Experience, ReplayBuffer, join_experiences_batch
from config import TrainConfig, load_config
from data import load_prompt_records
from loss import compute_total_loss
from reward_judge import JUDGE_MODEL_NAME, score_group as judge_score_group
from runtime_env import load_runtime_env


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def group_advantages(rewards: list[float], eps: float = 1e-8) -> torch.Tensor:
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    return (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std(unbiased=False) + eps)


def render_basic_chatml(messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
    """Chat ML template because our policy model doesnt have one"""
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        parts.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def format_user_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return render_basic_chatml(messages, add_generation_prompt=True)


@torch.no_grad()
def sample_group_responses(
    model,
    tokenizer,
    prompt: str,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[
    list[str],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[int],
]:
    prompt_text = format_user_prompt(tokenizer, prompt)

    model_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    #insert token type ids directly so the model doesnt break
    if "token_type_ids" not in model_inputs:
        model_inputs["token_type_ids"] = torch.zeros_like(model_inputs["input_ids"], dtype=torch.long)
    prompt_len = model_inputs["input_ids"].shape[1]

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    responses: list[str] = []
    sequence_list: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    action_masks: list[torch.Tensor] = []
    token_type_ids_list: list[torch.Tensor] = []
    response_lengths: list[int] = []

    for _ in range(group_size):
        generated = model.generate(
            **model_inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
        )

        sequence_ids = generated[0].detach().cpu()
        completion_ids = sequence_ids[prompt_len:]
        response_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        attention_mask = torch.ones(sequence_ids.size(0), dtype=torch.bool)
        action_mask = torch.zeros(sequence_ids.size(0) - 1, dtype=torch.bool)
        token_type_ids = torch.zeros(sequence_ids.size(0), dtype=torch.long)
        if prompt_len > 0 and action_mask.numel() >= prompt_len:
            action_mask[prompt_len - 1 :] = True

        responses.append(response_text)
        sequence_list.append(sequence_ids)
        attention_masks.append(attention_mask)
        action_masks.append(action_mask)
        token_type_ids_list.append(token_type_ids)
        response_lengths.append(int(completion_ids.numel()))

    return responses, sequence_list, attention_masks, action_masks, token_type_ids_list, response_lengths


def compute_token_log_probs_and_logits(
    model,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :].to(torch.float32)
    log_probs = F.log_softmax(logits, dim=-1)
    targets = sequence_ids[:, 1:].unsqueeze(-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=targets).squeeze(-1)
    return token_log_probs, logits


def save_checkpoint(model, tokenizer, output_dir: str, step: int, cfg: TrainConfig) -> None:
    ckpt_dir = Path(output_dir) / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    (ckpt_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))


def run_training(cfg: TrainConfig) -> None:
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        config=asdict(cfg),
    )

    set_seed(cfg.seed)
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "token_type_ids" not in tokenizer.model_input_names:
        tokenizer.model_input_names = list(tokenizer.model_input_names) + ["token_type_ids"]

    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=dtype).to(device)

    ref_name = cfg.reference_model_name or cfg.model_name
    ref_model = AutoModelForCausalLM.from_pretrained(ref_name, dtype=dtype).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )
    responses_per_step = cfg.prompts_per_step * cfg.group_size
    batches_per_step = math.ceil(responses_per_step / cfg.train_batch_size)
    optimizer_updates_per_step = math.ceil(batches_per_step / cfg.grad_accum_steps)
    total_optimizer_updates = max(1, cfg.steps * optimizer_updates_per_step)
    warmup_updates = int(total_optimizer_updates * cfg.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if warmup_updates <= 0:
            return 1.0
        if current_step < warmup_updates:
            return float(current_step + 1) / float(warmup_updates)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    prompt_records = load_prompt_records(
        dataset_path=cfg.dataset_path,
        dataset_name=cfg.dataset_name,
        dataset_split=cfg.dataset_split,
        dataset_max_samples=cfg.dataset_max_samples,
    )

    #Create replay buffer to store rollouts from policy
    replay_buffer = ReplayBuffer()

    print(f"Device: {device}")
    print(f"Policy model: {cfg.model_name}")
    print(f"Reference model: {ref_name}")
    print(f"Loaded prompts: {len(prompt_records)}")
    print(f"Judge model: {JUDGE_MODEL_NAME}")
    print(f"Parse-fail fallback score: {cfg.parse_fail_score}")
    print(
        f"Optimizer updates: total={total_optimizer_updates}, "
        f"warmup={warmup_updates} ({cfg.warmup_ratio:.2%})"
    )

    for step in range(cfg.steps):
        #set policy to the inference mode 
        model.eval()
        #clear buffer
        replay_buffer.clear()
        #sample generation prompts
        step_records = [random.choice(prompt_records) for _ in range(cfg.prompts_per_step)]
        step_rewards: list[float] = []
        step_parse_fail_count = 0

        for prompt_index, record in enumerate(step_records):
            # generate responses
            (
                responses,
                sequence_list,
                attention_masks,
                action_masks,
                token_type_ids_list,
                response_lengths,
            ) = sample_group_responses(
                model=model,
                tokenizer=tokenizer,
                prompt=record.prompt,
                group_size=cfg.group_size,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )

            #score the responses

            rewards, parse_fail_count, first_judge_response_text = judge_score_group(
                prompt=record.prompt,
                responses=responses,
                gold_response=record.gold_response,
                default_score=cfg.parse_fail_score,
            )
            step_parse_fail_count += parse_fail_count

            #calcualte group advantage
            advantages = group_advantages(rewards)

            # Sanity check - show sample reponse/judge evaluation

            first_response_text = responses[0] if responses else ""
            first_score_text = f"{rewards[0]:.2f}" if rewards else "n/a"
            print(f"\n=== Step {step + 1} | Prompt {prompt_index + 1}/{cfg.prompts_per_step} ===")
            print("Sampled policy response [0]:")
            print(first_response_text)
            print("\nJudge full response [0]:")
            print(first_judge_response_text)
            print(f"\nJudge score [0]: {first_score_text}/10")
            print("=== End Prompt ===\n")

            prompt_id = step * cfg.prompts_per_step + prompt_index
            group_total_len = float(sum(response_lengths))

            # Build experience per reposnse
            for i in range(cfg.group_size):
                experience = Experience(
                    sequence_ids=sequence_list[i],
                    attention_mask=attention_masks[i],
                    action_mask=action_masks[i],
                    token_type_ids=token_type_ids_list[i],
                    reward=torch.tensor(rewards[i], dtype=torch.float32),
                    advantage=advantages[i].detach().clone(),
                    prompt_id=torch.tensor(prompt_id, dtype=torch.long),
                    group_size=torch.tensor(cfg.group_size, dtype=torch.long),
                    response_len=torch.tensor(float(response_lengths[i]), dtype=torch.float32),
                    group_total_response_len=torch.tensor(group_total_len, dtype=torch.float32),
                    policy_version=torch.tensor(step, dtype=torch.long),
                )
                #add experience after scoring complete 
                replay_buffer.add(experience)
                step_rewards.append(float(rewards[i]))

        model.train()
        #Creat data loader from the buffer
        train_loader = DataLoader(
            dataset=replay_buffer.buffer,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=join_experiences_batch,
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        kl_loss_sum = 0.0

        for batch_idx, experience_batch in enumerate(train_loader):
            experience_batch = experience_batch.to(device)

            token_log_probs, policy_logits = compute_token_log_probs_and_logits(
                model=model,
                sequence_ids=experience_batch.sequence_ids,
                attention_mask=experience_batch.attention_mask,
                token_type_ids=experience_batch.token_type_ids,
            )

            with torch.no_grad():
                _, ref_logits = compute_token_log_probs_and_logits(
                    model=ref_model,
                    sequence_ids=experience_batch.sequence_ids,
                    attention_mask=experience_batch.attention_mask,
                    token_type_ids=experience_batch.token_type_ids,
                )

            # compute loss 
            loss_out = compute_total_loss(
                token_log_probs=token_log_probs,
                experience=experience_batch,
                beta=cfg.beta,
                policy_logits=policy_logits,
                ref_logits=ref_logits,
            )

            scaled_loss = loss_out.loss / cfg.grad_accum_steps
            scaled_loss.backward()

            total_loss_sum += float(loss_out.loss.item())
            policy_loss_sum += float(loss_out.policy_loss.item())
            kl_loss_sum += float(loss_out.kl_loss.item())

            if (batch_idx + 1) % cfg.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        # banch of metrics for logging

        num_batches = max(1, len(train_loader))
        avg_reward = sum(step_rewards) / max(1, len(step_rewards))
        avg_loss = total_loss_sum / num_batches
        avg_policy_loss = policy_loss_sum / num_batches
        avg_kl_loss = kl_loss_sum / num_batches
        parse_fail_rate = step_parse_fail_count / max(1, len(step_rewards))

        wandb.log(
            {
                "step": step + 1,
                "avg_reward": avg_reward,
                "loss": avg_loss,
                "policy_loss": avg_policy_loss,
                "kl_loss": avg_kl_loss,
                "buffer_size": len(replay_buffer),
                "lr": optimizer.param_groups[0]["lr"],
                "parse_fail_count": step_parse_fail_count,
                "parse_fail_rate": parse_fail_rate,
                "parse_fail_score": cfg.parse_fail_score,
            },
            step=step + 1,
        )

        if (step + 1) % cfg.print_every == 0:
            print(
                f"step {step + 1}/{cfg.steps} | "
                f"buffer={len(replay_buffer)} | "
                f"avg_reward={avg_reward:.3f} | "
                f"loss={avg_loss:.4f} | "
                f"policy={avg_policy_loss:.4f} | "
                f"kl={avg_kl_loss:.4f} | "
                f"parse_fail={step_parse_fail_count} ({parse_fail_rate:.2%})"
            )

        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            save_checkpoint(model, tokenizer, cfg.output_dir, step + 1, cfg)

    save_checkpoint(model, tokenizer, cfg.output_dir, cfg.steps, cfg)
    print(f"Training finished. Final checkpoint saved to {Path(cfg.output_dir) / f'step_{cfg.steps}'}")
    wandb.finish()


def main() -> None:
    load_runtime_env()
    cfg = load_config()
    run_training(cfg)


if __name__ == "__main__":
    main()
