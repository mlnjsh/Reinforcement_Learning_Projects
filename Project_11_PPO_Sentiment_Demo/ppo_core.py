"""From-scratch PPO building blocks for language-model RLHF.

We deliberately avoid the `trl` library so the audience can see exactly:
  - how responses are sampled and their log-probabilities collected,
  - how the clipped surrogate objective is computed,
  - how the KL-to-reference penalty is added.

For demo simplicity, no value head — advantage = reward - batch_mean(reward).
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn


# =============================================================================
# 1. Manual generation + log-prob collection
# =============================================================================
@torch.no_grad()
def generate_with_logprobs(
    policy: nn.Module,
    input_ids: torch.Tensor,        # (B, T_q)
    attention_mask: torch.Tensor,   # (B, T_q)
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_k: int = 0,
    eos_token_id: int | None = None,
):
    """Sample `max_new_tokens` per sequence and record log-probs of sampled tokens.

    Returns:
        full_ids       : (B, T_q + T_r)  prompt followed by generated tokens
        response_ids   : (B, T_r)        just the generated part
        response_logp  : (B, T_r)        log π(a_t | s_<=t) for each sampled token
        response_mask  : (B, T_r)        1 = real token, 0 = past-EOS padding
    """
    B = input_ids.size(0)
    device = input_ids.device

    cur_ids = input_ids
    cur_mask = attention_mask
    sampled = []
    sampled_logp = []
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        out = policy(input_ids=cur_ids, attention_mask=cur_mask)
        logits = out.logits[:, -1, :] / max(temperature, 1e-6)  # (B, V)

        if top_k > 0:
            topk_vals, _ = torch.topk(logits, k=top_k, dim=-1)
            cutoff = topk_vals[:, -1, None]
            logits = torch.where(logits < cutoff, torch.full_like(logits, -1e10), logits)

        log_probs = F.log_softmax(logits, dim=-1)              # (B, V)
        probs = log_probs.exp()
        next_token = torch.multinomial(probs, num_samples=1)   # (B, 1)
        token_logp = log_probs.gather(-1, next_token).squeeze(-1)  # (B,)

        # Once a sequence has emitted EOS, freeze it: reuse last token, logp = 0.
        if eos_token_id is not None:
            token_logp = torch.where(finished, torch.zeros_like(token_logp), token_logp)
            next_token_safe = torch.where(
                finished.unsqueeze(-1), cur_ids[:, -1:], next_token
            )
        else:
            next_token_safe = next_token

        sampled.append(next_token_safe)
        sampled_logp.append(token_logp)

        cur_ids = torch.cat([cur_ids, next_token_safe], dim=1)
        cur_mask = torch.cat([cur_mask, (~finished).long().unsqueeze(-1)], dim=1)

        if eos_token_id is not None:
            finished = finished | (next_token_safe.squeeze(-1) == eos_token_id)
            if bool(finished.all()):
                break

    response_ids = torch.cat(sampled, dim=1)                   # (B, T_r)
    response_logp = torch.stack(sampled_logp, dim=1)           # (B, T_r)
    response_mask = (cur_mask[:, input_ids.size(1):]).float()  # (B, T_r) 1/0
    return cur_ids, response_ids, response_logp, response_mask


# =============================================================================
# 2. Recompute log-probs of an already-generated sequence under a model
#    (used for both the *current* policy during PPO updates, and the *reference*
#    model for the KL penalty)
# =============================================================================
def compute_response_logprobs(
    model: nn.Module,
    full_ids: torch.Tensor,         # (B, T_q + T_r)
    attention_mask: torch.Tensor,   # (B, T_q + T_r)
    response_len: int,
) -> torch.Tensor:
    """log π(token_t | tokens_<t) for each token in the response slice."""
    out = model(input_ids=full_ids, attention_mask=attention_mask)
    # logits[:, t, :] predicts token at position t+1.
    # Response tokens occupy positions [T_q, T_q+T_r); to score them we need
    # logits at positions [T_q-1, T_q+T_r-1).
    T = full_ids.size(1)
    T_q = T - response_len
    pred_logits = out.logits[:, T_q - 1 : T - 1, :]     # (B, T_r, V)
    target = full_ids[:, T_q:T]                          # (B, T_r)
    log_probs = F.log_softmax(pred_logits, dim=-1)
    return log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (B, T_r)


# =============================================================================
# 3. PPO update — clipped surrogate + KL-to-reference penalty
# =============================================================================
@dataclass
class PPOConfig:
    clip_range: float = 0.2
    kl_coef: float = 0.2
    epochs_per_step: int = 4
    learning_rate: float = 1e-5
    grad_clip: float = 1.0


@dataclass
class PPOStats:
    loss: float
    policy_loss: float
    kl: float
    mean_ratio: float
    clip_fraction: float
    mean_reward: float
    std_reward: float


def ppo_update(
    policy: nn.Module,
    ref_policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_ids: torch.Tensor,
    old_logprobs: torch.Tensor,        # (B, T_r) log-probs at sample time (frozen)
    response_mask: torch.Tensor,       # (B, T_r)
    rewards: torch.Tensor,             # (B,) scalar reward per response
    cfg: PPOConfig,
) -> PPOStats:
    """Run K epochs of PPO updates on a single rollout batch."""
    T_r = response_ids.size(1)

    # 1) advantages = centered rewards (no critic in this demo)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    advantages = advantages.unsqueeze(-1).expand(-1, T_r)            # (B, T_r)

    # 2) reference log-probs (frozen) — used for KL penalty
    with torch.no_grad():
        ref_logprobs = compute_response_logprobs(
            ref_policy, full_ids, attention_mask, T_r
        )

    last_stats = None
    for _ in range(cfg.epochs_per_step):
        new_logprobs = compute_response_logprobs(
            policy, full_ids, attention_mask, T_r
        )                                                            # (B, T_r)

        # Per-token policy ratio
        ratio = (new_logprobs - old_logprobs).exp()                  # (B, T_r)

        # Clipped surrogate (negative because we MINIMIZE)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * advantages
        surrogate = -torch.min(unclipped, clipped)                   # (B, T_r)
        policy_loss = (surrogate * response_mask).sum() / response_mask.sum().clamp(min=1)

        # KL(new || ref) approximated with (new_logp - ref_logp) per token
        kl_per_token = (new_logprobs - ref_logprobs)
        kl = (kl_per_token * response_mask).sum() / response_mask.sum().clamp(min=1)
        kl_loss = cfg.kl_coef * kl

        loss = policy_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        optimizer.step()

        with torch.no_grad():
            clip_frac = (
                ((ratio - 1.0).abs() > cfg.clip_range).float() * response_mask
            ).sum() / response_mask.sum().clamp(min=1)
            last_stats = PPOStats(
                loss=float(loss),
                policy_loss=float(policy_loss),
                kl=float(kl),
                mean_ratio=float((ratio * response_mask).sum() / response_mask.sum().clamp(min=1)),
                clip_fraction=float(clip_frac),
                mean_reward=float(rewards.mean()),
                std_reward=float(rewards.std()),
            )
    return last_stats


# =============================================================================
# 4. Reference-model utility
# =============================================================================
def make_reference_copy(policy: nn.Module) -> nn.Module:
    """Frozen deep copy of the policy, used for the KL penalty."""
    import copy
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    return ref
