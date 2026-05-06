"""End-to-end RLHF demo: GPT-2 + PPO + pre-trained sentiment reward model.

Run:
    python ppo_demo.py                  # full demo
    python ppo_demo.py --baseline-only  # just show before-training samples
    python ppo_demo.py --steps 10       # shorter training (faster on CPU)
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

from reward import RewardModel
from ppo_core import (
    PPOConfig,
    generate_with_logprobs,
    make_reference_copy,
    ppo_update,
)


# =============================================================================
# Configuration  (edit these for your demo)
# =============================================================================
POLICY_MODEL_NAME = "lvwerra/gpt2-imdb"        # GPT-2 fine-tuned on IMDB
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

NUM_PPO_STEPS = 20
BATCH_SIZE = 8
MAX_NEW_TOKENS = 20
PROMPT_TOKENS = 6                               # short prompt = clear behavior change
NUM_EVAL_SAMPLES = 8
SAMPLES_PER_PROMPT = 8                          # for the BEFORE/AFTER reward distribution

PPO_CFG = PPOConfig(
    clip_range=0.2,
    kl_coef=0.2,        # <-- the key knob to discuss live
    epochs_per_step=4,
    learning_rate=1.0e-5,
    grad_clip=1.0,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"


# Hand-picked prompt seeds (movie-review beginnings) — short so behavior
# change is obvious.
PROMPT_SEEDS = [
    "I watched this movie and",
    "The acting in this film was",
    "Honestly, this movie made me",
    "From start to finish, the story",
    "The director clearly wanted to",
    "I went into this expecting",
    "The cinematography in this one",
    "Every single scene felt",
]


# =============================================================================
# Helpers
# =============================================================================
def encode_prompts(tokenizer, prompts):
    """Pad/truncate prompts to the same fixed length so they batch cleanly."""
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=PROMPT_TOKENS,
    )
    return enc["input_ids"], enc["attention_mask"]


def decode_responses(tokenizer, full_ids, prompt_len):
    """Decode the response slice, skipping pad/eos."""
    return [
        tokenizer.decode(full_ids[i, prompt_len:], skip_special_tokens=True)
        for i in range(full_ids.size(0))
    ]


def banner(title: str):
    print("\n" + "=" * 70 + f"\n  {title}\n" + "=" * 70)


# =============================================================================
# Phase A — generate samples and score them with the reward model
# =============================================================================
@torch.no_grad()
def evaluate(policy, tokenizer, reward_model, prompts, label, n_samples: int = 1):
    """Generate `n_samples` responses per prompt, score each, return grouped data.

    Returns a list of dicts (one per prompt):
        {prompt, samples: [{response, reward}, ...], mean_reward, std_reward}
    The `samples` list lets us build a box-plot of the reward distribution.
    """
    # Tile each prompt n_samples times so they batch together
    expanded = [p for p in prompts for _ in range(n_samples)]
    input_ids, attn_mask = encode_prompts(tokenizer, expanded)
    input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)

    full_ids, _, _, _ = generate_with_logprobs(
        policy,
        input_ids,
        attn_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
    )
    responses = decode_responses(tokenizer, full_ids, input_ids.size(1))
    full_texts = [p + r for p, r in zip(expanded, responses)]
    rewards = reward_model.score(full_texts).cpu().tolist()

    grouped = []
    for i, p in enumerate(prompts):
        sl = slice(i * n_samples, (i + 1) * n_samples)
        sample_rewards = rewards[sl]
        sample_responses = responses[sl]
        mean = sum(sample_rewards) / n_samples
        std = (sum((r - mean) ** 2 for r in sample_rewards) / n_samples) ** 0.5
        # Pick a representative response: the median-reward sample
        order = sorted(range(n_samples), key=lambda j: sample_rewards[j])
        repr_idx = order[n_samples // 2]
        grouped.append({
            "prompt": p,
            "response": sample_responses[repr_idx],   # representative for printing
            "reward": float(sample_rewards[repr_idx]),# representative for printing
            "mean_reward": float(mean),
            "std_reward": float(std),
            "samples": [
                {"response": r, "reward": float(s)}
                for r, s in zip(sample_responses, sample_rewards)
            ],
        })

    print(f"\n--- {label} (n={n_samples} samples per prompt) ---")
    for g in grouped:
        sign = "+" if g["mean_reward"] >= 0 else " "
        print(f"  mean={sign}{g['mean_reward']:6.2f} std={g['std_reward']:5.2f} | "
              f"{g['prompt']!r} -> repr: {g['response']!r}")
    overall = sum(g["mean_reward"] for g in grouped) / len(grouped)
    print(f"  overall mean reward: {overall:+.3f}")
    return grouped


# =============================================================================
# Phase B — PPO training loop
# =============================================================================
def train_ppo(policy, ref_policy, tokenizer, reward_model, num_steps):
    optimizer = AdamW(policy.parameters(), lr=PPO_CFG.learning_rate)

    log = []
    pad = tokenizer.pad_token_id

    for step in range(1, num_steps + 1):
        # Sample a batch of prompts (cycle through PROMPT_SEEDS)
        prompts = [PROMPT_SEEDS[(step * BATCH_SIZE + i) % len(PROMPT_SEEDS)]
                   for i in range(BATCH_SIZE)]
        input_ids, attn_mask = encode_prompts(tokenizer, prompts)
        input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)

        # 1) Rollout: sample responses + record old log-probs
        policy.eval()
        full_ids, response_ids, old_logp, response_mask = generate_with_logprobs(
            policy, input_ids, attn_mask,
            max_new_tokens=MAX_NEW_TOKENS, temperature=1.0,
        )
        full_attn = (full_ids != pad).long().to(DEVICE)

        # 2) Reward
        responses = decode_responses(tokenizer, full_ids, input_ids.size(1))
        full_texts = [p + r for p, r in zip(prompts, responses)]
        rewards = reward_model.score(full_texts).to(DEVICE)

        # 3) PPO update (K inner epochs of clipped surrogate + KL penalty)
        policy.train()
        stats = ppo_update(
            policy, ref_policy, optimizer,
            full_ids, full_attn, response_ids,
            old_logp, response_mask, rewards,
            PPO_CFG,
        )

        log.append({
            "step": step,
            "mean_reward": stats.mean_reward,
            "kl": stats.kl,
            "clip_fraction": stats.clip_fraction,
            "loss": stats.loss,
        })
        print(
            f"  step {step:3d}/{num_steps} | "
            f"reward {stats.mean_reward:+6.3f} ± {stats.std_reward:5.3f} | "
            f"KL {stats.kl:+6.4f} | "
            f"clip-frac {stats.clip_fraction:.2%} | "
            f"loss {stats.loss:+7.4f}"
        )
    return log


# =============================================================================
# Phase C — side-by-side rendering
# =============================================================================
def print_comparison(before, after):
    banner("BEFORE vs AFTER (same prompts)")
    print(f"{'PROMPT':<30}  {'r_before':>9}  {'r_after':>9}  {'delta':>7}")
    print("-" * 90)
    deltas = []
    for b, a in zip(before, after):
        d = a["reward"] - b["reward"]
        deltas.append(d)
        print(f"{b['prompt'][:28]!r:<30}  {b['reward']:+9.3f}  {a['reward']:+9.3f}  {d:+7.3f}")
        print(f"    BEFORE: {b['response']!r}")
        print(f"    AFTER : {a['response']!r}")
        print()
    avg = sum(deltas) / len(deltas)
    print(f"average reward shift: {avg:+.3f}   "
          f"({'positive shift!' if avg > 0 else 'no improvement'})")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true",
                        help="only run baseline + reward scoring, skip training")
    parser.add_argument("--steps", type=int, default=NUM_PPO_STEPS,
                        help="number of PPO training steps")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    banner(f"Loading models on {DEVICE}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # important for causal LMs

    policy = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_NAME).to(DEVICE)
    ref_policy = make_reference_copy(policy)
    reward_model = RewardModel(device=DEVICE)
    print(f"  loaded in {time.time() - t0:.1f}s")
    print(f"  policy params : {sum(p.numel() for p in policy.parameters())/1e6:.1f}M")
    print(f"  reward params : {sum(p.numel() for p in reward_model.model.parameters())/1e6:.1f}M")

    # ----- BEFORE -----
    banner("BASELINE — random IMDB-style GPT-2 (untrained policy)")
    eval_prompts = PROMPT_SEEDS[:NUM_EVAL_SAMPLES]
    before = evaluate(policy, tokenizer, reward_model, eval_prompts,
                      "BEFORE PPO", n_samples=SAMPLES_PER_PROMPT)
    (OUTPUT_DIR / "before.json").write_text(json.dumps(before, indent=2))

    if args.baseline_only:
        print("\n--baseline-only set; stopping.")
        return

    # ----- TRAIN -----
    banner(f"TRAINING — {args.steps} PPO steps, batch={BATCH_SIZE}, "
           f"KL_coef={PPO_CFG.kl_coef}, clip={PPO_CFG.clip_range}")
    t0 = time.time()
    log = train_ppo(policy, ref_policy, tokenizer, reward_model, args.steps)
    print(f"\n  trained in {(time.time()-t0)/60:.1f} min")
    (OUTPUT_DIR / "training_log.json").write_text(json.dumps(log, indent=2))

    # ----- AFTER -----
    banner("AFTER PPO — same prompts, trained policy")
    after = evaluate(policy, tokenizer, reward_model, eval_prompts,
                     "AFTER PPO", n_samples=SAMPLES_PER_PROMPT)
    (OUTPUT_DIR / "after.json").write_text(json.dumps(after, indent=2))

    # ----- COMPARE -----
    print_comparison(before, after)

    # ----- PLOT 1: training curves -----
    try:
        from plot_training import load_data, plot
        log, b, a = load_data()
        plot(log, b, a, OUTPUT_DIR / "training_curves.png")
        print(f"\nTraining curves saved to: "
              f"{(OUTPUT_DIR / 'training_curves.png').resolve()}")
    except Exception as e:
        print(f"\n(training-curves plot skipped: {e})")

    # ----- PLOT 2: per-prompt reward distribution -----
    try:
        from plot_distribution import plot_distribution
        plot_distribution(before, after, OUTPUT_DIR / "reward_distribution.png")
        print(f"Reward distribution saved to: "
              f"{(OUTPUT_DIR / 'reward_distribution.png').resolve()}")
    except Exception as e:
        print(f"(distribution plot skipped: {e})")

    print(f"\nAll artifacts saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
