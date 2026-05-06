"""One-shot builder: produces Project_11_PPO_Sentiment_Demo.ipynb (Colab-ready).

Run once:  python build_notebook.py
Then delete this file — it's scaffolding, not part of the project.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "Project_11_PPO_Sentiment_Demo.ipynb"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src,
    }


CELLS = []

# ---------------------------------------------------------------- intro
CELLS.append(md(r"""# Project 11 — PPO + Reward Modeling: Positive Review Generator

A **from-scratch** implementation of Proximal Policy Optimization for RLHF-style
fine-tuning of a small language model. **No `trl` library** — every line of the
PPO loss (clipped surrogate + KL-to-reference penalty) is visible in plain
PyTorch.

> Take a pre-trained GPT-2 (`lvwerra/gpt2-imdb`) that writes neutral / mixed
> movie reviews. Use a pre-trained sentiment classifier (`lvwerra/distilbert-imdb`)
> as the reward model. After a handful of PPO steps, GPT-2 has been nudged
> toward writing **positive** reviews — *without changing the reward model
> at all*.

## Pipeline

```
+---------------+    prompt     +-----------+   response    +----------------+
|  IMDB         | ------------> |  Policy   | ------------> |   Reward       |
|  beginning    |               |  (GPT-2)  |               |  classifier    |
|  fragment     |               |  trained  |               | (DistilBERT)   |
+---------------+               +-----+-----+               +-------+--------+
                                      ^                             |
                                      |     scalar reward r         |
                                      +-----------------------------+
                                          (PPO update + KL penalty)
```

## The PPO loss

```
L_PPO = -E[ min( ρ_t · A_t , clip(ρ_t, 1-ε, 1+ε) · A_t ) ]   +   β · KL(π_new || π_ref)
```

- **`ρ_t = π_new(a_t | s_t) / π_old(a_t | s_t)`** — per-token policy ratio
- **clip** prevents one bad batch from destroying the policy
- **`β · KL`** controls cumulative drift from the original GPT-2

This notebook bundles together: `reward.py`, `ppo_core.py`, `ppo_demo.py`,
`plot_training.py`, `plot_distribution.py`, `show_comparison.py` from the
project directory."""))

# ---------------------------------------------------------------- install
CELLS.append(md("## 1. Install dependencies\n\nNo `trl` needed — just `transformers`, `torch`, `matplotlib`, `numpy`."))

CELLS.append(code("""!pip install -q transformers torch matplotlib numpy"""))

CELLS.append(md("## 2. Imports, seeds, device"))

CELLS.append(code(r"""# On some Windows + corporate-cert setups the HF download breaks. Safe to keep.
import os
for v in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    os.environ.pop(v, None)

import copy, json, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"device: {DEVICE}")
print(f"output dir: {OUTPUT_DIR.resolve()}")"""))

# ---------------------------------------------------------------- config
CELLS.append(md(r"""## 3. Configuration knobs

These are the dials you'll touch during a live demo. The most interesting one
is **`KL_COEF`** — it's the lever between *aggressive learning* (low KL) and
*staying close to the base model* (high KL). Reward hacking lives in the
low-KL regime."""))

CELLS.append(code(r"""POLICY_MODEL_NAME  = "lvwerra/gpt2-imdb"        # GPT-2 fine-tuned on IMDB
REWARD_MODEL_NAME  = "lvwerra/distilbert-imdb"  # 2-class sentiment classifier

NUM_PPO_STEPS      = 5      # bump to 20 for the published curves (~12 min CPU)
BATCH_SIZE         = 8
MAX_NEW_TOKENS     = 20
PROMPT_TOKENS      = 6      # short prompt = clear behavior change
NUM_EVAL_SAMPLES   = 8
SAMPLES_PER_PROMPT = 8      # for the BEFORE/AFTER reward distribution

# PPO hyperparameters
CLIP_RANGE       = 0.2
KL_COEF          = 0.2      # <-- the key knob to discuss live
EPOCHS_PER_STEP  = 4
LEARNING_RATE    = 1.0e-5
GRAD_CLIP        = 1.0"""))

# ---------------------------------------------------------------- reward
CELLS.append(md(r"""## 4. Reward model

We wrap `lvwerra/distilbert-imdb` (a 2-class IMDB sentiment classifier) and
turn its logits into a scalar reward. Three reasonable choices for the
*shaping function*, with different gradient behavior:

| Option | Formula | Pros | Cons |
|---|---|---|---|
| A | raw positive logit | simple | unbounded, can explode |
| B | positive probability | bounded `[0, 1]` | saturates → vanishing gradient |
| **C (default)** | **log-odds (`pos − neg`)** | symmetric, smooth, no saturation | none in practice |

The difference shows up clearly in the training curves; try swapping in (A) or
(B) after a baseline run to see reward hacking vs. saturation effects."""))

CELLS.append(code(r"""class RewardModel:
    \"\"\"Pre-trained sentiment classifier wrapped as a scalar reward function.\"\"\"

    def __init__(self, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME)
            .to(device)
            .eval()
        )
        self.device = device
        self.pos_idx = self.model.config.label2id.get("POSITIVE", 1)
        self.neg_idx = self.model.config.label2id.get("NEGATIVE", 0)

    @torch.no_grad()
    def score(self, texts):
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits  # (B, 2)
        return shape_reward(logits, pos_idx=self.pos_idx, neg_idx=self.neg_idx)


def shape_reward(logits, pos_idx=1, neg_idx=0):
    \"\"\"Convert classifier logits to a scalar reward signal.\"\"\"
    # A) raw logit
    # return logits[:, pos_idx]
    # B) probability
    # return torch.softmax(logits, dim=-1)[:, pos_idx]
    # C) log-odds (DEFAULT) — symmetric, smooth, no saturation
    return logits[:, pos_idx] - logits[:, neg_idx]"""))

CELLS.append(md("### Smoke-test the reward model"))

CELLS.append(code(r"""rm = RewardModel(device=DEVICE)
sample_texts = [
    "This movie was absolutely wonderful, I loved every minute of it!",
    "Total waste of time. The acting was terrible and the plot made no sense.",
    "It was okay, I guess. Not really memorable.",
]
scores = rm.score(sample_texts)
print("Reward scores (log-odds; > 0 means positive):")
for text, s in zip(sample_texts, scores.tolist()):
    print(f"  {s:+7.3f}  {text[:60]}")"""))

# ---------------------------------------------------------------- ppo core: generation
CELLS.append(md(r"""## 5. PPO core — manual generation with log-probs

In `trl` this is one line. Here we do it by hand so the audience can see how
log-probs of *sampled* tokens are stored at sampling time and reused later as
the **frozen `old_logprobs`** in the PPO ratio."""))

CELLS.append(code(r"""@torch.no_grad()
def generate_with_logprobs(
    policy, input_ids, attention_mask,
    max_new_tokens=20, temperature=1.0, top_k=0, eos_token_id=None,
):
    \"\"\"Sample max_new_tokens per sequence and record log-probs of sampled tokens.\"\"\"
    B = input_ids.size(0)
    device = input_ids.device

    cur_ids = input_ids
    cur_mask = attention_mask
    sampled, sampled_logp = [], []
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        out = policy(input_ids=cur_ids, attention_mask=cur_mask)
        logits = out.logits[:, -1, :] / max(temperature, 1e-6)

        if top_k > 0:
            topk_vals, _ = torch.topk(logits, k=top_k, dim=-1)
            cutoff = topk_vals[:, -1, None]
            logits = torch.where(logits < cutoff, torch.full_like(logits, -1e10), logits)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        next_token = torch.multinomial(probs, num_samples=1)
        token_logp = log_probs.gather(-1, next_token).squeeze(-1)

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

    response_ids = torch.cat(sampled, dim=1)
    response_logp = torch.stack(sampled_logp, dim=1)
    response_mask = (cur_mask[:, input_ids.size(1):]).float()
    return cur_ids, response_ids, response_logp, response_mask"""))

# ---------------------------------------------------------------- ppo core: logprob recompute
CELLS.append(md(r"""## 6. Recompute log-probs under a model

Used twice during a PPO step:
1. **Current policy `π_new`** — to form the ratio `π_new / π_old` for the
   clipped surrogate.
2. **Reference policy `π_ref`** — the frozen copy of the original GPT-2 for
   the KL penalty `KL(π_new || π_ref)`.

Index alignment is the only tricky bit: `logits[:, t, :]` predicts token `t+1`,
so to score the response slice `[T_q : T_q + T_r]` we need logits at
`[T_q - 1 : T - 1]`."""))

CELLS.append(code(r"""def compute_response_logprobs(model, full_ids, attention_mask, response_len):
    \"\"\"log π(token_t | tokens_<t) for each token in the response slice.\"\"\"
    out = model(input_ids=full_ids, attention_mask=attention_mask)
    T = full_ids.size(1)
    T_q = T - response_len
    pred_logits = out.logits[:, T_q - 1 : T - 1, :]
    target = full_ids[:, T_q:T]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    return log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)"""))

# ---------------------------------------------------------------- ppo update
CELLS.append(md(r"""## 7. The PPO update — clipped surrogate + KL penalty

This is the heart of the algorithm. Per inner epoch:

1. Recompute `new_logprobs` under the current policy.
2. Form the per-token ratio `ρ = exp(new_logp − old_logp)`.
3. Clipped surrogate: `min(ρ·A, clip(ρ, 1−ε, 1+ε)·A)` — the **clip** caps how
   far a single update can move per token.
4. KL penalty `β · (new_logp − ref_logp)` — keeps the policy from drifting
   far from the original GPT-2 over many steps.

> **Demo simplification**: no critic / value head. Advantage is just centered
> batch reward `(r − mean) / std`. Real production PPO uses a learned value
> head + GAE; we skip ~80 lines for clarity."""))

CELLS.append(code(r"""@dataclass
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
    policy, ref_policy, optimizer,
    full_ids, attention_mask, response_ids,
    old_logprobs, response_mask, rewards, cfg,
):
    T_r = response_ids.size(1)

    # advantages = centered rewards (no critic in this demo)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    advantages = advantages.unsqueeze(-1).expand(-1, T_r)

    # reference log-probs (frozen) for KL penalty
    with torch.no_grad():
        ref_logprobs = compute_response_logprobs(
            ref_policy, full_ids, attention_mask, T_r
        )

    last_stats = None
    for _ in range(cfg.epochs_per_step):
        new_logprobs = compute_response_logprobs(policy, full_ids, attention_mask, T_r)

        ratio = (new_logprobs - old_logprobs).exp()

        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * advantages
        surrogate = -torch.min(unclipped, clipped)
        policy_loss = (surrogate * response_mask).sum() / response_mask.sum().clamp(min=1)

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


def make_reference_copy(policy):
    \"\"\"Frozen deep copy of the policy, used for the KL penalty.\"\"\"
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    return ref


PPO_CFG = PPOConfig(
    clip_range=CLIP_RANGE, kl_coef=KL_COEF, epochs_per_step=EPOCHS_PER_STEP,
    learning_rate=LEARNING_RATE, grad_clip=GRAD_CLIP,
)"""))

# ---------------------------------------------------------------- helpers + prompts
CELLS.append(md(r"""## 8. Demo orchestration helpers

Short, hand-picked review-beginning prompts so the behavior change is obvious
in a few PPO steps. Long prompts make the change harder to see."""))

CELLS.append(code(r"""PROMPT_SEEDS = [
    "I watched this movie and",
    "The acting in this film was",
    "Honestly, this movie made me",
    "From start to finish, the story",
    "The director clearly wanted to",
    "I went into this expecting",
    "The cinematography in this one",
    "Every single scene felt",
]


def encode_prompts(tokenizer, prompts):
    enc = tokenizer(
        prompts, return_tensors="pt", padding="max_length",
        truncation=True, max_length=PROMPT_TOKENS,
    )
    return enc["input_ids"], enc["attention_mask"]


def decode_responses(tokenizer, full_ids, prompt_len):
    return [
        tokenizer.decode(full_ids[i, prompt_len:], skip_special_tokens=True)
        for i in range(full_ids.size(0))
    ]


def banner(title):
    print("\n" + "=" * 70 + f"\n  {title}\n" + "=" * 70)"""))

CELLS.append(code(r"""@torch.no_grad()
def evaluate(policy, tokenizer, reward_model, prompts, label, n_samples=1):
    \"\"\"Generate n_samples per prompt, score each, return grouped data.\"\"\"
    expanded = [p for p in prompts for _ in range(n_samples)]
    input_ids, attn_mask = encode_prompts(tokenizer, expanded)
    input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)

    full_ids, _, _, _ = generate_with_logprobs(
        policy, input_ids, attn_mask,
        max_new_tokens=MAX_NEW_TOKENS, temperature=1.0,
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
        order = sorted(range(n_samples), key=lambda j: sample_rewards[j])
        repr_idx = order[n_samples // 2]
        grouped.append({
            "prompt": p,
            "response": sample_responses[repr_idx],
            "reward": float(sample_rewards[repr_idx]),
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
    return grouped"""))

CELLS.append(code(r"""def train_ppo(policy, ref_policy, tokenizer, reward_model, num_steps):
    optimizer = AdamW(policy.parameters(), lr=PPO_CFG.learning_rate)
    log = []
    pad = tokenizer.pad_token_id

    for step in range(1, num_steps + 1):
        prompts = [PROMPT_SEEDS[(step * BATCH_SIZE + i) % len(PROMPT_SEEDS)]
                   for i in range(BATCH_SIZE)]
        input_ids, attn_mask = encode_prompts(tokenizer, prompts)
        input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)

        # 1) rollout
        policy.eval()
        full_ids, response_ids, old_logp, response_mask = generate_with_logprobs(
            policy, input_ids, attn_mask,
            max_new_tokens=MAX_NEW_TOKENS, temperature=1.0,
        )
        full_attn = (full_ids != pad).long().to(DEVICE)

        # 2) reward
        responses = decode_responses(tokenizer, full_ids, input_ids.size(1))
        full_texts = [p + r for p, r in zip(prompts, responses)]
        rewards = reward_model.score(full_texts).to(DEVICE)

        # 3) PPO update
        policy.train()
        stats = ppo_update(
            policy, ref_policy, optimizer,
            full_ids, full_attn, response_ids,
            old_logp, response_mask, rewards, PPO_CFG,
        )

        log.append({
            "step": step,
            "mean_reward": stats.mean_reward,
            "kl": stats.kl,
            "clip_fraction": stats.clip_fraction,
            "loss": stats.loss,
        })
        print(f"  step {step:3d}/{num_steps} | "
              f"reward {stats.mean_reward:+6.3f} ± {stats.std_reward:5.3f} | "
              f"KL {stats.kl:+6.4f} | "
              f"clip-frac {stats.clip_fraction:.2%} | "
              f"loss {stats.loss:+7.4f}")
    return log


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
          f"({'positive shift!' if avg > 0 else 'no improvement'})")"""))

# ---------------------------------------------------------------- load models
CELLS.append(md(r"""## 9. Load models

- **Policy**: `lvwerra/gpt2-imdb` — GPT-2 fine-tuned on IMDB movie reviews
  (~124M params). Trainable.
- **Reference**: a frozen deep copy of the policy at t=0. Used only for the
  KL penalty.
- **Reward**: `lvwerra/distilbert-imdb` — a 2-class sentiment classifier
  (~67M params). Frozen."""))

CELLS.append(code(r"""banner(f"Loading models on {DEVICE}")
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
print(f"  reward params : {sum(p.numel() for p in reward_model.model.parameters())/1e6:.1f}M")"""))

# ---------------------------------------------------------------- baseline
CELLS.append(md("## 10. Baseline (BEFORE PPO)\n\nSample 8 responses per prompt and score them."))

CELLS.append(code(r"""eval_prompts = PROMPT_SEEDS[:NUM_EVAL_SAMPLES]
before = evaluate(policy, tokenizer, reward_model, eval_prompts,
                  "BEFORE PPO", n_samples=SAMPLES_PER_PROMPT)
(OUTPUT_DIR / "before.json").write_text(json.dumps(before, indent=2))"""))

# ---------------------------------------------------------------- training
CELLS.append(md(r"""## 11. PPO training

Default is **5 steps** for fast iteration. For the published-quality curves,
set `NUM_PPO_STEPS = 20` in the config cell and re-run from the top.

Watch the **clip-fraction** — values around 40–60% mean PPO's clip is
genuinely doing work (preventing large per-token updates that would have
hurt the policy)."""))

CELLS.append(code(r"""banner(f"TRAINING — {NUM_PPO_STEPS} PPO steps, batch={BATCH_SIZE}, "
       f"KL_coef={PPO_CFG.kl_coef}, clip={PPO_CFG.clip_range}")
t0 = time.time()
log = train_ppo(policy, ref_policy, tokenizer, reward_model, NUM_PPO_STEPS)
print(f"\n  trained in {(time.time()-t0)/60:.1f} min")
(OUTPUT_DIR / "training_log.json").write_text(json.dumps(log, indent=2))"""))

# ---------------------------------------------------------------- after
CELLS.append(md("## 12. After PPO — same prompts, trained policy"))

CELLS.append(code(r"""after = evaluate(policy, tokenizer, reward_model, eval_prompts,
                 "AFTER PPO", n_samples=SAMPLES_PER_PROMPT)
(OUTPUT_DIR / "after.json").write_text(json.dumps(after, indent=2))

print_comparison(before, after)"""))

# ---------------------------------------------------------------- plot training
CELLS.append(md(r"""## 13. Training curves

Four panels: reward, KL drift from reference, clip-fraction, total loss.

- **Reward** should trend up.
- **KL** rising = the policy is moving away from the original GPT-2 — that's
  the cost of behavior change. If it grows unbounded, raise `KL_COEF`.
- **Clip fraction**: how often per-token ratios fell outside `[1−ε, 1+ε]`.
  High values (40%+) mean PPO's clip is actively saving you from instability.
- **Loss**: the surrogate + KL combination. Going down ≠ getting better
  (it's not a real likelihood); use **reward** as the success metric."""))

CELLS.append(code(r"""def plot_training_curves(log, before_mean, after_mean):
    steps     = [row["step"] for row in log]
    reward    = [row["mean_reward"] for row in log]
    kl        = [row["kl"] for row in log]
    clip_frac = [row["clip_fraction"] for row in log]
    loss      = [row["loss"] for row in log]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fig.suptitle("PPO + RLHF: training dynamics  (GPT-2-imdb, DistilBERT-imdb reward)",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(steps, reward, marker="o", color="#1f77b4", linewidth=2,
            label="mean reward (during PPO rollout)")
    if before_mean is not None:
        ax.axhline(before_mean, ls="--", color="#d62728", alpha=0.7,
                   label=f"BEFORE (held-out): {before_mean:+.2f}")
    if after_mean is not None:
        ax.axhline(after_mean, ls="--", color="#2ca02c", alpha=0.7,
                   label=f"AFTER  (held-out): {after_mean:+.2f}")
    ax.set_title("Mean reward per PPO step")
    ax.set_xlabel("PPO step"); ax.set_ylabel("log-odds(positive)")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)

    ax = axes[0, 1]
    ax.plot(steps, kl, marker="o", color="#ff7f0e", linewidth=2)
    ax.axhline(0, ls=":", color="gray", alpha=0.5)
    ax.set_title("KL(policy || reference)  —  drift from base model")
    ax.set_xlabel("PPO step"); ax.set_ylabel("nats / token (single-sample est.)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.bar(steps, [c * 100 for c in clip_frac], color="#9467bd", alpha=0.85)
    ax.axhline(50, ls=":", color="gray", alpha=0.5, label="50% (clip is heavily active)")
    ax.set_title("Clip fraction  —  share of tokens hitting the PPO clip")
    ax.set_xlabel("PPO step"); ax.set_ylabel("% of tokens")
    ax.set_ylim(0, 100); ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1, 1]
    ax.plot(steps, loss, marker="o", color="#2ca02c", linewidth=2)
    ax.axhline(0, ls=":", color="gray", alpha=0.5)
    ax.set_title("PPO loss (policy clipped surrogate + KL penalty)")
    ax.set_xlabel("PPO step"); ax.set_ylabel("loss")
    ax.grid(alpha=0.3)

    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=140, bbox_inches="tight")
    plt.show()


before_mean = sum(r["reward"] for r in before) / len(before)
after_mean  = sum(r["reward"] for r in after)  / len(after)
plot_training_curves(log, before_mean, after_mean)"""))

# ---------------------------------------------------------------- plot distribution
CELLS.append(md(r"""## 14. Reward distribution: BEFORE vs AFTER

Two panels:
- **Left**: per-prompt boxes with raw samples overlaid. Look for prompts that
  *failed* — they reveal prompt-specific failure modes worth discussing.
- **Right**: pooled across prompts — the population-level shift.

The diamond markers on the right are the means."""))

CELLS.append(code(r"""def _rewards_per_prompt(rows):
    out = []
    for r in rows:
        if "samples" in r:
            out.append([s["reward"] for s in r["samples"]])
        else:
            out.append([r["reward"]])
    return out


def plot_reward_distribution(before, after):
    BEFORE_COLOR = "#d62728"
    AFTER_COLOR  = "#2ca02c"

    n = len(before)
    prompts = [r["prompt"] for r in before]
    before_rewards = _rewards_per_prompt(before)
    after_rewards  = _rewards_per_prompt(after)

    fig = plt.figure(figsize=(14, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

    # ---- left: per-prompt boxes
    ax = fig.add_subplot(gs[0, 0])
    width = 0.35
    positions = np.arange(n)

    bp_b = ax.boxplot(before_rewards, positions=positions - width/2, widths=width,
                      patch_artist=True, showfliers=False,
                      medianprops=dict(color="black", linewidth=1.5))
    bp_a = ax.boxplot(after_rewards,  positions=positions + width/2, widths=width,
                      patch_artist=True, showfliers=False,
                      medianprops=dict(color="black", linewidth=1.5))
    for patch in bp_b["boxes"]:
        patch.set_facecolor(BEFORE_COLOR); patch.set_alpha(0.55)
    for patch in bp_a["boxes"]:
        patch.set_facecolor(AFTER_COLOR);  patch.set_alpha(0.55)

    rng = np.random.default_rng(0)
    for i in range(n):
        jitter = (rng.random(len(before_rewards[i])) - 0.5) * 0.18
        ax.scatter(np.full(len(before_rewards[i]), positions[i] - width/2) + jitter,
                   before_rewards[i], s=18, color=BEFORE_COLOR,
                   edgecolors="black", linewidths=0.4, alpha=0.85, zorder=3)
        jitter = (rng.random(len(after_rewards[i])) - 0.5) * 0.18
        ax.scatter(np.full(len(after_rewards[i]), positions[i] + width/2) + jitter,
                   after_rewards[i], s=18, color=AFTER_COLOR,
                   edgecolors="black", linewidths=0.4, alpha=0.85, zorder=3)

    ax.axhline(0, ls=":", color="gray", alpha=0.6)
    ax.set_xticks(positions)
    short_labels = [p if len(p) <= 22 else p[:20] + "..." for p in prompts]
    ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("reward (log-odds positive)")
    ax.set_title("Per-prompt reward distribution  (n=8 samples/prompt)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=BEFORE_COLOR, alpha=0.55, label="BEFORE PPO"),
        Patch(facecolor=AFTER_COLOR,  alpha=0.55, label="AFTER PPO"),
    ], loc="lower right")

    # ---- right: pooled
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
    flat_before = [v for lst in before_rewards for v in lst]
    flat_after  = [v for lst in after_rewards  for v in lst]
    bp = ax2.boxplot([flat_before, flat_after], positions=[0, 1], widths=0.55,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", linewidth=1.5))
    bp["boxes"][0].set_facecolor(BEFORE_COLOR); bp["boxes"][0].set_alpha(0.55)
    bp["boxes"][1].set_facecolor(AFTER_COLOR);  bp["boxes"][1].set_alpha(0.55)

    for i, data in enumerate([flat_before, flat_after]):
        jitter = (rng.random(len(data)) - 0.5) * 0.32
        ax2.scatter(np.full(len(data), i) + jitter, data, s=14,
                    color=[BEFORE_COLOR, AFTER_COLOR][i],
                    edgecolors="black", linewidths=0.3, alpha=0.7, zorder=3)

    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["BEFORE", "AFTER"])
    ax2.axhline(0, ls=":", color="gray", alpha=0.6)
    ax2.set_title("All prompts pooled", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    mean_b = np.mean(flat_before); mean_a = np.mean(flat_after)
    ax2.scatter([0], [mean_b], marker="D", s=70, color="white",
                edgecolors="black", zorder=4)
    ax2.scatter([1], [mean_a], marker="D", s=70, color="white",
                edgecolors="black", zorder=4)
    ax2.text(0, mean_b, f"  {mean_b:+.2f}", va="center", fontsize=9)
    ax2.text(1, mean_a, f"  {mean_a:+.2f}", va="center", fontsize=9)

    fig.suptitle("PPO + RLHF: reward distribution shift  (GPT-2-imdb, DistilBERT-imdb reward)",
                 fontsize=13, fontweight="bold")
    plt.savefig(OUTPUT_DIR / "reward_distribution.png", dpi=140, bbox_inches="tight")
    plt.show()


plot_reward_distribution(before, after)"""))

# ---------------------------------------------------------------- shaping comparison
CELLS.append(md(r"""## 15. (Optional) Reward-shaping comparison

Re-run **5 PPO steps from the same fresh GPT-2-imdb init** with each of the
three reward-shaping options, then overlay the curves to make the failure
modes visible.

| Option | Formula | Expected behavior |
|---|---|---|
| **A** raw positive logit | `logits[:, pos]` | unbounded — reward can explode, large KL drift |
| **B** probability | `softmax(logits)[:, pos]` | bounded `[0, 1]` — saturates → vanishing gradient → flat |
| **C** log-odds (default) | `logits[:, pos] − logits[:, neg]` | symmetric, smooth growth |

> **Cost:** ~5–8 min on CPU (3 fresh trainings × 5 steps each).
> Skip this section if you only wanted the headline result."""))

CELLS.append(code(r"""# Three shaping functions to compare
def _shape_A(logits, pos, neg):
    return logits[:, pos]                                # raw positive logit

def _shape_B(logits, pos, neg):
    return torch.softmax(logits, dim=-1)[:, pos]         # probability (saturates)

def _shape_C(logits, pos, neg):
    return logits[:, pos] - logits[:, neg]               # log-odds (default)

SHAPING_OPTIONS = [
    ("A_raw_logit",   _shape_A),
    ("B_probability", _shape_B),
    ("C_log_odds",    _shape_C),
]


@torch.no_grad()
def score_with_shape(rm, texts, shape_fn):
    enc = rm.tokenizer(
        texts, padding=True, truncation=True, max_length=512,
        return_tensors="pt",
    ).to(rm.device)
    logits = rm.model(**enc).logits
    return shape_fn(logits, rm.pos_idx, rm.neg_idx)


def train_one_shaping(shape_fn, num_steps=5, seed=SEED):
    \"\"\"Train a fresh policy from the original GPT-2-imdb weights with shape_fn.\"\"\"
    torch.manual_seed(seed)
    fresh = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_NAME).to(DEVICE)
    fresh_ref = make_reference_copy(fresh)
    optimizer = AdamW(fresh.parameters(), lr=PPO_CFG.learning_rate)
    pad = tokenizer.pad_token_id
    log = []

    for step in range(1, num_steps + 1):
        prompts = [PROMPT_SEEDS[(step * BATCH_SIZE + i) % len(PROMPT_SEEDS)]
                   for i in range(BATCH_SIZE)]
        input_ids, attn_mask = encode_prompts(tokenizer, prompts)
        input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)

        fresh.eval()
        full_ids, response_ids, old_logp, response_mask = generate_with_logprobs(
            fresh, input_ids, attn_mask,
            max_new_tokens=MAX_NEW_TOKENS, temperature=1.0,
        )
        full_attn = (full_ids != pad).long().to(DEVICE)

        responses = decode_responses(tokenizer, full_ids, input_ids.size(1))
        full_texts = [p + r for p, r in zip(prompts, responses)]
        rewards = score_with_shape(reward_model, full_texts, shape_fn).to(DEVICE)

        fresh.train()
        stats = ppo_update(
            fresh, fresh_ref, optimizer,
            full_ids, full_attn, response_ids,
            old_logp, response_mask, rewards, PPO_CFG,
        )
        log.append({
            "step": step,
            "mean_reward": stats.mean_reward,
            "kl": stats.kl,
            "clip_fraction": stats.clip_fraction,
        })
        print(f"  step {step}/{num_steps} | reward {stats.mean_reward:+7.3f} | "
              f"KL {stats.kl:+6.4f} | clip-frac {stats.clip_fraction:.2%}")

    del fresh, fresh_ref, optimizer
    return log


results = {}
for name, fn in SHAPING_OPTIONS:
    banner(f"Training with shaping = {name}")
    results[name] = train_one_shaping(fn, num_steps=5)
(OUTPUT_DIR / "shaping_comparison.json").write_text(json.dumps(results, indent=2))
print("\nsaved logs to outputs/shaping_comparison.json")"""))

CELLS.append(md(r"""### Plot the comparison

Three panels:

1. **Raw reward** — note the y-axis differs *in scale* across shapings (probability is bounded `[0, 1]`, the others aren't), so don't compare absolute values across colors.
2. **Reward growth (Δ from step 1)** — apples-to-apples *shape* of the trajectory.
3. **KL drift** — directly comparable. Big KL on (A) means the policy moved fast; tiny KL on (B) means the gradient flatlined.

What you should see: **(A)** climbs fastest with the largest KL drift (often
chasing reward-hacky tokens), **(B)** plateaus quickly because the gradient
vanishes once samples score near 1.0, **(C)** steady moderate growth with
moderate drift."""))

CELLS.append(code(r"""def plot_shaping_comparison(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    fig.suptitle("Reward shaping comparison — 5 PPO steps from same init",
                 fontsize=13, fontweight="bold")

    colors = {
        "A_raw_logit":   "#d62728",
        "B_probability": "#ff7f0e",
        "C_log_odds":    "#2ca02c",
    }

    # 1) raw reward (scales differ — don't compare absolute values across colors)
    ax = axes[0]
    for name, log in results.items():
        steps  = [r["step"] for r in log]
        reward = [r["mean_reward"] for r in log]
        ax.plot(steps, reward, marker="o", color=colors[name], linewidth=2, label=name)
    ax.set_title("Raw reward  (scales differ!)")
    ax.set_xlabel("PPO step"); ax.set_ylabel("reward")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # 2) reward growth (delta from step 1) — apples-to-apples
    ax = axes[1]
    for name, log in results.items():
        steps  = [r["step"] for r in log]
        reward = [r["mean_reward"] for r in log]
        delta  = [r - reward[0] for r in reward]
        ax.plot(steps, delta, marker="o", color=colors[name], linewidth=2, label=name)
    ax.axhline(0, ls=":", color="gray", alpha=0.5)
    ax.set_title("Reward growth  (Δ from step 1)")
    ax.set_xlabel("PPO step"); ax.set_ylabel("reward − reward[1]")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # 3) KL drift — same units across shapings, directly comparable
    ax = axes[2]
    for name, log in results.items():
        steps = [r["step"] for r in log]
        kl    = [r["kl"] for r in log]
        ax.plot(steps, kl, marker="o", color=colors[name], linewidth=2, label=name)
    ax.axhline(0, ls=":", color="gray", alpha=0.5)
    ax.set_title("KL drift from reference")
    ax.set_xlabel("PPO step"); ax.set_ylabel("nats / token")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    plt.savefig(OUTPUT_DIR / "shaping_comparison.png", dpi=140, bbox_inches="tight")
    plt.show()


plot_shaping_comparison(results)"""))


# ---------------------------------------------------------------- replay / takeaways
CELLS.append(md(r"""## 16. Educational takeaways

1. **PPO is just two things on top of REINFORCE**: a clipped policy ratio
   (per-batch safety net) and a KL-to-reference penalty (cumulative drift
   safety net).
2. **Reward hacking is real.** AFTER samples sometimes look like
   `' and or my History all my all and my I and the all my my I the All I my'` —
   the model learned the classifier likes certain tokens, not that humans
   like good prose. Motivates **harder reward models** and **stronger
   KL constraints**.
3. **Visible behavior change in <5 minutes on CPU.** The whole RLHF pipeline
   is now tractable on a laptop — which is what makes it teachable.

### What this demo intentionally does **not** do

- **No critic / value head.** Advantage is just centered batch reward.
  Production PPO uses a learned value head + GAE.
- **No real reward-model training.** We use the off-the-shelf
  `lvwerra/distilbert-imdb` and treat its log-odds as reward. Reward-model
  training from preference data is a separate (Project 01) topic.
- **No safety filtering.** Real RLHF pipelines include harmlessness
  classifiers, refusal training, and red-teaming.

### Things to try

| Change | Where | Expected effect |
|---|---|---|
| `NUM_PPO_STEPS = 20` | config cell | smoother curves, stronger positivity, more reward-hacking |
| `KL_COEF = 0.05` | config cell | aggressive learning, larger drift, gibberish risk |
| `KL_COEF = 1.0` | config cell | almost no movement — KL dominates the loss |
| Switch shaping to (B) | `shape_reward` | watch reward saturate near 1.0 → vanishing grad |
| Add your own prompts | `PROMPT_SEEDS` | see if the policy generalizes to new beginnings |"""))


# ---------------------------------------------------------------- assemble
notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# ---- the source strings used Python-escape `\"\"\"` for triple quotes.
# Before writing JSON, convert those back to real triple quotes inside cell sources.
for cell in CELLS:
    cell["source"] = cell["source"].replace('\\"\\"\\"', '"""')

OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT.resolve()}  ({OUT.stat().st_size/1024:.1f} KB, {len(CELLS)} cells)")
