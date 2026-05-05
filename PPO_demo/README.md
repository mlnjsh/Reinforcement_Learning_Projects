# PPO + RLHF Demo — Positive Review Generator

This demo shows the **RLHF (Reinforcement Learning from Human Feedback) pipeline in
miniature**, using PPO (Proximal Policy Optimization).

## What you'll see

A pre-trained GPT-2 (`lvwerra/gpt2-imdb`) starts out writing **neutral or negative**
movie reviews. After ~20 PPO steps, the same model writes **positive** reviews.
The reward signal comes from a **pre-trained sentiment classifier**
(`lvwerra/distilbert-imdb`) — no reward model training required.

```
+-------------+     prompt      +------------+     response       +--------------+
|  IMDB       | --------------> |  Policy    | -----------------> |   Reward     |
|  beginning  |                 |  (GPT-2)   |                    |  classifier  |
|  fragment   |                 |            |                    | (DistilBERT) |
+-------------+                 +-----+------+                    +------+-------+
                                      ^                                  |
                                      |     scalar reward r              |
                                      +----------------------------------+
                                            (PPO update + KL penalty)
```

## The PPO loss (what the audience should remember)

For each (prompt, response) pair we compute three things:

1. **Reward** `r` — scalar from the classifier (high if response is positive sentiment)
2. **Policy ratio** `ρ_t = π_new(a_t|s_t) / π_old(a_t|s_t)` per token
3. **KL penalty** `β · KL(π_new || π_ref)` to keep the model close to the original

PPO minimizes:

```
L_PPO = -E[ min( ρ_t · A_t , clip(ρ_t, 1-ε, 1+ε) · A_t ) ]   +   β · KL(π_new || π_ref)
```

The `clip` is the heart of PPO — it stops one bad batch from destroying the policy.

## Files

- `reward.py`        Reward model wrapper + shaping function
- `ppo_core.py`      Manual generation w/ logprobs, PPO update step
- `ppo_demo.py`      Main script: baseline → train → after, side-by-side print
- `outputs/`         Saved JSON of before/after samples (for replay)

## How to run

```bash
# One-time: install dependencies
pip install transformers torch accelerate

# Full demo (downloads ~600MB on first run, then trains)
python ppo_demo.py

# Quick test (skip training, just baseline samples + reward scores)
python ppo_demo.py --baseline-only
```

## Demo-time knobs (edit at top of `ppo_demo.py`)

| Knob | Default | Effect |
|------|---------|--------|
| `NUM_PPO_STEPS` | 20 | More steps = stronger positivity, but mode-collapse risk |
| `KL_COEF` | 0.2 | **The key knob.** Low = aggressive learning + drift to gibberish; High = safe but slow |
| `CLIP_RANGE` | 0.2 | Standard PPO clip. Rarely changed |
| `BATCH_SIZE` | 8 | Bigger = lower variance but slower on CPU |
| `MAX_NEW_TOKENS` | 20 | Longer = harder for reward model (and slower) |

Discuss the **KL coefficient** with your audience — it's the central RLHF tuning knob.
