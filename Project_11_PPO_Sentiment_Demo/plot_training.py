"""Plot PPO training curves: reward, KL, clip-fraction, loss.

Reads outputs/training_log.json + outputs/before.json + outputs/after.json
and writes outputs/training_curves.png.

Designed to look good projected on a screen during the demo.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).parent / "outputs"


def load_data():
    log_path = OUTPUT_DIR / "training_log.json"
    if not log_path.exists():
        raise FileNotFoundError(
            f"{log_path} not found — run `python ppo_demo.py` first."
        )
    log = json.loads(log_path.read_text())

    before_mean = after_mean = None
    bp = OUTPUT_DIR / "before.json"
    ap = OUTPUT_DIR / "after.json"
    if bp.exists():
        rows = json.loads(bp.read_text())
        before_mean = sum(r["reward"] for r in rows) / len(rows)
    if ap.exists():
        rows = json.loads(ap.read_text())
        after_mean = sum(r["reward"] for r in rows) / len(rows)
    return log, before_mean, after_mean


def plot(log, before_mean, after_mean, save_path):
    steps = [row["step"] for row in log]
    reward = [row["mean_reward"] for row in log]
    kl = [row["kl"] for row in log]
    clip_frac = [row["clip_fraction"] for row in log]
    loss = [row["loss"] for row in log]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fig.suptitle(
        "PPO + RLHF: training dynamics  (GPT-2-imdb, DistilBERT-imdb reward)",
        fontsize=14, fontweight="bold",
    )

    # --- (0,0) reward -----------------------------------------------------
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
    ax.set_xlabel("PPO step")
    ax.set_ylabel("log-odds(positive)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # --- (0,1) KL ---------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(steps, kl, marker="o", color="#ff7f0e", linewidth=2)
    ax.axhline(0, ls=":", color="gray", alpha=0.5)
    ax.set_title("KL(policy || reference)  —  drift from base model")
    ax.set_xlabel("PPO step")
    ax.set_ylabel("nats / token (single-sample est.)")
    ax.grid(alpha=0.3)

    # --- (1,0) clip fraction ---------------------------------------------
    ax = axes[1, 0]
    ax.bar(steps, [c * 100 for c in clip_frac], color="#9467bd", alpha=0.85)
    ax.axhline(50, ls=":", color="gray", alpha=0.5, label="50% (clip is heavily active)")
    ax.set_title("Clip fraction  —  share of tokens hitting the PPO clip")
    ax.set_xlabel("PPO step")
    ax.set_ylabel("% of tokens")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=9)

    # --- (1,1) loss -------------------------------------------------------
    ax = axes[1, 1]
    ax.plot(steps, loss, marker="o", color="#2ca02c", linewidth=2)
    ax.axhline(0, ls=":", color="gray", alpha=0.5)
    ax.set_title("PPO loss (policy clipped surrogate + KL penalty)")
    ax.set_xlabel("PPO step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)

    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    log, before_mean, after_mean = load_data()
    save_path = OUTPUT_DIR / "training_curves.png"
    plot(log, before_mean, after_mean, save_path)
    print(f"Wrote {save_path.resolve()}")
    print(f"  steps logged   : {len(log)}")
    print(f"  reward range   : {min(r['mean_reward'] for r in log):+.2f} "
          f"-> {max(r['mean_reward'] for r in log):+.2f}")
    if before_mean is not None and after_mean is not None:
        print(f"  held-out shift : {before_mean:+.2f} -> {after_mean:+.2f} "
              f"(delta {after_mean - before_mean:+.2f})")


if __name__ == "__main__":
    main()
