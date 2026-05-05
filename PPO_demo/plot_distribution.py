"""Per-prompt reward distribution: BEFORE vs AFTER side-by-side box plots.

Reads outputs/before.json + outputs/after.json (both produced with
`SAMPLES_PER_PROMPT > 1` in ppo_demo.py) and writes
outputs/reward_distribution.png.

The plot has two panels:
  Left:  one (BEFORE, AFTER) box pair PER PROMPT, with raw samples overlaid.
  Right: one combined box for BEFORE and one for AFTER (overall view).
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "outputs"

BEFORE_COLOR = "#d62728"   # red
AFTER_COLOR = "#2ca02c"    # green


def _rewards_per_prompt(rows):
    """Pull the list of sample rewards for each prompt row."""
    out = []
    for r in rows:
        if "samples" in r:
            out.append([s["reward"] for s in r["samples"]])
        else:
            out.append([r["reward"]])
    return out


def plot_distribution(before, after, save_path):
    """Render the side-by-side box plot. `before` and `after` are lists from JSON."""
    assert len(before) == len(after), "BEFORE and AFTER must have same number of prompts"
    n = len(before)
    prompts = [r["prompt"] for r in before]

    before_rewards = _rewards_per_prompt(before)
    after_rewards = _rewards_per_prompt(after)

    fig = plt.figure(figsize=(14, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

    # =========================================================
    # LEFT: per-prompt boxes (BEFORE & AFTER side by side)
    # =========================================================
    ax = fig.add_subplot(gs[0, 0])
    width = 0.35
    positions = np.arange(n)

    bp_b = ax.boxplot(
        before_rewards, positions=positions - width/2, widths=width,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
    )
    bp_a = ax.boxplot(
        after_rewards, positions=positions + width/2, widths=width,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch in bp_b["boxes"]:
        patch.set_facecolor(BEFORE_COLOR); patch.set_alpha(0.55)
    for patch in bp_a["boxes"]:
        patch.set_facecolor(AFTER_COLOR); patch.set_alpha(0.55)

    # raw sample scatter overlay (strip-plot)
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

    # custom legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=BEFORE_COLOR, alpha=0.55, label="BEFORE PPO"),
        Patch(facecolor=AFTER_COLOR, alpha=0.55, label="AFTER PPO"),
    ], loc="lower right")

    # =========================================================
    # RIGHT: aggregated boxes (all prompts pooled)
    # =========================================================
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
    flat_before = [v for lst in before_rewards for v in lst]
    flat_after = [v for lst in after_rewards for v in lst]
    bp = ax2.boxplot(
        [flat_before, flat_after], positions=[0, 1], widths=0.55,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
    )
    bp["boxes"][0].set_facecolor(BEFORE_COLOR); bp["boxes"][0].set_alpha(0.55)
    bp["boxes"][1].set_facecolor(AFTER_COLOR);  bp["boxes"][1].set_alpha(0.55)

    # scatter
    for i, data in enumerate([flat_before, flat_after]):
        jitter = (rng.random(len(data)) - 0.5) * 0.32
        ax2.scatter(np.full(len(data), i) + jitter, data, s=14,
                    color=[BEFORE_COLOR, AFTER_COLOR][i],
                    edgecolors="black", linewidths=0.3, alpha=0.7, zorder=3)

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["BEFORE", "AFTER"])
    ax2.axhline(0, ls=":", color="gray", alpha=0.6)
    ax2.set_title("All prompts pooled", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    # annotate means
    mean_b = np.mean(flat_before)
    mean_a = np.mean(flat_after)
    ax2.scatter([0], [mean_b], marker="D", s=70, color="white",
                edgecolors="black", zorder=4, label=f"mean = {mean_b:+.2f}")
    ax2.scatter([1], [mean_a], marker="D", s=70, color="white",
                edgecolors="black", zorder=4)
    ax2.text(0, mean_b, f"  {mean_b:+.2f}", va="center", fontsize=9)
    ax2.text(1, mean_a, f"  {mean_a:+.2f}", va="center", fontsize=9)

    fig.suptitle(
        "PPO + RLHF: reward distribution shift  (GPT-2-imdb, DistilBERT-imdb reward)",
        fontsize=13, fontweight="bold",
    )
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main():
    before = json.loads((OUTPUT_DIR / "before.json").read_text())
    after = json.loads((OUTPUT_DIR / "after.json").read_text())
    save_path = OUTPUT_DIR / "reward_distribution.png"
    plot_distribution(before, after, save_path)
    print(f"Wrote {save_path.resolve()}")

    # quick numeric summary
    flat_b = [s for r in before for s in (r.get("samples") or [{"reward": r["reward"]}])
              for s in [s["reward"]]]
    # The above one-liner is brittle; do it cleanly:
    flat_b = []
    flat_a = []
    for r in before:
        flat_b.extend([s["reward"] for s in r["samples"]] if "samples" in r else [r["reward"]])
    for r in after:
        flat_a.extend([s["reward"] for s in r["samples"]] if "samples" in r else [r["reward"]])
    print(f"  BEFORE: n={len(flat_b)}, "
          f"mean={np.mean(flat_b):+.3f}, median={np.median(flat_b):+.3f}, "
          f"min={min(flat_b):+.3f}, max={max(flat_b):+.3f}")
    print(f"  AFTER : n={len(flat_a)}, "
          f"mean={np.mean(flat_a):+.3f}, median={np.median(flat_a):+.3f}, "
          f"min={min(flat_a):+.3f}, max={max(flat_a):+.3f}")


if __name__ == "__main__":
    main()
