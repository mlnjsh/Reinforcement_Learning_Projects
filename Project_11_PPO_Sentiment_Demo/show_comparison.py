"""Replay a saved before/after comparison without re-running training.

Reads outputs/before.json + outputs/after.json. Works with both the old
single-sample layout and the new multi-sample layout (one row per prompt
with `samples: [{response, reward}, ...]`).
"""
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "outputs"


def mean_reward(row: dict) -> float:
    """Robust to both single-sample and multi-sample JSONs."""
    if "mean_reward" in row:
        return float(row["mean_reward"])
    return float(row["reward"])


def repr_response(row: dict) -> str:
    return row.get("response", "")


def main():
    before = json.loads((OUTPUT_DIR / "before.json").read_text())
    after = json.loads((OUTPUT_DIR / "after.json").read_text())

    print("\n" + "=" * 70)
    print("  BEFORE vs AFTER (same prompts)")
    print("=" * 70)
    print(f"{'PROMPT':<32}  {'r_before':>9}  {'r_after':>9}  {'delta':>7}")
    print("-" * 90)
    deltas = []
    for b, a in zip(before, after):
        b_r = mean_reward(b)
        a_r = mean_reward(a)
        d = a_r - b_r
        deltas.append(d)
        print(f"{b['prompt'][:30]!r:<32}  {b_r:+9.3f}  {a_r:+9.3f}  {d:+7.3f}")
        print(f"    BEFORE: {repr_response(b)!r}")
        print(f"    AFTER : {repr_response(a)!r}")
        print()
    print(f"  mean reward BEFORE : {sum(mean_reward(r) for r in before)/len(before):+.3f}")
    print(f"  mean reward AFTER  : {sum(mean_reward(r) for r in after )/len(after ):+.3f}")
    print(f"  average shift      : {sum(deltas)/len(deltas):+.3f}")


if __name__ == "__main__":
    main()
