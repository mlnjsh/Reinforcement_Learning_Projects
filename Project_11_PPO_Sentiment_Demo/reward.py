"""Reward model wrapper for the RLHF demo.

Uses `lvwerra/distilbert-imdb`, a DistilBERT classifier pre-trained on IMDB
sentiment. We turn its 2-class output into a scalar reward suitable for PPO.
"""
from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


REWARD_MODEL_NAME = "lvwerra/distilbert-imdb"


class RewardModel:
    """Pre-trained sentiment classifier wrapped as a scalar reward function."""

    def __init__(self, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME)
            .to(device)
            .eval()
        )
        self.device = device
        # The label index for POSITIVE — confirm from the model config to be safe.
        # distilbert-imdb uses {0: NEGATIVE, 1: POSITIVE}.
        self.pos_idx = self.model.config.label2id.get("POSITIVE", 1)
        self.neg_idx = self.model.config.label2id.get("NEGATIVE", 0)

    @torch.no_grad()
    def score(self, texts: list[str]) -> torch.Tensor:
        """Return a (B,) tensor of scalar rewards (log-odds of POSITIVE)."""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits  # (B, 2)
        return shape_reward(logits, pos_idx=self.pos_idx, neg_idx=self.neg_idx)


# ----------------------------------------------------------------------
# Reward shaping — converts 2-class logits to a scalar reward.
# Edit this function during the demo to discuss design trade-offs.
# ----------------------------------------------------------------------
def shape_reward(
    logits: torch.Tensor,
    pos_idx: int = 1,
    neg_idx: int = 0,
) -> torch.Tensor:
    """Convert classifier logits to a scalar reward signal.

    Three reasonable choices (uncomment one):

    A) Raw positive logit:       large but unbounded; can explode
    B) Positive probability:     bounded [0,1] but saturates (vanishing grad)
    C) Log-odds (DEFAULT):       symmetric, smooth — what we use here

    Args:
        logits: (B, 2) classifier output
    Returns:
        (B,) scalar reward per example
    """
    # A) raw logit
    # return logits[:, pos_idx]

    # B) probability
    # return torch.softmax(logits, dim=-1)[:, pos_idx]

    # C) log-odds (symmetric around 0; no saturation)
    return logits[:, pos_idx] - logits[:, neg_idx]


if __name__ == "__main__":
    # Tiny smoke test
    rm = RewardModel(device="cpu")
    sample_texts = [
        "This movie was absolutely wonderful, I loved every minute of it!",
        "Total waste of time. The acting was terrible and the plot made no sense.",
        "It was okay, I guess. Not really memorable.",
    ]
    scores = rm.score(sample_texts)
    print("Reward scores (log-odds; > 0 means positive):")
    for text, s in zip(sample_texts, scores.tolist()):
        print(f"  {s:+7.3f}  {text[:60]}")
