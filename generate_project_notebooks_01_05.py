import json
import os
import uuid


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def md(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().split("\n"),
        "id": str(uuid.uuid4())[:8],
    }


def code(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.strip().split("\n"),
        "execution_count": None,
        "outputs": [],
        "id": str(uuid.uuid4())[:8],
    }


def save(path, cells):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for cell in cells:
        lines = cell["source"]
        cell["source"] = [
            l + "\n" if i < len(lines) - 1 else l for i, l in enumerate(lines)
        ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb(cells), f, indent=1, ensure_ascii=False)
    print(f"Created {path}")


# ---------------------------------------------------------------------------
# Base directory
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))

# ===================================================================
# Notebook 1: RLHF for LLM Alignment
# ===================================================================
def make_notebook_01():
    cells = []

    cells.append(md("""# Project 1: Reinforcement Learning from Human Feedback (RLHF) for LLM Alignment

**Author:** Milan Amrut Joshi

## Overview
This notebook implements a simplified RLHF pipeline from scratch using only NumPy and Matplotlib.
We simulate the three core stages of RLHF as used in systems like InstructGPT and ChatGPT:

1. **Supervised Fine-Tuning (SFT):** Train a base policy on demonstration data
2. **Reward Model Training:** Learn a reward function from pairwise human preferences (Bradley-Terry model)
3. **PPO Optimization:** Fine-tune the SFT policy using PPO with a KL penalty to maximize the learned reward

The key insight of RLHF is that human preferences are easier to collect than demonstrations,
and RL can optimize a policy to produce outputs that humans prefer."""))

    cells.append(md("""## 1. Mathematical Foundation

### Bradley-Terry Preference Model
Given two responses $y_1$ and $y_2$ to the same prompt $x$, the probability that a human prefers $y_1$ over $y_2$ is:

$$P(y_1 \\succ y_2 | x) = \\sigma(r_\\theta(x, y_1) - r_\\theta(x, y_2))$$

where $\\sigma$ is the sigmoid function and $r_\\theta$ is the learned reward model.

### Reward Model Loss
The negative log-likelihood loss for pairwise comparisons:

$$\\mathcal{L}_{RM}(\\theta) = -\\mathbb{E}_{(x, y_w, y_l)} [\\log \\sigma(r_\\theta(x, y_w) - r_\\theta(x, y_l))]$$

where $y_w$ is the preferred (winning) response and $y_l$ is the rejected (losing) response.

### PPO Objective with KL Penalty
$$\\mathcal{L}_{PPO}(\\phi) = \\mathbb{E}_{x, y \\sim \\pi_\\phi} [r_\\theta(x, y) - \\beta \\, D_{KL}(\\pi_\\phi(\\cdot|x) \\| \\pi_{ref}(\\cdot|x))]$$

The KL penalty $\\beta$ prevents the policy from deviating too far from the reference (SFT) policy,
which helps maintain generation quality and diversity."""))

    cells.append(md("""## 2. Setup and Imports"""))

    cells.append(code("""import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Environment Configuration ---
VOCAB_SIZE = 20          # Simplified vocabulary size
SEQ_LEN = 5             # Sequence length (number of tokens generated)
NUM_PROMPTS = 50         # Number of different prompts
NUM_COMPARISONS = 200    # Number of pairwise comparisons for reward model
HIDDEN_DIM = 32          # Hidden dimension for reward model

print("RLHF Pipeline Configuration:")
print(f"  Vocabulary size: {VOCAB_SIZE}")
print(f"  Sequence length: {SEQ_LEN}")
print(f"  Number of prompts: {NUM_PROMPTS}")
print(f"  Number of comparisons: {NUM_COMPARISONS}")"""))

    cells.append(md("""## 3. Simulating Text Generation as Token Selection

We model text generation as a sequential token selection problem.
Each "prompt" is represented as a context vector, and a "response" is a sequence of token indices
sampled from a categorical distribution (the policy).

The **ground truth reward** is a hidden function that scores sequences based on
token diversity, specific token preferences, and sequence patterns."""))

    cells.append(code("""def softmax(x, axis=-1):
    \"\"\"Numerically stable softmax.\"\"\"
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def generate_prompt_contexts(num_prompts, dim=10):
    \"\"\"Generate random prompt context vectors.\"\"\"
    return np.random.randn(num_prompts, dim)


def true_reward_function(sequence):
    \"\"\"
    Hidden ground truth reward that simulates human preference.
    Rewards: token diversity, presence of 'good' tokens, penalizes repetition.
    \"\"\"
    unique_ratio = len(set(sequence)) / len(sequence)
    good_tokens = {3, 7, 11, 15, 18}
    good_count = sum(1 for t in sequence if t in good_tokens) / len(sequence)
    # Penalize consecutive repeated tokens
    repeat_penalty = sum(1 for i in range(1, len(sequence)) if sequence[i] == sequence[i-1])
    reward = 2.0 * unique_ratio + 3.0 * good_count - 1.5 * repeat_penalty
    return reward


# Generate prompt contexts
prompt_contexts = generate_prompt_contexts(NUM_PROMPTS)
print(f"Generated {NUM_PROMPTS} prompt contexts with shape {prompt_contexts.shape}")

# Test the true reward function
test_seq_good = [3, 7, 11, 15, 18]
test_seq_bad = [0, 0, 0, 0, 0]
print(f"Reward for diverse+good tokens: {true_reward_function(test_seq_good):.3f}")
print(f"Reward for repeated bad tokens: {true_reward_function(test_seq_bad):.3f}")"""))

    cells.append(md("""## 4. Stage 1: Supervised Fine-Tuning (SFT) Simulation

We create a simple policy network that maps prompt contexts to token distributions.
We first generate demonstration data using a near-optimal policy and then train on it."""))

    cells.append(code("""class PolicyNetwork:
    \"\"\"Simple linear policy: context -> token logits for each position.\"\"\"

    def __init__(self, context_dim=10, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN):
        self.context_dim = context_dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        # Parameters: one weight matrix per position
        scale = 0.1
        self.W = [np.random.randn(context_dim, vocab_size) * scale for _ in range(seq_len)]
        self.b = [np.zeros(vocab_size) for _ in range(seq_len)]

    def get_logits(self, context):
        \"\"\"Get token logits for each position given a context.\"\"\"
        logits = []
        for t in range(self.seq_len):
            logit = context @ self.W[t] + self.b[t]
            logits.append(logit)
        return logits  # list of (vocab_size,) arrays

    def get_probs(self, context):
        \"\"\"Get token probabilities for each position.\"\"\"
        logits = self.get_logits(context)
        return [softmax(l) for l in logits]

    def sample(self, context, temperature=1.0):
        \"\"\"Sample a sequence given a context.\"\"\"
        logits = self.get_logits(context)
        sequence = []
        for l in logits:
            probs = softmax(l / temperature)
            token = np.random.choice(self.vocab_size, p=probs)
            sequence.append(token)
        return sequence

    def log_prob(self, context, sequence):
        \"\"\"Compute log probability of a sequence given context.\"\"\"
        probs = self.get_probs(context)
        lp = 0.0
        for t, token in enumerate(sequence):
            lp += np.log(probs[t][token] + 1e-10)
        return lp

    def copy(self):
        \"\"\"Create a deep copy of this policy.\"\"\"
        new_policy = PolicyNetwork(self.context_dim, self.vocab_size, self.seq_len)
        new_policy.W = [w.copy() for w in self.W]
        new_policy.b = [b.copy() for b in self.b]
        return new_policy


# Create demonstration data from a biased distribution favoring good tokens
def generate_demonstrations(prompt_contexts, num_demos_per_prompt=3):
    \"\"\"Generate expert demonstrations that favor good tokens.\"\"\"
    demos = []
    good_tokens = [3, 7, 11, 15, 18]
    for ctx in prompt_contexts:
        for _ in range(num_demos_per_prompt):
            seq = []
            for _ in range(SEQ_LEN):
                if np.random.rand() < 0.6:
                    token = np.random.choice(good_tokens)
                else:
                    token = np.random.randint(VOCAB_SIZE)
                seq.append(token)
            demos.append((ctx, seq))
    return demos


# Generate demos and train SFT policy
demos = generate_demonstrations(prompt_contexts)
print(f"Generated {len(demos)} demonstration sequences")

# SFT training via cross-entropy
sft_policy = PolicyNetwork()
lr_sft = 0.01
sft_losses = []

for epoch in range(30):
    epoch_loss = 0.0
    np.random.shuffle(demos)
    for ctx, seq in demos:
        probs_list = sft_policy.get_probs(ctx)
        for t, token in enumerate(seq):
            # Gradient of cross-entropy loss
            grad = probs_list[t].copy()
            grad[token] -= 1.0  # softmax grad for correct class
            sft_policy.W[t] -= lr_sft * np.outer(ctx, grad)
            sft_policy.b[t] -= lr_sft * grad
            epoch_loss -= np.log(probs_list[t][token] + 1e-10)
    avg_loss = epoch_loss / len(demos)
    sft_losses.append(avg_loss)

print(f"SFT training complete. Final loss: {sft_losses[-1]:.4f}")

# Keep a reference copy for KL penalty
ref_policy = sft_policy.copy()"""))

    cells.append(code("""# Plot SFT training loss
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(sft_losses, 'b-', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax.set_title('SFT Training Loss', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 5. Stage 2: Reward Model Training (Bradley-Terry)

We collect pairwise comparisons using the true reward function (simulating human annotators)
and train a reward model to predict which response is preferred.

The reward model takes a (context, sequence) pair and outputs a scalar reward."""))

    cells.append(code("""class RewardModel:
    \"\"\"Neural reward model: (context, sequence_embedding) -> scalar reward.\"\"\"

    def __init__(self, context_dim=10, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        # Sequence embedding: average of one-hot vectors
        input_dim = context_dim + vocab_size
        scale = 0.1
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * scale
        self.b2 = np.zeros(1)

    def _embed_sequence(self, sequence):
        \"\"\"Create a bag-of-tokens embedding.\"\"\"
        emb = np.zeros(self.vocab_size)
        for token in sequence:
            emb[token] += 1.0
        return emb / self.seq_len

    def predict(self, context, sequence):
        \"\"\"Predict reward for a (context, sequence) pair.\"\"\"
        seq_emb = self._embed_sequence(sequence)
        x = np.concatenate([context, seq_emb])
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        reward = (h @ self.W2 + self.b2)[0]
        return reward, h, x  # return intermediates for backprop

    def forward(self, context, sequence):
        \"\"\"Forward pass returning just the reward.\"\"\"
        r, _, _ = self.predict(context, sequence)
        return r


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# Generate pairwise comparison data
comparisons = []
for _ in range(NUM_COMPARISONS):
    idx = np.random.randint(NUM_PROMPTS)
    ctx = prompt_contexts[idx]
    # Generate two responses using the SFT policy
    seq1 = sft_policy.sample(ctx, temperature=1.5)
    seq2 = sft_policy.sample(ctx, temperature=1.5)
    r1 = true_reward_function(seq1)
    r2 = true_reward_function(seq2)
    # Human labels the one with higher true reward as preferred
    if r1 > r2:
        comparisons.append((ctx, seq1, seq2))  # seq1 preferred
    else:
        comparisons.append((ctx, seq2, seq1))  # seq2 preferred

print(f"Generated {len(comparisons)} pairwise comparisons")

# Train reward model
reward_model = RewardModel()
lr_rm = 0.001
rm_losses = []
rm_accuracies = []

for epoch in range(50):
    epoch_loss = 0.0
    correct = 0
    np.random.shuffle(comparisons)

    for ctx, seq_w, seq_l in comparisons:
        # Forward pass for winner and loser
        r_w, h_w, x_w = reward_model.predict(ctx, seq_w)
        r_l, h_l, x_l = reward_model.predict(ctx, seq_l)

        # Bradley-Terry loss: -log(sigmoid(r_w - r_l))
        diff = r_w - r_l
        prob = sigmoid(diff)
        loss = -np.log(prob + 1e-10)
        epoch_loss += loss

        if diff > 0:
            correct += 1

        # Backpropagation
        # d_loss/d_diff = -(1 - prob)
        d_diff = -(1.0 - prob)

        # Gradients for winner path
        d_r_w = d_diff
        d_h_w = d_r_w * reward_model.W2.flatten()
        d_h_w = d_h_w * (h_w > 0).astype(float)  # ReLU grad

        # Gradients for loser path
        d_r_l = -d_diff
        d_h_l = d_r_l * reward_model.W2.flatten()
        d_h_l = d_h_l * (h_l > 0).astype(float)

        # Update W2, b2
        reward_model.W2 -= lr_rm * (h_w.reshape(-1, 1) * d_r_w + h_l.reshape(-1, 1) * d_r_l)
        reward_model.b2 -= lr_rm * (d_r_w + d_r_l)

        # Update W1, b1
        reward_model.W1 -= lr_rm * np.outer(x_w, d_h_w)
        reward_model.W1 -= lr_rm * np.outer(x_l, d_h_l)
        reward_model.b1 -= lr_rm * (d_h_w + d_h_l)

    avg_loss = epoch_loss / len(comparisons)
    accuracy = correct / len(comparisons)
    rm_losses.append(avg_loss)
    rm_accuracies.append(accuracy)

print(f"Reward model training complete.")
print(f"  Final loss: {rm_losses[-1]:.4f}")
print(f"  Final accuracy: {rm_accuracies[-1]:.4f}")"""))

    cells.append(code("""# Plot reward model training
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(rm_losses, 'r-', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Bradley-Terry Loss', fontsize=12)
axes[0].set_title('Reward Model Loss', fontsize=14)
axes[0].grid(True, alpha=0.3)

axes[1].plot(rm_accuracies, 'g-', linewidth=2)
axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Preference Accuracy', fontsize=12)
axes[1].set_title('Reward Model Accuracy', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 6. Stage 3: PPO with KL Penalty

Now we use PPO to optimize the SFT policy using the learned reward model.
The KL penalty keeps the policy close to the reference (SFT) policy.

$$\\text{objective} = r_{\\theta}(x, y) - \\beta \\sum_t D_{KL}(\\pi_\\phi(\\cdot|x, t) \\| \\pi_{ref}(\\cdot|x, t))$$"""))

    cells.append(code("""# PPO Training
ppo_policy = sft_policy.copy()
lr_ppo = 0.005
beta_kl = 0.2  # KL penalty coefficient
num_ppo_epochs = 40
samples_per_epoch = 30

ppo_rewards = []
ppo_kl_divs = []
ppo_true_rewards = []

for epoch in range(num_ppo_epochs):
    epoch_reward = 0.0
    epoch_kl = 0.0
    epoch_true_reward = 0.0

    for _ in range(samples_per_epoch):
        idx = np.random.randint(NUM_PROMPTS)
        ctx = prompt_contexts[idx]

        # Sample from current policy
        sequence = ppo_policy.sample(ctx)

        # Compute learned reward
        learned_reward = reward_model.forward(ctx, sequence)

        # Compute KL divergence between current and reference policy
        curr_probs = ppo_policy.get_probs(ctx)
        ref_probs = ref_policy.get_probs(ctx)

        kl = 0.0
        for t in range(SEQ_LEN):
            for v in range(VOCAB_SIZE):
                if curr_probs[t][v] > 1e-10:
                    kl += curr_probs[t][v] * np.log(curr_probs[t][v] / (ref_probs[t][v] + 1e-10))

        # Total objective: reward - beta * KL
        advantage = learned_reward - beta_kl * kl

        # Policy gradient update (REINFORCE-style within PPO)
        for t in range(SEQ_LEN):
            token = sequence[t]
            grad = curr_probs[t].copy()
            grad[token] -= 1.0
            # Gradient ascent on advantage
            ppo_policy.W[t] -= lr_ppo * advantage * np.outer(ctx, grad)
            ppo_policy.b[t] -= lr_ppo * advantage * grad

        epoch_reward += learned_reward
        epoch_kl += kl
        epoch_true_reward += true_reward_function(sequence)

    ppo_rewards.append(epoch_reward / samples_per_epoch)
    ppo_kl_divs.append(epoch_kl / samples_per_epoch)
    ppo_true_rewards.append(epoch_true_reward / samples_per_epoch)

print(f"PPO training complete.")
print(f"  Final learned reward: {ppo_rewards[-1]:.4f}")
print(f"  Final KL divergence: {ppo_kl_divs[-1]:.4f}")
print(f"  Final true reward: {ppo_true_rewards[-1]:.4f}")"""))

    cells.append(code("""# Plot PPO training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(ppo_rewards, 'b-', linewidth=2, label='Learned Reward')
axes[0].plot(ppo_true_rewards, 'g--', linewidth=2, label='True Reward')
axes[0].set_xlabel('PPO Epoch', fontsize=12)
axes[0].set_ylabel('Reward', fontsize=12)
axes[0].set_title('Reward During PPO Training', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(ppo_kl_divs, 'r-', linewidth=2)
axes[1].set_xlabel('PPO Epoch', fontsize=12)
axes[1].set_ylabel('KL Divergence', fontsize=12)
axes[1].set_title('KL Divergence (Policy vs Reference)', fontsize=14)
axes[1].grid(True, alpha=0.3)

axes[2].set_title('Before vs After RLHF: Token Distribution', fontsize=14)
# Compare token distributions for a sample prompt
sample_ctx = prompt_contexts[0]
ref_probs_sample = ref_policy.get_probs(sample_ctx)
ppo_probs_sample = ppo_policy.get_probs(sample_ctx)

x_tokens = np.arange(VOCAB_SIZE)
width = 0.35
# Average across positions
ref_avg = np.mean([ref_probs_sample[t] for t in range(SEQ_LEN)], axis=0)
ppo_avg = np.mean([ppo_probs_sample[t] for t in range(SEQ_LEN)], axis=0)
axes[2].bar(x_tokens - width/2, ref_avg, width, label='SFT (Before)', alpha=0.7)
axes[2].bar(x_tokens + width/2, ppo_avg, width, label='RLHF (After)', alpha=0.7)
axes[2].set_xlabel('Token ID', fontsize=12)
axes[2].set_ylabel('Average Probability', fontsize=12)
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)

# Mark good tokens
for gt in [3, 7, 11, 15, 18]:
    axes[2].axvline(x=gt, color='green', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 7. Evaluation: Before vs After RLHF

Let us compare the SFT policy (before RLHF) with the PPO-optimized policy (after RLHF)
across multiple metrics."""))

    cells.append(code("""# Evaluate both policies
num_eval = 200
sft_true_rewards = []
ppo_eval_true_rewards = []
sft_good_token_fracs = []
ppo_good_token_fracs = []
good_tokens_set = {3, 7, 11, 15, 18}

for _ in range(num_eval):
    idx = np.random.randint(NUM_PROMPTS)
    ctx = prompt_contexts[idx]

    seq_sft = ref_policy.sample(ctx)
    seq_ppo = ppo_policy.sample(ctx)

    sft_true_rewards.append(true_reward_function(seq_sft))
    ppo_eval_true_rewards.append(true_reward_function(seq_ppo))

    sft_good_token_fracs.append(sum(1 for t in seq_sft if t in good_tokens_set) / SEQ_LEN)
    ppo_good_token_fracs.append(sum(1 for t in seq_ppo if t in good_tokens_set) / SEQ_LEN)

print("=== Evaluation Results ===")
print(f"SFT  - Mean True Reward: {np.mean(sft_true_rewards):.4f} +/- {np.std(sft_true_rewards):.4f}")
print(f"RLHF - Mean True Reward: {np.mean(ppo_eval_true_rewards):.4f} +/- {np.std(ppo_eval_true_rewards):.4f}")
print(f"SFT  - Good Token Fraction: {np.mean(sft_good_token_fracs):.4f}")
print(f"RLHF - Good Token Fraction: {np.mean(ppo_good_token_fracs):.4f}")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(sft_true_rewards, bins=20, alpha=0.6, label='SFT', color='blue')
axes[0].hist(ppo_eval_true_rewards, bins=20, alpha=0.6, label='RLHF', color='green')
axes[0].set_xlabel('True Reward', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('True Reward Distribution: SFT vs RLHF', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].hist(sft_good_token_fracs, bins=10, alpha=0.6, label='SFT', color='blue')
axes[1].hist(ppo_good_token_fracs, bins=10, alpha=0.6, label='RLHF', color='green')
axes[1].set_xlabel('Fraction of Good Tokens', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Good Token Usage: SFT vs RLHF', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 8. Summary

In this notebook we implemented a complete RLHF pipeline:

1. **SFT Stage:** Trained a base policy on expert demonstrations using cross-entropy loss
2. **Reward Model:** Learned human preferences via the Bradley-Terry model from pairwise comparisons
3. **PPO Stage:** Optimized the policy using the learned reward with KL penalty for stability

Key findings:
- The reward model successfully learned to predict human preferences (accuracy >> 50%)
- PPO optimization increased the true reward compared to the SFT baseline
- The KL penalty kept the policy from diverging too far from the reference
- The RLHF policy learned to generate sequences with more preferred tokens

This simplified pipeline captures the essential mechanics of real RLHF systems used to align LLMs."""))

    path = os.path.join(BASE, "Project_01_RLHF_LLM_Alignment", "rlhf_notebook.ipynb")
    save(path, cells)


# ===================================================================
# Notebook 2: Offline RL for Healthcare
# ===================================================================
def make_notebook_02():
    cells = []

    cells.append(md("""# Project 2: Offline RL for Healthcare Treatment Optimization

**Author:** Milan Amrut Joshi

## Overview
This notebook explores **Offline Reinforcement Learning** applied to healthcare treatment optimization.
In healthcare, we cannot run trial-and-error experiments on patients. Instead, we must learn
optimal treatment policies from previously collected clinical data.

We implement:
1. **Data Generation:** Simulated patient trajectories with vitals, treatments, and outcomes
2. **Behavior Cloning (BC):** Supervised learning baseline that imitates the data collection policy
3. **Conservative Q-Learning (CQL):** Offline RL method that penalizes overestimation of Q-values for unseen state-action pairs
4. **Policy Comparison:** Analyzing how CQL avoids dangerous out-of-distribution (OOD) actions"""))

    cells.append(md("""## 1. Mathematical Foundation

### The Offline RL Problem
Given a static dataset $\\mathcal{D} = \\{(s_t, a_t, r_t, s_{t+1})\\}$ collected by a behavior policy $\\pi_\\beta$,
find an optimal policy $\\pi^*$ without further environment interaction.

### Behavior Cloning
Directly learn the behavior policy via maximum likelihood:
$$\\pi_{BC}(a|s) = \\arg\\max_\\pi \\mathbb{E}_{(s,a) \\sim \\mathcal{D}} [\\log \\pi(a|s)]$$

### Conservative Q-Learning (CQL)
CQL adds a regularization term that penalizes Q-values for OOD actions:

$$\\mathcal{L}_{CQL}(\\theta) = \\alpha \\left( \\mathbb{E}_{s \\sim \\mathcal{D}, a \\sim \\mu} [Q_\\theta(s,a)] - \\mathbb{E}_{(s,a) \\sim \\mathcal{D}} [Q_\\theta(s,a)] \\right) + \\frac{1}{2} \\mathbb{E}_{\\mathcal{D}} \\left[(Q_\\theta(s,a) - \\hat{B}^\\pi Q_{\\bar{\\theta}}(s,a))^2\\right]$$

where $\\mu$ is a broad distribution over actions (e.g., uniform) and $\\alpha$ controls the conservatism.

The first term pushes down Q-values for actions sampled uniformly (including OOD),
while pulling up Q-values for actions actually observed in the dataset."""))

    cells.append(md("""## 2. Setup and Environment"""))

    cells.append(code("""import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Patient/Environment Configuration ---
NUM_VITALS = 4          # Blood pressure, heart rate, temperature, oxygen
NUM_TREATMENTS = 5      # Different treatment options
NUM_PATIENTS = 500      # Number of patient trajectories
EPISODE_LENGTH = 10     # Timesteps per patient episode
GAMMA = 0.95            # Discount factor

VITAL_NAMES = ['BloodPressure', 'HeartRate', 'Temperature', 'OxygenSat']
TREATMENT_NAMES = ['Drug_A', 'Drug_B', 'Drug_C', 'Rest', 'Surgery']

print("Offline RL Healthcare Configuration:")
print(f"  Number of vitals: {NUM_VITALS}")
print(f"  Number of treatments: {NUM_TREATMENTS}")
print(f"  Patient trajectories: {NUM_PATIENTS}")
print(f"  Episode length: {EPISODE_LENGTH}")"""))

    cells.append(md("""## 3. Patient Simulator and Data Generation

We simulate a simplified patient model where:
- **State:** 4 vital signs (normalized to [0, 1] range)
- **Actions:** 5 possible treatments
- **Dynamics:** Each treatment has different effects on vitals
- **Reward:** Based on how close vitals are to healthy ranges"""))

    cells.append(code("""class PatientSimulator:
    \"\"\"Simulated patient environment for treatment optimization.\"\"\"

    def __init__(self):
        # Healthy ranges for each vital (normalized)
        self.healthy_low = np.array([0.4, 0.3, 0.45, 0.85])
        self.healthy_high = np.array([0.6, 0.5, 0.55, 0.98])
        self.healthy_center = (self.healthy_low + self.healthy_high) / 2

        # Treatment effects on vitals: shape (num_treatments, num_vitals)
        self.treatment_effects = np.array([
            [-0.05,  0.02, -0.01,  0.03],  # Drug A: lowers BP, slight HR increase
            [ 0.01, -0.04,  0.00,  0.02],  # Drug B: lowers HR
            [-0.02, -0.01, -0.03,  0.01],  # Drug C: lowers temperature
            [ 0.01,  0.01,  0.01,  0.02],  # Rest: mild recovery
            [-0.08, -0.05, -0.02,  0.05],  # Surgery: aggressive, high effect
        ])

    def reset(self):
        \"\"\"Initialize a new patient with random vitals.\"\"\"
        self.state = np.random.uniform(0.2, 0.8, NUM_VITALS)
        return self.state.copy()

    def step(self, action):
        \"\"\"Apply treatment and return next state, reward.\"\"\"
        effect = self.treatment_effects[action]
        noise = np.random.randn(NUM_VITALS) * 0.02
        self.state = np.clip(self.state + effect + noise, 0.0, 1.0)

        # Reward: negative distance from healthy center
        dist = np.abs(self.state - self.healthy_center)
        in_range = np.all((self.state >= self.healthy_low) & (self.state <= self.healthy_high))
        reward = -np.sum(dist) + (1.0 if in_range else 0.0)
        return self.state.copy(), reward


def behavior_policy(state):
    \"\"\"Suboptimal behavior policy used for data collection (e.g., a rule-based doctor).\"\"\"
    # Simple heuristic: pick treatment based on which vital is most abnormal
    healthy_center = np.array([0.5, 0.4, 0.50, 0.92])
    deviations = state - healthy_center

    # Mostly correct but noisy decisions
    if np.random.rand() < 0.3:
        return np.random.randint(NUM_TREATMENTS)  # Random exploration 30% of time

    max_dev_idx = np.argmax(np.abs(deviations))
    if max_dev_idx == 0:
        return 0  # Drug A for BP
    elif max_dev_idx == 1:
        return 1  # Drug B for HR
    elif max_dev_idx == 2:
        return 2  # Drug C for Temperature
    else:
        return 3  # Rest for Oxygen


# Generate dataset
env = PatientSimulator()
dataset = []

for patient_id in range(NUM_PATIENTS):
    state = env.reset()
    for t in range(EPISODE_LENGTH):
        action = behavior_policy(state)
        next_state, reward = env.step(action)
        dataset.append({
            'state': state.copy(),
            'action': action,
            'reward': reward,
            'next_state': next_state.copy(),
            'done': (t == EPISODE_LENGTH - 1)
        })
        state = next_state

print(f"Dataset size: {len(dataset)} transitions")

# Convert to arrays
states = np.array([d['state'] for d in dataset])
actions = np.array([d['action'] for d in dataset])
rewards = np.array([d['reward'] for d in dataset])
next_states = np.array([d['next_state'] for d in dataset])
dones = np.array([d['done'] for d in dataset]).astype(float)

print(f"States shape: {states.shape}")
print(f"Mean reward: {rewards.mean():.4f}")
print(f"Action distribution: {np.bincount(actions, minlength=NUM_TREATMENTS) / len(actions)}")"""))

    cells.append(code("""# Visualize the dataset
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i in range(NUM_VITALS):
    ax = axes[i // 2][i % 2]
    ax.hist(states[:, i], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=env.healthy_low[i], color='green', linestyle='--', linewidth=2, label='Healthy range')
    ax.axvline(x=env.healthy_high[i], color='green', linestyle='--', linewidth=2)
    ax.set_title(f'{VITAL_NAMES[i]} Distribution', fontsize=13)
    ax.set_xlabel('Value', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Patient Vitals Distribution in Dataset', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()

# Action distribution
fig, ax = plt.subplots(figsize=(8, 4))
action_counts = np.bincount(actions, minlength=NUM_TREATMENTS)
ax.bar(range(NUM_TREATMENTS), action_counts, color='coral', edgecolor='black')
ax.set_xticks(range(NUM_TREATMENTS))
ax.set_xticklabels(TREATMENT_NAMES, rotation=20)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Treatment Distribution in Dataset (Behavior Policy)', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 4. Behavior Cloning (Baseline)

Behavior cloning simply imitates the data collection policy using supervised learning.
It learns $\\pi_{BC}(a|s)$ by maximizing the likelihood of observed actions."""))

    cells.append(code("""def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class BehaviorCloning:
    \"\"\"Behavior cloning: supervised learning on (state, action) pairs.\"\"\"

    def __init__(self, state_dim, num_actions, hidden_dim=32):
        scale = 0.1
        self.W1 = np.random.randn(state_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, num_actions) * scale
        self.b2 = np.zeros(num_actions)

    def predict_probs(self, state):
        h = np.maximum(0, state @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return softmax(logits), h

    def get_action(self, state):
        probs, _ = self.predict_probs(state)
        return np.argmax(probs)

    def train(self, states, actions, epochs=100, lr=0.01, batch_size=64):
        losses = []
        n = len(states)
        for epoch in range(epochs):
            indices = np.random.permutation(n)
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                s_batch = states[batch_idx]
                a_batch = actions[batch_idx]

                # Forward pass
                h = np.maximum(0, s_batch @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                probs = softmax(logits)

                # Cross-entropy loss
                log_probs = np.log(probs[np.arange(len(a_batch)), a_batch] + 1e-10)
                loss = -np.mean(log_probs)
                epoch_loss += loss
                num_batches += 1

                # Backward pass
                grad_logits = probs.copy()
                grad_logits[np.arange(len(a_batch)), a_batch] -= 1.0
                grad_logits /= len(a_batch)

                grad_W2 = h.T @ grad_logits
                grad_b2 = np.sum(grad_logits, axis=0)

                grad_h = grad_logits @ self.W2.T
                grad_h = grad_h * (h > 0).astype(float)

                grad_W1 = s_batch.T @ grad_h
                grad_b1 = np.sum(grad_h, axis=0)

                self.W1 -= lr * grad_W1
                self.b1 -= lr * grad_b1
                self.W2 -= lr * grad_W2
                self.b2 -= lr * grad_b2

            losses.append(epoch_loss / num_batches)
        return losses


# Train behavior cloning
bc_agent = BehaviorCloning(NUM_VITALS, NUM_TREATMENTS)
bc_losses = bc_agent.train(states, actions, epochs=80)

print(f"BC training complete. Final loss: {bc_losses[-1]:.4f}")

# Plot BC loss
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(bc_losses, 'b-', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax.set_title('Behavior Cloning Training Loss', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 5. Conservative Q-Learning (CQL)

CQL is a key offline RL algorithm that addresses the **distributional shift** problem.
Standard Q-learning can overestimate Q-values for state-action pairs not seen in the dataset,
leading to dangerous OOD actions. CQL adds a conservative penalty:

$$Q_{CQL}(s,a) \\leq Q^{\\pi}(s,a) \\quad \\text{for all } (s,a)$$

This ensures the learned policy stays close to the data distribution."""))

    cells.append(code("""class CQLAgent:
    \"\"\"Conservative Q-Learning for offline RL.\"\"\"

    def __init__(self, state_dim, num_actions, hidden_dim=32, alpha=1.0):
        scale = 0.1
        self.num_actions = num_actions
        self.alpha = alpha  # CQL conservatism coefficient

        # Q-network
        self.W1 = np.random.randn(state_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, num_actions) * scale
        self.b2 = np.zeros(num_actions)

        # Target network (copy)
        self.W1_target = self.W1.copy()
        self.b1_target = self.b1.copy()
        self.W2_target = self.W2.copy()
        self.b2_target = self.b2.copy()

    def predict_q(self, state, target=False):
        if target:
            h = np.maximum(0, state @ self.W1_target + self.b1_target)
            q = h @ self.W2_target + self.b2_target
        else:
            h = np.maximum(0, state @ self.W1 + self.b1)
            q = h @ self.W2 + self.b2
        return q, h

    def get_action(self, state):
        q, _ = self.predict_q(state.reshape(1, -1))
        return np.argmax(q[0])

    def update_target(self, tau=0.01):
        self.W1_target = tau * self.W1 + (1 - tau) * self.W1_target
        self.b1_target = tau * self.b1 + (1 - tau) * self.b1_target
        self.W2_target = tau * self.W2 + (1 - tau) * self.W2_target
        self.b2_target = tau * self.b2 + (1 - tau) * self.b2_target

    def train(self, states, actions, rewards, next_states, dones,
             epochs=100, lr=0.005, batch_size=64, gamma=0.95):
        losses = []
        cql_penalties = []
        n = len(states)

        for epoch in range(epochs):
            indices = np.random.permutation(n)
            epoch_loss = 0.0
            epoch_cql = 0.0
            num_batches = 0

            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                s = states[batch_idx]
                a = actions[batch_idx]
                r = rewards[batch_idx]
                ns = next_states[batch_idx]
                d = dones[batch_idx]

                bs = len(s)

                # Current Q-values
                q_vals, h = self.predict_q(s)
                q_selected = q_vals[np.arange(bs), a]

                # Target Q-values
                q_next, _ = self.predict_q(ns, target=True)
                q_next_max = np.max(q_next, axis=1)
                targets = r + gamma * (1 - d) * q_next_max

                # TD loss
                td_error = q_selected - targets
                td_loss = np.mean(td_error ** 2)

                # CQL penalty: logsumexp(Q(s, .)) - Q(s, a_data)
                q_logsumexp = np.log(np.sum(np.exp(q_vals - np.max(q_vals, axis=1, keepdims=True)), axis=1))
                q_logsumexp += np.max(q_vals, axis=1)
                cql_penalty = np.mean(q_logsumexp - q_selected)

                total_loss = td_loss + self.alpha * cql_penalty
                epoch_loss += total_loss
                epoch_cql += cql_penalty
                num_batches += 1

                # Backprop for TD loss
                grad_q = np.zeros_like(q_vals)
                grad_q[np.arange(bs), a] = 2.0 * td_error / bs

                # Backprop for CQL penalty
                q_softmax = softmax(q_vals)
                grad_cql = self.alpha * (q_softmax.copy()) / bs
                grad_cql[np.arange(bs), a] -= self.alpha / bs

                grad_q_total = grad_q + grad_cql

                # Update network
                grad_W2 = h.T @ grad_q_total
                grad_b2 = np.sum(grad_q_total, axis=0)

                grad_h = grad_q_total @ self.W2.T
                grad_h = grad_h * (h > 0).astype(float)

                grad_W1 = s.T @ grad_h
                grad_b1 = np.sum(grad_h, axis=0)

                self.W1 -= lr * grad_W1
                self.b1 -= lr * grad_b1
                self.W2 -= lr * grad_W2
                self.b2 -= lr * grad_b2

                self.update_target()

            losses.append(epoch_loss / num_batches)
            cql_penalties.append(epoch_cql / num_batches)

        return losses, cql_penalties


# Train CQL agent
cql_agent = CQLAgent(NUM_VITALS, NUM_TREATMENTS, alpha=2.0)
cql_losses, cql_penalties = cql_agent.train(
    states, actions, rewards, next_states, dones, epochs=80
)

print(f"CQL training complete.")
print(f"  Final total loss: {cql_losses[-1]:.4f}")
print(f"  Final CQL penalty: {cql_penalties[-1]:.4f}")"""))

    cells.append(code("""# Plot CQL training
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(cql_losses, 'r-', linewidth=2, label='Total Loss')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('CQL Training Loss', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(cql_penalties, 'purple', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('CQL Penalty', fontsize=12)
axes[1].set_title('Conservative Penalty Over Training', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 6. Policy Comparison and OOD Analysis

Let us compare the learned policies and analyze how CQL handles out-of-distribution actions."""))

    cells.append(code("""# Compare policies on test patients
env_test = PatientSimulator()
num_test = 100

def evaluate_policy(policy_fn, env, num_episodes, episode_length):
    total_rewards = []
    all_actions = []
    for _ in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0
        for t in range(episode_length):
            action = policy_fn(state)
            next_state, reward = env.step(action)
            ep_reward += reward
            all_actions.append(action)
            state = next_state
        total_rewards.append(ep_reward)
    return np.array(total_rewards), np.array(all_actions)

# Evaluate behavior policy, BC, CQL
beh_rewards, beh_actions = evaluate_policy(behavior_policy, env_test, num_test, EPISODE_LENGTH)
bc_rewards, bc_actions = evaluate_policy(bc_agent.get_action, env_test, num_test, EPISODE_LENGTH)
cql_rewards, cql_actions = evaluate_policy(cql_agent.get_action, env_test, num_test, EPISODE_LENGTH)

print("=== Policy Evaluation ===")
print(f"Behavior Policy: {np.mean(beh_rewards):.3f} +/- {np.std(beh_rewards):.3f}")
print(f"Behavior Cloning: {np.mean(bc_rewards):.3f} +/- {np.std(bc_rewards):.3f}")
print(f"CQL:              {np.mean(cql_rewards):.3f} +/- {np.std(cql_rewards):.3f}")"""))

    cells.append(code("""# Visualize Q-value distributions and action comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Reward comparison boxplot
ax = axes[0][0]
bp = ax.boxplot([beh_rewards, bc_rewards, cql_rewards],
                labels=['Behavior', 'BC', 'CQL'],
                patch_artist=True)
colors = ['lightblue', 'lightgreen', 'salmon']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_ylabel('Episode Reward', fontsize=12)
ax.set_title('Policy Performance Comparison', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

# 2. Action distributions
ax = axes[0][1]
width = 0.25
x_pos = np.arange(NUM_TREATMENTS)
beh_dist = np.bincount(beh_actions, minlength=NUM_TREATMENTS) / len(beh_actions)
bc_dist = np.bincount(bc_actions, minlength=NUM_TREATMENTS) / len(bc_actions)
cql_dist = np.bincount(cql_actions, minlength=NUM_TREATMENTS) / len(cql_actions)
ax.bar(x_pos - width, beh_dist, width, label='Behavior', alpha=0.8)
ax.bar(x_pos, bc_dist, width, label='BC', alpha=0.8)
ax.bar(x_pos + width, cql_dist, width, label='CQL', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(TREATMENT_NAMES, rotation=20, fontsize=9)
ax.set_ylabel('Proportion', fontsize=12)
ax.set_title('Action Distribution Comparison', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# 3. Q-value distribution for CQL
ax = axes[1][0]
sample_states = states[np.random.choice(len(states), 200)]
all_q = []
for s in sample_states:
    q, _ = cql_agent.predict_q(s.reshape(1, -1))
    all_q.append(q[0])
all_q = np.array(all_q)
for a in range(NUM_TREATMENTS):
    ax.hist(all_q[:, a], bins=30, alpha=0.5, label=TREATMENT_NAMES[a])
ax.set_xlabel('Q-value', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('CQL Q-value Distributions by Treatment', fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 4. Q-value heatmap for a representative state
ax = axes[1][1]
grid_bp = np.linspace(0.2, 0.8, 10)
grid_hr = np.linspace(0.2, 0.8, 10)
q_map = np.zeros((10, 10))
for i, bp_val in enumerate(grid_bp):
    for j, hr_val in enumerate(grid_hr):
        test_state = np.array([bp_val, hr_val, 0.5, 0.9])
        q, _ = cql_agent.predict_q(test_state.reshape(1, -1))
        q_map[j, i] = np.max(q[0])
im = ax.imshow(q_map, origin='lower', aspect='auto', cmap='viridis',
               extent=[0.2, 0.8, 0.2, 0.8])
ax.set_xlabel('Blood Pressure', fontsize=12)
ax.set_ylabel('Heart Rate', fontsize=12)
ax.set_title('Max Q-value Landscape (CQL)', fontsize=14)
plt.colorbar(im, ax=ax, label='Max Q')

plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 7. Summary

This notebook demonstrated offline RL for healthcare treatment optimization:

1. **Data Generation:** Created a realistic patient simulator with vitals, treatments, and outcomes
2. **Behavior Cloning:** Trained a supervised baseline that imitates the (suboptimal) data collection policy
3. **Conservative Q-Learning:** Implemented CQL which learns conservative Q-values to avoid OOD actions
4. **Analysis:** Compared policies and showed how CQL's conservative penalty affects action selection

Key takeaways:
- Offline RL is crucial for healthcare where online exploration is unsafe
- Behavior cloning can only match the data policy, not improve upon it
- CQL's conservative penalty prevents overestimation for unseen state-action pairs
- The trade-off between conservatism and optimality is controlled by alpha"""))

    path = os.path.join(BASE, "Project_02_Offline_RL_Healthcare", "offline_rl_notebook.ipynb")
    save(path, cells)


# ===================================================================
# Notebook 3: Multi-Agent RL
# ===================================================================
def make_notebook_03():
    cells = []

    cells.append(md("""# Project 3: Multi-Agent Reinforcement Learning - Predator-Prey

**Author:** Milan Amrut Joshi

## Overview
This notebook implements a **Multi-Agent RL** system for a predator-prey grid world:
- **2 predators** must cooperate to capture **1 prey**
- Each predator uses **Independent Q-Learning (IQL)** -- treating other agents as part of the environment
- The prey follows a simple escape heuristic

We analyze:
1. How capture rates improve over training
2. Whether implicit cooperation emerges between independent learners
3. Cooperative vs independent strategies in multi-agent settings"""))

    cells.append(md("""## 1. Multi-Agent RL Background

### Independent Q-Learning (IQL)
Each agent $i$ maintains its own Q-table $Q_i(s, a_i)$ and updates it independently:

$$Q_i(s, a_i) \\leftarrow Q_i(s, a_i) + \\alpha [r_i + \\gamma \\max_{a_i'} Q_i(s', a_i') - Q_i(s, a_i)]$$

The key challenge is that from each agent's perspective, the environment is **non-stationary**
because other agents are simultaneously learning and changing their policies.

### Predator-Prey Game
- **Grid:** $N \\times N$ toroidal grid (wraps around)
- **Capture condition:** A predator occupies the same cell as the prey
- **Cooperative reward:** Both predators get reward when either captures the prey
- **State:** Relative positions of prey from each predator's perspective"""))

    cells.append(md("""## 2. Setup"""))

    cells.append(code("""import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

GRID_SIZE = 7
MAX_STEPS = 50          # Max steps per episode
NUM_EPISODES = 5000
ALPHA = 0.2             # Learning rate
GAMMA = 0.95            # Discount factor
EPSILON_START = 1.0     # Initial exploration
EPSILON_END = 0.05
EPSILON_DECAY = 0.998

# Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
ACTION_NAMES = ['Stay', 'Up', 'Down', 'Left', 'Right']
NUM_ACTIONS = 5
# Movement deltas (row, col)
MOVES = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

print("Predator-Prey MARL Configuration:")
print(f"  Grid size: {GRID_SIZE}x{GRID_SIZE}")
print(f"  Max steps per episode: {MAX_STEPS}")
print(f"  Training episodes: {NUM_EPISODES}")"""))

    cells.append(md("""## 3. Predator-Prey Environment"""))

    cells.append(code("""class PredatorPreyEnv:
    \"\"\"Grid world with 2 predators and 1 prey.\"\"\"

    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size

    def reset(self):
        \"\"\"Place agents at random non-overlapping positions.\"\"\"
        positions = []
        while len(positions) < 3:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if pos not in positions:
                positions.append(pos)
        self.pred1_pos = positions[0]
        self.pred2_pos = positions[1]
        self.prey_pos = positions[2]
        return self._get_obs()

    def _wrap(self, pos):
        return (pos[0] % self.grid_size, pos[1] % self.grid_size)

    def _manhattan_dist(self, p1, p2):
        \"\"\"Toroidal Manhattan distance.\"\"\"
        dr = min(abs(p1[0] - p2[0]), self.grid_size - abs(p1[0] - p2[0]))
        dc = min(abs(p1[1] - p2[1]), self.grid_size - abs(p1[1] - p2[1]))
        return dr + dc

    def _relative_pos(self, agent_pos, target_pos):
        \"\"\"Get relative position (wrapped to [-grid//2, grid//2]).\"\"\"
        dr = (target_pos[0] - agent_pos[0] + self.grid_size // 2) % self.grid_size - self.grid_size // 2
        dc = (target_pos[1] - agent_pos[1] + self.grid_size // 2) % self.grid_size - self.grid_size // 2
        return (dr, dc)

    def _get_obs(self):
        \"\"\"
        State for each predator: (relative_prey_row, relative_prey_col,
                                   relative_other_pred_row, relative_other_pred_col)
        Discretized to grid offsets.
        \"\"\"
        prey_rel_1 = self._relative_pos(self.pred1_pos, self.prey_pos)
        other_rel_1 = self._relative_pos(self.pred1_pos, self.pred2_pos)

        prey_rel_2 = self._relative_pos(self.pred2_pos, self.prey_pos)
        other_rel_2 = self._relative_pos(self.pred2_pos, self.pred1_pos)

        obs1 = (prey_rel_1[0], prey_rel_1[1], other_rel_1[0], other_rel_1[1])
        obs2 = (prey_rel_2[0], prey_rel_2[1], other_rel_2[0], other_rel_2[1])
        return obs1, obs2

    def _prey_policy(self):
        \"\"\"Simple escape heuristic: move away from closest predator.\"\"\"
        d1 = self._manhattan_dist(self.prey_pos, self.pred1_pos)
        d2 = self._manhattan_dist(self.prey_pos, self.pred2_pos)

        # Choose the closer predator as threat
        threat = self.pred1_pos if d1 <= d2 else self.pred2_pos

        # Move away from threat (with some randomness)
        if np.random.rand() < 0.2:
            return np.random.randint(NUM_ACTIONS)

        rel = self._relative_pos(self.prey_pos, threat)
        # Move opposite to relative direction of threat
        best_actions = []
        if rel[0] > 0:
            best_actions.append(1)  # up (away)
        elif rel[0] < 0:
            best_actions.append(2)  # down
        if rel[1] > 0:
            best_actions.append(3)  # left
        elif rel[1] < 0:
            best_actions.append(4)  # right

        if len(best_actions) == 0:
            return np.random.randint(1, NUM_ACTIONS)
        return np.random.choice(best_actions)

    def step(self, action1, action2):
        \"\"\"Execute actions and return observations, rewards, done.\"\"\"
        # Move predators
        m1 = MOVES[action1]
        m2 = MOVES[action2]
        self.pred1_pos = self._wrap((self.pred1_pos[0] + m1[0], self.pred1_pos[1] + m1[1]))
        self.pred2_pos = self._wrap((self.pred2_pos[0] + m2[0], self.pred2_pos[1] + m2[1]))

        # Check capture before prey moves
        captured = (self.pred1_pos == self.prey_pos) or (self.pred2_pos == self.prey_pos)

        if not captured:
            # Move prey
            prey_action = self._prey_policy()
            pm = MOVES[prey_action]
            self.prey_pos = self._wrap((self.prey_pos[0] + pm[0], self.prey_pos[1] + pm[1]))
            # Check capture after prey moves
            captured = (self.pred1_pos == self.prey_pos) or (self.pred2_pos == self.prey_pos)

        obs1, obs2 = self._get_obs()

        if captured:
            return obs1, obs2, 10.0, True

        # Small penalty per step to encourage faster capture
        return obs1, obs2, -0.1, False


env = PredatorPreyEnv()
obs = env.reset()
print(f"Predator 1 observation shape: {len(obs[0])} values (relative positions)")
print(f"Sample obs: pred1={obs[0]}, pred2={obs[1]}")"""))

    cells.append(md("""## 4. Independent Q-Learning Agents"""))

    cells.append(code("""class IQLAgent:
    \"\"\"Independent Q-Learning agent using a hash-based Q-table.\"\"\"

    def __init__(self, num_actions=NUM_ACTIONS, alpha=ALPHA, gamma=GAMMA):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}

    def _get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        return self.q_table[state]

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        q = self._get_q(state)
        return np.argmax(q)

    def update(self, state, action, reward, next_state, done):
        q = self._get_q(state)
        q_next = self._get_q(next_state)
        target = reward + (0 if done else self.gamma * np.max(q_next))
        q[action] += self.alpha * (target - q[action])


# Create agents
agent1 = IQLAgent()
agent2 = IQLAgent()

# Training
epsilon = EPSILON_START
capture_history = []
reward_history = []
steps_history = []

for ep in range(NUM_EPISODES):
    obs1, obs2 = env.reset()
    total_reward = 0.0
    captured = False

    for step in range(MAX_STEPS):
        a1 = agent1.get_action(obs1, epsilon)
        a2 = agent2.get_action(obs2, epsilon)

        next_obs1, next_obs2, reward, done = env.step(a1, a2)

        agent1.update(obs1, a1, reward, next_obs1, done)
        agent2.update(obs2, a2, reward, next_obs2, done)

        total_reward += reward
        obs1, obs2 = next_obs1, next_obs2

        if done:
            captured = True
            break

    capture_history.append(1 if captured else 0)
    reward_history.append(total_reward)
    steps_history.append(step + 1)
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

print(f"Training complete!")
print(f"  Q-table sizes: Agent1={len(agent1.q_table)}, Agent2={len(agent2.q_table)}")
print(f"  Final epsilon: {epsilon:.4f}")
print(f"  Overall capture rate: {np.mean(capture_history):.4f}")"""))

    cells.append(code("""# Smooth and plot training metrics
def moving_average(data, window=100):
    return np.convolve(data, np.ones(window)/window, mode='valid')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Capture rate
cap_smooth = moving_average(capture_history, 200)
axes[0].plot(cap_smooth, 'g-', linewidth=2)
axes[0].set_xlabel('Episode', fontsize=12)
axes[0].set_ylabel('Capture Rate', fontsize=12)
axes[0].set_title('Capture Rate Over Training (Moving Avg)', fontsize=14)
axes[0].set_ylim([0, 1.05])
axes[0].grid(True, alpha=0.3)

# Reward
rew_smooth = moving_average(reward_history, 200)
axes[1].plot(rew_smooth, 'b-', linewidth=2)
axes[1].set_xlabel('Episode', fontsize=12)
axes[1].set_ylabel('Episode Reward', fontsize=12)
axes[1].set_title('Episode Reward Over Training', fontsize=14)
axes[1].grid(True, alpha=0.3)

# Steps to capture
steps_smooth = moving_average(steps_history, 200)
axes[2].plot(steps_smooth, 'r-', linewidth=2)
axes[2].set_xlabel('Episode', fontsize=12)
axes[2].set_ylabel('Steps', fontsize=12)
axes[2].set_title('Steps per Episode Over Training', fontsize=14)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 5. Visualization: Grid World and Trajectories"""))

    cells.append(code("""def visualize_episode(env, agent1, agent2, max_steps=30):
    \"\"\"Run and visualize a single episode.\"\"\"
    obs1, obs2 = env.reset()
    trajectory = {
        'pred1': [env.pred1_pos],
        'pred2': [env.pred2_pos],
        'prey': [env.prey_pos]
    }

    for step in range(max_steps):
        a1 = agent1.get_action(obs1, epsilon=0.0)
        a2 = agent2.get_action(obs2, epsilon=0.0)
        obs1, obs2, reward, done = env.step(a1, a2)
        trajectory['pred1'].append(env.pred1_pos)
        trajectory['pred2'].append(env.pred2_pos)
        trajectory['prey'].append(env.prey_pos)
        if done:
            break

    return trajectory, done


# Run a few episodes and visualize
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx in range(6):
    ax = axes[idx // 3][idx % 3]
    traj, captured = visualize_episode(env, agent1, agent2)

    # Draw grid
    for i in range(GRID_SIZE + 1):
        ax.axhline(y=i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        ax.axvline(x=i - 0.5, color='gray', linewidth=0.5, alpha=0.3)

    # Plot trajectories
    p1 = np.array(traj['pred1'])
    p2 = np.array(traj['pred2'])
    pr = np.array(traj['prey'])

    ax.plot(p1[:, 1], p1[:, 0], 'b-o', markersize=4, alpha=0.6, label='Predator 1')
    ax.plot(p2[:, 1], p2[:, 0], 'r-s', markersize=4, alpha=0.6, label='Predator 2')
    ax.plot(pr[:, 1], pr[:, 0], 'g-^', markersize=4, alpha=0.6, label='Prey')

    # Mark start and end
    ax.plot(p1[0, 1], p1[0, 0], 'bD', markersize=10)
    ax.plot(p2[0, 1], p2[0, 0], 'rD', markersize=10)
    ax.plot(pr[0, 1], pr[0, 0], 'gD', markersize=10)

    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_aspect('equal')
    result = "CAPTURED" if captured else "ESCAPED"
    ax.set_title(f'Episode {idx+1}: {result} ({len(traj["pred1"])-1} steps)', fontsize=12)
    ax.invert_yaxis()
    if idx == 0:
        ax.legend(fontsize=9, loc='upper right')

plt.suptitle('Sample Episodes: Predator-Prey Chase', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 6. Cooperation Analysis

Let us analyze whether cooperation emerges by looking at:
- Distance between predators when capture occurs
- Coordination patterns (approaching prey from different directions)"""))

    cells.append(code("""# Analyze cooperation patterns
capture_distances = []  # Distance between predators at capture
approach_angles = []     # Whether predators approach from different sides

num_analysis = 500
for _ in range(num_analysis):
    obs1, obs2 = env.reset()
    for step in range(MAX_STEPS):
        a1 = agent1.get_action(obs1, epsilon=0.0)
        a2 = agent2.get_action(obs2, epsilon=0.0)
        obs1, obs2, reward, done = env.step(a1, a2)

        if done:
            d = env._manhattan_dist(env.pred1_pos, env.pred2_pos)
            capture_distances.append(d)
            # Check if approaching from different sides
            rel1 = env._relative_pos(env.prey_pos, env.pred1_pos)
            rel2 = env._relative_pos(env.prey_pos, env.pred2_pos)
            # Different sides if relative positions are in opposite directions
            same_side = (np.sign(rel1[0]) == np.sign(rel2[0])) and (np.sign(rel1[1]) == np.sign(rel2[1]))
            approach_angles.append(0 if same_side else 1)
            break

print(f"Capture analysis ({len(capture_distances)} captures out of {num_analysis} episodes):")
print(f"  Mean distance between predators at capture: {np.mean(capture_distances):.2f}")
print(f"  Fraction approaching from different sides: {np.mean(approach_angles):.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(capture_distances, bins=range(GRID_SIZE + 2), align='left',
             color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Manhattan Distance Between Predators', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Predator Distance at Capture', fontsize=14)
axes[0].grid(True, alpha=0.3, axis='y')

# Cooperation over training: compare early vs late episodes
early_cap = np.mean(capture_history[:500])
late_cap = np.mean(capture_history[-500:])
categories = ['Early Training\n(first 500 ep)', 'Late Training\n(last 500 ep)']
values = [early_cap, late_cap]
bars = axes[1].bar(categories, values, color=['salmon', 'lightgreen'], edgecolor='black')
axes[1].set_ylabel('Capture Rate', fontsize=12)
axes[1].set_title('Capture Rate: Early vs Late Training', fontsize=14)
axes[1].set_ylim([0, 1.1])
for bar, val in zip(bars, values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 7. Summary

This notebook demonstrated Multi-Agent RL in a predator-prey setting:

1. **Environment:** Toroidal grid world with 2 predators chasing 1 prey
2. **Algorithm:** Independent Q-Learning (IQL) for each predator
3. **Results:** Predators learned to capture prey with increasing efficiency
4. **Cooperation:** Even with independent learning, implicit cooperation emerged

Key insights:
- IQL is simple but suffers from non-stationarity (each agent's environment changes as others learn)
- Despite this, emergent cooperation can arise when agents share a common reward signal
- The capture rate improved significantly over training as agents learned complementary strategies
- Multi-agent credit assignment remains challenging -- shared rewards help but are not always sufficient"""))

    path = os.path.join(BASE, "Project_03_Multi_Agent_RL", "marl_notebook.ipynb")
    save(path, cells)


# ===================================================================
# Notebook 4: Safe RL
# ===================================================================
def make_notebook_04():
    cells = []

    cells.append(md("""# Project 4: Safe Reinforcement Learning with Constrained Optimization

**Author:** Milan Amrut Joshi

## Overview
This notebook implements **Safe RL** in a grid world with dangerous zones.
We compare:
1. **Standard Q-Learning:** Maximizes reward, ignoring safety constraints
2. **Lagrangian Q-Learning:** Incorporates safety constraints via Lagrangian relaxation

The goal is to reach a target while avoiding dangerous cells, subject to a
constraint on the cumulative cost (visits to dangerous zones).

$$\\max_\\pi \\mathbb{E}[\\sum_t \\gamma^t r_t] \\quad \\text{s.t.} \\quad \\mathbb{E}[\\sum_t \\gamma^t c_t] \\leq d$$

where $c_t$ is the cost at time $t$ and $d$ is the cost threshold."""))

    cells.append(md("""## 1. Constrained MDP Framework

### Standard (Unconstrained) MDP
$$\\pi^* = \\arg\\max_\\pi \\mathbb{E}_\\pi \\left[\\sum_{t=0}^\\infty \\gamma^t r(s_t, a_t)\\right]$$

### Constrained MDP (CMDP)
$$\\pi^* = \\arg\\max_\\pi \\mathbb{E}_\\pi \\left[\\sum_{t=0}^\\infty \\gamma^t r_t\\right] \\quad \\text{subject to} \\quad \\mathbb{E}_\\pi \\left[\\sum_{t=0}^\\infty \\gamma^t c_t\\right] \\leq d$$

### Lagrangian Relaxation
Convert the constrained problem to an unconstrained one:

$$\\mathcal{L}(\\pi, \\lambda) = \\mathbb{E}_\\pi \\left[\\sum_t \\gamma^t r_t\\right] - \\lambda \\left(\\mathbb{E}_\\pi \\left[\\sum_t \\gamma^t c_t\\right] - d\\right)$$

The Lagrange multiplier $\\lambda \\geq 0$ is updated via dual gradient ascent:

$$\\lambda \\leftarrow \\max(0, \\lambda + \\eta_\\lambda (\\hat{C} - d))$$

where $\\hat{C}$ is the observed episode cost and $d$ is the threshold."""))

    cells.append(md("""## 2. Setup"""))

    cells.append(code("""import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Grid world configuration
GRID_H = 8
GRID_W = 8
START = (0, 0)
GOAL = (7, 7)
COST_THRESHOLD = 2.0     # Maximum allowed cumulative cost

# Actions: up, down, left, right
NUM_ACTIONS = 4
ACTION_NAMES = ['Up', 'Down', 'Left', 'Right']
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Training parameters
NUM_EPISODES = 3000
ALPHA = 0.2
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.998

print("Safe RL Grid World Configuration:")
print(f"  Grid: {GRID_H}x{GRID_W}")
print(f"  Start: {START}, Goal: {GOAL}")
print(f"  Cost threshold: {COST_THRESHOLD}")"""))

    cells.append(md("""## 3. Grid World with Dangerous Zones"""))

    cells.append(code("""class SafeGridWorld:
    \"\"\"Grid world with reward and cost (danger) signals.\"\"\"

    def __init__(self, grid_h=GRID_H, grid_w=GRID_W):
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Define danger zones (cost = 1 when visited)
        self.danger_map = np.zeros((grid_h, grid_w))
        # Create a dangerous corridor
        danger_cells = [
            (1, 2), (1, 3), (1, 4), (1, 5),
            (2, 2), (2, 5),
            (3, 2), (3, 3), (3, 4), (3, 5),
            (4, 4), (4, 5), (4, 6),
            (5, 1), (5, 2), (5, 3),
            (6, 3), (6, 4), (6, 5),
        ]
        for r, c in danger_cells:
            self.danger_map[r, c] = 1.0

        # Reward: -0.1 per step, +10 at goal, bonus for approaching goal
        self.goal = GOAL

    def reset(self):
        self.state = START
        return self.state

    def step(self, action):
        r, c = self.state
        dr, dc = MOVES[action]
        nr, nc = r + dr, c + dc

        # Check boundaries
        if 0 <= nr < self.grid_h and 0 <= nc < self.grid_w:
            self.state = (nr, nc)
        # else stay in place

        reward = -0.1
        cost = self.danger_map[self.state[0], self.state[1]]
        done = (self.state == self.goal)
        if done:
            reward = 10.0

        return self.state, reward, cost, done


env = SafeGridWorld()

# Visualize the grid
fig, ax = plt.subplots(figsize=(8, 8))
grid_display = np.zeros((GRID_H, GRID_W, 3))

for r in range(GRID_H):
    for c in range(GRID_W):
        if env.danger_map[r, c] > 0:
            grid_display[r, c] = [1.0, 0.3, 0.3]  # Red for danger
        else:
            grid_display[r, c] = [0.9, 0.9, 0.9]  # Light gray for safe

# Mark start and goal
grid_display[START[0], START[1]] = [0.3, 0.7, 0.3]  # Green for start
grid_display[GOAL[0], GOAL[1]] = [0.3, 0.3, 1.0]    # Blue for goal

ax.imshow(grid_display, origin='upper')
for r in range(GRID_H):
    for c in range(GRID_W):
        if env.danger_map[r, c] > 0:
            ax.text(c, r, 'X', ha='center', va='center', fontsize=14, color='darkred', fontweight='bold')
        if (r, c) == START:
            ax.text(c, r, 'S', ha='center', va='center', fontsize=16, color='darkgreen', fontweight='bold')
        if (r, c) == GOAL:
            ax.text(c, r, 'G', ha='center', va='center', fontsize=16, color='white', fontweight='bold')

ax.set_title('Grid World: S=Start, G=Goal, X=Danger', fontsize=14)
ax.set_xticks(range(GRID_W))
ax.set_yticks(range(GRID_H))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 4. Standard Q-Learning (Ignores Constraints)"""))

    cells.append(code("""class StandardQLearning:
    \"\"\"Standard Q-learning that maximizes reward only.\"\"\"

    def __init__(self, grid_h, grid_w, num_actions):
        self.q_table = np.zeros((grid_h, grid_w, num_actions))
        self.num_actions = num_actions

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.q_table[state[0], state[1]])

    def update(self, state, action, reward, next_state, done, alpha=ALPHA, gamma=GAMMA):
        r, c = state
        nr, nc = next_state
        target = reward + (0 if done else gamma * np.max(self.q_table[nr, nc]))
        self.q_table[r, c, action] += alpha * (target - self.q_table[r, c, action])


# Train standard Q-learning
std_agent = StandardQLearning(GRID_H, GRID_W, NUM_ACTIONS)
std_rewards = []
std_costs = []
epsilon = EPSILON_START

for ep in range(NUM_EPISODES):
    state = env.reset()
    ep_reward = 0.0
    ep_cost = 0.0

    for step in range(200):
        action = std_agent.get_action(state, epsilon)
        next_state, reward, cost, done = env.step(action)
        std_agent.update(state, action, reward, next_state, done)
        ep_reward += reward
        ep_cost += cost
        state = next_state
        if done:
            break

    std_rewards.append(ep_reward)
    std_costs.append(ep_cost)
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

print(f"Standard Q-Learning complete.")
print(f"  Mean reward (last 500): {np.mean(std_rewards[-500:]):.3f}")
print(f"  Mean cost (last 500): {np.mean(std_costs[-500:]):.3f}")
print(f"  Cost threshold: {COST_THRESHOLD}")"""))

    cells.append(md("""## 5. Lagrangian Q-Learning (Safety-Constrained)

The Lagrangian approach modifies the effective reward:

$$r_{\\text{eff}}(s, a) = r(s, a) - \\lambda \\cdot c(s, a)$$

The multiplier $\\lambda$ increases when the cost exceeds the threshold and decreases otherwise."""))

    cells.append(code("""class LagrangianQLearning:
    \"\"\"Q-learning with Lagrangian constraint for safe RL.\"\"\"

    def __init__(self, grid_h, grid_w, num_actions, cost_threshold):
        self.q_table = np.zeros((grid_h, grid_w, num_actions))
        self.q_cost = np.zeros((grid_h, grid_w, num_actions))  # Cost Q-function
        self.num_actions = num_actions
        self.lam = 0.0                   # Lagrange multiplier
        self.cost_threshold = cost_threshold
        self.lr_lambda = 0.05            # Learning rate for lambda

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        r, c = state
        # Effective Q = Q_reward - lambda * Q_cost
        effective_q = self.q_table[r, c] - self.lam * self.q_cost[r, c]
        return np.argmax(effective_q)

    def update(self, state, action, reward, cost, next_state, done,
               alpha=ALPHA, gamma=GAMMA):
        r, c = state
        nr, nc = next_state

        # Update reward Q-function
        target_r = reward + (0 if done else gamma * np.max(self.q_table[nr, nc]))
        self.q_table[r, c, action] += alpha * (target_r - self.q_table[r, c, action])

        # Update cost Q-function
        target_c = cost + (0 if done else gamma * np.max(self.q_cost[nr, nc]))
        self.q_cost[r, c, action] += alpha * (target_c - self.q_cost[r, c, action])

    def update_lambda(self, episode_cost):
        \"\"\"Dual gradient ascent on the Lagrange multiplier.\"\"\"
        self.lam = max(0.0, self.lam + self.lr_lambda * (episode_cost - self.cost_threshold))


# Train Lagrangian Q-learning
lag_agent = LagrangianQLearning(GRID_H, GRID_W, NUM_ACTIONS, COST_THRESHOLD)
lag_rewards = []
lag_costs = []
lag_lambdas = []
epsilon = EPSILON_START

for ep in range(NUM_EPISODES):
    state = env.reset()
    ep_reward = 0.0
    ep_cost = 0.0

    for step in range(200):
        action = lag_agent.get_action(state, epsilon)
        next_state, reward, cost, done = env.step(action)
        lag_agent.update(state, action, reward, cost, next_state, done)
        ep_reward += reward
        ep_cost += cost
        state = next_state
        if done:
            break

    # Update Lagrange multiplier
    lag_agent.update_lambda(ep_cost)

    lag_rewards.append(ep_reward)
    lag_costs.append(ep_cost)
    lag_lambdas.append(lag_agent.lam)
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

print(f"Lagrangian Q-Learning complete.")
print(f"  Mean reward (last 500): {np.mean(lag_rewards[-500:]):.3f}")
print(f"  Mean cost (last 500): {np.mean(lag_costs[-500:]):.3f}")
print(f"  Final lambda: {lag_agent.lam:.4f}")"""))

    cells.append(code("""# Compare training curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

def smooth(data, w=100):
    return np.convolve(data, np.ones(w)/w, mode='valid')

# Rewards
axes[0][0].plot(smooth(std_rewards), 'b-', linewidth=2, label='Standard Q')
axes[0][0].plot(smooth(lag_rewards), 'r-', linewidth=2, label='Lagrangian Q')
axes[0][0].set_xlabel('Episode', fontsize=12)
axes[0][0].set_ylabel('Episode Reward', fontsize=12)
axes[0][0].set_title('Reward Comparison', fontsize=14)
axes[0][0].legend(fontsize=11)
axes[0][0].grid(True, alpha=0.3)

# Costs
axes[0][1].plot(smooth(std_costs), 'b-', linewidth=2, label='Standard Q')
axes[0][1].plot(smooth(lag_costs), 'r-', linewidth=2, label='Lagrangian Q')
axes[0][1].axhline(y=COST_THRESHOLD, color='k', linestyle='--', linewidth=2, label=f'Threshold={COST_THRESHOLD}')
axes[0][1].set_xlabel('Episode', fontsize=12)
axes[0][1].set_ylabel('Episode Cost', fontsize=12)
axes[0][1].set_title('Cost Comparison (Safety)', fontsize=14)
axes[0][1].legend(fontsize=11)
axes[0][1].grid(True, alpha=0.3)

# Lambda evolution
axes[1][0].plot(lag_lambdas, 'purple', linewidth=2)
axes[1][0].set_xlabel('Episode', fontsize=12)
axes[1][0].set_ylabel('Lambda', fontsize=12)
axes[1][0].set_title('Lagrange Multiplier Evolution', fontsize=14)
axes[1][0].grid(True, alpha=0.3)

# Reward-cost tradeoff scatter
window = 200
std_r_bins = [np.mean(std_rewards[i:i+window]) for i in range(0, len(std_rewards)-window, window)]
std_c_bins = [np.mean(std_costs[i:i+window]) for i in range(0, len(std_costs)-window, window)]
lag_r_bins = [np.mean(lag_rewards[i:i+window]) for i in range(0, len(lag_rewards)-window, window)]
lag_c_bins = [np.mean(lag_costs[i:i+window]) for i in range(0, len(lag_costs)-window, window)]

axes[1][1].scatter(std_c_bins, std_r_bins, c='blue', s=60, alpha=0.7, label='Standard Q')
axes[1][1].scatter(lag_c_bins, lag_r_bins, c='red', s=60, alpha=0.7, label='Lagrangian Q')
axes[1][1].axvline(x=COST_THRESHOLD, color='k', linestyle='--', linewidth=2, label='Cost Threshold')
axes[1][1].set_xlabel('Mean Episode Cost', fontsize=12)
axes[1][1].set_ylabel('Mean Episode Reward', fontsize=12)
axes[1][1].set_title('Reward-Cost Tradeoff', fontsize=14)
axes[1][1].legend(fontsize=11)
axes[1][1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 6. Visualize Safe vs Unsafe Paths"""))

    cells.append(code("""def extract_path(env, agent, max_steps=100, use_lagrangian=False):
    \"\"\"Extract the greedy path from start to goal.\"\"\"
    state = env.reset()
    path = [state]
    total_cost = 0.0
    for _ in range(max_steps):
        if use_lagrangian:
            action = agent.get_action(state, epsilon=0.0)
        else:
            action = agent.get_action(state, epsilon=0.0)
        next_state, reward, cost, done = env.step(action)
        path.append(next_state)
        total_cost += cost
        state = next_state
        if done:
            break
    return path, total_cost


std_path, std_path_cost = extract_path(env, std_agent)
lag_path, lag_path_cost = extract_path(env, lag_agent, use_lagrangian=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, path, path_cost, title in [
    (axes[0], std_path, std_path_cost, 'Standard Q-Learning (Unsafe)'),
    (axes[1], lag_path, lag_path_cost, 'Lagrangian Q-Learning (Safe)')
]:
    # Draw grid
    grid_display = np.zeros((GRID_H, GRID_W, 3))
    for r in range(GRID_H):
        for c in range(GRID_W):
            if env.danger_map[r, c] > 0:
                grid_display[r, c] = [1.0, 0.85, 0.85]
            else:
                grid_display[r, c] = [0.95, 0.95, 0.95]

    ax.imshow(grid_display, origin='upper')

    # Draw danger zones
    for r in range(GRID_H):
        for c in range(GRID_W):
            if env.danger_map[r, c] > 0:
                ax.text(c, r, 'X', ha='center', va='center', fontsize=10,
                       color='red', alpha=0.5)

    # Draw path
    path_arr = np.array(path)
    ax.plot(path_arr[:, 1], path_arr[:, 0], 'b-o', markersize=6, linewidth=2.5,
           alpha=0.8, zorder=5)
    ax.plot(path_arr[0, 1], path_arr[0, 0], 'gs', markersize=15, zorder=6, label='Start')
    ax.plot(path_arr[-1, 1], path_arr[-1, 0], 'r*', markersize=18, zorder=6, label='End')

    ax.set_title(f'{title}\nPath cost: {path_cost:.1f}, Steps: {len(path)-1}', fontsize=13)
    ax.set_xlim(-0.5, GRID_W - 0.5)
    ax.set_ylim(GRID_H - 0.5, -0.5)
    ax.set_xticks(range(GRID_W))
    ax.set_yticks(range(GRID_H))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 7. Constraint Satisfaction Analysis"""))

    cells.append(code("""# Run many test episodes and check constraint satisfaction
num_test = 500
std_test_costs = []
lag_test_costs = []

for _ in range(num_test):
    # Standard Q
    state = env.reset()
    cost_sum = 0.0
    for step in range(200):
        action = std_agent.get_action(state, epsilon=0.0)
        next_state, _, cost, done = env.step(action)
        cost_sum += cost
        state = next_state
        if done:
            break
    std_test_costs.append(cost_sum)

    # Lagrangian Q
    state = env.reset()
    cost_sum = 0.0
    for step in range(200):
        action = lag_agent.get_action(state, epsilon=0.0)
        next_state, _, cost, done = env.step(action)
        cost_sum += cost
        state = next_state
        if done:
            break
    lag_test_costs.append(cost_sum)

std_violation_rate = np.mean(np.array(std_test_costs) > COST_THRESHOLD)
lag_violation_rate = np.mean(np.array(lag_test_costs) > COST_THRESHOLD)

print(f"=== Constraint Satisfaction (threshold={COST_THRESHOLD}) ===")
print(f"Standard Q: mean cost={np.mean(std_test_costs):.3f}, violation rate={std_violation_rate:.3f}")
print(f"Lagrangian Q: mean cost={np.mean(lag_test_costs):.3f}, violation rate={lag_violation_rate:.3f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(std_test_costs, bins=30, alpha=0.6, label='Standard Q', color='blue')
ax.hist(lag_test_costs, bins=30, alpha=0.6, label='Lagrangian Q', color='red')
ax.axvline(x=COST_THRESHOLD, color='black', linestyle='--', linewidth=2, label=f'Threshold={COST_THRESHOLD}')
ax.set_xlabel('Episode Cost', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Cost Distribution: Standard vs Lagrangian Q-Learning', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 8. Summary

This notebook demonstrated Safe RL using constrained optimization:

1. **Standard Q-Learning** finds the shortest path but may traverse dangerous zones
2. **Lagrangian Q-Learning** learns to avoid danger by incorporating cost constraints
3. The Lagrange multiplier **automatically adjusts** the safety-reward tradeoff
4. Constraint satisfaction analysis shows the Lagrangian approach respects the cost budget

Key takeaways:
- Safety constraints are critical in real-world RL (robotics, healthcare, autonomous driving)
- The Lagrangian approach is simple yet effective for constrained MDPs
- The lambda update rule provides an adaptive mechanism for balancing reward and safety
- There is an inherent tradeoff: safer policies may achieve lower reward"""))

    path = os.path.join(BASE, "Project_04_Safe_RL", "safe_rl_notebook.ipynb")
    save(path, cells)


# ===================================================================
# Notebook 5: World Models and Model-Based RL
# ===================================================================
def make_notebook_05():
    cells = []

    cells.append(md("""# Project 5: World Models and Model-Based RL (Dyna-Q)

**Author:** Milan Amrut Joshi

## Overview
This notebook implements **World Models** and **Model-Based RL** using the Dyna-Q architecture.
The key idea is to learn a model of the environment's dynamics and use it for planning,
which dramatically improves sample efficiency compared to model-free methods.

We implement:
1. **Stochastic Grid World:** An environment with probabilistic transitions
2. **Dynamics Model:** A learned model that predicts next states from (state, action) pairs
3. **Dyna-Q:** Combines real experience with simulated experience from the learned model
4. **Comparison:** Model-free Q-learning vs Dyna-Q learning curves"""))

    cells.append(md("""## 1. Model-Based RL Theory

### Dyna-Q Architecture
Dyna-Q (Sutton, 1991) integrates learning and planning:

For each real experience $(s, a, r, s')$:
1. **Direct RL:** Update Q-values from real experience
2. **Model Learning:** Update the dynamics model $\\hat{T}(s'|s,a)$ and $\\hat{R}(s,a)$
3. **Planning:** Sample $k$ previously visited $(s,a)$ pairs, use the model to generate
   simulated transitions, and update Q-values from these imagined experiences

### Dynamics Model
$$\\hat{s}' = f_\\theta(s, a), \\quad \\hat{r} = g_\\theta(s, a)$$

The model predicts the next state and reward given the current state and action.

### Planning Updates
For $k$ planning steps:
$$Q(s, a) \\leftarrow Q(s, a) + \\alpha [\\hat{r} + \\gamma \\max_{a'} Q(\\hat{s}', a') - Q(s, a)]$$

### Sample Efficiency
Model-based methods can achieve $k+1$ Q-updates per real interaction,
compared to 1 update for model-free methods."""))

    cells.append(md("""## 2. Setup"""))

    cells.append(code("""import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Grid world configuration
GRID_H = 8
GRID_W = 8
START = (0, 0)
GOAL = (7, 7)

# Actions: up, down, left, right
NUM_ACTIONS = 4
ACTION_NAMES = ['Up', 'Down', 'Left', 'Right']
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Training config
NUM_EPISODES = 300
ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
PLANNING_STEPS = [0, 5, 20, 50]  # Different amounts of planning

print("World Models / Dyna-Q Configuration:")
print(f"  Grid: {GRID_H}x{GRID_W}")
print(f"  Episodes: {NUM_EPISODES}")
print(f"  Planning steps tested: {PLANNING_STEPS}")"""))

    cells.append(md("""## 3. Stochastic Grid World Environment

This grid world has:
- **Stochastic transitions:** 80% chance of intended move, 20% chance of perpendicular slip
- **Walls:** Internal walls that block movement
- **Reward:** -0.1 per step, +10 at goal"""))

    cells.append(code("""class StochasticGridWorld:
    \"\"\"Grid world with stochastic transitions and internal walls.\"\"\"

    def __init__(self):
        self.grid_h = GRID_H
        self.grid_w = GRID_W
        self.goal = GOAL
        self.slip_prob = 0.2  # Probability of slipping

        # Internal walls (blocked transitions)
        # Format: set of ((r1,c1), (r2,c2)) meaning movement between these cells is blocked
        self.walls = set()
        # Horizontal wall segment
        for c in range(2, 6):
            self.walls.add(((3, c), (4, c)))
            self.walls.add(((4, c), (3, c)))
        # Vertical wall segment
        for r in range(1, 4):
            self.walls.add(((r, 5), (r, 6)))
            self.walls.add(((r, 6), (r, 5)))

    def reset(self):
        self.state = START
        return self.state

    def _is_valid_move(self, from_pos, to_pos):
        \"\"\"Check if movement is valid (within bounds and no wall).\"\"\"
        r, c = to_pos
        if r < 0 or r >= self.grid_h or c < 0 or c >= self.grid_w:
            return False
        if (from_pos, to_pos) in self.walls:
            return False
        return True

    def step(self, action):
        r, c = self.state

        # Stochastic transition
        if np.random.rand() < self.slip_prob:
            # Slip to a perpendicular direction
            if action in [0, 1]:  # up/down -> slip left/right
                action = np.random.choice([2, 3])
            else:  # left/right -> slip up/down
                action = np.random.choice([0, 1])

        dr, dc = MOVES[action]
        new_pos = (r + dr, c + dc)

        if self._is_valid_move(self.state, new_pos):
            self.state = new_pos

        reward = -0.1
        done = (self.state == self.goal)
        if done:
            reward = 10.0

        return self.state, reward, done

    def get_wall_segments(self):
        \"\"\"Return wall segments for visualization.\"\"\"
        segments = []
        processed = set()
        for (r1, c1), (r2, c2) in self.walls:
            key = (min(r1, r2), min(c1, c2), max(r1, r2), max(c1, c2))
            if key not in processed:
                processed.add(key)
                segments.append(((r1, c1), (r2, c2)))
        return segments


env = StochasticGridWorld()

# Visualize the environment
fig, ax = plt.subplots(figsize=(8, 8))
grid_display = np.ones((GRID_H, GRID_W, 3)) * 0.95

grid_display[START[0], START[1]] = [0.3, 0.8, 0.3]
grid_display[GOAL[0], GOAL[1]] = [0.3, 0.3, 1.0]

ax.imshow(grid_display, origin='upper')

# Draw walls
for (r1, c1), (r2, c2) in env.get_wall_segments():
    if r1 != r2:  # Horizontal wall between rows
        row = max(r1, r2)
        ax.plot([c1 - 0.5, c1 + 0.5], [row - 0.5, row - 0.5], 'k-', linewidth=3)
    else:  # Vertical wall between columns
        col = max(c1, c2)
        ax.plot([col - 0.5, col - 0.5], [r1 - 0.5, r1 + 0.5], 'k-', linewidth=3)

ax.text(START[1], START[0], 'S', ha='center', va='center', fontsize=18, fontweight='bold', color='darkgreen')
ax.text(GOAL[1], GOAL[0], 'G', ha='center', va='center', fontsize=18, fontweight='bold', color='white')

ax.set_title('Stochastic Grid World with Walls', fontsize=14)
ax.set_xticks(range(GRID_W))
ax.set_yticks(range(GRID_H))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 4. Learned Dynamics Model

We learn a tabular model that stores observed transitions.
For each (state, action), the model remembers the resulting (next_state, reward)."""))

    cells.append(code("""class TabularDynamicsModel:
    \"\"\"Tabular dynamics model that stores observed transitions.\"\"\"

    def __init__(self):
        self.transitions = {}  # (state, action) -> list of (next_state, reward)
        self.visited_sa = []   # List of visited (state, action) pairs

    def update(self, state, action, next_state, reward):
        \"\"\"Store an observed transition.\"\"\"
        key = (state, action)
        if key not in self.transitions:
            self.transitions[key] = []
            self.visited_sa.append(key)
        self.transitions[key].append((next_state, reward))

    def predict(self, state, action):
        \"\"\"Predict next state and reward by sampling from stored transitions.\"\"\"
        key = (state, action)
        if key in self.transitions:
            idx = np.random.randint(len(self.transitions[key]))
            return self.transitions[key][idx]
        return None

    def sample_visited(self):
        \"\"\"Sample a random previously visited (state, action) pair.\"\"\"
        if len(self.visited_sa) == 0:
            return None
        idx = np.random.randint(len(self.visited_sa))
        return self.visited_sa[idx]

    def prediction_accuracy(self, test_transitions):
        \"\"\"Compute prediction accuracy on a test set.\"\"\"
        correct = 0
        total = 0
        for state, action, true_next, true_reward in test_transitions:
            pred = self.predict(state, action)
            if pred is not None:
                pred_next, pred_reward = pred
                if pred_next == true_next:
                    correct += 1
                total += 1
        return correct / max(total, 1)


print("Dynamics model class ready.")
print("The model stores observed (s, a) -> (s', r) transitions.")
print("For planning, it samples from stored transitions to simulate experience.")"""))

    cells.append(md("""## 5. Dyna-Q Implementation

Dyna-Q combines direct RL, model learning, and planning in a single loop."""))

    cells.append(code("""def run_dyna_q(env, num_episodes, planning_steps, alpha=ALPHA, gamma=GAMMA):
    \"\"\"Run Dyna-Q with a given number of planning steps.\"\"\"
    q_table = np.zeros((GRID_H, GRID_W, NUM_ACTIONS))
    model = TabularDynamicsModel()
    epsilon = EPSILON_START

    episode_rewards = []
    episode_steps = []
    model_accuracies = []
    test_buffer = []

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0
        step_count = 0

        for step in range(500):
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(NUM_ACTIONS)
            else:
                action = np.argmax(q_table[state[0], state[1]])

            next_state, reward, done = env.step(action)

            # Direct RL update
            td_target = reward + (0 if done else gamma * np.max(q_table[next_state[0], next_state[1]]))
            q_table[state[0], state[1], action] += alpha * (td_target - q_table[state[0], state[1], action])

            # Model learning
            model.update(state, action, next_state, reward)

            # Store for accuracy testing
            if len(test_buffer) < 1000:
                test_buffer.append((state, action, next_state, reward))

            # Planning: use model for simulated updates
            for _ in range(planning_steps):
                sa = model.sample_visited()
                if sa is not None:
                    sim_state, sim_action = sa
                    result = model.predict(sim_state, sim_action)
                    if result is not None:
                        sim_next, sim_reward = result
                        sim_target = sim_reward + gamma * np.max(q_table[sim_next[0], sim_next[1]])
                        q_table[sim_state[0], sim_state[1], sim_action] += alpha * (
                            sim_target - q_table[sim_state[0], sim_state[1], sim_action])

            ep_reward += reward
            step_count += 1
            state = next_state

            if done:
                break

        episode_rewards.append(ep_reward)
        episode_steps.append(step_count)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Periodically check model accuracy
        if (ep + 1) % 10 == 0 and len(test_buffer) > 0:
            acc = model.prediction_accuracy(test_buffer[-100:])
            model_accuracies.append((ep, acc))

    return episode_rewards, episode_steps, model_accuracies, q_table, model


# Run experiments with different planning steps
results = {}
for k in PLANNING_STEPS:
    print(f"Running Dyna-Q with k={k} planning steps...")
    env_run = StochasticGridWorld()
    rewards, steps, acc, q, mdl = run_dyna_q(env_run, NUM_EPISODES, k)
    results[k] = {
        'rewards': rewards,
        'steps': steps,
        'accuracy': acc,
        'q_table': q,
        'model': mdl
    }
    print(f"  k={k}: Mean reward (last 50): {np.mean(rewards[-50:]):.3f}, "
          f"Mean steps (last 50): {np.mean(steps[-50:]):.1f}")"""))

    cells.append(code("""# Plot learning curves comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = ['gray', 'blue', 'green', 'red']
def smooth(data, w=20):
    if len(data) < w:
        return data
    return np.convolve(data, np.ones(w)/w, mode='valid')

for i, k in enumerate(PLANNING_STEPS):
    label = f'k={k} (Model-Free)' if k == 0 else f'k={k} planning'
    axes[0].plot(smooth(results[k]['rewards']), color=colors[i], linewidth=2, label=label, alpha=0.8)

axes[0].set_xlabel('Episode', fontsize=12)
axes[0].set_ylabel('Episode Reward', fontsize=12)
axes[0].set_title('Learning Curves: Model-Free vs Dyna-Q', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

for i, k in enumerate(PLANNING_STEPS):
    label = f'k={k}'
    axes[1].plot(smooth(results[k]['steps']), color=colors[i], linewidth=2, label=label, alpha=0.8)

axes[1].set_xlabel('Episode', fontsize=12)
axes[1].set_ylabel('Steps to Goal', fontsize=12)
axes[1].set_title('Steps per Episode: Faster Convergence with Planning', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 6. Model Prediction Accuracy Over Training

The dynamics model improves as more transitions are observed.
Let us track how accurately the model predicts next states."""))

    cells.append(code("""# Plot model accuracy for the k=20 case
fig, ax = plt.subplots(figsize=(10, 5))

for k in [5, 20, 50]:
    if len(results[k]['accuracy']) > 0:
        eps, accs = zip(*results[k]['accuracy'])
        ax.plot(eps, accs, 'o-', linewidth=2, markersize=4, label=f'k={k}', alpha=0.8)

ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Prediction Accuracy', fontsize=12)
ax.set_title('Dynamics Model Accuracy Over Training', fontsize=14)
ax.set_ylim([0, 1.05])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 7. Learned Value Function and Policy Visualization"""))

    cells.append(code("""# Visualize value functions for model-free (k=0) and Dyna-Q (k=20)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, k, title in [(axes[0], 0, 'Model-Free (k=0)'), (axes[1], 20, 'Dyna-Q (k=20)')]:
    q = results[k]['q_table']
    v = np.max(q, axis=2)

    im = ax.imshow(v, origin='upper', cmap='viridis', aspect='equal')
    plt.colorbar(im, ax=ax, label='V(s)')

    # Draw policy arrows
    arrow_map = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}
    for r in range(GRID_H):
        for c in range(GRID_W):
            best_a = np.argmax(q[r, c])
            if np.max(q[r, c]) > 0.01:  # Only draw if meaningful
                dx, dy = arrow_map[best_a]
                ax.arrow(c, r, dx, dy, head_width=0.15, head_length=0.08,
                        fc='white', ec='white', alpha=0.7)

    ax.set_title(f'{title}\nValue Function & Policy', fontsize=13)
    ax.set_xticks(range(GRID_W))
    ax.set_yticks(range(GRID_H))
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()"""))

    cells.append(md("""## 8. Sample Efficiency Analysis"""))

    cells.append(code("""# Compute cumulative reward over real environment steps
fig, ax = plt.subplots(figsize=(10, 6))

for i, k in enumerate(PLANNING_STEPS):
    cumulative = np.cumsum(results[k]['rewards'])
    real_steps = np.cumsum(results[k]['steps'])
    ax.plot(real_steps, cumulative, color=colors[i], linewidth=2,
           label=f'k={k} planning steps', alpha=0.8)

ax.set_xlabel('Total Real Environment Steps', fontsize=12)
ax.set_ylabel('Cumulative Reward', fontsize=12)
ax.set_title('Sample Efficiency: Cumulative Reward vs Real Steps', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Summary statistics
print("=== Sample Efficiency Summary ===")
for k in PLANNING_STEPS:
    total_steps = sum(results[k]['steps'])
    total_reward = sum(results[k]['rewards'])
    final_perf = np.mean(results[k]['rewards'][-50:])
    print(f"k={k:2d}: Total real steps={total_steps:6d}, "
          f"Total reward={total_reward:8.1f}, "
          f"Final performance={final_perf:.3f}")"""))

    cells.append(md("""## 9. Summary

This notebook demonstrated World Models and Model-Based RL:

1. **Stochastic Grid World:** Environment with probabilistic transitions and internal walls
2. **Dynamics Model:** Tabular model that stores and replays observed transitions
3. **Dyna-Q:** Combined real experience with model-based planning for faster learning
4. **Results:** More planning steps led to dramatically better sample efficiency

Key takeaways:
- Model-based RL learns much faster by reusing experience through planning
- The dynamics model accuracy improves as more transitions are observed
- There is a computation-vs-sample tradeoff: more planning = more compute per step but fewer steps needed
- In stochastic environments, the model naturally captures transition probabilities
- Dyna-Q is elegant: each real step generates k+1 learning updates instead of just 1"""))

    path = os.path.join(BASE, "Project_05_World_Models", "world_models_notebook.ipynb")
    save(path, cells)


# ===================================================================
# Main: Generate all notebooks
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating RL Project Notebooks 01-05")
    print("=" * 60)
    print()

    make_notebook_01()
    make_notebook_02()
    make_notebook_03()
    make_notebook_04()
    make_notebook_05()

    print()
    print("=" * 60)
    print("All 5 notebooks generated successfully!")
    print("=" * 60)
