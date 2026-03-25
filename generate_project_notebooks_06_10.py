import json
import os
import uuid

def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells
    }

def md(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().split("\n"),
        "id": str(uuid.uuid4())[:8]
    }

def code(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.strip().split("\n"),
        "execution_count": None,
        "outputs": [],
        "id": str(uuid.uuid4())[:8]
    }

def save(path, cells):
    for cell in cells:
        lines = cell["source"]
        cell["source"] = [l + "\n" if i < len(lines) - 1 else l for i, l in enumerate(lines)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb(cells), f, indent=1, ensure_ascii=False)
    print(f"Created {path}")


# =============================================================================
# Notebook 6: RLAIF
# =============================================================================
def make_notebook_06():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "Project_06_RLAIF", "rlaif_notebook.ipynb")
    cells = [
        md("""# Project 6: RLAIF - Reinforcement Learning from AI Feedback
**Author:** Milan Amrut Joshi

## Overview
This notebook explores **Reinforcement Learning from AI Feedback (RLAIF)** and compares it with the more traditional **RLHF (Reinforcement Learning from Human Feedback)** approach.

### Key Concepts
- **RLHF:** Uses human annotators to rank/score model outputs, then trains a reward model on those preferences.
- **RLAIF:** Replaces (or supplements) human annotators with an AI judge that scores outputs according to specified criteria.

### Why RLAIF?
1. **Scalability** - AI judges can label millions of examples cheaply
2. **Consistency** - Less inter-annotator disagreement
3. **Speed** - No waiting for human labeling pipelines

### Trade-offs
- AI judges have **systematic biases** (e.g., verbosity preference)
- Humans are **noisy** but capture nuances AI might miss
- Combining both may yield the best results"""),

        md(r"""## Mathematical Framework

### Response Representation
Each response $y$ is represented as a feature vector in a quality space:

$$y = [h, o, s] \in \mathbb{R}^3$$

where:
- $h$ = helpfulness score $\in [0, 1]$
- $o$ = honesty score $\in [0, 1]$
- $s$ = harmlessness (safety) score $\in [0, 1]$

### True Quality Function
The ground-truth quality of a response is:

$$Q^*(y) = w_h \cdot h + w_o \cdot o + w_s \cdot s$$

where $w_h, w_o, w_s$ are the true importance weights.

### Human Annotator Model
Human labels are noisy observations of true quality:

$$r_{\text{human}}(y) = Q^*(y) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_{\text{human}}^2)$$

### AI Judge Model
The AI judge has systematic bias but lower noise:

$$r_{\text{AI}}(y) = Q^*(y) + b(y) + \delta, \quad \delta \sim \mathcal{N}(0, \sigma_{\text{AI}}^2)$$

where $b(y)$ is a systematic bias term (e.g., preference for longer responses) and $\sigma_{\text{AI}} < \sigma_{\text{human}}$.

### Reward Model Training
We train reward models $\hat{R}_\theta$ by minimizing MSE on labeled data:

$$\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^N \left(\hat{R}_\theta(y_i) - r_i\right)^2$$

### Policy Optimization via REINFORCE
The policy gradient is:

$$\nabla_\phi J(\phi) = \mathbb{E}_{\pi_\phi}\left[\nabla_\phi \log \pi_\phi(y) \cdot \hat{R}_\theta(y)\right]$$"""),

        md("""## Setup and Imports"""),

        code("""import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
print("RLAIF vs RLHF Comparison - Setup Complete")"""),

        md("""## 1. Define the Response Space and True Quality Function

We simulate a pool of candidate responses, each characterized by three quality dimensions."""),

        code("""# Ground-truth importance weights
TRUE_WEIGHTS = np.array([0.4, 0.35, 0.25])  # helpfulness, honesty, harmlessness

def true_quality(features):
    '''Compute ground-truth quality of response feature vectors.'''
    return features @ TRUE_WEIGHTS

# Generate a pool of candidate responses
N_RESPONSES = 500
response_features = np.random.beta(2, 2, size=(N_RESPONSES, 3))  # features in [0,1]
true_scores = true_quality(response_features)

print(f"Response pool: {N_RESPONSES} responses")
print(f"Feature dimensions: helpfulness, honesty, harmlessness")
print(f"True quality range: [{true_scores.min():.3f}, {true_scores.max():.3f}]")
print(f"True quality mean: {true_scores.mean():.3f}")"""),

        md("""## 2. Human Annotator Simulation

Humans provide noisy but unbiased labels. We simulate inter-annotator variability."""),

        code("""HUMAN_NOISE_STD = 0.15  # substantial noise

def human_annotate(features, n_annotators=3):
    '''Simulate human annotations - noisy but unbiased on average.'''
    true_q = true_quality(features)
    annotations = []
    for _ in range(n_annotators):
        noise = np.random.normal(0, HUMAN_NOISE_STD, size=len(features))
        annotations.append(true_q + noise)
    # Average across annotators
    mean_annotation = np.mean(annotations, axis=0)
    return np.clip(mean_annotation, 0, 1)

human_labels = human_annotate(response_features)

# Measure annotation quality
human_error = np.mean((human_labels - true_scores)**2)
human_corr = np.corrcoef(human_labels, true_scores)[0, 1]
print(f"Human annotation MSE: {human_error:.4f}")
print(f"Human-True correlation: {human_corr:.4f}")"""),

        md("""## 3. AI Judge Simulation

The AI judge is more consistent (lower noise) but has systematic biases - it tends to overweight helpfulness and underweight harmlessness."""),

        code("""AI_NOISE_STD = 0.05  # much less noise than humans
AI_BIAS_WEIGHTS = np.array([0.55, 0.30, 0.15])  # biased toward helpfulness

def ai_judge(features):
    '''Simulate AI judge - systematic bias but low noise.'''
    biased_score = features @ AI_BIAS_WEIGHTS
    noise = np.random.normal(0, AI_NOISE_STD, size=len(features))
    return np.clip(biased_score + noise, 0, 1)

ai_labels = ai_judge(response_features)

# Measure AI judge quality
ai_error = np.mean((ai_labels - true_scores)**2)
ai_corr = np.corrcoef(ai_labels, true_scores)[0, 1]
print(f"AI judge MSE: {ai_error:.4f}")
print(f"AI-True correlation: {ai_corr:.4f}")
print(f"\nComparison:")
print(f"  Human noise (unbiased): std={HUMAN_NOISE_STD}")
print(f"  AI noise (biased):      std={AI_NOISE_STD}")
print(f"  AI bias weights: {AI_BIAS_WEIGHTS} vs true: {TRUE_WEIGHTS}")"""),

        md("""## 4. Visualize Label Quality"""),

        code("""fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Human vs True
axes[0].scatter(true_scores, human_labels, alpha=0.3, s=10, color='steelblue')
axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
axes[0].set_xlabel('True Quality')
axes[0].set_ylabel('Human Label')
axes[0].set_title(f'Human Labels (corr={human_corr:.3f})')
axes[0].legend()
axes[0].set_aspect('equal')

# AI vs True
axes[1].scatter(true_scores, ai_labels, alpha=0.3, s=10, color='darkorange')
axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
axes[1].set_xlabel('True Quality')
axes[1].set_ylabel('AI Label')
axes[1].set_title(f'AI Judge Labels (corr={ai_corr:.3f})')
axes[1].legend()
axes[1].set_aspect('equal')

# Label agreement
axes[2].scatter(human_labels, ai_labels, alpha=0.3, s=10, color='green')
axes[2].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect agreement')
agreement_corr = np.corrcoef(human_labels, ai_labels)[0, 1]
axes[2].set_xlabel('Human Label')
axes[2].set_ylabel('AI Label')
axes[2].set_title(f'Label Agreement (corr={agreement_corr:.3f})')
axes[2].legend()
axes[2].set_aspect('equal')

plt.tight_layout()
plt.savefig('rlaif_label_quality.png', dpi=150, bbox_inches='tight')
plt.show()
print("Label quality comparison plotted.")"""),

        md("""## 5. Train Reward Models

We train simple linear reward models on both human and AI labels using gradient descent."""),

        code("""class LinearRewardModel:
    '''Simple linear reward model: R(y) = y^T w + b'''

    def __init__(self, input_dim=3, lr=0.01):
        self.w = np.random.randn(input_dim) * 0.1
        self.b = 0.0
        self.lr = lr

    def predict(self, features):
        return features @ self.w + self.b

    def train_step(self, features, labels):
        preds = self.predict(features)
        errors = preds - labels
        grad_w = (2.0 / len(labels)) * (features.T @ errors)
        grad_b = (2.0 / len(labels)) * np.sum(errors)
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b
        mse = np.mean(errors**2)
        return mse

def train_reward_model(features, labels, n_epochs=200, lr=0.01):
    '''Train a reward model and return loss history.'''
    model = LinearRewardModel(input_dim=features.shape[1], lr=lr)
    losses = []
    for epoch in range(n_epochs):
        loss = model.train_step(features, labels)
        losses.append(loss)
    return model, losses

# Split data
split = int(0.8 * N_RESPONSES)
train_feat, val_feat = response_features[:split], response_features[split:]
train_human, val_human = human_labels[:split], human_labels[split:]
train_ai, val_ai = ai_labels[:split], ai_labels[split:]
val_true = true_scores[split:]

# Train both reward models
rm_human, losses_human = train_reward_model(train_feat, train_human, n_epochs=300)
rm_ai, losses_ai = train_reward_model(train_feat, train_ai, n_epochs=300)

print("Reward Model Training Complete")
print(f"  RLHF model weights: {rm_human.w}")
print(f"  RLAIF model weights: {rm_ai.w}")
print(f"  True weights:        {TRUE_WEIGHTS}")"""),

        md("""## 6. Reward Model Convergence"""),

        code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training loss curves
axes[0].plot(losses_human, label='RLHF (Human Labels)', color='steelblue', linewidth=2)
axes[0].plot(losses_ai, label='RLAIF (AI Labels)', color='darkorange', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Reward Model Training Loss')
axes[0].legend()
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

# Weight comparison
x_pos = np.arange(3)
width = 0.25
axes[1].bar(x_pos - width, TRUE_WEIGHTS, width, label='True', color='green', alpha=0.8)
axes[1].bar(x_pos, rm_human.w, width, label='RLHF', color='steelblue', alpha=0.8)
axes[1].bar(x_pos + width, rm_ai.w, width, label='RLAIF', color='darkorange', alpha=0.8)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(['Helpfulness', 'Honesty', 'Harmlessness'])
axes[1].set_ylabel('Weight')
axes[1].set_title('Learned Reward Weights vs True Weights')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('rlaif_reward_models.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## 7. Policy Optimization via REINFORCE

We optimize a stochastic policy that generates response feature vectors. The policy parameterizes the mean of a Gaussian from which features are sampled."""),

        code("""class GaussianPolicy:
    '''Policy that outputs mean features, sampled with Gaussian noise.'''

    def __init__(self, dim=3, lr=0.005):
        self.mu = np.array([0.5, 0.5, 0.5])
        self.log_std = np.array([-0.5, -0.5, -0.5])
        self.lr = lr
        self.dim = dim

    def sample(self, n=1):
        std = np.exp(self.log_std)
        samples = self.mu + std * np.random.randn(n, self.dim)
        return np.clip(samples, 0, 1)

    def log_prob(self, x):
        std = np.exp(self.log_std)
        var = std ** 2
        lp = -0.5 * np.sum(((x - self.mu)**2) / var + 2 * self.log_std + np.log(2 * np.pi), axis=1)
        return lp

    def update(self, samples, rewards):
        log_probs = self.log_prob(samples)
        baseline = np.mean(rewards)
        advantages = rewards - baseline

        std = np.exp(self.log_std)
        var = std ** 2

        # Gradient for mu
        diff = samples - self.mu
        grad_mu = np.mean(advantages[:, None] * (diff / var), axis=0)
        # Gradient for log_std
        grad_log_std = np.mean(advantages[:, None] * ((diff**2 / var) - 1.0), axis=0)

        self.mu += self.lr * grad_mu
        self.mu = np.clip(self.mu, 0.05, 0.95)
        self.log_std += self.lr * 0.1 * grad_log_std
        self.log_std = np.clip(self.log_std, -3, 0)

def train_policy(reward_model, n_episodes=300, batch_size=32):
    '''Train a policy using REINFORCE with the given reward model.'''
    policy = GaussianPolicy(lr=0.005)
    history = {'true_quality': [], 'model_reward': [], 'mu': []}

    for ep in range(n_episodes):
        samples = policy.sample(batch_size)
        model_rewards = reward_model.predict(samples)
        true_q = true_quality(samples)
        policy.update(samples, model_rewards)

        history['true_quality'].append(np.mean(true_q))
        history['model_reward'].append(np.mean(model_rewards))
        history['mu'].append(policy.mu.copy())

    return policy, history

# Train policies with both reward models
print("Training RLHF policy...")
policy_rlhf, hist_rlhf = train_policy(rm_human, n_episodes=400)
print("Training RLAIF policy...")
policy_rlaif, hist_rlaif = train_policy(rm_ai, n_episodes=400)

print(f"\nFinal RLHF policy mu:  {policy_rlhf.mu}")
print(f"Final RLAIF policy mu: {policy_rlaif.mu}")"""),

        md("""## 8. Policy Training Curves"""),

        code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# True quality over training
axes[0].plot(hist_rlhf['true_quality'], label='RLHF', color='steelblue', linewidth=2, alpha=0.8)
axes[0].plot(hist_rlaif['true_quality'], label='RLAIF', color='darkorange', linewidth=2, alpha=0.8)
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('True Quality (ground truth)')
axes[0].set_title('Policy True Quality During Training')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Model reward over training
axes[1].plot(hist_rlhf['model_reward'], label='RLHF', color='steelblue', linewidth=2, alpha=0.8)
axes[1].plot(hist_rlaif['model_reward'], label='RLAIF', color='darkorange', linewidth=2, alpha=0.8)
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Model Reward')
axes[1].set_title('Policy Model Reward During Training')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rlaif_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## 9. Final Evaluation and Comparison"""),

        code("""# Generate final samples from each policy
N_EVAL = 1000
samples_rlhf = policy_rlhf.sample(N_EVAL)
samples_rlaif = policy_rlaif.sample(N_EVAL)

# Compute quality metrics
q_rlhf = true_quality(samples_rlhf)
q_rlaif = true_quality(samples_rlaif)

# Baseline: random policy
samples_random = np.random.beta(2, 2, size=(N_EVAL, 3))
q_random = true_quality(samples_random)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Quality distributions
axes[0].hist(q_random, bins=30, alpha=0.5, label=f'Random ({q_random.mean():.3f})', color='gray', density=True)
axes[0].hist(q_rlhf, bins=30, alpha=0.5, label=f'RLHF ({q_rlhf.mean():.3f})', color='steelblue', density=True)
axes[0].hist(q_rlaif, bins=30, alpha=0.5, label=f'RLAIF ({q_rlaif.mean():.3f})', color='darkorange', density=True)
axes[0].set_xlabel('True Quality')
axes[0].set_ylabel('Density')
axes[0].set_title('Final Quality Distributions')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Per-dimension comparison
dims = ['Helpfulness', 'Honesty', 'Harmlessness']
rlhf_means = samples_rlhf.mean(axis=0)
rlaif_means = samples_rlaif.mean(axis=0)
random_means = samples_random.mean(axis=0)

x_pos = np.arange(3)
width = 0.25
axes[1].bar(x_pos - width, random_means, width, label='Random', color='gray', alpha=0.7)
axes[1].bar(x_pos, rlhf_means, width, label='RLHF', color='steelblue', alpha=0.7)
axes[1].bar(x_pos + width, rlaif_means, width, label='RLAIF', color='darkorange', alpha=0.7)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(dims)
axes[1].set_ylabel('Mean Score')
axes[1].set_title('Per-Dimension Policy Quality')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('rlaif_final_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("=== Final Summary ===")
print(f"Random policy true quality: {q_random.mean():.4f} +/- {q_random.std():.4f}")
print(f"RLHF policy true quality:   {q_rlhf.mean():.4f} +/- {q_rlhf.std():.4f}")
print(f"RLAIF policy true quality:  {q_rlaif.mean():.4f} +/- {q_rlaif.std():.4f}")"""),

        md("""## 10. Label Efficiency Analysis

How does performance scale with the number of labels?"""),

        code("""label_counts = [25, 50, 100, 200, 300, 400]
human_quality_by_n = []
ai_quality_by_n = []

for n_labels in label_counts:
    # Train reward models with limited data
    rm_h, _ = train_reward_model(response_features[:n_labels], human_labels[:n_labels], n_epochs=300)
    rm_a, _ = train_reward_model(response_features[:n_labels], ai_labels[:n_labels], n_epochs=300)

    # Train policies
    _, h_hist = train_policy(rm_h, n_episodes=200)
    _, a_hist = train_policy(rm_a, n_episodes=200)

    human_quality_by_n.append(np.mean(h_hist['true_quality'][-50:]))
    ai_quality_by_n.append(np.mean(a_hist['true_quality'][-50:]))

plt.figure(figsize=(10, 6))
plt.plot(label_counts, human_quality_by_n, 'o-', label='RLHF (Human)', color='steelblue', linewidth=2, markersize=8)
plt.plot(label_counts, ai_quality_by_n, 's-', label='RLAIF (AI)', color='darkorange', linewidth=2, markersize=8)
plt.axhline(y=q_random.mean(), color='gray', linestyle='--', label='Random baseline')
plt.xlabel('Number of Labels')
plt.ylabel('True Quality of Trained Policy')
plt.title('Label Efficiency: RLHF vs RLAIF')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('rlaif_label_efficiency.png', dpi=150, bbox_inches='tight')
plt.show()

print("Label efficiency analysis complete.")"""),

        md("""## Conclusions

1. **RLAIF converges faster** due to lower label noise, especially with few labels.
2. **RLHF achieves slightly higher true quality** because human labels are unbiased.
3. **AI judge bias** causes the RLAIF policy to overweight helpfulness at the expense of harmlessness.
4. **Label efficiency**: RLAIF is more sample-efficient, reaching good performance with fewer labels.
5. In practice, **combining both approaches** (e.g., AI pre-labeling with human verification) often works best.

### Key Takeaway
RLAIF is a powerful scalability tool, but practitioners must be aware of and mitigate systematic biases in the AI judge."""),
    ]
    save(path, cells)


# =============================================================================
# Notebook 7: Hierarchical RL
# =============================================================================
def make_notebook_07():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "Project_07_Hierarchical_RL", "hierarchical_rl_notebook.ipynb")
    cells = [
        md("""# Project 7: Hierarchical Reinforcement Learning - Four-Room Grid World
**Author:** Milan Amrut Joshi

## Overview
This notebook implements **Hierarchical Reinforcement Learning (HRL)** using the **Options Framework** in a classic four-room grid world environment.

### Key Concepts
- **Flat RL**: Standard Q-learning treats every step equally
- **Hierarchical RL**: Decomposes tasks into subtasks at multiple levels of abstraction
- **Options Framework** (Sutton, Precup, Singh 1999): Extends actions to temporally extended "options"

### The Four-Room Environment
A grid world divided into 4 rooms connected by doorways. The agent must navigate from a start position to a goal, passing through doorways between rooms."""),

        md(r"""## Mathematical Framework

### Options Framework
An **option** $\omega = (I_\omega, \pi_\omega, \beta_\omega)$ consists of:
- $I_\omega \subseteq \mathcal{S}$: initiation set (states where option can start)
- $\pi_\omega$: intra-option policy (how to act within the option)
- $\beta_\omega: \mathcal{S} \to [0,1]$: termination function

### Two-Level Hierarchy
**High-level policy** $\pi_H(s)$: selects which subgoal (doorway) to pursue
$$\pi_H: \mathcal{S} \to \{\text{doorway}_1, \text{doorway}_2, \text{doorway}_3, \text{doorway}_4, \text{goal}\}$$

**Low-level policy** $\pi_L(s, g)$: navigates to the selected subgoal $g$
$$\pi_L: \mathcal{S} \times \mathcal{G} \to \{up, down, left, right\}$$

### Q-Learning over Options (SMDP Q-Learning)
For options, the Q-update uses cumulated reward over multiple steps:

$$Q(\omega, s) \leftarrow Q(\omega, s) + \alpha \left[r + \gamma^k \max_{\omega'} Q(\omega', s') - Q(\omega, s)\right]$$

where $k$ is the number of primitive steps the option took and $r$ is the cumulated discounted reward.

### Flat Q-Learning Baseline
Standard one-step Q-learning:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$"""),

        md("""## Setup"""),

        code("""import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

np.random.seed(42)
print("Hierarchical RL - Four-Room Grid World - Setup Complete")"""),

        md("""## 1. Four-Room Grid World Environment"""),

        code("""class FourRoomGrid:
    '''Four-room grid world with doorways between rooms.'''

    def __init__(self, size=13):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self._build_walls()
        self.start = (1, 1)
        self.goal = (11, 11)
        self.doorways = [(6, 2), (2, 6), (6, 9), (9, 6)]
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        self.action_names = ['Up', 'Down', 'Left', 'Right']
        self.reset()

    def _build_walls(self):
        s = self.size
        mid = s // 2  # 6
        # Borders
        self.grid[0, :] = 1
        self.grid[s-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, s-1] = 1
        # Horizontal wall
        self.grid[mid, :] = 1
        # Vertical wall
        self.grid[:, mid] = 1

    def _open_doorways(self):
        for d in self.doorways:
            self.grid[d] = 0

    def reset(self):
        self._build_walls()
        self._open_doorways()
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        dr, dc = self.actions[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] == 0:
            self.agent_pos = (nr, nc)
        done = (self.agent_pos == self.goal)
        reward = 1.0 if done else -0.01
        return self.agent_pos, reward, done

    def get_valid_states(self):
        states = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == 0:
                    states.append((r, c))
        return states

    def get_room(self, pos):
        r, c = pos
        mid = self.size // 2
        if r < mid and c < mid: return 0  # top-left
        if r < mid and c > mid: return 1  # top-right
        if r > mid and c < mid: return 2  # bottom-left
        if r > mid and c > mid: return 3  # bottom-right
        return -1  # on wall/doorway

env = FourRoomGrid()
print(f"Grid size: {env.size}x{env.size}")
print(f"Start: {env.start}, Goal: {env.goal}")
print(f"Doorways: {env.doorways}")
print(f"Valid states: {len(env.get_valid_states())}")"""),

        md("""## 2. Visualize the Environment"""),

        code("""def plot_grid(env, path=None, title="Four-Room Grid World"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    display = env.grid.copy().astype(float)

    # Color walls
    cmap = ListedColormap(['white', 'black'])
    ax.imshow(display, cmap=cmap, origin='upper')

    # Draw grid lines
    for i in range(env.size + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        ax.axvline(i - 0.5, color='gray', linewidth=0.5)

    # Mark doorways
    for d in env.doorways:
        ax.add_patch(Rectangle((d[1]-0.5, d[0]-0.5), 1, 1, fill=True,
                                facecolor='lightblue', edgecolor='blue', linewidth=2))

    # Mark start and goal
    ax.plot(env.start[1], env.start[0], 'gs', markersize=15, label='Start')
    ax.plot(env.goal[1], env.goal[0], 'r*', markersize=20, label='Goal')

    # Draw path if provided
    if path is not None and len(path) > 1:
        path = np.array(path)
        ax.plot(path[:, 1], path[:, 0], 'b-', linewidth=2, alpha=0.7)
        ax.plot(path[:, 1], path[:, 0], 'b.', markersize=4)

    # Room labels
    for room_idx, (rr, rc) in enumerate([(3, 3), (3, 9), (9, 3), (9, 9)]):
        ax.text(rc, rr, f'Room {room_idx}', ha='center', va='center',
                fontsize=10, color='gray', fontweight='bold')

    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)
    return fig, ax

fig, ax = plot_grid(env)
plt.savefig('four_room_grid.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## 3. Flat Q-Learning Baseline"""),

        code("""def flat_q_learning(env, n_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.3,
                      max_steps=500, epsilon_decay=0.995):
    '''Standard flat Q-learning.'''
    Q = {}
    for s in env.get_valid_states():
        Q[s] = np.zeros(4)

    episode_steps = []
    episode_rewards = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        path = [state]

        for step in range(max_steps):
            # Epsilon-greedy
            if np.random.random() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q.get(state, np.zeros(4)))

            next_state, reward, done = env.step(action)
            total_reward += reward

            # Q-update
            best_next = np.max(Q.get(next_state, np.zeros(4)))
            Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

            state = next_state
            path.append(state)

            if done:
                break

        episode_steps.append(step + 1)
        episode_rewards.append(total_reward)
        epsilon = max(0.01, epsilon * epsilon_decay)

    return Q, episode_steps, episode_rewards

print("Training flat Q-learning agent...")
Q_flat, steps_flat, rewards_flat = flat_q_learning(env, n_episodes=800)
print(f"Final avg steps (last 50): {np.mean(steps_flat[-50:]):.1f}")
print(f"Final avg reward (last 50): {np.mean(rewards_flat[-50:]):.3f}")"""),

        md("""## 4. Hierarchical RL with Options Framework"""),

        code("""class HierarchicalAgent:
    '''Two-level hierarchical agent with options.'''

    def __init__(self, env, alpha_h=0.1, alpha_l=0.1, gamma=0.99, epsilon=0.3):
        self.env = env
        self.alpha_h = alpha_h  # high-level learning rate
        self.alpha_l = alpha_l  # low-level learning rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.subgoals = list(env.doorways) + [env.goal]

        # High-level Q: state -> subgoal
        self.Q_high = {}
        for s in env.get_valid_states():
            self.Q_high[s] = np.zeros(len(self.subgoals))

        # Low-level Q: (state, subgoal_idx) -> action
        self.Q_low = {}
        for s in env.get_valid_states():
            for g_idx in range(len(self.subgoals)):
                self.Q_low[(s, g_idx)] = np.zeros(4)

    def select_subgoal(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.subgoals))
        return np.argmax(self.Q_high.get(state, np.zeros(len(self.subgoals))))

    def select_action(self, state, subgoal_idx):
        key = (state, subgoal_idx)
        if np.random.random() < self.epsilon * 0.5:
            return np.random.randint(4)
        return np.argmax(self.Q_low.get(key, np.zeros(4)))

    def subgoal_reached(self, state, subgoal_idx):
        return state == self.subgoals[subgoal_idx]

    def train_episode(self, max_steps=500):
        state = self.env.reset()
        total_reward = 0
        total_steps = 0
        path = [state]
        subgoals_used = []

        while total_steps < max_steps:
            # High-level: pick subgoal
            g_idx = self.select_subgoal(state)
            subgoals_used.append(g_idx)
            option_start = state
            option_reward = 0
            option_steps = 0

            # Low-level: navigate to subgoal
            while total_steps < max_steps:
                action = self.select_action(state, g_idx)
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                total_steps += 1
                option_reward += (self.gamma ** option_steps) * reward
                option_steps += 1
                path.append(next_state)

                # Low-level Q update
                intrinsic = 1.0 if self.subgoal_reached(next_state, g_idx) else -0.01
                key = (state, g_idx)
                next_key = (next_state, g_idx)
                best_next = np.max(self.Q_low.get(next_key, np.zeros(4)))
                self.Q_low[key][action] += self.alpha_l * (
                    intrinsic + self.gamma * best_next - self.Q_low[key][action]
                )

                state = next_state

                if done or self.subgoal_reached(state, g_idx) or option_steps > 50:
                    break

            # High-level Q update (SMDP)
            best_high = np.max(self.Q_high.get(state, np.zeros(len(self.subgoals))))
            self.Q_high[option_start][g_idx] += self.alpha_h * (
                option_reward + (self.gamma ** option_steps) * best_high
                - self.Q_high[option_start][g_idx]
            )

            if done:
                break

        return total_steps, total_reward, path, subgoals_used

def train_hierarchical(env, n_episodes=800, epsilon_decay=0.995):
    agent = HierarchicalAgent(env, epsilon=0.3)
    ep_steps = []
    ep_rewards = []
    all_subgoals = []

    for ep in range(n_episodes):
        steps, reward, path, sg = agent.train_episode()
        ep_steps.append(steps)
        ep_rewards.append(reward)
        all_subgoals.append(sg)
        agent.epsilon = max(0.01, agent.epsilon * epsilon_decay)

    return agent, ep_steps, ep_rewards, all_subgoals

print("Training hierarchical agent...")
h_agent, steps_hier, rewards_hier, subgoals_log = train_hierarchical(env, n_episodes=800)
print(f"Final avg steps (last 50): {np.mean(steps_hier[-50:]):.1f}")
print(f"Final avg reward (last 50): {np.mean(rewards_hier[-50:]):.3f}")"""),

        md("""## 5. Compare Learning Curves"""),

        code("""def smooth(data, window=30):
    return np.convolve(data, np.ones(window)/window, mode='valid')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Steps per episode
axes[0].plot(smooth(steps_flat), label='Flat Q-Learning', color='steelblue', linewidth=2)
axes[0].plot(smooth(steps_hier), label='Hierarchical RL', color='darkorange', linewidth=2)
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Steps to Goal')
axes[0].set_title('Steps per Episode (smoothed)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Cumulative reward
axes[1].plot(smooth(rewards_flat), label='Flat Q-Learning', color='steelblue', linewidth=2)
axes[1].plot(smooth(rewards_hier), label='Hierarchical RL', color='darkorange', linewidth=2)
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Episode Reward')
axes[1].set_title('Reward per Episode (smoothed)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hierarchical_learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## 6. Visualize Learned Paths"""),

        code("""def get_greedy_path(env, Q, max_steps=200):
    '''Extract greedy path from flat Q-table.'''
    state = env.reset()
    path = [state]
    for _ in range(max_steps):
        action = np.argmax(Q.get(state, np.zeros(4)))
        state, _, done = env.step(action)
        path.append(state)
        if done:
            break
    return path

def get_hierarchical_path(env, agent, max_steps=200):
    '''Extract greedy path from hierarchical agent.'''
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    state = env.reset()
    path = [state]
    for _ in range(max_steps):
        g_idx = agent.select_subgoal(state)
        for _ in range(50):
            action = agent.select_action(state, g_idx)
            state, _, done = env.step(action)
            path.append(state)
            if done or agent.subgoal_reached(state, g_idx):
                break
        if done:
            break
    agent.epsilon = old_eps
    return path

path_flat = get_greedy_path(env, Q_flat)
path_hier = get_hierarchical_path(env, h_agent)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Flat path
plt.sca(axes[0])
fig_f, ax_f = plot_grid(env, path=path_flat, title=f'Flat Q-Learning Path ({len(path_flat)} steps)')
# Copy content to subplot
axes[0].imshow(ax_f.images[0].get_array(), cmap=ListedColormap(['white', 'black']), origin='upper')
plt.close(fig_f)

# Just replot properly
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
for ax, path, title in [(ax1, path_flat, f'Flat Q-Learning ({len(path_flat)} steps)'),
                          (ax2, path_hier, f'Hierarchical RL ({len(path_hier)} steps)')]:
    display = env.grid.copy().astype(float)
    cmap = ListedColormap(['white', 'black'])
    ax.imshow(display, cmap=cmap, origin='upper')
    for i in range(env.size + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        ax.axvline(i - 0.5, color='gray', linewidth=0.5)
    for d in env.doorways:
        ax.add_patch(Rectangle((d[1]-0.5, d[0]-0.5), 1, 1, fill=True,
                                facecolor='lightblue', edgecolor='blue', linewidth=2))
    ax.plot(env.start[1], env.start[0], 'gs', markersize=15)
    ax.plot(env.goal[1], env.goal[0], 'r*', markersize=20)
    if len(path) > 1:
        p = np.array(path)
        ax.plot(p[:, 1], p[:, 0], 'b-', linewidth=2, alpha=0.7)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)

plt.close(fig)
plt.tight_layout()
plt.savefig('hierarchical_paths.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## 7. Subgoal Usage Analysis"""),

        code("""# Analyze which subgoals the hierarchical agent uses over training
subgoal_names = [f'Door {i} {d}' for i, d in enumerate(env.doorways)] + ['Goal']

# Count subgoal selections in windows
window = 50
subgoal_counts = []
for i in range(0, len(subgoals_log) - window, window):
    counts = np.zeros(len(subgoal_names))
    for sg_list in subgoals_log[i:i+window]:
        for sg in sg_list:
            counts[sg] += 1
    subgoal_counts.append(counts / counts.sum() if counts.sum() > 0 else counts)

subgoal_counts = np.array(subgoal_counts)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(subgoal_counts)) * window
for i, name in enumerate(subgoal_names):
    ax.plot(x, subgoal_counts[:, i], label=name, linewidth=2)

ax.set_xlabel('Episode')
ax.set_ylabel('Selection Frequency')
ax.set_title('Subgoal Selection Frequency Over Training')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hierarchical_subgoal_usage.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## 8. Q-Value Heatmaps"""),

        code("""def plot_q_heatmap(env, Q, title="Q-Value Heatmap"):
    '''Plot the max Q-value at each state.'''
    heatmap = np.full((env.size, env.size), np.nan)
    for s in env.get_valid_states():
        if s in Q:
            heatmap[s[0], s[1]] = np.max(Q[s])

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(heatmap, cmap='viridis', origin='upper')
    # Overlay walls
    for r in range(env.size):
        for c in range(env.size):
            if env.grid[r, c] == 1:
                ax.add_patch(Rectangle((c-0.5, r-0.5), 1, 1, fill=True,
                                        facecolor='black'))
    ax.plot(env.start[1], env.start[0], 'gs', markersize=12)
    ax.plot(env.goal[1], env.goal[0], 'r*', markersize=16)
    for d in env.doorways:
        ax.plot(d[1], d[0], 'co', markersize=10)
    plt.colorbar(im, ax=ax, label='Max Q-Value')
    ax.set_title(title, fontsize=13)
    return fig

fig = plot_q_heatmap(env, Q_flat, "Flat Q-Learning: Max Q-Values")
plt.savefig('hierarchical_q_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## Conclusions

1. **Hierarchical RL converges faster** by decomposing the long-horizon task into manageable subtasks.
2. **Subgoal discovery** is critical: doorways serve as natural subgoals in the four-room domain.
3. **Path efficiency**: The hierarchical agent learns structured, room-by-room navigation.
4. **The Options Framework** provides a principled way to incorporate temporal abstraction.
5. **Flat Q-learning** eventually catches up but requires significantly more episodes.

### Key Takeaway
Hierarchical RL excels when tasks have natural decomposition structure. The four-room domain is a classic example where temporal abstraction dramatically improves learning efficiency."""),
    ]
    save(path, cells)


# =============================================================================
# Notebook 8: Meta-RL
# =============================================================================
def make_notebook_08():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "Project_08_Meta_RL", "meta_rl_notebook.ipynb")
    cells = [
        md("""# Project 8: Meta-Reinforcement Learning on Multi-Armed Bandits
**Author:** Milan Amrut Joshi

## Overview
This notebook implements **Meta-Reinforcement Learning (Meta-RL)** using a MAML-inspired approach applied to a distribution of multi-armed bandit tasks.

### Key Concepts
- **Meta-Learning**: "Learning to learn" - train a model that can quickly adapt to new tasks
- **MAML (Model-Agnostic Meta-Learning)**: Inner loop adapts to specific task, outer loop learns good initialization
- **Task Distribution**: Each task is a different bandit with different reward distributions
- **Few-Shot Adaptation**: After meta-training, the agent adapts to new tasks with very few interactions"""),

        md(r"""## Mathematical Framework

### Multi-Armed Bandit Task
A task $\mathcal{T}_i$ is a $K$-armed bandit with reward distributions:
$$r_k \sim \mathcal{N}(\mu_k^{(i)}, \sigma^2), \quad k = 1, \ldots, K$$

where $\mu_k^{(i)}$ are the true arm means for task $i$.

### Task Distribution
Tasks are sampled from a distribution:
$$\mathcal{T}_i \sim p(\mathcal{T})$$
$$\mu_k^{(i)} \sim \text{Uniform}(0, 1), \quad k = 1, \ldots, K$$

### MAML-Style Meta-Learning

**Inner loop** (task-specific adaptation):
Given task $\mathcal{T}_i$ with $N$ interaction steps, update parameters:
$$\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)$$

**Outer loop** (meta-update across tasks):
$$\theta \leftarrow \theta - \beta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta_i')$$

### Policy Parameterization
The policy is a softmax over action preferences $\theta \in \mathbb{R}^K$:
$$\pi_\theta(a = k) = \frac{\exp(\theta_k)}{\sum_{j=1}^K \exp(\theta_j)}$$

### Loss Function (Negative Expected Reward)
$$\mathcal{L}_{\mathcal{T}}(\theta) = -\mathbb{E}_{a \sim \pi_\theta}[r(a)]$$"""),

        md("""## Setup"""),

        code("""import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
print("Meta-RL on Multi-Armed Bandits - Setup Complete")"""),

        md("""## 1. Multi-Armed Bandit Task Distribution"""),

        code("""class BanditTask:
    '''A single K-armed bandit task.'''

    def __init__(self, K=5, reward_std=0.3, means=None):
        self.K = K
        self.reward_std = reward_std
        if means is not None:
            self.means = means
        else:
            self.means = np.random.uniform(0, 1, size=K)
        self.best_arm = np.argmax(self.means)
        self.best_reward = self.means[self.best_arm]

    def pull(self, arm):
        return self.means[arm] + np.random.normal(0, self.reward_std)


class TaskDistribution:
    '''Distribution over bandit tasks.'''

    def __init__(self, K=5, reward_std=0.3):
        self.K = K
        self.reward_std = reward_std

    def sample_task(self):
        return BanditTask(self.K, self.reward_std)

    def sample_tasks(self, n):
        return [self.sample_task() for _ in range(n)]

K = 5  # number of arms
task_dist = TaskDistribution(K=K, reward_std=0.3)

# Visualize a few tasks
sample_tasks = task_dist.sample_tasks(6)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for idx, (ax, task) in enumerate(zip(axes.flat, sample_tasks)):
    colors = ['steelblue'] * K
    colors[task.best_arm] = 'darkorange'
    ax.bar(range(K), task.means, color=colors, edgecolor='black', alpha=0.8)
    ax.set_title(f'Task {idx}: best arm = {task.best_arm}')
    ax.set_xlabel('Arm')
    ax.set_ylabel('Mean Reward')
    ax.set_ylim(0, 1.1)

plt.suptitle('Sample Bandit Tasks from Distribution', fontsize=14)
plt.tight_layout()
plt.savefig('meta_rl_task_distribution.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## 2. Softmax Policy"""),

        code("""def softmax(logits):
    '''Numerically stable softmax.'''
    x = logits - np.max(logits)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def sample_action(theta):
    '''Sample an action from the softmax policy.'''
    probs = softmax(theta)
    return np.random.choice(len(theta), p=probs)

def policy_gradient(theta, action, reward, baseline=0):
    '''Compute REINFORCE gradient for softmax policy.'''
    probs = softmax(theta)
    # d log pi(a) / d theta
    grad_log = -probs.copy()
    grad_log[action] += 1.0
    return grad_log * (reward - baseline)

# Test
test_theta = np.zeros(K)
print(f"Uniform probs: {softmax(test_theta)}")
test_theta[2] = 2.0
print(f"Biased probs:  {softmax(test_theta)}")"""),

        md("""## 3. MAML-Style Meta-Learner"""),

        code("""class MAMLBanditAgent:
    '''MAML-style meta-learner for bandit tasks.'''

    def __init__(self, K=5, inner_lr=0.3, outer_lr=0.05, inner_steps=5):
        self.K = K
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        # Meta-parameters: initial action preferences
        self.theta_init = np.zeros(K)

    def inner_adapt(self, task, n_steps=None):
        '''Adapt to a specific task (inner loop).'''
        if n_steps is None:
            n_steps = self.inner_steps
        theta = self.theta_init.copy()
        rewards = []

        for step in range(n_steps):
            action = sample_action(theta)
            reward = task.pull(action)
            rewards.append(reward)
            baseline = np.mean(rewards) if len(rewards) > 0 else 0
            grad = policy_gradient(theta, action, reward, baseline)
            theta += self.inner_lr * grad

        return theta, rewards

    def meta_update(self, tasks, eval_steps=3):
        '''Outer loop: update theta_init across tasks.'''
        meta_grad = np.zeros(self.K)

        for task in tasks:
            # Inner loop: adapt
            theta_adapted, _ = self.inner_adapt(task)

            # Evaluate adapted policy on same task
            eval_rewards = []
            eval_grads = []
            for _ in range(eval_steps):
                action = sample_action(theta_adapted)
                reward = task.pull(action)
                eval_rewards.append(reward)
                grad = policy_gradient(theta_adapted, action, reward, np.mean(eval_rewards))
                eval_grads.append(grad)

            # Approximate meta-gradient (first-order MAML)
            meta_grad += np.mean(eval_grads, axis=0)

        meta_grad /= len(tasks)
        self.theta_init += self.outer_lr * meta_grad
        return meta_grad

    def evaluate(self, task, n_steps=20):
        '''Evaluate on a new task with adaptation.'''
        theta = self.theta_init.copy()
        rewards = []
        actions = []

        for step in range(n_steps):
            action = sample_action(theta)
            reward = task.pull(action)
            rewards.append(reward)
            actions.append(action)
            baseline = np.mean(rewards)
            grad = policy_gradient(theta, action, reward, baseline)
            theta += self.inner_lr * grad

        return rewards, actions

print("MAML Bandit Agent created.")
print(f"  Arms: {K}")
print(f"  Inner LR: 0.3, Outer LR: 0.05")
print(f"  Inner steps: 5")"""),

        md("""## 4. Regular (Non-Meta) Learner for Comparison"""),

        code("""class RegularBanditAgent:
    '''Standard learner that starts from scratch each task.'''

    def __init__(self, K=5, lr=0.3):
        self.K = K
        self.lr = lr

    def evaluate(self, task, n_steps=20):
        '''Evaluate on a task starting from scratch.'''
        theta = np.zeros(self.K)  # always start from scratch
        rewards = []
        actions = []

        for step in range(n_steps):
            action = sample_action(theta)
            reward = task.pull(action)
            rewards.append(reward)
            actions.append(action)
            baseline = np.mean(rewards)
            grad = policy_gradient(theta, action, reward, baseline)
            theta += self.lr * grad

        return rewards, actions

class EpsilonGreedyAgent:
    '''Epsilon-greedy baseline.'''

    def __init__(self, K=5, epsilon=0.1):
        self.K = K
        self.epsilon = epsilon

    def evaluate(self, task, n_steps=20):
        Q = np.zeros(self.K)
        N = np.zeros(self.K)
        rewards = []
        actions = []

        for step in range(n_steps):
            if np.random.random() < self.epsilon or step < self.K:
                action = step if step < self.K else np.random.randint(self.K)
            else:
                action = np.argmax(Q)
            reward = task.pull(action)
            rewards.append(reward)
            actions.append(action)
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]

        return rewards, actions

print("Baseline agents created.")"""),

        md("""## 5. Meta-Training"""),

        code("""meta_agent = MAMLBanditAgent(K=K, inner_lr=0.3, outer_lr=0.05, inner_steps=5)

N_META_EPOCHS = 150
TASKS_PER_EPOCH = 10
meta_rewards_history = []

for epoch in range(N_META_EPOCHS):
    tasks = task_dist.sample_tasks(TASKS_PER_EPOCH)
    meta_agent.meta_update(tasks)

    # Evaluate current meta-initialization on fresh tasks
    if epoch % 10 == 0:
        eval_tasks = task_dist.sample_tasks(20)
        avg_reward = 0
        for t in eval_tasks:
            r, _ = meta_agent.evaluate(t, n_steps=10)
            avg_reward += np.mean(r)
        avg_reward /= len(eval_tasks)
        meta_rewards_history.append((epoch, avg_reward))
        if epoch % 30 == 0:
            print(f"Epoch {epoch:3d}: avg reward = {avg_reward:.3f}, theta_init = {meta_agent.theta_init}")

print(f"\nFinal meta-initialization: {meta_agent.theta_init}")
print("Meta-training complete!")"""),

        md("""## 6. Few-Shot Adaptation Comparison"""),

        code("""# Compare adaptation speed on new tasks
N_EVAL_TASKS = 100
N_EVAL_STEPS = 25

regular = RegularBanditAgent(K=K, lr=0.3)
eps_greedy = EpsilonGreedyAgent(K=K, epsilon=0.1)

meta_rewards_all = np.zeros((N_EVAL_TASKS, N_EVAL_STEPS))
regular_rewards_all = np.zeros((N_EVAL_TASKS, N_EVAL_STEPS))
greedy_rewards_all = np.zeros((N_EVAL_TASKS, N_EVAL_STEPS))
optimal_rewards = np.zeros(N_EVAL_TASKS)

for i in range(N_EVAL_TASKS):
    task = task_dist.sample_task()
    optimal_rewards[i] = task.best_reward

    r_meta, _ = meta_agent.evaluate(task, N_EVAL_STEPS)
    r_reg, _ = regular.evaluate(task, N_EVAL_STEPS)
    r_eps, _ = eps_greedy.evaluate(task, N_EVAL_STEPS)

    meta_rewards_all[i] = r_meta
    regular_rewards_all[i] = r_reg
    greedy_rewards_all[i] = r_eps

# Average across tasks
meta_avg = meta_rewards_all.mean(axis=0)
regular_avg = regular_rewards_all.mean(axis=0)
greedy_avg = greedy_rewards_all.mean(axis=0)
optimal_avg = optimal_rewards.mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Cumulative reward per step
axes[0].plot(range(1, N_EVAL_STEPS+1), np.cumsum(meta_avg), label='Meta-RL (MAML)',
             color='darkorange', linewidth=2)
axes[0].plot(range(1, N_EVAL_STEPS+1), np.cumsum(regular_avg), label='Regular Learner',
             color='steelblue', linewidth=2)
axes[0].plot(range(1, N_EVAL_STEPS+1), np.cumsum(greedy_avg), label='Epsilon-Greedy',
             color='green', linewidth=2)
axes[0].plot(range(1, N_EVAL_STEPS+1), np.cumsum(np.full(N_EVAL_STEPS, optimal_avg)),
             'k--', linewidth=1.5, label='Optimal', alpha=0.7)
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Cumulative Reward')
axes[0].set_title('Few-Shot Adaptation: Cumulative Reward')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Per-step reward
window = 3
for arr, label, color in [(meta_avg, 'Meta-RL', 'darkorange'),
                            (regular_avg, 'Regular', 'steelblue'),
                            (greedy_avg, 'Eps-Greedy', 'green')]:
    smoothed = np.convolve(arr, np.ones(window)/window, mode='valid')
    axes[1].plot(smoothed, label=label, color=color, linewidth=2)
axes[1].axhline(y=optimal_avg, color='black', linestyle='--', label='Optimal', alpha=0.7)
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Average Reward')
axes[1].set_title('Per-Step Reward (smoothed)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('meta_rl_adaptation.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Average reward over {N_EVAL_STEPS} steps:")
print(f"  Meta-RL:        {meta_avg.mean():.4f}")
print(f"  Regular:        {regular_avg.mean():.4f}")
print(f"  Epsilon-Greedy: {greedy_avg.mean():.4f}")
print(f"  Optimal:        {optimal_avg:.4f}")"""),

        md("""## 7. Task-by-Task Performance Analysis"""),

        code("""# Compare total reward per task
meta_totals = meta_rewards_all.sum(axis=1)
regular_totals = regular_rewards_all.sum(axis=1)
greedy_totals = greedy_rewards_all.sum(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of total rewards
axes[0].hist(meta_totals, bins=20, alpha=0.6, label='Meta-RL', color='darkorange', density=True)
axes[0].hist(regular_totals, bins=20, alpha=0.6, label='Regular', color='steelblue', density=True)
axes[0].hist(greedy_totals, bins=20, alpha=0.6, label='Eps-Greedy', color='green', density=True)
axes[0].set_xlabel('Total Reward per Task')
axes[0].set_ylabel('Density')
axes[0].set_title('Distribution of Total Rewards')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Scatter: Meta vs Regular
axes[1].scatter(regular_totals, meta_totals, alpha=0.5, s=30, color='darkorange')
max_val = max(regular_totals.max(), meta_totals.max())
min_val = min(regular_totals.min(), meta_totals.min())
axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Equal')
axes[1].set_xlabel('Regular Learner Total Reward')
axes[1].set_ylabel('Meta-RL Total Reward')
axes[1].set_title('Meta-RL vs Regular (per task)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
meta_wins = np.sum(meta_totals > regular_totals)
axes[1].text(0.05, 0.95, f'Meta wins: {meta_wins}/{N_EVAL_TASKS}',
             transform=axes[1].transAxes, fontsize=11, verticalalignment='top')

plt.tight_layout()
plt.savefig('meta_rl_task_analysis.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## 8. Adaptation Dynamics: Watch the Meta-Learner Adapt"""),

        code("""# Show how the meta-learner's policy evolves on a single task
demo_task = task_dist.sample_task()
print(f"Demo task arm means: {demo_task.means}")
print(f"Best arm: {demo_task.best_arm} (mean={demo_task.best_reward:.3f})")

# Track policy evolution
n_demo_steps = 30
theta_trajectory = [meta_agent.theta_init.copy()]
theta = meta_agent.theta_init.copy()
demo_rewards = []

for step in range(n_demo_steps):
    action = sample_action(theta)
    reward = demo_task.pull(action)
    demo_rewards.append(reward)
    baseline = np.mean(demo_rewards)
    grad = policy_gradient(theta, action, reward, baseline)
    theta = theta + meta_agent.inner_lr * grad
    theta_trajectory.append(theta.copy())

theta_trajectory = np.array(theta_trajectory)
prob_trajectory = np.array([softmax(t) for t in theta_trajectory])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Arm means
colors_arms = plt.cm.tab10(np.arange(K))
axes[0].bar(range(K), demo_task.means, color=colors_arms, edgecolor='black')
axes[0].set_xlabel('Arm')
axes[0].set_ylabel('True Mean')
axes[0].set_title('Task Arm Means')

# Policy probability evolution
for arm in range(K):
    axes[1].plot(prob_trajectory[:, arm], label=f'Arm {arm}', color=colors_arms[arm], linewidth=2)
axes[1].set_xlabel('Adaptation Step')
axes[1].set_ylabel('Selection Probability')
axes[1].set_title('Policy Evolution During Adaptation')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Cumulative regret
optimal_per_step = demo_task.best_reward
cum_regret = np.cumsum([optimal_per_step - r for r in demo_rewards])
axes[2].plot(cum_regret, 'r-', linewidth=2)
axes[2].set_xlabel('Step')
axes[2].set_ylabel('Cumulative Regret')
axes[2].set_title('Cumulative Regret During Adaptation')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('meta_rl_adaptation_dynamics.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## 9. Meta-Training Progress"""),

        code("""epochs, rewards = zip(*meta_rewards_history)

plt.figure(figsize=(10, 5))
plt.plot(epochs, rewards, 'o-', color='darkorange', linewidth=2, markersize=6)
plt.xlabel('Meta-Training Epoch')
plt.ylabel('Average Reward on New Tasks')
plt.title('Meta-Training Progress')
plt.grid(True, alpha=0.3)
plt.savefig('meta_rl_training_progress.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## Conclusions

1. **Meta-RL enables fast adaptation**: The meta-learner achieves higher cumulative reward in the first few steps compared to baselines.
2. **Good initialization matters**: MAML finds an initial policy that is well-suited for the task distribution.
3. **Few-shot performance**: Even with just 5-10 interactions, the meta-learner identifies the best arm more reliably.
4. **Trade-off**: Meta-training requires many tasks, but each new task requires very few samples.
5. **The softmax policy** provides a smooth, differentiable action selection mechanism ideal for gradient-based meta-learning.

### Key Takeaway
Meta-RL is particularly powerful when agents face a distribution of related tasks and need to adapt quickly. The MAML approach provides a principled way to learn transferable initialization parameters."""),
    ]
    save(path, cells)


# =============================================================================
# Notebook 9: Drug Discovery RL
# =============================================================================
def make_notebook_09():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "Project_09_Drug_Discovery_RL", "drug_discovery_notebook.ipynb")
    cells = [
        md("""# Project 9: RL for Drug Discovery - Molecular Optimization
**Author:** Milan Amrut Joshi

## Overview
This notebook applies **Reinforcement Learning** to the problem of **de novo molecular design**, where an agent builds molecules atom-by-atom to optimize drug-like properties.

### Key Concepts
- **De Novo Drug Design**: Generating novel molecular structures from scratch
- **Multi-Objective Optimization**: Balancing competing drug properties
- **REINFORCE Algorithm**: Policy gradient method for discrete molecular construction
- **Simplified Molecular Representation**: Atom sequences with chemical properties

### Drug-Likeness Criteria (Lipinski's Rule of Five)
1. Molecular weight < 500 Da
2. LogP (lipophilicity) <= 5
3. H-bond donors <= 5
4. H-bond acceptors <= 10"""),

        md(r"""## Mathematical Framework

### Molecular Construction as an MDP
- **State** $s_t$: Partial molecule (sequence of atoms so far)
- **Action** $a_t$: Choose next atom from vocabulary $\{C, N, O, S, F\}$
- **Transition**: Deterministic append of atom to sequence
- **Episode**: Fixed-length molecule generation (e.g., 10 atoms)

### Atom Properties
Each atom type $a$ has properties:
- $w_a$: atomic weight
- $\ell_a$: LogP contribution
- $\tau_a$: toxicity factor
- $d_a$: drug-likeness contribution

### Multi-Objective Reward
The reward for a complete molecule $m = (a_1, \ldots, a_T)$ is:

$$R(m) = \lambda_1 \cdot R_{\text{logP}}(m) + \lambda_2 \cdot R_{\text{MW}}(m) + \lambda_3 \cdot R_{\text{tox}}(m) + \lambda_4 \cdot R_{\text{drug}}(m)$$

Each component is designed so higher is better:

$$R_{\text{logP}}(m) = \exp\left(-\frac{(\text{LogP}(m) - \text{target})^2}{2\sigma^2}\right)$$

$$R_{\text{MW}}(m) = \begin{cases} 1 & \text{if MW} < 500 \\ \exp(-(MW - 500)/100) & \text{otherwise} \end{cases}$$

$$R_{\text{tox}}(m) = 1 - \text{Toxicity}(m)$$

$$R_{\text{drug}}(m) = \text{sigmoid}(\text{DrugScore}(m) - \text{threshold})$$

### Policy Parameterization
The policy $\pi_\theta$ maps the current partial molecule state to a distribution over atom types:

$$\pi_\theta(a_t | s_t) = \text{softmax}(W \cdot \phi(s_t) + b)$$

where $\phi(s_t)$ is a feature representation of the current molecule state.

### REINFORCE Update
$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot (R(m) - b)\right]$$"""),

        md("""## Setup"""),

        code("""import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
print("RL for Drug Discovery - Molecular Optimization - Setup Complete")"""),

        md("""## 1. Define Atom Vocabulary and Properties"""),

        code("""# Atom vocabulary with simplified chemical properties
ATOMS = {
    'C': {'weight': 12.0, 'logp': 0.5,  'toxicity': 0.1, 'drug_score': 0.3, 'color': '#404040'},
    'N': {'weight': 14.0, 'logp': -0.5, 'toxicity': 0.15, 'drug_score': 0.5, 'color': '#3050F8'},
    'O': {'weight': 16.0, 'logp': -1.0, 'toxicity': 0.05, 'drug_score': 0.6, 'color': '#FF0D0D'},
    'S': {'weight': 32.1, 'logp': 0.8,  'toxicity': 0.3,  'drug_score': 0.2, 'color': '#FFFF30'},
    'F': {'weight': 19.0, 'logp': 0.4,  'toxicity': 0.25, 'drug_score': 0.4, 'color': '#90E050'},
}
ATOM_NAMES = list(ATOMS.keys())
N_ATOMS = len(ATOM_NAMES)
MOL_LENGTH = 10  # atoms per molecule

print("Atom Vocabulary:")
print(f"{'Atom':<6} {'Weight':<10} {'LogP':<8} {'Toxicity':<10} {'DrugScore':<10}")
print("-" * 44)
for name, props in ATOMS.items():
    print(f"{name:<6} {props['weight']:<10.1f} {props['logp']:<8.1f} {props['toxicity']:<10.2f} {props['drug_score']:<10.1f}")"""),

        md("""## 2. Molecular Property Calculators"""),

        code("""def compute_properties(molecule):
    '''Compute all properties for a molecule (list of atom indices).'''
    atoms = [ATOM_NAMES[a] for a in molecule]
    props = [ATOMS[a] for a in atoms]

    mw = sum(p['weight'] for p in props)
    logp = sum(p['logp'] for p in props)
    toxicity = 1.0 - np.prod([1.0 - p['toxicity'] for p in props])
    drug_score = np.mean([p['drug_score'] for p in props])

    # Bonus for diversity (having different atom types)
    unique_ratio = len(set(molecule)) / len(molecule)
    diversity_bonus = unique_ratio * 0.2

    return {
        'mw': mw,
        'logp': logp,
        'toxicity': toxicity,
        'drug_score': drug_score + diversity_bonus,
        'diversity': unique_ratio,
        'formula': ''.join(atoms)
    }

def compute_reward(properties, target_logp=1.5, logp_sigma=2.0):
    '''Multi-objective reward function.'''
    # LogP reward: Gaussian centered at target
    r_logp = np.exp(-((properties['logp'] - target_logp)**2) / (2 * logp_sigma**2))

    # Molecular weight reward: penalize if > 500
    mw = properties['mw']
    if mw < 200:
        r_mw = mw / 200.0
    elif mw < 500:
        r_mw = 1.0
    else:
        r_mw = np.exp(-(mw - 500) / 100.0)

    # Toxicity reward: lower is better
    r_tox = 1.0 - properties['toxicity']

    # Drug-likeness reward
    r_drug = 1.0 / (1.0 + np.exp(-(properties['drug_score'] - 0.4) * 5))

    # Weighted combination
    weights = [0.3, 0.2, 0.25, 0.25]
    total = weights[0]*r_logp + weights[1]*r_mw + weights[2]*r_tox + weights[3]*r_drug

    return total, {'logp': r_logp, 'mw': r_mw, 'tox': r_tox, 'drug': r_drug}

# Test with a random molecule
test_mol = [0, 1, 2, 0, 0, 1, 2, 3, 4, 0]  # C N O C C N O S F C
test_props = compute_properties(test_mol)
test_reward, test_components = compute_reward(test_props)

print(f"Test molecule: {test_props['formula']}")
print(f"  MW: {test_props['mw']:.1f}, LogP: {test_props['logp']:.2f}")
print(f"  Toxicity: {test_props['toxicity']:.3f}, Drug Score: {test_props['drug_score']:.3f}")
print(f"  Total Reward: {test_reward:.4f}")
print(f"  Components: {test_components}")"""),

        md("""## 3. Molecular Generation Policy"""),

        code("""class MolecularPolicy:
    '''Policy network for molecule generation.

    Uses a simple linear model: at each step, the state features (counts of
    atoms so far + position) are mapped to action logits via a weight matrix.
    '''

    def __init__(self, n_atoms=N_ATOMS, mol_length=MOL_LENGTH, lr=0.01):
        self.n_atoms = n_atoms
        self.mol_length = mol_length
        self.lr = lr
        # Feature dim: atom counts (5) + position fraction (1) + remaining fraction (1) = 7
        self.feat_dim = n_atoms + 2
        # Weight matrix: feat_dim -> n_atoms
        self.W = np.random.randn(self.feat_dim, n_atoms) * 0.1
        self.b = np.zeros(n_atoms)

    def _get_features(self, partial_mol, step):
        '''Extract features from partial molecule.'''
        counts = np.zeros(self.n_atoms)
        for a in partial_mol:
            counts[a] += 1
        if len(partial_mol) > 0:
            counts /= len(partial_mol)
        position_frac = step / self.mol_length
        remaining_frac = 1.0 - position_frac
        return np.concatenate([counts, [position_frac, remaining_frac]])

    def get_probs(self, partial_mol, step):
        '''Get action probabilities.'''
        feat = self._get_features(partial_mol, step)
        logits = feat @ self.W + self.b
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        return probs, feat

    def sample_molecule(self):
        '''Generate one molecule.'''
        molecule = []
        log_probs = []
        features = []

        for step in range(self.mol_length):
            probs, feat = self.get_probs(molecule, step)
            action = np.random.choice(self.n_atoms, p=probs)
            molecule.append(action)
            log_probs.append(np.log(probs[action] + 1e-10))
            features.append((feat, action, probs))

        return molecule, log_probs, features

    def update(self, episodes):
        '''REINFORCE update with baseline.'''
        rewards = [ep['reward'] for ep in episodes]
        baseline = np.mean(rewards)

        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)

        for ep in episodes:
            advantage = ep['reward'] - baseline
            for feat, action, probs in ep['features']:
                # Gradient of log softmax
                grad_logits = -probs.copy()
                grad_logits[action] += 1.0
                # Scale by advantage
                grad_W += advantage * np.outer(feat, grad_logits)
                grad_b += advantage * grad_logits

        n = len(episodes)
        self.W += self.lr * grad_W / n
        self.b += self.lr * grad_b / n

policy = MolecularPolicy(lr=0.02)
print("Molecular generation policy created.")
print(f"  Feature dimension: {policy.feat_dim}")
print(f"  Action space: {N_ATOMS} atom types")
print(f"  Molecule length: {MOL_LENGTH}")"""),

        md("""## 4. Training Loop"""),

        code("""N_ITERATIONS = 200
BATCH_SIZE = 64

reward_history = []
property_history = {'mw': [], 'logp': [], 'toxicity': [], 'drug_score': [], 'diversity': []}
component_history = {'logp': [], 'mw': [], 'tox': [], 'drug': []}
best_molecules = []
best_reward = -np.inf

for iteration in range(N_ITERATIONS):
    episodes = []

    for _ in range(BATCH_SIZE):
        mol, log_probs, features = policy.sample_molecule()
        props = compute_properties(mol)
        reward, components = compute_reward(props)

        episodes.append({
            'molecule': mol,
            'log_probs': log_probs,
            'features': features,
            'reward': reward,
            'properties': props,
            'components': components
        })

    # Update policy
    policy.update(episodes)

    # Track metrics
    batch_rewards = [ep['reward'] for ep in episodes]
    mean_reward = np.mean(batch_rewards)
    reward_history.append(mean_reward)

    for key in property_history:
        vals = [ep['properties'][key] for ep in episodes]
        property_history[key].append(np.mean(vals))

    for key in component_history:
        vals = [ep['components'][key] for ep in episodes]
        component_history[key].append(np.mean(vals))

    # Track best molecules
    for ep in episodes:
        if ep['reward'] > best_reward:
            best_reward = ep['reward']
            best_molecules.append({
                'formula': ep['properties']['formula'],
                'reward': ep['reward'],
                'props': ep['properties'],
                'iteration': iteration
            })

    if iteration % 40 == 0:
        print(f"Iter {iteration:3d}: reward={mean_reward:.4f}, "
              f"MW={property_history['mw'][-1]:.1f}, "
              f"LogP={property_history['logp'][-1]:.2f}, "
              f"Tox={property_history['toxicity'][-1]:.3f}")

print(f"\nTraining complete! Best reward: {best_reward:.4f}")
if best_molecules:
    bm = best_molecules[-1]
    print(f"Best molecule: {bm['formula']} (reward={bm['reward']:.4f})")"""),

        md("""## 5. Training Curves"""),

        code("""fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Overall reward
axes[0, 0].plot(reward_history, color='steelblue', linewidth=1, alpha=0.4)
# Smooth
window = 10
smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')
axes[0, 0].plot(smoothed, color='steelblue', linewidth=2)
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Mean Reward')
axes[0, 0].set_title('Total Reward Over Training')
axes[0, 0].grid(True, alpha=0.3)

# Reward components
for key, color, label in [('logp', 'blue', 'LogP'), ('mw', 'green', 'MW'),
                           ('tox', 'red', 'Toxicity'), ('drug', 'purple', 'Drug-likeness')]:
    smoothed = np.convolve(component_history[key], np.ones(window)/window, mode='valid')
    axes[0, 1].plot(smoothed, color=color, linewidth=2, label=label)
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Component Reward')
axes[0, 1].set_title('Reward Components Over Training')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Molecular properties
axes[1, 0].plot(property_history['mw'], label='Mol Weight', color='green', linewidth=2)
ax2 = axes[1, 0].twinx()
ax2.plot(property_history['logp'], label='LogP', color='blue', linewidth=2)
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Molecular Weight', color='green')
ax2.set_ylabel('LogP', color='blue')
axes[1, 0].set_title('Property Evolution')
axes[1, 0].grid(True, alpha=0.3)

# Toxicity and diversity
axes[1, 1].plot(property_history['toxicity'], label='Toxicity', color='red', linewidth=2)
axes[1, 1].plot(property_history['diversity'], label='Diversity', color='purple', linewidth=2)
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Toxicity & Diversity Over Training')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('drug_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## 6. Property Distributions of Generated Molecules"""),

        code("""# Generate a batch of molecules with trained policy
N_GEN = 500
gen_molecules = []
for _ in range(N_GEN):
    mol, _, _ = policy.sample_molecule()
    props = compute_properties(mol)
    reward, _ = compute_reward(props)
    props['reward'] = reward
    gen_molecules.append(props)

# Also generate random baseline
random_molecules = []
for _ in range(N_GEN):
    mol = list(np.random.randint(0, N_ATOMS, size=MOL_LENGTH))
    props = compute_properties(mol)
    reward, _ = compute_reward(props)
    props['reward'] = reward
    random_molecules.append(props)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for idx, (key, title, target) in enumerate([
    ('mw', 'Molecular Weight', None),
    ('logp', 'LogP', 1.5),
    ('toxicity', 'Toxicity', None),
    ('drug_score', 'Drug Score', None),
    ('diversity', 'Atom Diversity', None),
    ('reward', 'Total Reward', None),
]):
    ax = axes.flat[idx]
    gen_vals = [m[key] for m in gen_molecules]
    rand_vals = [m[key] for m in random_molecules]

    ax.hist(rand_vals, bins=25, alpha=0.5, label='Random', color='gray', density=True)
    ax.hist(gen_vals, bins=25, alpha=0.5, label='RL-Generated', color='steelblue', density=True)
    if target is not None:
        ax.axvline(x=target, color='red', linestyle='--', linewidth=2, label=f'Target={target}')
    ax.set_xlabel(title)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('drug_property_distributions.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## 7. Pareto Front Analysis"""),

        code("""# Multi-objective: plot Pareto fronts for key property pairs
gen_logp = [m['logp'] for m in gen_molecules]
gen_tox = [m['toxicity'] for m in gen_molecules]
gen_drug = [m['drug_score'] for m in gen_molecules]
gen_mw = [m['mw'] for m in gen_molecules]

rand_logp = [m['logp'] for m in random_molecules]
rand_tox = [m['toxicity'] for m in random_molecules]
rand_drug = [m['drug_score'] for m in random_molecules]

def find_pareto_front(obj1, obj2, maximize_both=True):
    '''Find Pareto-optimal points. obj1 and obj2 are arrays to maximize.'''
    points = np.column_stack([obj1, obj2])
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if maximize_both:
                if points[j, 0] >= points[i, 0] and points[j, 1] >= points[i, 1] and \
                   (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1]):
                    is_pareto[i] = False
                    break
            else:  # maximize obj1, minimize obj2
                if points[j, 0] >= points[i, 0] and points[j, 1] <= points[i, 1] and \
                   (points[j, 0] > points[i, 0] or points[j, 1] < points[i, 1]):
                    is_pareto[i] = False
                    break
    return is_pareto

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Drug Score vs Toxicity (want high drug, low tox)
ax = axes[0]
ax.scatter(rand_tox, rand_drug, alpha=0.3, s=10, color='gray', label='Random')
ax.scatter(gen_tox, gen_drug, alpha=0.3, s=10, color='steelblue', label='RL')
pareto = find_pareto_front([-t for t in gen_tox], gen_drug, maximize_both=True)
p_tox = np.array(gen_tox)[pareto]
p_drug = np.array(gen_drug)[pareto]
order = np.argsort(p_tox)
ax.plot(p_tox[order], p_drug[order], 'r-o', markersize=4, linewidth=2, label='Pareto Front')
ax.set_xlabel('Toxicity (lower is better)')
ax.set_ylabel('Drug Score (higher is better)')
ax.set_title('Drug Score vs Toxicity')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# LogP vs Drug Score
ax = axes[1]
ax.scatter(rand_logp, rand_drug, alpha=0.3, s=10, color='gray', label='Random')
ax.scatter(gen_logp, gen_drug, alpha=0.3, s=10, color='steelblue', label='RL')
ax.axvline(x=1.5, color='red', linestyle='--', alpha=0.5, label='Target LogP')
ax.set_xlabel('LogP')
ax.set_ylabel('Drug Score')
ax.set_title('LogP vs Drug Score')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# LogP vs Toxicity
ax = axes[2]
ax.scatter(rand_logp, rand_tox, alpha=0.3, s=10, color='gray', label='Random')
ax.scatter(gen_logp, gen_tox, alpha=0.3, s=10, color='steelblue', label='RL')
ax.set_xlabel('LogP')
ax.set_ylabel('Toxicity')
ax.set_title('LogP vs Toxicity')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('drug_pareto_front.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## 8. Top Molecules Analysis"""),

        code("""# Sort generated molecules by reward
gen_sorted = sorted(gen_molecules, key=lambda x: x['reward'], reverse=True)

print("Top 10 Generated Molecules:")
print(f"{'Rank':<6} {'Formula':<14} {'Reward':<10} {'MW':<8} {'LogP':<8} {'Tox':<8} {'Drug':<8}")
print("-" * 62)
for i, m in enumerate(gen_sorted[:10]):
    print(f"{i+1:<6} {m['formula']:<14} {m['reward']:<10.4f} {m['mw']:<8.1f} "
          f"{m['logp']:<8.2f} {m['toxicity']:<8.3f} {m['drug_score']:<8.3f}")

# Atom frequency in top molecules vs random
top_n = 50
top_mols = gen_sorted[:top_n]

top_atom_freq = np.zeros(N_ATOMS)
rand_atom_freq = np.zeros(N_ATOMS)

for m in top_mols:
    for ch in m['formula']:
        idx = ATOM_NAMES.index(ch)
        top_atom_freq[idx] += 1
top_atom_freq /= top_atom_freq.sum()

for m in random_molecules[:top_n]:
    for ch in m['formula']:
        idx = ATOM_NAMES.index(ch)
        rand_atom_freq[idx] += 1
rand_atom_freq /= rand_atom_freq.sum()

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(N_ATOMS)
width = 0.35
ax.bar(x - width/2, rand_atom_freq, width, label='Random', color='gray', alpha=0.7)
ax.bar(x + width/2, top_atom_freq, width, label=f'Top {top_n} RL', color='steelblue', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(ATOM_NAMES)
ax.set_ylabel('Frequency')
ax.set_title('Atom Usage: Top RL Molecules vs Random')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.savefig('drug_atom_frequency.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## 9. Quality Improvement Over Training"""),

        code("""# Track how the quality of the best molecule improves
best_per_iter = []
running_best = -np.inf
for i, r in enumerate(reward_history):
    running_best = max(running_best, r)
    best_per_iter.append(running_best)

plt.figure(figsize=(10, 5))
plt.plot(reward_history, alpha=0.4, color='steelblue', linewidth=1, label='Mean Reward')
plt.plot(best_per_iter, color='darkorange', linewidth=2, label='Best So Far')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.title('Quality Improvement Over Training')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('drug_quality_improvement.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## Conclusions

1. **RL successfully optimizes molecular properties**: The agent learns to generate molecules with better drug-likeness, lower toxicity, and target LogP.
2. **Multi-objective balancing**: The weighted reward function guides the agent toward Pareto-optimal molecules.
3. **Atom preferences emerge**: The agent learns which atoms contribute most to desired properties (e.g., more O for lower toxicity, more N for drug-likeness).
4. **Pareto front**: RL-generated molecules dominate random molecules on the trade-off between competing objectives.
5. **Limitations**: This simplified model uses atom sequences rather than molecular graphs - real drug discovery uses graph-based representations.

### Key Takeaway
RL provides a powerful framework for multi-objective molecular optimization. Even with a simplified representation, the agent discovers meaningful chemical patterns that optimize drug-like properties."""),
    ]
    save(path, cells)


# =============================================================================
# Notebook 10: Sim-to-Real Transfer
# =============================================================================
def make_notebook_10():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "Project_10_Sim_to_Real", "sim_to_real_notebook.ipynb")
    cells = [
        md("""# Project 10: Sim-to-Real Transfer with Domain Randomization
**Author:** Milan Amrut Joshi

## Overview
This notebook demonstrates **Sim-to-Real Transfer** using **Domain Randomization** on a simplified CartPole-like balancing task.

### Key Concepts
- **Sim-to-Real Gap**: Policies trained in simulation often fail in the real world due to modeling errors
- **Domain Randomization**: Randomize simulator parameters during training to produce robust policies
- **Transfer Evaluation**: Test sim-trained policies under "real" (noisy) conditions

### Approach
1. Build a CartPole simulator with configurable physics parameters
2. Train three policies:
   - **Sim-only**: Trained with default (clean) simulator parameters
   - **Domain-randomized**: Trained with randomized gravity, friction, and noise
   - **Real-only**: Trained directly in the noisy "real" environment
3. Evaluate all policies under various noise levels"""),

        md(r"""## Mathematical Framework

### CartPole Dynamics
The CartPole state is $s = [x, \dot{x}, \theta, \dot{\theta}]$:
- $x$: cart position
- $\dot{x}$: cart velocity
- $\theta$: pole angle (from vertical)
- $\dot{\theta}$: pole angular velocity

### Equations of Motion (Simplified)
$$\ddot{\theta} = \frac{g \sin\theta + \cos\theta \left(\frac{-F - m_p l \dot{\theta}^2 \sin\theta}{m_c + m_p}\right)}{l\left(\frac{4}{3} - \frac{m_p \cos^2\theta}{m_c + m_p}\right)}$$

$$\ddot{x} = \frac{F + m_p l (\dot{\theta}^2 \sin\theta - \ddot{\theta}\cos\theta)}{m_c + m_p}$$

where $F$ is the applied force, $g$ is gravity, $m_c$ is cart mass, $m_p$ is pole mass, $l$ is pole half-length.

### Domain Randomization
During training, physics parameters are sampled:
$$g \sim \mathcal{U}(g_0 - \Delta g, \; g_0 + \Delta g)$$
$$\mu_{\text{friction}} \sim \mathcal{U}(0, \; \mu_{\max})$$
$$\text{obs\_noise} \sim \mathcal{N}(0, \; \sigma_{\text{obs}}^2)$$

### Policy (Linear)
$$a = \text{sign}(\theta^T s)$$

where $\theta \in \mathbb{R}^4$ are learned weights.

### Real Environment
The "real" environment has fixed but different parameters plus observation noise:
$$s_{\text{observed}} = s_{\text{true}} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_{\text{real}}^2)$$"""),

        md("""## Setup"""),

        code("""import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
print("Sim-to-Real Transfer with Domain Randomization - Setup Complete")"""),

        md("""## 1. CartPole Environment (Numpy Implementation)"""),

        code("""class CartPole:
    '''Simplified CartPole environment with configurable physics.'''

    def __init__(self, gravity=9.8, cart_mass=1.0, pole_mass=0.1, pole_length=0.5,
                 friction=0.0, force_mag=10.0, dt=0.02, obs_noise=0.0):
        self.gravity = gravity
        self.cart_mass = cart_mass
        self.pole_mass = pole_mass
        self.pole_length = pole_length
        self.friction = friction
        self.force_mag = force_mag
        self.dt = dt
        self.obs_noise = obs_noise
        self.total_mass = cart_mass + pole_mass
        self.polemass_length = pole_mass * pole_length
        self.state = None

    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0]) + np.random.uniform(-0.05, 0.05, 4)
        return self._observe()

    def _observe(self):
        if self.obs_noise > 0:
            return self.state + np.random.normal(0, self.obs_noise, 4)
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag

        # Apply friction
        force -= self.friction * x_dot

        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sin_th) / self.total_mass
        theta_acc = (self.gravity * sin_th - cos_th * temp) / \
                    (self.pole_length * (4.0/3.0 - self.pole_mass * cos_th**2 / self.total_mass))
        x_acc = temp - self.polemass_length * theta_acc * cos_th / self.total_mass

        # Euler integration
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * x_acc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])

        # Termination conditions
        done = bool(abs(x) > 2.4 or abs(theta) > 0.2095)  # ~12 degrees
        reward = 1.0 if not done else 0.0

        return self._observe(), reward, done

    def __repr__(self):
        return (f"CartPole(g={self.gravity:.1f}, mc={self.cart_mass:.1f}, "
                f"mp={self.pole_mass:.2f}, l={self.pole_length:.1f}, "
                f"mu={self.friction:.2f}, noise={self.obs_noise:.3f})")

# Create environments
sim_env = CartPole()  # clean simulator
real_env = CartPole(gravity=10.5, friction=0.1, obs_noise=0.02,
                     pole_mass=0.12, pole_length=0.55)

print(f"Sim env:  {sim_env}")
print(f"Real env: {real_env}")"""),

        md("""## 2. Domain Randomization Environment Factory"""),

        code("""class DomainRandomizedCartPole:
    '''Wrapper that randomizes physics parameters each episode.'''

    def __init__(self, gravity_range=(8.0, 12.0), friction_range=(0.0, 0.2),
                 noise_range=(0.0, 0.03), mass_range=(0.08, 0.15),
                 length_range=(0.4, 0.6)):
        self.gravity_range = gravity_range
        self.friction_range = friction_range
        self.noise_range = noise_range
        self.mass_range = mass_range
        self.length_range = length_range
        self.env = None

    def reset(self):
        g = np.random.uniform(*self.gravity_range)
        mu = np.random.uniform(*self.friction_range)
        noise = np.random.uniform(*self.noise_range)
        mp = np.random.uniform(*self.mass_range)
        pl = np.random.uniform(*self.length_range)
        self.env = CartPole(gravity=g, friction=mu, obs_noise=noise,
                            pole_mass=mp, pole_length=pl)
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

dr_env = DomainRandomizedCartPole()
print("Domain Randomized Environment created.")
print(f"  Gravity range: {dr_env.gravity_range}")
print(f"  Friction range: {dr_env.friction_range}")
print(f"  Noise range: {dr_env.noise_range}")
print(f"  Pole mass range: {dr_env.mass_range}")
print(f"  Pole length range: {dr_env.length_range}")"""),

        md("""## 3. Linear Policy with Evolution Strategy (ES) Optimization

We use a simple evolution strategy to optimize a linear policy, which works well for low-dimensional problems."""),

        code("""class LinearPolicy:
    '''Linear policy: action = 1 if w^T s > 0, else 0.'''

    def __init__(self, state_dim=4):
        self.w = np.zeros(state_dim)

    def act(self, state):
        return 1 if np.dot(self.w, state) > 0 else 0

    def copy(self):
        p = LinearPolicy(len(self.w))
        p.w = self.w.copy()
        return p

def evaluate_policy(policy, env, n_episodes=5, max_steps=500):
    '''Evaluate a policy and return mean episode length.'''
    total_steps = 0
    for _ in range(n_episodes):
        obs = env.reset()
        for step in range(max_steps):
            action = policy.act(obs)
            obs, reward, done = env.step(action)
            if done:
                break
        total_steps += step + 1
    return total_steps / n_episodes

def evolution_strategy(env, n_generations=100, pop_size=50, sigma=0.1,
                       lr=0.03, max_steps=500, n_eval=3):
    '''Simple evolution strategy to optimize the linear policy.'''
    policy = LinearPolicy()
    best_reward = 0
    reward_history = []

    for gen in range(n_generations):
        # Generate perturbations
        noise = np.random.randn(pop_size, 4)
        rewards = np.zeros(pop_size)

        for i in range(pop_size):
            candidate = policy.copy()
            candidate.w = policy.w + sigma * noise[i]
            rewards[i] = evaluate_policy(candidate, env, n_episodes=n_eval, max_steps=max_steps)

        # Normalize rewards
        rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Update weights
        grad = np.dot(noise.T, rewards_norm) / pop_size
        policy.w += lr * grad

        mean_reward = rewards.mean()
        reward_history.append(mean_reward)
        best_reward = max(best_reward, mean_reward)

        if gen % 20 == 0:
            print(f"  Gen {gen:3d}: mean={mean_reward:.1f}, best={best_reward:.1f}, w={policy.w}")

    return policy, reward_history

print("Evolution Strategy optimizer ready.")"""),

        md("""## 4. Train Three Policies"""),

        code("""print("=" * 60)
print("Training Policy 1: Sim-Only (clean simulator)")
print("=" * 60)
policy_sim, hist_sim = evolution_strategy(sim_env, n_generations=80, pop_size=40)

print()
print("=" * 60)
print("Training Policy 2: Domain-Randomized")
print("=" * 60)
policy_dr, hist_dr = evolution_strategy(dr_env, n_generations=80, pop_size=40)

print()
print("=" * 60)
print("Training Policy 3: Real-Only (noisy environment)")
print("=" * 60)
policy_real, hist_real = evolution_strategy(real_env, n_generations=80, pop_size=40)

print("\nFinal weights:")
print(f"  Sim-only:   {policy_sim.w}")
print(f"  Domain-Rand:{policy_dr.w}")
print(f"  Real-only:  {policy_real.w}")"""),

        md("""## 5. Training Curves"""),

        code("""fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(hist_sim, label='Sim-Only', color='steelblue', linewidth=2)
ax.plot(hist_dr, label='Domain-Randomized', color='darkorange', linewidth=2)
ax.plot(hist_real, label='Real-Only', color='green', linewidth=2)

ax.set_xlabel('Generation')
ax.set_ylabel('Mean Episode Length')
ax.set_title('Training Curves: Three Policy Training Regimes')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sim2real_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## 6. Transfer Evaluation: Test All Policies in "Real" Environment"""),

        code("""def evaluate_transfer(policy, env, n_episodes=50, max_steps=500):
    '''Evaluate policy transfer to a target environment.'''
    episode_lengths = []
    for _ in range(n_episodes):
        obs = env.reset()
        for step in range(max_steps):
            action = policy.act(obs)
            obs, reward, done = env.step(action)
            if done:
                break
        episode_lengths.append(step + 1)
    return episode_lengths

# Evaluate all policies in real environment
print("Evaluating policies in 'real' environment...")
real_results_sim = evaluate_transfer(policy_sim, real_env)
real_results_dr = evaluate_transfer(policy_dr, real_env)
real_results_real = evaluate_transfer(policy_real, real_env)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
data = [real_results_sim, real_results_dr, real_results_real]
labels = ['Sim-Only', 'Domain-Rand', 'Real-Only']
colors = ['steelblue', 'darkorange', 'green']

bp = axes[0].boxplot(data, labels=labels, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0].set_ylabel('Episode Length in Real Env')
axes[0].set_title('Transfer Performance (Real Environment)')
axes[0].grid(True, alpha=0.3, axis='y')

# Bar chart with error bars
means = [np.mean(d) for d in data]
stds = [np.std(d) for d in data]
axes[1].bar(labels, means, yerr=stds, color=colors, alpha=0.7, edgecolor='black', capsize=5)
axes[1].set_ylabel('Mean Episode Length')
axes[1].set_title('Mean Transfer Performance')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('sim2real_transfer_eval.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Real env performance:")
print(f"  Sim-Only:      {np.mean(real_results_sim):.1f} +/- {np.std(real_results_sim):.1f}")
print(f"  Domain-Rand:   {np.mean(real_results_dr):.1f} +/- {np.std(real_results_dr):.1f}")
print(f"  Real-Only:     {np.mean(real_results_real):.1f} +/- {np.std(real_results_real):.1f}")"""),

        md("""## 7. Robustness Across Noise Levels"""),

        code("""noise_levels = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
n_eval = 30

sim_performance = []
dr_performance = []
real_performance = []

for noise in noise_levels:
    test_env = CartPole(gravity=10.5, friction=0.1, obs_noise=noise,
                        pole_mass=0.12, pole_length=0.55)

    sim_perf = np.mean(evaluate_transfer(policy_sim, test_env, n_eval))
    dr_perf = np.mean(evaluate_transfer(policy_dr, test_env, n_eval))
    real_perf = np.mean(evaluate_transfer(policy_real, test_env, n_eval))

    sim_performance.append(sim_perf)
    dr_performance.append(dr_perf)
    real_performance.append(real_perf)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(noise_levels, sim_performance, 'o-', label='Sim-Only', color='steelblue',
        linewidth=2, markersize=8)
ax.plot(noise_levels, dr_performance, 's-', label='Domain-Randomized', color='darkorange',
        linewidth=2, markersize=8)
ax.plot(noise_levels, real_performance, '^-', label='Real-Only', color='green',
        linewidth=2, markersize=8)

ax.set_xlabel('Observation Noise Level (std)')
ax.set_ylabel('Mean Episode Length')
ax.set_title('Robustness to Observation Noise')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sim2real_robustness_noise.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## 8. Robustness Across Gravity Variations"""),

        code("""gravity_values = [7.0, 8.0, 9.0, 9.8, 10.5, 11.0, 12.0, 13.0]
n_eval = 30

sim_grav = []
dr_grav = []
real_grav = []

for g in gravity_values:
    test_env = CartPole(gravity=g, friction=0.05, obs_noise=0.01,
                        pole_mass=0.11, pole_length=0.5)

    sim_grav.append(np.mean(evaluate_transfer(policy_sim, test_env, n_eval)))
    dr_grav.append(np.mean(evaluate_transfer(policy_dr, test_env, n_eval)))
    real_grav.append(np.mean(evaluate_transfer(policy_real, test_env, n_eval)))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(gravity_values, sim_grav, 'o-', label='Sim-Only', color='steelblue',
        linewidth=2, markersize=8)
ax.plot(gravity_values, dr_grav, 's-', label='Domain-Randomized', color='darkorange',
        linewidth=2, markersize=8)
ax.plot(gravity_values, real_grav, '^-', label='Real-Only', color='green',
        linewidth=2, markersize=8)

ax.axvline(x=9.8, color='steelblue', linestyle=':', alpha=0.5, label='Sim default (9.8)')
ax.axvline(x=10.5, color='green', linestyle=':', alpha=0.5, label='Real default (10.5)')

ax.set_xlabel('Gravity')
ax.set_ylabel('Mean Episode Length')
ax.set_title('Robustness to Gravity Variations')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sim2real_robustness_gravity.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## 9. Comprehensive Robustness Heatmap"""),

        code("""# Test across combinations of noise and gravity
noise_test = [0.0, 0.01, 0.02, 0.04, 0.06]
gravity_test = [8.0, 9.0, 10.0, 11.0, 12.0]
n_eval = 20

def build_heatmap(policy, noise_vals, gravity_vals, n_eval):
    heatmap = np.zeros((len(gravity_vals), len(noise_vals)))
    for i, g in enumerate(gravity_vals):
        for j, n in enumerate(noise_vals):
            test_env = CartPole(gravity=g, friction=0.05, obs_noise=n,
                                pole_mass=0.11, pole_length=0.5)
            heatmap[i, j] = np.mean(evaluate_transfer(policy, test_env, n_eval))
    return heatmap

hm_sim = build_heatmap(policy_sim, noise_test, gravity_test, n_eval)
hm_dr = build_heatmap(policy_dr, noise_test, gravity_test, n_eval)
hm_real = build_heatmap(policy_real, noise_test, gravity_test, n_eval)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
vmin = min(hm_sim.min(), hm_dr.min(), hm_real.min())
vmax = max(hm_sim.max(), hm_dr.max(), hm_real.max())

for ax, hm, title in [(axes[0], hm_sim, 'Sim-Only'),
                        (axes[1], hm_dr, 'Domain-Randomized'),
                        (axes[2], hm_real, 'Real-Only')]:
    im = ax.imshow(hm, cmap='YlOrRd', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(noise_test)))
    ax.set_xticklabels([f'{n:.2f}' for n in noise_test])
    ax.set_yticks(range(len(gravity_test)))
    ax.set_yticklabels([f'{g:.0f}' for g in gravity_test])
    ax.set_xlabel('Observation Noise')
    ax.set_ylabel('Gravity')
    ax.set_title(title)
    for i in range(len(gravity_test)):
        for j in range(len(noise_test)):
            ax.text(j, i, f'{hm[i,j]:.0f}', ha='center', va='center', fontsize=9)

plt.colorbar(im, ax=axes, label='Mean Episode Length', shrink=0.8)
plt.suptitle('Transfer Robustness Heatmaps', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('sim2real_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()"""),

        md("""## 10. Summary Statistics"""),

        code("""print("=" * 70)
print("SIM-TO-REAL TRANSFER SUMMARY")
print("=" * 70)

print("\nPolicy Weights (linear policy: action = sign(w^T state)):")
print(f"  Sim-Only:         {policy_sim.w}")
print(f"  Domain-Randomized:{policy_dr.w}")
print(f"  Real-Only:        {policy_real.w}")

print(f"\nPerformance in REAL Environment (50 episodes):")
print(f"  Sim-Only:          {np.mean(real_results_sim):6.1f} +/- {np.std(real_results_sim):.1f}")
print(f"  Domain-Randomized: {np.mean(real_results_dr):6.1f} +/- {np.std(real_results_dr):.1f}")
print(f"  Real-Only:         {np.mean(real_results_real):6.1f} +/- {np.std(real_results_real):.1f}")

print(f"\nAverage Robustness (across all noise levels):")
print(f"  Sim-Only:          {np.mean(sim_performance):6.1f}")
print(f"  Domain-Randomized: {np.mean(dr_performance):6.1f}")
print(f"  Real-Only:         {np.mean(real_performance):6.1f}")

print(f"\nAverage Robustness (across gravity variations):")
print(f"  Sim-Only:          {np.mean(sim_grav):6.1f}")
print(f"  Domain-Randomized: {np.mean(dr_grav):6.1f}")
print(f"  Real-Only:         {np.mean(real_grav):6.1f}")"""),

        md("""## Conclusions

1. **Sim-to-real gap is real**: The sim-only policy performs well in simulation but degrades under real-world conditions (noise, different physics).
2. **Domain randomization bridges the gap**: By training across varied simulator settings, the domain-randomized policy becomes robust to parameter uncertainty.
3. **Real-only training is sample-efficient for the target**: But it does not generalize well to conditions different from training.
4. **Domain randomization provides the best overall robustness**: It performs competitively across the widest range of conditions.
5. **The trade-off**: Domain-randomized policies may sacrifice peak performance for broader robustness.

### Key Takeaway
Domain randomization is a simple yet powerful technique for sim-to-real transfer. By making the training environment more varied than reality, the policy learns invariant features that transfer better to the real world. This is foundational for robotics, autonomous vehicles, and other physical AI applications."""),
    ]
    save(path, cells)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Generating RL Project Notebooks 06-10...\n")
    make_notebook_06()
    make_notebook_07()
    make_notebook_08()
    make_notebook_09()
    make_notebook_10()
    print("\nAll 5 notebooks generated successfully!")
