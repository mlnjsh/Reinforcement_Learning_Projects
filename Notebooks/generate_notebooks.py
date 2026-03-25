"""Generate all 20 RL basics notebooks."""
import json
import os
import uuid

def nb(cells):
    """Create notebook structure."""
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
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().split("\n"), "id": str(uuid.uuid4())[:8]}

def code(source, outputs=None):
    return {"cell_type": "code", "metadata": {}, "source": source.strip().split("\n"), "execution_count": None, "outputs": outputs or [], "id": str(uuid.uuid4())[:8]}

def save(name, cells):
    path = os.path.join(os.path.dirname(__file__), f"{name}.ipynb")
    # Fix: each line needs \n except last
    for cell in cells:
        lines = cell["source"]
        cell["source"] = [l + "\n" if i < len(lines)-1 else l for i, l in enumerate(lines)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb(cells), f, indent=1, ensure_ascii=False)
    print(f"Created {name}.ipynb")

# ============================================================
# NOTEBOOK 01: Introduction to Reinforcement Learning
# ============================================================
save("01_Introduction_to_Reinforcement_Learning", [
md("""# Notebook 01: Introduction to Reinforcement Learning

**Author:** Milan Amrut Joshi
**Series:** Reinforcement Learning Basics (1/20)

---

## 1. What is Reinforcement Learning?

**Reinforcement Learning (RL)** is a branch of machine learning where an **agent** learns to make decisions by interacting with an **environment** to maximize **cumulative reward**.

### The Three Paradigms of Machine Learning

| Paradigm | Data | Signal | Example |
|----------|------|--------|---------|
| **Supervised Learning** | Labeled $(x, y)$ pairs | Correct answer | Image classification |
| **Unsupervised Learning** | Unlabeled $x$ | Structure/patterns | Clustering |
| **Reinforcement Learning** | States, actions, rewards | Reward signal (delayed) | Game playing, robotics |

**Key difference:** In RL, there is no supervisor — only a **reward signal** that may be **delayed**. The agent must discover which actions yield the most reward through **trial and error**.
"""),
md("""## 2. The RL Framework

The agent-environment interaction loop:

```
Agent ──action a_t──> Environment
  ^                      |
  |                      v
  <──state s_{t+1}, reward r_{t+1}──
```

### Core Components

| Symbol | Name | Definition |
|--------|------|------------|
| $\\mathcal{S}$ | **State space** | Set of all possible states |
| $\\mathcal{A}$ | **Action space** | Set of all possible actions |
| $R$ | **Reward function** | $R: \\mathcal{S} \\times \\mathcal{A} \\times \\mathcal{S} \\to \\mathbb{R}$ |
| $P$ | **Transition function** | $P(s'|s,a) = \\Pr[S_{t+1}=s'|S_t=s, A_t=a]$ |
| $\\gamma$ | **Discount factor** | $\\gamma \\in [0, 1]$ — weights future rewards |
| $\\pi$ | **Policy** | $\\pi(a|s) = \\Pr[A_t=a|S_t=s]$ — agent's strategy |

### The Interaction Loop

At each time step $t$:
1. Agent observes state $S_t \\in \\mathcal{S}$
2. Agent selects action $A_t \\sim \\pi(\\cdot|S_t)$
3. Environment transitions to $S_{t+1} \\sim P(\\cdot|S_t, A_t)$
4. Agent receives reward $R_{t+1} = R(S_t, A_t, S_{t+1})$
"""),
md("""## 3. The Reward Hypothesis

> **"All of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward)."**
> — Sutton & Barto

This is the foundational assumption of RL: any goal can be expressed as maximizing expected cumulative reward.

### Return (Cumulative Reward)

The **return** $G_t$ is the total discounted reward from time step $t$:

$$G_t = R_{t+1} + \\gamma R_{t+2} + \\gamma^2 R_{t+3} + \\cdots = \\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1}$$

### Why Discount? ($\\gamma < 1$)

1. **Mathematical convergence:** Ensures $G_t < \\infty$ for bounded rewards: $G_t \\leq \\frac{R_{\\max}}{1 - \\gamma}$
2. **Uncertainty:** Future rewards are less certain
3. **Immediate preference:** A reward now is worth more than the same reward later
4. **Avoids infinite returns** in continuing (non-episodic) tasks

| $\\gamma$ | Behavior |
|-----------|----------|
| $\\gamma = 0$ | **Myopic** — only cares about immediate reward |
| $\\gamma = 1$ | **Far-sighted** — all rewards equally important |
| $\\gamma = 0.99$ | Typical — balances short and long term |
"""),
md("""## 4. Episodes vs Continuing Tasks

### Episodic Tasks
- Have a natural **terminal state** (end of game, end of episode)
- Return: $G_t = \\sum_{k=0}^{T-t-1} \\gamma^k R_{t+k+1}$ where $T$ is the terminal time
- Examples: Chess game, maze navigation, Atari games

### Continuing Tasks
- Go on **forever** — no terminal state
- Must use $\\gamma < 1$ to keep return finite
- Examples: Robot control, stock trading, temperature regulation

## 5. Types of RL

| Criterion | Type A | Type B |
|-----------|--------|--------|
| **Model knowledge** | **Model-Based** (knows $P, R$) | **Model-Free** (learns from experience) |
| **What to learn** | **Value-Based** (learn $V$ or $Q$) | **Policy-Based** (learn $\\pi$ directly) |
| **Data source** | **On-Policy** (learn from own policy) | **Off-Policy** (learn from other policy) |
"""),
md("""## 6. Historical Timeline

| Year | Milestone |
|------|-----------|
| 1950s | **Bellman** — Dynamic Programming, Bellman Equation |
| 1972 | **Klopf** — Trial-and-error learning in adaptive systems |
| 1988 | **Sutton** — Temporal Difference learning |
| 1989 | **Watkins** — Q-Learning |
| 1992 | **Tesauro** — TD-Gammon (backgammon) |
| 1998 | **Sutton & Barto** — RL: An Introduction (1st edition) |
| 2013 | **Mnih et al.** — DQN plays Atari |
| 2016 | **Silver et al.** — AlphaGo defeats Lee Sedol |
| 2017 | **Schulman et al.** — PPO |
| 2019 | **OpenAI Five** — Dota 2 |
| 2022 | **InstructGPT/ChatGPT** — RLHF for language models |
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Simple Grid World Environment from Scratch
# ============================================

class GridWorld:
    \"\"\"
    4x4 Grid World.
    States: 0-15 (4x4 grid)
    Actions: 0=up, 1=right, 2=down, 3=left
    Terminal states: 0 (top-left) and 15 (bottom-right)
    Reward: -1 for every step (encourages shortest path)
    \"\"\"
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.terminal_states = [0, self.n_states - 1]
        self.state = None

    def reset(self):
        # Start in a random non-terminal state
        while True:
            self.state = np.random.randint(self.n_states)
            if self.state not in self.terminal_states:
                break
        return self.state

    def step(self, action):
        if self.state in self.terminal_states:
            return self.state, 0, True

        row, col = divmod(self.state, self.size)

        if action == 0:    # up
            row = max(0, row - 1)
        elif action == 1:  # right
            col = min(self.size - 1, col + 1)
        elif action == 2:  # down
            row = min(self.size - 1, row + 1)
        elif action == 3:  # left
            col = max(0, col - 1)

        self.state = row * self.size + col
        reward = -1
        done = self.state in self.terminal_states
        return self.state, reward, done

    def render(self):
        grid = np.full((self.size, self.size), '.')
        row, col = divmod(self.state, self.size)
        grid[row, col] = 'A'
        grid[0, 0] = 'T'
        grid[self.size-1, self.size-1] = 'T'
        if self.state in self.terminal_states:
            row, col = divmod(self.state, self.size)
            grid[row, col] = 'X'
        for r in grid:
            print(' '.join(r))
        print()

# --- Run a random agent ---
env = GridWorld(size=4)
n_episodes = 1000
episode_lengths = []

for ep in range(n_episodes):
    state = env.reset()
    steps = 0
    done = False
    while not done and steps < 100:
        action = np.random.randint(env.n_actions)  # random policy
        state, reward, done = env.step(action)
        steps += 1
    episode_lengths.append(steps)

print(f"Random agent over {n_episodes} episodes:")
print(f"  Mean steps: {np.mean(episode_lengths):.1f}")
print(f"  Min steps:  {np.min(episode_lengths)}")
print(f"  Max steps:  {np.max(episode_lengths)}")

# Visualize episode lengths
plt.figure(figsize=(10, 4))
plt.plot(episode_lengths, alpha=0.3, color='steelblue')
plt.axhline(np.mean(episode_lengths), color='red', linestyle='--', label=f'Mean = {np.mean(episode_lengths):.1f}')
plt.xlabel('Episode')
plt.ylabel('Steps to Terminal State')
plt.title('Random Agent Performance on 4x4 GridWorld')
plt.legend()
plt.tight_layout()
plt.show()
"""),
code("""# --- Demonstrate the discount factor effect ---
gamma_values = [0.0, 0.5, 0.9, 0.99, 1.0]
rewards = [1] * 20  # constant reward of 1 for 20 steps

plt.figure(figsize=(10, 5))
for gamma in gamma_values:
    discounted = [gamma**k * r for k, r in enumerate(rewards)]
    cumulative = np.cumsum(discounted)
    plt.plot(cumulative, label=f'$\\gamma$ = {gamma}', linewidth=2)

plt.xlabel('Time Step')
plt.ylabel('Cumulative Discounted Return $G_t$')
plt.title('Effect of Discount Factor $\\gamma$ on Return')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Show that G_t converges for gamma < 1
print("Theoretical max return G = R_max / (1 - gamma):")
for gamma in [0.5, 0.9, 0.95, 0.99]:
    print(f"  gamma={gamma}: G_max = 1/(1-{gamma}) = {1/(1-gamma):.1f}")
"""),
md("""## Summary

| Concept | Key Formula |
|---------|------------|
| **Return** | $G_t = \\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1}$ |
| **Policy** | $\\pi(a|s) = \\Pr[A_t = a | S_t = s]$ |
| **Transition** | $P(s'|s,a) = \\Pr[S_{t+1}=s'|S_t=s, A_t=a]$ |
| **Goal** | Maximize $\\mathbb{E}_\\pi[G_t]$ |

**Next:** Notebook 02 — Markov Decision Processes (MDPs)
"""),
])

# ============================================================
# NOTEBOOK 02: Markov Decision Processes
# ============================================================
save("02_Markov_Decision_Processes", [
md("""# Notebook 02: Markov Decision Processes (MDPs)

**Series:** Reinforcement Learning Basics (2/20)

---

## 1. The Markov Property

A state $S_t$ is **Markov** if and only if:

$$P[S_{t+1} | S_t] = P[S_{t+1} | S_1, S_2, \\ldots, S_t]$$

**"The future is independent of the past given the present."**

The current state captures all relevant information from the history. No need to remember the past — the state is a **sufficient statistic**.

### Why is this important?
- Makes the problem **tractable** — we only need to consider current state
- Enables **recursive** solutions (Bellman equations)
- Most real-world problems can be approximately Markov with proper state representation
"""),
md("""## 2. Formal MDP Definition

A **Markov Decision Process** is a tuple $(\\mathcal{S}, \\mathcal{A}, P, R, \\gamma)$:

| Component | Symbol | Definition |
|-----------|--------|------------|
| State space | $\\mathcal{S}$ | Finite set of states |
| Action space | $\\mathcal{A}$ | Finite set of actions |
| Transition probability | $P$ | $P(s'|s,a) = \\Pr[S_{t+1}=s'|S_t=s, A_t=a]$ |
| Reward function | $R$ | $R(s,a,s') = \\mathbb{E}[R_{t+1}|S_t=s, A_t=a, S_{t+1}=s']$ |
| Discount factor | $\\gamma$ | $\\gamma \\in [0,1]$ |

### Transition Probability Properties
For all $s \\in \\mathcal{S}, a \\in \\mathcal{A}$:
$$\\sum_{s' \\in \\mathcal{S}} P(s'|s,a) = 1 \\quad \\text{and} \\quad P(s'|s,a) \\geq 0$$
"""),
md("""## 3. Policy

A **policy** $\\pi$ defines the agent's behavior:

**Stochastic policy:** $\\pi(a|s) = \\Pr[A_t = a | S_t = s]$

**Deterministic policy:** $\\pi(s) = a$ (maps each state to one action)

A policy is **stationary** if it depends only on the current state, not on time: $\\pi(a|s)$ is the same for all $t$.

## 4. Value Functions

### State-Value Function
$$V^\\pi(s) = \\mathbb{E}_\\pi[G_t | S_t = s] = \\mathbb{E}_\\pi\\left[\\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1} \\,\\Big|\\, S_t = s\\right]$$

*"How good is it to be in state $s$ under policy $\\pi$?"*

### Action-Value Function
$$Q^\\pi(s, a) = \\mathbb{E}_\\pi[G_t | S_t = s, A_t = a] = \\mathbb{E}_\\pi\\left[\\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1} \\,\\Big|\\, S_t = s, A_t = a\\right]$$

*"How good is it to take action $a$ in state $s$ under policy $\\pi$?"*

### Relationship between V and Q
$$V^\\pi(s) = \\sum_{a \\in \\mathcal{A}} \\pi(a|s) \\, Q^\\pi(s, a)$$

$$Q^\\pi(s,a) = \\sum_{s'} P(s'|s,a)\\left[R(s,a,s') + \\gamma V^\\pi(s')\\right]$$
"""),
md("""## 5. Optimal Value Functions

$$V^*(s) = \\max_\\pi V^\\pi(s) \\quad \\forall s \\in \\mathcal{S}$$

$$Q^*(s,a) = \\max_\\pi Q^\\pi(s,a) \\quad \\forall s \\in \\mathcal{S}, a \\in \\mathcal{A}$$

An **optimal policy** $\\pi^*$ achieves the optimal value in every state:
$$\\pi^*(s) = \\arg\\max_a Q^*(s, a)$$

### Theorem: Existence of Optimal Policy
For any finite MDP, there exists at least one deterministic optimal policy $\\pi^*$ such that:
- $V^{\\pi^*}(s) = V^*(s) \\quad \\forall s$
- $Q^{\\pi^*}(s,a) = Q^*(s,a) \\quad \\forall s, a$
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# ============================================
# MDP Class Implementation
# ============================================

class MDP:
    \"\"\"
    Finite Markov Decision Process.

    Parameters:
        n_states: number of states
        n_actions: number of actions
        P: transition probabilities P[s, a, s'] = P(s'|s,a)
        R: reward function R[s, a, s']
        gamma: discount factor
    \"\"\"
    def __init__(self, n_states, n_actions, P, R, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = P  # shape: (n_states, n_actions, n_states)
        self.R = R  # shape: (n_states, n_actions, n_states)
        self.gamma = gamma

    def expected_reward(self, s, a):
        \"\"\"E[R | s, a] = sum_s' P(s'|s,a) * R(s,a,s')\"\"\"
        return np.sum(self.P[s, a] * self.R[s, a])

    def compute_v_pi(self, pi):
        \"\"\"
        Compute V^pi by solving the linear system:
        V = R^pi + gamma * P^pi * V
        => (I - gamma * P^pi) V = R^pi
        \"\"\"
        # Build P^pi and R^pi
        P_pi = np.zeros((self.n_states, self.n_states))
        R_pi = np.zeros(self.n_states)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                P_pi[s] += pi[s, a] * self.P[s, a]
                R_pi[s] += pi[s, a] * self.expected_reward(s, a)
        # Solve: (I - gamma * P_pi) V = R_pi
        A = np.eye(self.n_states) - self.gamma * P_pi
        V = np.linalg.solve(A, R_pi)
        return V

    def compute_q_pi(self, pi):
        \"\"\"Compute Q^pi from V^pi.\"\"\"
        V = self.compute_v_pi(pi)
        Q = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                Q[s, a] = np.sum(self.P[s, a] * (self.R[s, a] + self.gamma * V))
        return Q


# --- Example: Student MDP (simplified) ---
# States: 0=Class1, 1=Class2, 2=Class3, 3=Pass, 4=Facebook, 5=Sleep
# Simplified to 3 states for clarity
n_s, n_a = 3, 2  # 3 states, 2 actions (study/slack)
P = np.zeros((n_s, n_a, n_s))
R = np.zeros((n_s, n_a, n_s))

# State 0: action 0 (study) -> state 1, reward -2
P[0, 0, 1] = 1.0;  R[0, 0, 1] = -2
# State 0: action 1 (slack) -> state 0, reward -1
P[0, 1, 0] = 1.0;  R[0, 1, 0] = -1
# State 1: action 0 (study) -> state 2, reward -2
P[1, 0, 2] = 1.0;  R[1, 0, 2] = -2
# State 1: action 1 (slack) -> state 0, reward -1
P[1, 1, 0] = 1.0;  R[1, 1, 0] = -1
# State 2 (terminal): absorbing
P[2, 0, 2] = 1.0;  R[2, 0, 2] = 10
P[2, 1, 2] = 1.0;  R[2, 1, 2] = 10

mdp = MDP(n_s, n_a, P, R, gamma=0.9)

# Random policy
pi_random = np.ones((n_s, n_a)) / n_a
V_random = mdp.compute_v_pi(pi_random)
Q_random = mdp.compute_q_pi(pi_random)

# Study-always policy
pi_study = np.zeros((n_s, n_a))
pi_study[:, 0] = 1.0
V_study = mdp.compute_v_pi(pi_study)

print("Student MDP (3 states: Class1, Class2, Passed)")
print(f"\\nV^pi (random policy): {np.round(V_random, 2)}")
print(f"V^pi (study policy):  {np.round(V_study, 2)}")
print(f"\\nQ^pi (random policy):\\n{np.round(Q_random, 2)}")

# Visualize
states = ['Class 1', 'Class 2', 'Passed']
x = np.arange(len(states))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, V_random, width, label='Random Policy', color='salmon')
ax.bar(x + width/2, V_study, width, label='Study Policy', color='steelblue')
ax.set_xlabel('State')
ax.set_ylabel('$V^\\pi(s)$')
ax.set_title('State-Value Function Under Different Policies')
ax.set_xticks(x)
ax.set_xticklabels(states)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""),
md("""## Summary

| Concept | Formula |
|---------|---------|
| **Markov Property** | $P[S_{t+1}|S_t] = P[S_{t+1}|S_1,\\ldots,S_t]$ |
| **State-Value** | $V^\\pi(s) = \\mathbb{E}_\\pi[G_t|S_t=s]$ |
| **Action-Value** | $Q^\\pi(s,a) = \\mathbb{E}_\\pi[G_t|S_t=s, A_t=a]$ |
| **V-Q Relationship** | $V^\\pi(s) = \\sum_a \\pi(a|s) Q^\\pi(s,a)$ |
| **Optimal Value** | $V^*(s) = \\max_\\pi V^\\pi(s)$ |

**Next:** Notebook 03 — Bellman Equations
"""),
])

# ============================================================
# NOTEBOOK 03: Bellman Equations
# ============================================================
save("03_Bellman_Equations", [
md("""# Notebook 03: Bellman Equations

**Series:** Reinforcement Learning Basics (3/20)

---

## 1. Why Bellman Equations?

Bellman equations are the **foundation of all RL algorithms**. They express a recursive relationship:

> *"The value of a state equals the immediate reward plus the discounted value of the next state."*

This recursive structure lets us:
- **Solve** for value functions (DP)
- **Estimate** value functions (MC, TD)
- **Optimize** policies (Q-learning, Policy Gradient)
"""),
md("""## 2. Bellman Expectation Equation for $V^\\pi$

### Derivation

Starting from the definition:
$$V^\\pi(s) = \\mathbb{E}_\\pi[G_t | S_t = s]$$

Expand $G_t$:
$$= \\mathbb{E}_\\pi[R_{t+1} + \\gamma G_{t+1} | S_t = s]$$

$$= \\mathbb{E}_\\pi[R_{t+1} | S_t = s] + \\gamma \\, \\mathbb{E}_\\pi[G_{t+1} | S_t = s]$$

Condition on action and next state:

$$\\boxed{V^\\pi(s) = \\sum_{a} \\pi(a|s) \\sum_{s'} P(s'|s,a) \\Big[ R(s,a,s') + \\gamma \\, V^\\pi(s') \\Big]}$$

**In words:** Value of $s$ = weighted average over actions of (expected immediate reward + discounted value of next state).

## 3. Bellman Expectation Equation for $Q^\\pi$

$$\\boxed{Q^\\pi(s,a) = \\sum_{s'} P(s'|s,a) \\Big[ R(s,a,s') + \\gamma \\sum_{a'} \\pi(a'|s') Q^\\pi(s',a') \\Big]}$$
"""),
md("""## 4. Bellman Optimality Equations

For the **optimal** value functions:

$$\\boxed{V^*(s) = \\max_a \\sum_{s'} P(s'|s,a) \\Big[ R(s,a,s') + \\gamma \\, V^*(s') \\Big]}$$

$$\\boxed{Q^*(s,a) = \\sum_{s'} P(s'|s,a) \\Big[ R(s,a,s') + \\gamma \\max_{a'} Q^*(s',a') \\Big]}$$

**Key difference:** Expectation equations average over $\\pi$; Optimality equations take the **max** over actions.

## 5. Matrix Form

For a fixed policy $\\pi$, the Bellman equation is **linear**:

$$\\mathbf{v}^\\pi = \\mathbf{r}^\\pi + \\gamma \\mathbf{P}^\\pi \\mathbf{v}^\\pi$$

where:
- $\\mathbf{v}^\\pi \\in \\mathbb{R}^{|\\mathcal{S}|}$ is the value vector
- $\\mathbf{P}^\\pi_{ss'} = \\sum_a \\pi(a|s) P(s'|s,a)$ is the transition matrix under $\\pi$
- $\\mathbf{r}^\\pi_s = \\sum_a \\pi(a|s) \\sum_{s'} P(s'|s,a) R(s,a,s')$

**Closed-form solution:**
$$\\mathbf{v}^\\pi = (\\mathbf{I} - \\gamma \\mathbf{P}^\\pi)^{-1} \\mathbf{r}^\\pi$$

**Complexity:** $O(|\\mathcal{S}|^3)$ for matrix inversion — only feasible for small MDPs.
"""),
md("""## 6. Contraction Mapping (Intuition)

The **Bellman operator** $T^\\pi$ defined by:
$$(T^\\pi V)(s) = \\sum_a \\pi(a|s) \\sum_{s'} P(s'|s,a)[R(s,a,s') + \\gamma V(s')]$$

is a **$\\gamma$-contraction** in the max-norm:
$$\\| T^\\pi V_1 - T^\\pi V_2 \\|_\\infty \\leq \\gamma \\| V_1 - V_2 \\|_\\infty$$

By the **Banach Fixed-Point Theorem**, $T^\\pi$ has a unique fixed point $V^\\pi$, and repeated application converges:
$$V_0 \\xrightarrow{T^\\pi} V_1 \\xrightarrow{T^\\pi} V_2 \\xrightarrow{T^\\pi} \\cdots \\to V^\\pi$$
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Solving Bellman Equations via Matrix Inversion
# ============================================

# 4x4 GridWorld MDP
size = 4
n_states = size * size
n_actions = 4  # up, right, down, left
gamma = 0.99  # Use 0.99 to avoid singular matrix with terminal absorbing states
terminal_states = [0, 15]

# Build transition model
P = np.zeros((n_states, n_actions, n_states))
R = np.zeros((n_states, n_actions, n_states))

moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

for s in range(n_states):
    if s in terminal_states:
        for a in range(n_actions):
            P[s, a, s] = 1.0
            R[s, a, s] = 0.0
        continue
    row, col = divmod(s, size)
    for a in range(n_actions):
        dr, dc = moves[a]
        new_row = max(0, min(size-1, row + dr))
        new_col = max(0, min(size-1, col + dc))
        s_next = new_row * size + new_col
        P[s, a, s_next] = 1.0
        R[s, a, s_next] = -1.0

# Uniform random policy
pi = np.ones((n_states, n_actions)) / n_actions

# Build P^pi and r^pi
P_pi = np.zeros((n_states, n_states))
r_pi = np.zeros(n_states)
for s in range(n_states):
    for a in range(n_actions):
        P_pi[s] += pi[s, a] * P[s, a]
        r_pi[s] += pi[s, a] * np.sum(P[s, a] * R[s, a])

# Solve: v = (I - gamma * P_pi)^{-1} * r_pi
V_exact = np.linalg.solve(np.eye(n_states) - gamma * P_pi, r_pi)

print("Exact V^pi (uniform random policy, gamma=1.0):")
print(V_exact.reshape(size, size).round(1))
print()

# Visualize
fig, ax = plt.subplots(figsize=(6, 6))
V_grid = V_exact.reshape(size, size)
im = ax.imshow(V_grid, cmap='RdYlGn')
for i in range(size):
    for j in range(size):
        ax.text(j, i, f'{V_grid[i,j]:.1f}', ha='center', va='center', fontsize=14, fontweight='bold')
ax.set_title('$V^\\pi$ for Random Policy (Bellman Exact Solution)', fontsize=14)
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(im)
plt.tight_layout()
plt.show()
"""),
md("""## Summary

| Equation | Formula |
|----------|---------|
| **Bellman Expectation (V)** | $V^\\pi(s) = \\sum_a \\pi(a|s) \\sum_{s'} P(s'|s,a)[R + \\gamma V^\\pi(s')]$ |
| **Bellman Expectation (Q)** | $Q^\\pi(s,a) = \\sum_{s'} P(s'|s,a)[R + \\gamma \\sum_{a'} \\pi(a'|s') Q^\\pi(s',a')]$ |
| **Bellman Optimality (V)** | $V^*(s) = \\max_a \\sum_{s'} P(s'|s,a)[R + \\gamma V^*(s')]$ |
| **Bellman Optimality (Q)** | $Q^*(s,a) = \\sum_{s'} P(s'|s,a)[R + \\gamma \\max_{a'} Q^*(s',a')]$ |
| **Matrix Form** | $\\mathbf{v}^\\pi = (I - \\gamma P^\\pi)^{-1} \\mathbf{r}^\\pi$ |

**Next:** Notebook 04 — Dynamic Programming
"""),
])

# ============================================================
# NOTEBOOK 04: Dynamic Programming
# ============================================================
save("04_Dynamic_Programming", [
md("""# Notebook 04: Dynamic Programming

**Series:** Reinforcement Learning Basics (4/20)

---

## 1. What is Dynamic Programming?

**Dynamic Programming (DP)** solves MDPs when we have a **perfect model** (know $P$ and $R$).

**Requirements:**
- Complete knowledge of the MDP
- Finite state and action spaces

**Key idea:** Use Bellman equations as **update rules** — iterate until convergence.
"""),
md("""## 2. Policy Evaluation (Prediction)

**Goal:** Compute $V^\\pi$ for a given policy $\\pi$.

**Iterative update:**
$$V_{k+1}(s) = \\sum_a \\pi(a|s) \\sum_{s'} P(s'|s,a) \\Big[ R(s,a,s') + \\gamma V_k(s') \\Big]$$

Start with arbitrary $V_0$, repeat until $\\|V_{k+1} - V_k\\|_\\infty < \\theta$.

**Why it converges:** The Bellman operator is a $\\gamma$-contraction. Each iteration reduces the error by at least a factor of $\\gamma$:
$$\\|V_{k+1} - V^\\pi\\|_\\infty \\leq \\gamma \\|V_k - V^\\pi\\|_\\infty$$

## 3. Policy Improvement

**Policy Improvement Theorem:** If for all $s$:
$$Q^\\pi(s, \\pi'(s)) \\geq V^\\pi(s)$$

then $\\pi'$ is at least as good as $\\pi$: $V^{\\pi'}(s) \\geq V^\\pi(s)$ for all $s$.

**Greedy improvement:**
$$\\pi'(s) = \\arg\\max_a Q^\\pi(s, a) = \\arg\\max_a \\sum_{s'} P(s'|s,a)[R(s,a,s') + \\gamma V^\\pi(s')]$$

## 4. Policy Iteration

1. **Initialize** $\\pi_0$ arbitrarily
2. **Evaluate:** Compute $V^{\\pi_k}$ (policy evaluation)
3. **Improve:** $\\pi_{k+1}(s) = \\arg\\max_a Q^{\\pi_k}(s,a)$ (greedy)
4. If $\\pi_{k+1} = \\pi_k$, stop. Else go to step 2.

**Guaranteed to converge** to $\\pi^*$ in finite MDPs (finite number of deterministic policies).

## 5. Value Iteration

Combine evaluation and improvement into **one step**:

$$V_{k+1}(s) = \\max_a \\sum_{s'} P(s'|s,a) \\Big[ R(s,a,s') + \\gamma V_k(s') \\Big]$$

This is the **Bellman optimality operator**. Converges to $V^*$.

Extract optimal policy at the end:
$$\\pi^*(s) = \\arg\\max_a \\sum_{s'} P(s'|s,a)[R(s,a,s') + \\gamma V^*(s')]$$

## 6. Generalized Policy Iteration (GPI)

Almost all RL methods follow the GPI pattern:
- **Evaluation:** Make value function more accurate for current policy
- **Improvement:** Make policy more greedy w.r.t. current value function

These two processes interact and converge to optimality.
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# ============================================
# GridWorld Setup
# ============================================
size = 4
n_states = size * size
n_actions = 4
gamma = 1.0
terminal_states = [0, 15]
moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

P = np.zeros((n_states, n_actions, n_states))
R = np.zeros((n_states, n_actions, n_states))

for s in range(n_states):
    if s in terminal_states:
        for a in range(n_actions):
            P[s, a, s] = 1.0
        continue
    row, col = divmod(s, size)
    for a in range(n_actions):
        dr, dc = moves[a]
        nr, nc = max(0, min(size-1, row+dr)), max(0, min(size-1, col+dc))
        s_next = nr * size + nc
        P[s, a, s_next] = 1.0
        R[s, a, s_next] = -1.0

# ============================================
# Policy Evaluation
# ============================================
def policy_evaluation(pi, P, R, gamma, theta=1e-6):
    V = np.zeros(n_states)
    history = [V.copy()]
    while True:
        V_new = np.zeros(n_states)
        for s in range(n_states):
            for a in range(n_actions):
                for s_next in range(n_states):
                    V_new[s] += pi[s, a] * P[s, a, s_next] * (R[s, a, s_next] + gamma * V[s_next])
        if np.max(np.abs(V_new - V)) < theta:
            break
        V = V_new
        history.append(V.copy())
    return V_new, history

# Evaluate random policy
pi_random = np.ones((n_states, n_actions)) / n_actions
V_pi, pe_history = policy_evaluation(pi_random, P, R, gamma)

print("Policy Evaluation — Random Policy:")
print(V_pi.reshape(size, size).round(1))
print(f"Converged in {len(pe_history)} iterations")
"""),
code("""# ============================================
# Policy Iteration
# ============================================
def policy_iteration(P, R, gamma, theta=1e-6):
    pi = np.ones((n_states, n_actions)) / n_actions
    iteration = 0
    while True:
        # Evaluate
        V, _ = policy_evaluation(pi, P, R, gamma, theta)
        # Improve
        policy_stable = True
        for s in range(n_states):
            old_action = np.argmax(pi[s])
            Q_s = np.zeros(n_actions)
            for a in range(n_actions):
                for s_next in range(n_states):
                    Q_s[a] += P[s, a, s_next] * (R[s, a, s_next] + gamma * V[s_next])
            best_a = np.argmax(Q_s)
            pi[s] = 0
            pi[s, best_a] = 1.0
            if old_action != best_a:
                policy_stable = False
        iteration += 1
        if policy_stable:
            break
    return pi, V, iteration

pi_star, V_star_pi, pi_iters = policy_iteration(P, R, gamma)
print(f"Policy Iteration converged in {pi_iters} iterations")
print("Optimal V*:")
print(V_star_pi.reshape(size, size).round(1))

# Show optimal policy
arrows = ['↑', '→', '↓', '←']
policy_grid = np.array([arrows[np.argmax(pi_star[s])] for s in range(n_states)]).reshape(size, size)
policy_grid[0, 0] = 'T'
policy_grid[size-1, size-1] = 'T'
print("\\nOptimal Policy:")
for row in policy_grid:
    print(' '.join(row))
"""),
code("""# ============================================
# Value Iteration
# ============================================
def value_iteration(P, R, gamma, theta=1e-6):
    V = np.zeros(n_states)
    history = [V.copy()]
    iteration = 0
    while True:
        V_new = np.zeros(n_states)
        for s in range(n_states):
            Q_s = np.zeros(n_actions)
            for a in range(n_actions):
                for s_next in range(n_states):
                    Q_s[a] += P[s, a, s_next] * (R[s, a, s_next] + gamma * V[s_next])
            V_new[s] = np.max(Q_s)
        iteration += 1
        history.append(V_new.copy())
        if np.max(np.abs(V_new - V)) < theta:
            break
        V = V_new

    # Extract policy
    pi = np.zeros((n_states, n_actions))
    for s in range(n_states):
        Q_s = np.zeros(n_actions)
        for a in range(n_actions):
            for s_next in range(n_states):
                Q_s[a] += P[s, a, s_next] * (R[s, a, s_next] + gamma * V_new[s_next])
        pi[s, np.argmax(Q_s)] = 1.0
    return pi, V_new, history

pi_vi, V_star_vi, vi_history = value_iteration(P, R, gamma)
print(f"Value Iteration converged in {len(vi_history)-1} iterations")
print("V* (Value Iteration):")
print(V_star_vi.reshape(size, size).round(1))

# Convergence plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Value Iteration convergence
errors = [np.max(np.abs(vi_history[i] - V_star_vi)) for i in range(len(vi_history))]
axes[0].semilogy(errors, 'b-o', markersize=4)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('$\\|V_k - V^*\\|_\\infty$')
axes[0].set_title('Value Iteration Convergence')
axes[0].grid(True, alpha=0.3)

# Heatmap of V*
im = axes[1].imshow(V_star_vi.reshape(size, size), cmap='RdYlGn')
for i in range(size):
    for j in range(size):
        axes[1].text(j, i, f'{V_star_vi[i*size+j]:.1f}', ha='center', va='center', fontsize=12, fontweight='bold')
axes[1].set_title('Optimal Value Function $V^*$')
axes[1].set_xticks([]); axes[1].set_yticks([])
plt.colorbar(im, ax=axes[1])
plt.tight_layout()
plt.show()
"""),
md("""## Summary

| Algorithm | Update | Complexity per Sweep |
|-----------|--------|---------------------|
| **Policy Evaluation** | $V_{k+1}(s) = \\sum_a \\pi(a|s) \\sum_{s'} P[R + \\gamma V_k(s')]$ | $O(|\\mathcal{S}|^2 |\\mathcal{A}|)$ |
| **Policy Iteration** | Evaluate $\\to$ Improve $\\to$ repeat | Converges in few iterations |
| **Value Iteration** | $V_{k+1}(s) = \\max_a \\sum_{s'} P[R + \\gamma V_k(s')]$ | $O(|\\mathcal{S}|^2 |\\mathcal{A}|)$ |

**Limitation:** DP requires a complete model. Next we learn **model-free** methods.

**Next:** Notebook 05 — Monte Carlo Methods
"""),
])

# ============================================================
# NOTEBOOK 05: Monte Carlo Methods
# ============================================================
save("05_Monte_Carlo_Methods", [
md("""# Notebook 05: Monte Carlo Methods

**Series:** Reinforcement Learning Basics (5/20)

---

## 1. Why Monte Carlo?

**Key insight:** We don't need a model! Learn directly from **episodes of experience**.

MC methods estimate value functions by **averaging sample returns**.

| Property | DP | MC |
|----------|----|----|
| Model needed? | Yes | **No** |
| Bootstrapping? | Yes | **No** |
| Works online? | Yes | **No** (need complete episodes) |
| Bias | None | **None** (unbiased) |
| Variance | Low | **High** |
"""),
md("""## 2. MC Prediction

**Goal:** Estimate $V^\\pi(s) = \\mathbb{E}_\\pi[G_t | S_t = s]$

### First-Visit MC
For each episode:
1. Generate episode: $S_0, A_0, R_1, S_1, A_1, R_2, \\ldots, S_T$
2. For each state $s$ appearing in the episode (first visit only):
   - Compute return $G_t = \\sum_{k=0}^{T-t-1} \\gamma^k R_{t+k+1}$
   - Append $G_t$ to $\\text{Returns}(s)$
3. $V(s) = \\text{average}(\\text{Returns}(s))$

### Every-Visit MC
Same, but count **every** visit to $s$, not just the first.

### Incremental Update (more practical)
$$V(S_t) \\leftarrow V(S_t) + \\alpha \\Big[ G_t - V(S_t) \\Big]$$

**Why it works:** By the **Law of Large Numbers**, the sample average converges to the expected value.

## 3. MC Control (On-Policy)

Use **$\\varepsilon$-greedy** policy for exploration:

$$\\pi(a|s) = \\begin{cases} 1 - \\varepsilon + \\frac{\\varepsilon}{|\\mathcal{A}|} & \\text{if } a = \\arg\\max_a Q(s,a) \\\\ \\frac{\\varepsilon}{|\\mathcal{A}|} & \\text{otherwise} \\end{cases}$$

## 4. Off-Policy MC: Importance Sampling

Learn about target policy $\\pi$ from data generated by behavior policy $b$.

**Importance Sampling Ratio:**
$$\\rho_{t:T-1} = \\prod_{k=t}^{T-1} \\frac{\\pi(A_k|S_k)}{b(A_k|S_k)}$$

**Ordinary IS:** $V(s) = \\frac{\\sum_{t \\in \\mathcal{T}(s)} \\rho_{t:T(t)-1} G_t}{|\\mathcal{T}(s)|}$ — unbiased but high variance

**Weighted IS:** $V(s) = \\frac{\\sum_{t \\in \\mathcal{T}(s)} \\rho_{t:T(t)-1} G_t}{\\sum_{t \\in \\mathcal{T}(s)} \\rho_{t:T(t)-1}}$ — biased but much lower variance
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Blackjack Environment
# ============================================

def draw_card():
    card = min(np.random.randint(1, 14), 10)
    return card

def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21

def hand_value(hand):
    val = sum(hand)
    if usable_ace(hand):
        val += 10
    return val

def play_blackjack(policy, initial_state=None):
    \"\"\"Play one episode of blackjack. Returns list of (state, action, reward).\"\"\"
    # Deal
    player = [draw_card(), draw_card()]
    dealer_showing = draw_card()
    dealer_hidden = draw_card()

    trajectory = []

    # Player's turn
    while True:
        p_sum = hand_value(player)
        if p_sum > 21:
            trajectory.append(((p_sum, dealer_showing, usable_ace(player)), None, -1))
            return trajectory
        if p_sum < 12:
            player.append(draw_card())
            continue
        state = (p_sum, dealer_showing, usable_ace(player))
        action = policy(state)  # 0 = stick, 1 = hit
        trajectory.append((state, action, None))
        if action == 0:
            break
        player.append(draw_card())

    # Dealer's turn (hits until >= 17)
    dealer = [dealer_showing, dealer_hidden]
    while hand_value(dealer) < 17:
        dealer.append(draw_card())

    p_val = hand_value(player)
    d_val = hand_value(dealer)

    if d_val > 21 or p_val > d_val:
        reward = 1
    elif p_val < d_val:
        reward = -1
    else:
        reward = 0

    # Update last entry with reward
    trajectory[-1] = (trajectory[-1][0], trajectory[-1][1], reward)
    return trajectory

# ============================================
# First-Visit MC Prediction
# ============================================

def mc_prediction(policy, n_episodes=500000, gamma=1.0):
    V = {}
    returns_count = {}
    returns_sum = {}

    for _ in range(n_episodes):
        episode = play_blackjack(policy)
        states_in_episode = set()
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            if reward is not None:
                G = reward + gamma * G
            if state not in states_in_episode and state[0] <= 21:
                states_in_episode.add(state)
                returns_sum[state] = returns_sum.get(state, 0) + G
                returns_count[state] = returns_count.get(state, 0) + 1
                V[state] = returns_sum[state] / returns_count[state]
    return V

# Policy: stick on 20 or 21
def simple_policy(state):
    return 0 if state[0] >= 20 else 1

V = mc_prediction(simple_policy, n_episodes=500000)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, has_ace in enumerate([False, True]):
    grid = np.zeros((10, 10))
    for p_sum in range(12, 22):
        for d_show in range(1, 11):
            state = (p_sum, d_show, has_ace)
            grid[p_sum - 12, d_show - 1] = V.get(state, 0)
    im = axes[idx].imshow(grid, cmap='RdYlGn', origin='lower', aspect='auto')
    axes[idx].set_xlabel('Dealer Showing')
    axes[idx].set_ylabel('Player Sum')
    axes[idx].set_xticks(range(10))
    axes[idx].set_xticklabels(range(1, 11))
    axes[idx].set_yticks(range(10))
    axes[idx].set_yticklabels(range(12, 22))
    axes[idx].set_title(f'V(s) — {"Usable" if has_ace else "No Usable"} Ace')
    plt.colorbar(im, ax=axes[idx])
plt.suptitle('MC Prediction: V^π for Stick-on-20 Policy (500K episodes)', fontsize=13)
plt.tight_layout()
plt.show()
"""),
md("""## Summary

| Method | Formula | Properties |
|--------|---------|------------|
| **First-Visit MC** | $V(s) = \\text{avg}(G_t)$ for first visits | Unbiased, high variance |
| **Incremental MC** | $V(s) \\leftarrow V(s) + \\alpha[G_t - V(s)]$ | Online-ready |
| **$\\varepsilon$-greedy MC** | Explore with probability $\\varepsilon$ | On-policy control |
| **Importance Sampling** | Weight by $\\rho = \\prod \\pi/b$ | Off-policy |

**Next:** Notebook 06 — Temporal Difference Learning
"""),
])

# ============================================================
# NOTEBOOK 06: Temporal Difference Learning
# ============================================================
save("06_Temporal_Difference_Learning", [
md("""# Notebook 06: Temporal Difference Learning

**Series:** Reinforcement Learning Basics (6/20)

---

## 1. TD Learning: The Best of Both Worlds

TD combines **MC** (learn from experience, no model) with **DP** (bootstrap, update before episode ends).

### TD(0) Update Rule

$$\\boxed{V(S_t) \\leftarrow V(S_t) + \\alpha \\Big[ \\underbrace{R_{t+1} + \\gamma V(S_{t+1})}_{\\text{TD Target}} - V(S_t) \\Big]}$$

| Term | Name | Meaning |
|------|------|---------|
| $R_{t+1} + \\gamma V(S_{t+1})$ | **TD Target** | Estimated return |
| $\\delta_t = R_{t+1} + \\gamma V(S_{t+1}) - V(S_t)$ | **TD Error** | How wrong our estimate was |
| $\\alpha$ | **Learning rate** | Step size |

### Why TD?
- Can learn **online** (after every step, not just episode end)
- Can learn from **incomplete** sequences
- Works for **continuing** tasks (no episodes needed)
- **Lower variance** than MC (bootstrapping reduces variance)
"""),
md("""## 2. Bias-Variance Tradeoff

| Method | Target | Bias | Variance |
|--------|--------|------|----------|
| **MC** | $G_t$ (actual return) | **Zero** bias | **High** variance |
| **TD(0)** | $R_{t+1} + \\gamma V(S_{t+1})$ | **Some** bias (depends on $V$) | **Lower** variance |

**Why MC has high variance:** $G_t$ depends on many random actions, transitions, and rewards over the entire episode.

**Why TD has lower variance:** Only depends on one random action, transition, and reward (one step).

**Why TD has bias:** We use our current estimate $V(S_{t+1})$ instead of the true value — if $V$ is wrong, the target is biased.

## 3. Batch TD vs Batch MC

Given a finite batch of episodes, TD and MC converge to different answers:

- **Batch MC:** Minimizes mean-squared error on training data
- **Batch TD:** Finds the maximum likelihood MDP model, then solves it (**certainty equivalence**)

TD exploits the Markov property and is usually more data-efficient.
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Random Walk Environment (5 states)
# ============================================
# States: 0(terminal-left), 1, 2, 3, 4, 5, 6(terminal-right)
# Start in state 3 (center)
# Actions: left (-1) or right (+1) with equal probability
# Reward: +1 at right terminal, 0 everywhere else

TRUE_VALUES = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 0])  # including terminals

def td0_prediction(n_episodes=100, alpha=0.1, gamma=1.0):
    V = np.full(7, 0.5)
    V[0] = 0; V[6] = 0  # terminal states
    history = [V[1:6].copy()]

    for _ in range(n_episodes):
        state = 3  # start in center
        while state not in [0, 6]:
            action = np.random.choice([-1, 1])
            next_state = state + action
            reward = 1.0 if next_state == 6 else 0.0
            # TD(0) update
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state
        history.append(V[1:6].copy())
    return V, history

def mc_prediction(n_episodes=100, alpha=0.1, gamma=1.0):
    V = np.full(7, 0.5)
    V[0] = 0; V[6] = 0
    history = [V[1:6].copy()]

    for _ in range(n_episodes):
        state = 3
        trajectory = [state]
        while state not in [0, 6]:
            action = np.random.choice([-1, 1])
            state = state + action
            trajectory.append(state)

        # Return: 1 if ended at right terminal, 0 otherwise
        G = 1.0 if trajectory[-1] == 6 else 0.0

        # Update all visited states
        for s in trajectory[:-1]:
            V[s] += alpha * (G - V[s])
        history.append(V[1:6].copy())
    return V, history

# Run both
np.random.seed(42)
V_td, td_hist = td0_prediction(n_episodes=100, alpha=0.1)
np.random.seed(42)
V_mc, mc_hist = mc_prediction(n_episodes=100, alpha=0.1)

# Plot learned values at different episodes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
episodes_to_show = [0, 1, 10, 50, 100]
colors = plt.cm.viridis(np.linspace(0, 1, len(episodes_to_show)))

for ax, hist, title in [(axes[0], td_hist, 'TD(0)'), (axes[1], mc_hist, 'MC')]:
    for i, ep in enumerate(episodes_to_show):
        if ep < len(hist):
            ax.plot(range(1, 6), hist[ep], 'o-', color=colors[i], label=f'Episode {ep}')
    ax.plot(range(1, 6), TRUE_VALUES[1:6], 'k--', linewidth=2, label='True Values')
    ax.set_xlabel('State')
    ax.set_ylabel('Estimated Value')
    ax.set_title(f'{title} Prediction (α=0.1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(['A', 'B', 'C', 'D', 'E'])

plt.tight_layout()
plt.show()

# RMSE comparison
n_runs = 100
n_episodes = 100
rmse_td = np.zeros(n_episodes + 1)
rmse_mc = np.zeros(n_episodes + 1)

for run in range(n_runs):
    _, td_h = td0_prediction(n_episodes, alpha=0.1)
    _, mc_h = mc_prediction(n_episodes, alpha=0.01)
    for ep in range(min(len(td_h), n_episodes + 1)):
        rmse_td[ep] += np.sqrt(np.mean((td_h[ep] - TRUE_VALUES[1:6])**2))
    for ep in range(min(len(mc_h), n_episodes + 1)):
        rmse_mc[ep] += np.sqrt(np.mean((mc_h[ep] - TRUE_VALUES[1:6])**2))

rmse_td /= n_runs
rmse_mc /= n_runs

plt.figure(figsize=(10, 5))
plt.plot(rmse_td, label='TD(0) α=0.1', linewidth=2)
plt.plot(rmse_mc, label='MC α=0.01', linewidth=2)
plt.xlabel('Episodes')
plt.ylabel('RMSE (averaged over 100 runs)')
plt.title('TD(0) vs MC: Learning Curves on Random Walk')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""),
md("""## Summary

| Property | MC | TD(0) |
|----------|----| ------|
| **Target** | $G_t$ (actual return) | $R_{t+1} + \\gamma V(S_{t+1})$ |
| **Bias** | None | Some (bootstrapping) |
| **Variance** | High | Lower |
| **Model needed** | No | No |
| **Complete episodes** | Yes | **No** |
| **Bootstrapping** | No | **Yes** |

**Next:** Notebook 07 — SARSA
"""),
])

# ============================================================
# NOTEBOOK 07: SARSA
# ============================================================
save("07_SARSA_On_Policy_TD_Control", [
md("""# Notebook 07: SARSA — On-Policy TD Control

**Series:** Reinforcement Learning Basics (7/20)

---

## 1. From Prediction to Control

TD(0) estimates $V^\\pi$. For **control**, we need $Q(s,a)$ to improve the policy.

## 2. SARSA Update

$$\\boxed{Q(S_t, A_t) \\leftarrow Q(S_t, A_t) + \\alpha \\Big[ R_{t+1} + \\gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \\Big]}$$

**Name origin:** Uses the quintuple $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$

**On-Policy:** The next action $A_{t+1}$ is chosen by the **same** $\\varepsilon$-greedy policy we're improving.

## 3. Expected SARSA

$$Q(S_t, A_t) \\leftarrow Q(S_t, A_t) + \\alpha \\Big[ R_{t+1} + \\gamma \\sum_a \\pi(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t) \\Big]$$

Uses the **expected value** over next actions instead of a sample. Lower variance than SARSA.

## 4. SARSA is "Safe"

Because SARSA is on-policy, it accounts for the possibility of future exploration (random actions). This means it learns to **avoid dangerous states** even if the optimal path goes near them — it knows it might accidentally step into them.
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Cliff Walking Environment
# ============================================
class CliffWalking:
    def __init__(self, height=4, width=12):
        self.height = height
        self.width = width
        self.start = (height-1, 0)
        self.goal = (height-1, width-1)
        self.cliff = [(height-1, c) for c in range(1, width-1)]
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        r, c = self.state
        if action == 0: r = max(0, r-1)       # up
        elif action == 1: c = min(self.width-1, c+1)  # right
        elif action == 2: r = min(self.height-1, r+1)  # down
        elif action == 3: c = max(0, c-1)       # left

        self.state = (r, c)
        if self.state in self.cliff:
            self.state = self.start
            return self.state, -100, False
        elif self.state == self.goal:
            return self.state, -1, True
        else:
            return self.state, -1, False

def epsilon_greedy(Q, state, epsilon, n_actions=4):
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])

# ============================================
# SARSA
# ============================================
def sarsa(env, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    Q = {}
    rewards_per_episode = []
    for ep in range(n_episodes):
        state = env.reset()
        if state not in Q: Q[state] = np.zeros(4)
        action = epsilon_greedy(Q, state, epsilon)
        total_reward = 0
        for _ in range(10000):
            next_state, reward, done = env.step(action)
            if next_state not in Q: Q[next_state] = np.zeros(4)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state, action = next_state, next_action
            total_reward += reward
            if done: break
        rewards_per_episode.append(total_reward)
    return Q, rewards_per_episode

# ============================================
# Q-Learning (for comparison)
# ============================================
def q_learning(env, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    Q = {}
    rewards_per_episode = []
    for ep in range(n_episodes):
        state = env.reset()
        if state not in Q: Q[state] = np.zeros(4)
        total_reward = 0
        for _ in range(10000):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done = env.step(action)
            if next_state not in Q: Q[next_state] = np.zeros(4)
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            state = next_state
            total_reward += reward
            if done: break
        rewards_per_episode.append(total_reward)
    return Q, rewards_per_episode

env = CliffWalking()

# Average over multiple runs
n_runs = 50
sarsa_rewards = np.zeros(500)
ql_rewards = np.zeros(500)

for _ in range(n_runs):
    _, sr = sarsa(CliffWalking(), 500)
    _, qr = q_learning(CliffWalking(), 500)
    sarsa_rewards += np.array(sr)
    ql_rewards += np.array(qr)

sarsa_rewards /= n_runs
ql_rewards /= n_runs

# Smooth
def smooth(x, w=10):
    return np.convolve(x, np.ones(w)/w, mode='valid')

plt.figure(figsize=(12, 5))
plt.plot(smooth(sarsa_rewards, 20), label='SARSA', linewidth=2)
plt.plot(smooth(ql_rewards, 20), label='Q-Learning', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Sum of Rewards (smoothed)')
plt.title('Cliff Walking: SARSA vs Q-Learning')
plt.legend()
plt.ylim(-100, 0)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Show learned paths
Q_s, _ = sarsa(CliffWalking(), 500)
Q_q, _ = q_learning(CliffWalking(), 500)

def extract_path(Q, env):
    state = env.reset()
    path = [state]
    for _ in range(50):
        if state not in Q:
            break
        action = np.argmax(Q[state])
        state, _, done = env.step(action)
        path.append(state)
        if done: break
    return path

path_sarsa = extract_path(Q_s, CliffWalking())
path_ql = extract_path(Q_q, CliffWalking())

print("SARSA path (safe):", path_sarsa[:15], "...")
print("Q-Learning path (optimal):", path_ql[:15], "...")
"""),
md("""## Summary

| Algorithm | Update | On/Off-Policy |
|-----------|--------|--------------|
| **SARSA** | $Q(s,a) \\leftarrow Q(s,a) + \\alpha[R + \\gamma Q(s',a') - Q(s,a)]$ | On-Policy |
| **Expected SARSA** | $Q(s,a) \\leftarrow Q(s,a) + \\alpha[R + \\gamma \\sum_a \\pi(a|s')Q(s',a) - Q(s,a)]$ | On-Policy |

**Key insight:** SARSA finds the **safe** path; Q-Learning finds the **optimal** path.

**Next:** Notebook 08 — Q-Learning
"""),
])

# ============================================================
# NOTEBOOK 08: Q-Learning
# ============================================================
save("08_Q_Learning_Off_Policy_TD_Control", [
md("""# Notebook 08: Q-Learning — Off-Policy TD Control

**Series:** Reinforcement Learning Basics (8/20)

---

## 1. Q-Learning Update

$$\\boxed{Q(S_t, A_t) \\leftarrow Q(S_t, A_t) + \\alpha \\Big[ R_{t+1} + \\gamma \\max_a Q(S_{t+1}, a) - Q(S_t, A_t) \\Big]}$$

**Off-Policy:** Uses $\\max_a Q(S_{t+1}, a)$ regardless of what action is actually taken. Learns $Q^*$ while following any exploration policy.

## 2. Why Off-Policy is Powerful

- Learn from **demonstrations** (watching an expert)
- **Experience replay** (reuse old data)
- Learn about **multiple policies** simultaneously
- Learn **optimal** policy while exploring

## 3. Maximization Bias

$$\\mathbb{E}[\\max_a Q(s,a)] \\geq \\max_a \\mathbb{E}[Q(s,a)]$$

Q-Learning can **overestimate** values because it uses the max operator on noisy estimates.

## 4. Double Q-Learning

Use **two** Q-tables to decouple selection and evaluation:

1. With probability 0.5, update $Q_1$:
$$Q_1(S_t, A_t) \\leftarrow Q_1(S_t, A_t) + \\alpha[R_{t+1} + \\gamma Q_2(S_{t+1}, \\arg\\max_a Q_1(S_{t+1}, a)) - Q_1(S_t, A_t)]$$

2. Otherwise, update $Q_2$ analogously (swap roles).
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Maximization Bias Example
# ============================================
# State A: left or right
# Left -> State B, reward 0
# From B: many actions, each gives N(-0.1, 1) reward, then terminal
# Right -> terminal, reward 0
# Optimal: go right (expected reward 0 > -0.1)
# But Q-learning initially overestimates left due to max bias

def run_experiment(n_episodes=300, n_runs=1000, double=False):
    left_counts = np.zeros(n_episodes)

    for run in range(n_runs):
        Q1_A = np.array([0.0, 0.0])  # [left, right]
        Q1_B = np.zeros(10)  # 10 actions from state B
        if double:
            Q2_A = np.array([0.0, 0.0])
            Q2_B = np.zeros(10)

        alpha = 0.1
        epsilon = 0.1

        for ep in range(n_episodes):
            # State A: choose action
            if np.random.random() < epsilon:
                action_A = np.random.randint(2)
            else:
                if double:
                    action_A = np.argmax(Q1_A + Q2_A)
                else:
                    action_A = np.argmax(Q1_A)

            if action_A == 0:  # left
                left_counts[ep] += 1

            if action_A == 1:  # right -> terminal, reward 0
                if double:
                    if np.random.random() < 0.5:
                        Q1_A[1] += alpha * (0 - Q1_A[1])
                    else:
                        Q2_A[1] += alpha * (0 - Q2_A[1])
                else:
                    Q1_A[1] += alpha * (0 - Q1_A[1])
            else:  # left -> state B
                # Choose action in B
                if np.random.random() < epsilon:
                    action_B = np.random.randint(10)
                else:
                    if double:
                        action_B = np.argmax(Q1_B + Q2_B)
                    else:
                        action_B = np.argmax(Q1_B)

                reward_B = np.random.normal(-0.1, 1.0)

                if double:
                    if np.random.random() < 0.5:
                        Q1_B[action_B] += alpha * (reward_B - Q1_B[action_B])
                        best_B = np.argmax(Q1_B)
                        Q1_A[0] += alpha * (0 + Q2_B[best_B] - Q1_A[0])
                    else:
                        Q2_B[action_B] += alpha * (reward_B - Q2_B[action_B])
                        best_B = np.argmax(Q2_B)
                        Q2_A[0] += alpha * (0 + Q1_B[best_B] - Q2_A[0])
                else:
                    Q1_B[action_B] += alpha * (reward_B - Q1_B[action_B])
                    Q1_A[0] += alpha * (0 + np.max(Q1_B) - Q1_A[0])

    return left_counts / n_runs * 100  # percentage

pct_left_ql = run_experiment(double=False)
pct_left_dql = run_experiment(double=True)

plt.figure(figsize=(10, 5))
plt.plot(pct_left_ql, label='Q-Learning', linewidth=2, color='red')
plt.plot(pct_left_dql, label='Double Q-Learning', linewidth=2, color='blue')
plt.axhline(5, color='black', linestyle='--', label='Optimal (≈5% with ε=0.1)')
plt.xlabel('Episodes')
plt.ylabel('% Left Actions from State A')
plt.title('Maximization Bias: Q-Learning vs Double Q-Learning')
plt.legend()
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("Q-Learning overestimates 'left' because max over noisy Q-values is biased upward.")
print("Double Q-Learning uses separate tables to break this correlation.")
"""),
md("""## Summary

| Algorithm | Update | Key Property |
|-----------|--------|-------------|
| **Q-Learning** | $Q(s,a) \\leftarrow Q(s,a) + \\alpha[R + \\gamma \\max_{a'} Q(s',a') - Q(s,a)]$ | Off-policy, learns $Q^*$ |
| **Double Q-Learning** | Decouple selection ($Q_1$) and evaluation ($Q_2$) | Reduces maximization bias |

**Next:** Notebook 09 — N-Step Bootstrapping
"""),
])

# ============================================================
# NOTEBOOKS 09-20: Generate remaining notebooks
# ============================================================

save("09_N_Step_Bootstrapping", [
md("""# Notebook 09: N-Step Bootstrapping & Eligibility Traces

**Series:** Reinforcement Learning Basics (9/20)

---

## 1. Unifying MC and TD

| Method | Target | $n$ |
|--------|--------|-----|
| TD(0) | $R_{t+1} + \\gamma V(S_{t+1})$ | $n = 1$ |
| 2-step TD | $R_{t+1} + \\gamma R_{t+2} + \\gamma^2 V(S_{t+2})$ | $n = 2$ |
| MC | $R_{t+1} + \\gamma R_{t+2} + \\cdots + \\gamma^{T-t-1} R_T$ | $n = \\infty$ |

## 2. N-Step Return

$$G_t^{(n)} = R_{t+1} + \\gamma R_{t+2} + \\cdots + \\gamma^{n-1} R_{t+n} + \\gamma^n V(S_{t+n})$$

$$= \\sum_{k=0}^{n-1} \\gamma^k R_{t+k+1} + \\gamma^n V(S_{t+n})$$

**Update:** $V(S_t) \\leftarrow V(S_t) + \\alpha [G_t^{(n)} - V(S_t)]$

## 3. TD(λ) — The λ-Return

The **$\\lambda$-return** is a weighted average of all n-step returns:

$$G_t^\\lambda = (1 - \\lambda) \\sum_{n=1}^{\\infty} \\lambda^{n-1} G_t^{(n)}$$

- $\\lambda = 0$: TD(0)
- $\\lambda = 1$: MC
- $0 < \\lambda < 1$: smooth interpolation

## 4. Eligibility Traces (Backward View)

Instead of looking forward (λ-return), we maintain a **trace** for each state:

$$e_t(s) = \\gamma \\lambda \\, e_{t-1}(s) + \\mathbb{1}(S_t = s)$$

**Update all states simultaneously:**
$$V(s) \\leftarrow V(s) + \\alpha \\, \\delta_t \\, e_t(s) \\quad \\forall s$$

where $\\delta_t = R_{t+1} + \\gamma V(S_{t+1}) - V(S_t)$

**Key insight:** The eligibility trace answers "which states are responsible for the current TD error?"
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# Random Walk (19 states + 2 terminals)
n_states = 19
TRUE_VALUES = np.arange(-18, 20, 2) / 20.0  # true values for 19 states

def random_walk_episode():
    state = 9  # center (0-indexed, states 0-18)
    trajectory = [(state, 0)]
    while True:
        step = np.random.choice([-1, 1])
        next_state = state + step
        if next_state < 0:  # left terminal
            trajectory.append((state, -1))
            return trajectory, -1
        elif next_state >= n_states:  # right terminal
            trajectory.append((state, 1))
            return trajectory, 1
        else:
            trajectory.append((next_state, 0))
            state = next_state

def n_step_td(n, n_episodes=10, alpha=0.1, gamma=1.0):
    V = np.zeros(n_states)
    errors = []
    for _ in range(n_episodes):
        traj, final_reward = random_walk_episode()
        states = [s for s, _ in traj[:-1]]
        T = len(states)
        rewards = [0] + [r for _, r in traj[1:]]  # rewards[1] = R_1, etc.

        for t in range(T):
            end = min(t + n, T)
            G = sum(gamma**(k-t-1) * rewards[k] for k in range(t+1, end+1))
            if end < T:
                G += gamma**n * V[states[end]]
            elif final_reward != 0:
                G = sum(gamma**(k-t-1) * rewards[k] for k in range(t+1, len(rewards)))
            V[states[t]] += alpha * (G - V[states[t]])
        errors.append(np.sqrt(np.mean((V - TRUE_VALUES)**2)))
    return V, errors

# Compare different n values
fig, ax = plt.subplots(figsize=(10, 6))
for n in [1, 2, 4, 8, 16, 128]:
    rmses = []
    for _ in range(100):
        _, errs = n_step_td(n, n_episodes=10, alpha=0.2)
        rmses.append(errs[-1])
    ax.bar(str(n), np.mean(rmses), yerr=np.std(rmses)/10, capsize=5, alpha=0.7)
ax.set_xlabel('n (number of steps)')
ax.set_ylabel('RMSE after 10 episodes')
ax.set_title('N-Step TD: Effect of n on Random Walk (19 states)')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
"""),
code("""# ============================================
# TD(λ) with Eligibility Traces
# ============================================
def td_lambda(lam, n_episodes=10, alpha=0.05, gamma=1.0):
    V = np.zeros(n_states)
    for _ in range(n_episodes):
        state = 9
        e = np.zeros(n_states)  # eligibility traces
        while True:
            step = np.random.choice([-1, 1])
            next_state = state + step
            if next_state < 0:
                reward = -1; delta = reward - V[state]
                e[state] += 1
                V += alpha * delta * e
                break
            elif next_state >= n_states:
                reward = 1; delta = reward - V[state]
                e[state] += 1
                V += alpha * delta * e
                break
            else:
                reward = 0
                delta = reward + gamma * V[next_state] - V[state]
                e[state] += 1
                V += alpha * delta * e
                e *= gamma * lam
                state = next_state
    return V

# Compare different lambda values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
lambdas = [0.0, 0.4, 0.8, 0.95, 1.0]

for lam in lambdas:
    rmse_list = []
    for _ in range(200):
        V = td_lambda(lam, n_episodes=10, alpha=0.05)
        rmse_list.append(np.sqrt(np.mean((V - TRUE_VALUES)**2)))
    axes[0].bar(str(lam), np.mean(rmse_list), alpha=0.7)

axes[0].set_xlabel('λ')
axes[0].set_ylabel('RMSE')
axes[0].set_title('TD(λ) Performance vs λ')
axes[0].grid(True, alpha=0.3, axis='y')

# Show eligibility trace decay
steps = np.arange(20)
for lam in [0.0, 0.5, 0.8, 0.95]:
    trace = (0.9 * lam) ** steps  # gamma * lambda decay
    axes[1].plot(steps, trace, 'o-', label=f'λ={lam}', markersize=4)
axes[1].set_xlabel('Steps Since Visit')
axes[1].set_ylabel('Eligibility Trace Value')
axes[1].set_title('Eligibility Trace Decay (γ=0.9)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""),
md("""## Summary

| Method | Formula | Bias-Variance |
|--------|---------|--------------|
| **n-step TD** | $G_t^{(n)} = \\sum_{k=0}^{n-1} \\gamma^k R_{t+k+1} + \\gamma^n V(S_{t+n})$ | Tunable via $n$ |
| **TD(λ)** | $G_t^\\lambda = (1-\\lambda)\\sum_{n=1}^\\infty \\lambda^{n-1} G_t^{(n)}$ | Tunable via $\\lambda$ |
| **Eligibility Traces** | $e_t(s) = \\gamma\\lambda e_{t-1}(s) + \\mathbb{1}(S_t=s)$ | Efficient backward view |

**Next:** Notebook 10 — Function Approximation
"""),
])

save("10_Function_Approximation", [
md("""# Notebook 10: Function Approximation in RL

**Series:** Reinforcement Learning Basics (10/20)

---

## 1. Why Function Approximation?

Tabular methods need a separate entry for each state. Real-world problems have:
- **Continuous** state spaces (robot joint angles)
- **Huge** discrete spaces (Go: $3^{361}$ states)

**Solution:** Approximate $V(s) \\approx \\hat{V}(s, \\mathbf{w})$ with a parameterized function.

## 2. Linear Function Approximation

$$\\hat{V}(s, \\mathbf{w}) = \\mathbf{w}^\\top \\mathbf{x}(s) = \\sum_{j=1}^d w_j x_j(s)$$

where $\\mathbf{x}(s) = [x_1(s), \\ldots, x_d(s)]^\\top$ is a **feature vector**.

### Objective: Mean Squared Value Error
$$\\overline{VE}(\\mathbf{w}) = \\sum_s \\mu(s) \\Big[ V^\\pi(s) - \\hat{V}(s, \\mathbf{w}) \\Big]^2$$

### SGD Update
$$\\mathbf{w} \\leftarrow \\mathbf{w} + \\alpha \\Big[ V^\\pi(S_t) - \\hat{V}(S_t, \\mathbf{w}) \\Big] \\nabla_\\mathbf{w} \\hat{V}(S_t, \\mathbf{w})$$

For linear: $\\nabla_\\mathbf{w} \\hat{V}(S_t, \\mathbf{w}) = \\mathbf{x}(S_t)$

### Semi-Gradient TD(0)
$$\\mathbf{w} \\leftarrow \\mathbf{w} + \\alpha \\Big[ R_{t+1} + \\gamma \\hat{V}(S_{t+1}, \\mathbf{w}) - \\hat{V}(S_t, \\mathbf{w}) \\Big] \\nabla_\\mathbf{w} \\hat{V}(S_t, \\mathbf{w})$$

**"Semi-gradient"** because we don't differentiate through the target $\\hat{V}(S_{t+1}, \\mathbf{w})$.

## 3. Feature Construction

| Method | Features | Properties |
|--------|----------|------------|
| **Polynomial** | $1, s, s^2, \\ldots$ | Simple, global |
| **Fourier** | $\\cos(k\\pi s)$ | Smooth, global |
| **Tile Coding** | Binary, overlapping tiles | Local, efficient |
| **RBF** | $\\exp(-\\|s - c_j\\|^2 / 2\\sigma^2)$ | Local, smooth |

## 4. Convergence

For linear function approximation with semi-gradient TD(0):
$$\\overline{VE}(\\mathbf{w}_{TD}) \\leq \\frac{1}{1 - \\gamma} \\min_\\mathbf{w} \\overline{VE}(\\mathbf{w})$$

The TD solution is within a factor of $\\frac{1}{1-\\gamma}$ of the best possible linear approximation.
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Tile Coding Implementation
# ============================================
class TileCoding:
    def __init__(self, n_tilings=8, tiles_per_dim=8, state_bounds=None):
        self.n_tilings = n_tilings
        self.tiles_per_dim = tiles_per_dim
        self.state_bounds = state_bounds  # list of (low, high) for each dim
        self.n_dims = len(state_bounds)
        self.n_tiles = n_tilings * (tiles_per_dim ** self.n_dims)

    def get_features(self, state):
        features = np.zeros(self.n_tiles)
        for tiling in range(self.n_tilings):
            indices = []
            for dim in range(self.n_dims):
                low, high = self.state_bounds[dim]
                width = (high - low) / self.tiles_per_dim
                offset = tiling * width / self.n_tilings
                idx = int((state[dim] - low + offset) / width)
                idx = max(0, min(self.tiles_per_dim - 1, idx))
                indices.append(idx)
            # Flatten multi-dim index
            flat_idx = tiling * (self.tiles_per_dim ** self.n_dims)
            for d, idx in enumerate(indices):
                flat_idx += idx * (self.tiles_per_dim ** (self.n_dims - 1 - d))
            features[flat_idx] = 1.0
        return features

# ============================================
# Mountain Car with Semi-Gradient SARSA
# ============================================
class MountainCar:
    def __init__(self):
        self.pos_bounds = (-1.2, 0.5)
        self.vel_bounds = (-0.07, 0.07)

    def reset(self):
        self.pos = np.random.uniform(-0.6, -0.4)
        self.vel = 0.0
        return (self.pos, self.vel)

    def step(self, action):
        # action: 0=reverse, 1=neutral, 2=forward
        force = action - 1  # -1, 0, or 1
        self.vel += 0.001 * force - 0.0025 * np.cos(3 * self.pos)
        self.vel = np.clip(self.vel, *self.vel_bounds)
        self.pos += self.vel
        self.pos = np.clip(self.pos, *self.pos_bounds)
        if self.pos == self.pos_bounds[0]:
            self.vel = 0.0
        done = self.pos >= 0.5
        return (self.pos, self.vel), -1.0, done

# Train
env = MountainCar()
tc = TileCoding(n_tilings=8, tiles_per_dim=8,
                state_bounds=[(-1.2, 0.5), (-0.07, 0.07)])

n_actions = 3
w = np.zeros((n_actions, tc.n_tiles))
alpha = 0.1 / 8  # divide by n_tilings
gamma = 1.0
epsilon = 0.1

def get_q(state, action):
    return np.dot(w[action], tc.get_features(state))

def choose_action(state):
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    qs = [get_q(state, a) for a in range(n_actions)]
    return np.argmax(qs)

steps_per_episode = []
for ep in range(200):
    state = env.reset()
    action = choose_action(state)
    steps = 0
    for _ in range(5000):
        next_state, reward, done = env.step(action)
        steps += 1
        if done:
            w[action] += alpha * (reward - get_q(state, action)) * tc.get_features(state)
            break
        next_action = choose_action(next_state)
        td_error = reward + gamma * get_q(next_state, next_action) - get_q(state, action)
        w[action] += alpha * td_error * tc.get_features(state)
        state, action = next_state, next_action
    steps_per_episode.append(steps)

plt.figure(figsize=(10, 5))
plt.plot(steps_per_episode, alpha=0.4, color='steelblue')
window = 10
smoothed = np.convolve(steps_per_episode, np.ones(window)/window, mode='valid')
plt.plot(range(window-1, len(steps_per_episode)), smoothed, 'r-', linewidth=2, label='Smoothed')
plt.xlabel('Episode')
plt.ylabel('Steps per Episode')
plt.title('Semi-Gradient SARSA with Tile Coding on Mountain Car')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""),
md("""## Summary

| Concept | Formula |
|---------|---------|
| **Linear Approximation** | $\\hat{V}(s,\\mathbf{w}) = \\mathbf{w}^\\top \\mathbf{x}(s)$ |
| **Semi-Gradient TD** | $\\mathbf{w} \\leftarrow \\mathbf{w} + \\alpha \\delta_t \\nabla_\\mathbf{w} \\hat{V}(S_t, \\mathbf{w})$ |
| **TD Error** | $\\delta_t = R_{t+1} + \\gamma \\hat{V}(S_{t+1}, \\mathbf{w}) - \\hat{V}(S_t, \\mathbf{w})$ |

**Next:** Notebook 11 — Policy Gradient Methods
"""),
])

save("11_Policy_Gradient_REINFORCE", [
md("""# Notebook 11: Policy Gradient Methods — REINFORCE

**Series:** Reinforcement Learning Basics (11/20)

---

## 1. Why Policy Gradients?

Value-based methods learn $Q(s,a)$ then derive a policy. Policy gradient methods **directly** parameterize the policy $\\pi_\\theta(a|s)$.

| Advantage | Explanation |
|-----------|-------------|
| **Stochastic policies** | Can naturally represent mixed strategies |
| **Continuous actions** | Output parameters of a distribution (e.g., Gaussian) |
| **Convergence** | Guaranteed to converge to a local optimum |

## 2. Policy Parameterizations

**Softmax (discrete):**
$$\\pi_\\theta(a|s) = \\frac{\\exp(\\theta^\\top \\mathbf{x}(s,a))}{\\sum_b \\exp(\\theta^\\top \\mathbf{x}(s,b))}$$

**Gaussian (continuous):**
$$\\pi_\\theta(a|s) = \\frac{1}{\\sigma\\sqrt{2\\pi}} \\exp\\left(-\\frac{(a - \\mu_\\theta(s))^2}{2\\sigma^2}\\right)$$

## 3. Policy Gradient Theorem

**Objective:** $J(\\theta) = V^{\\pi_\\theta}(s_0) = \\mathbb{E}_{\\pi_\\theta}\\left[\\sum_t \\gamma^t R_{t+1}\\right]$

$$\\boxed{\\nabla_\\theta J(\\theta) = \\mathbb{E}_{\\pi_\\theta}\\left[\\sum_t \\gamma^t \\nabla_\\theta \\log \\pi_\\theta(A_t|S_t) \\, Q^{\\pi_\\theta}(S_t, A_t)\\right]}$$

### Derivation (Key Steps)

**Log-derivative trick:** $\\nabla_\\theta \\pi_\\theta = \\pi_\\theta \\nabla_\\theta \\log \\pi_\\theta$

This allows us to write the gradient as an expectation under $\\pi_\\theta$, which we can estimate with samples!

## 4. REINFORCE Algorithm

Replace $Q^{\\pi_\\theta}(S_t, A_t)$ with the sample return $G_t$:

$$\\theta \\leftarrow \\theta + \\alpha \\gamma^t G_t \\nabla_\\theta \\log \\pi_\\theta(A_t|S_t)$$

## 5. REINFORCE with Baseline

$$\\theta \\leftarrow \\theta + \\alpha \\gamma^t (G_t - b(S_t)) \\nabla_\\theta \\log \\pi_\\theta(A_t|S_t)$$

**Common baseline:** $b(S_t) = V(S_t)$

**Why baseline helps:** Reduces variance without changing the expected gradient:
$$\\mathbb{E}_{\\pi_\\theta}[b(S_t) \\nabla_\\theta \\log \\pi_\\theta(A_t|S_t)] = 0$$
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# ============================================
# CartPole Environment (simplified)
# ============================================
class CartPole:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold = 12 * np.pi / 180
        self.x_threshold = 2.4

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, 4)
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4/3 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x += self.tau * x_dot
        x_dot += self.tau * xacc
        theta += self.tau * theta_dot
        theta_dot += self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])
        done = (abs(x) > self.x_threshold or abs(theta) > self.theta_threshold)
        reward = 1.0 if not done else 0.0
        return self.state.copy(), reward, done

# ============================================
# REINFORCE with Baseline
# ============================================
class PolicyNetwork:
    def __init__(self, n_features=4, n_actions=2, lr=0.01):
        self.theta = np.zeros((n_features, n_actions))
        self.lr = lr

    def softmax(self, state):
        logits = state @ self.theta
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def choose_action(self, state):
        probs = self.softmax(state)
        return np.random.choice(len(probs), p=probs)

    def update(self, states, actions, advantages):
        for s, a, adv in zip(states, actions, advantages):
            probs = self.softmax(s)
            grad_log = -probs.copy()
            grad_log[a] += 1  # ∇log π = (1(a) - π(a))
            self.theta += self.lr * adv * np.outer(s, grad_log)

class ValueBaseline:
    def __init__(self, n_features=4, lr=0.01):
        self.w = np.zeros(n_features)
        self.lr = lr

    def predict(self, state):
        return state @ self.w

    def update(self, states, returns):
        for s, G in zip(states, returns):
            self.w += self.lr * (G - self.predict(s)) * s

# Train
env = CartPole()
policy = PolicyNetwork(lr=0.002)
baseline = ValueBaseline(lr=0.01)
gamma = 0.99

episode_rewards = []
for ep in range(1000):
    state = env.reset()
    states, actions, rewards = [], [], []
    for _ in range(500):
        action = policy.choose_action(state)
        next_state, reward, done = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        if done:
            break

    # Compute returns
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    # Compute advantages
    advantages = [G - baseline.predict(s) for G, s in zip(returns, states)]

    # Update
    policy.update(states, actions, advantages)
    baseline.update(states, returns)
    episode_rewards.append(sum(rewards))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, alpha=0.3, color='steelblue')
smoothed = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
plt.plot(range(49, len(episode_rewards)), smoothed, 'r-', linewidth=2, label='50-ep moving avg')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('REINFORCE with Baseline on CartPole')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print(f"Last 100 episodes avg: {np.mean(episode_rewards[-100:]):.1f}")
"""),
md("""## Summary

| Concept | Formula |
|---------|---------|
| **Policy Gradient Theorem** | $\\nabla J(\\theta) = \\mathbb{E}[\\sum_t \\gamma^t \\nabla \\log \\pi_\\theta(A_t|S_t) Q^\\pi(S_t,A_t)]$ |
| **REINFORCE** | $\\theta \\leftarrow \\theta + \\alpha \\gamma^t G_t \\nabla \\log \\pi_\\theta(A_t|S_t)$ |
| **With Baseline** | $\\theta \\leftarrow \\theta + \\alpha \\gamma^t (G_t - b(S_t)) \\nabla \\log \\pi_\\theta(A_t|S_t)$ |

**Next:** Notebook 12 — Actor-Critic Methods
"""),
])

save("12_Actor_Critic_Methods", [
md("""# Notebook 12: Actor-Critic Methods

**Series:** Reinforcement Learning Basics (12/20)

---

## 1. Motivation

REINFORCE has **high variance** because $G_t$ depends on many random variables. The Actor-Critic framework reduces variance by using a **learned value function** (critic) instead of sample returns.

## 2. Architecture

- **Actor:** Policy $\\pi_\\theta(a|s)$ — decides what to do
- **Critic:** Value function $\\hat{V}_w(s)$ — evaluates how good the state is

## 3. One-Step Actor-Critic

**TD Error (serves as advantage estimate):**
$$\\delta_t = R_{t+1} + \\gamma \\hat{V}_w(S_{t+1}) - \\hat{V}_w(S_t)$$

**Actor Update:**
$$\\theta \\leftarrow \\theta + \\alpha_\\theta \\, \\delta_t \\, \\nabla_\\theta \\log \\pi_\\theta(A_t|S_t)$$

**Critic Update:**
$$w \\leftarrow w + \\alpha_w \\, \\delta_t \\, \\nabla_w \\hat{V}_w(S_t)$$

## 4. A2C (Advantage Actor-Critic)

Uses the advantage $A(s,a) = Q(s,a) - V(s) \\approx \\delta_t$:

$$\\nabla_\\theta J(\\theta) \\approx \\sum_t \\nabla_\\theta \\log \\pi_\\theta(A_t|S_t) \\, \\hat{A}_t$$

where $\\hat{A}_t = \\delta_t = R_{t+1} + \\gamma V_w(S_{t+1}) - V_w(S_t)$

## 5. Why Actor-Critic Works

- $\\delta_t$ is an **unbiased estimate** of the advantage $A^\\pi(S_t, A_t)$
- Lower variance than MC returns $G_t$
- Can learn **online** (every step, not just episode end)
- **Two-timescale** convergence: critic learns faster, actor uses critic's estimates
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# Reuse CartPole from Notebook 11
class CartPole:
    def __init__(self):
        self.gravity = 9.8; self.masscart = 1.0; self.masspole = 0.1
        self.total_mass = 1.1; self.length = 0.5
        self.polemass_length = 0.05; self.force_mag = 10.0; self.tau = 0.02
        self.theta_threshold = 12 * np.pi / 180; self.x_threshold = 2.4
    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, 4); return self.state.copy()
    def step(self, action):
        x, xd, th, thd = self.state; f = self.force_mag if action == 1 else -self.force_mag
        ct, st = np.cos(th), np.sin(th)
        tmp = (f + self.polemass_length * thd**2 * st) / self.total_mass
        thacc = (self.gravity*st - ct*tmp) / (self.length*(4/3 - self.masspole*ct**2/self.total_mass))
        xacc = tmp - self.polemass_length*thacc*ct/self.total_mass
        self.state = np.array([x+self.tau*xd, xd+self.tau*xacc, th+self.tau*thd, thd+self.tau*thacc])
        done = abs(self.state[0]) > self.x_threshold or abs(self.state[2]) > self.theta_threshold
        return self.state.copy(), 1.0 if not done else 0.0, done

# ============================================
# One-Step Actor-Critic
# ============================================
def actor_critic(n_episodes=1000, alpha_theta=0.002, alpha_w=0.01, gamma=0.99):
    env = CartPole()
    theta = np.zeros((4, 2))  # actor weights
    w = np.zeros(4)            # critic weights
    rewards_history = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        for t in range(500):
            # Actor: softmax policy
            logits = state @ theta
            logits -= np.max(logits)
            probs = np.exp(logits) / np.sum(np.exp(logits))
            action = np.random.choice(2, p=probs)

            next_state, reward, done = env.step(action)
            total_reward += reward

            # TD Error
            v_current = state @ w
            v_next = 0 if done else next_state @ w
            delta = reward + gamma * v_next - v_current

            # Critic update
            w += alpha_w * delta * state

            # Actor update
            grad_log = -probs.copy()
            grad_log[action] += 1
            theta += alpha_theta * delta * np.outer(state, grad_log)

            state = next_state
            if done:
                break
        rewards_history.append(total_reward)
    return rewards_history

# Run and compare with REINFORCE
np.random.seed(42)
ac_rewards = actor_critic(1000)

plt.figure(figsize=(10, 5))
window = 50
smoothed = np.convolve(ac_rewards, np.ones(window)/window, mode='valid')
plt.plot(ac_rewards, alpha=0.2, color='steelblue')
plt.plot(range(window-1, len(ac_rewards)), smoothed, 'r-', linewidth=2, label='Actor-Critic (50-ep avg)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('One-Step Actor-Critic on CartPole')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 520)
plt.tight_layout()
plt.show()
print(f"Last 100 episodes avg: {np.mean(ac_rewards[-100:]):.1f}")
"""),
md("""## Summary

| Component | Update |
|-----------|--------|
| **Critic** | $w \\leftarrow w + \\alpha_w \\delta_t \\nabla_w \\hat{V}_w(S_t)$ |
| **Actor** | $\\theta \\leftarrow \\theta + \\alpha_\\theta \\delta_t \\nabla_\\theta \\log \\pi_\\theta(A_t|S_t)$ |
| **TD Error** | $\\delta_t = R_{t+1} + \\gamma \\hat{V}_w(S_{t+1}) - \\hat{V}_w(S_t)$ |

**Next:** Notebook 13 — Advantage Functions & GAE
"""),
])

# Notebooks 13-20 follow the same pattern
save("13_Advantage_Functions_and_GAE", [
md("""# Notebook 13: Advantage Functions & Generalized Advantage Estimation

**Series:** Reinforcement Learning Basics (13/20)

---

## 1. The Advantage Function

$$A^\\pi(s, a) = Q^\\pi(s, a) - V^\\pi(s)$$

**Meaning:** How much **better** is action $a$ compared to the **average** action in state $s$?

**Properties:**
- $\\sum_a \\pi(a|s) A^\\pi(s,a) = 0$ (zero mean under policy)
- $A^\\pi(s,a) > 0$: action $a$ is better than average
- $A^\\pi(s,a) < 0$: action $a$ is worse than average

## 2. TD Error as Advantage Estimate

$$\\delta_t = R_{t+1} + \\gamma V(S_{t+1}) - V(S_t)$$

$\\mathbb{E}[\\delta_t | S_t, A_t] = Q^\\pi(S_t, A_t) - V^\\pi(S_t) = A^\\pi(S_t, A_t)$

So $\\delta_t$ is an **unbiased** estimate of the advantage!

## 3. Generalized Advantage Estimation (GAE)

$$\\hat{A}_t^{GAE(\\gamma, \\lambda)} = \\sum_{l=0}^{\\infty} (\\gamma\\lambda)^l \\delta_{t+l}$$

**Special cases:**
- $\\lambda = 0$: $\\hat{A}_t = \\delta_t$ (TD, low variance, high bias)
- $\\lambda = 1$: $\\hat{A}_t = G_t - V(S_t)$ (MC, high variance, low bias)

### Efficient Backward Computation

$$\\hat{A}_T = \\delta_T$$
$$\\hat{A}_t = \\delta_t + \\gamma\\lambda \\, \\hat{A}_{t+1}$$

This makes GAE $O(T)$ — just one backward pass!

## 4. Why GAE Matters

GAE is used in **PPO**, **TRPO**, and most modern policy gradient methods. It provides a principled way to trade off bias and variance via $\\lambda$.
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# ============================================
# GAE Computation Demo
# ============================================
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    \"\"\"Compute GAE advantages using backward recursion.\"\"\"
    T = len(rewards)
    advantages = np.zeros(T)
    last_gae = 0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0  # terminal
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = last_gae = delta + gamma * lam * last_gae

    returns = advantages + values
    return advantages, returns

# Simulate an episode
np.random.seed(42)
T = 50
rewards = np.random.randn(T) * 0.1 + 0.5  # noisy positive rewards
values = np.cumsum(np.random.randn(T) * 0.05) + 5  # approximate value function

# Compare different lambda values
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

for lam in [0.0, 0.5, 0.9, 0.95, 1.0]:
    adv, ret = compute_gae(rewards, values, gamma=0.99, lam=lam)
    axes[0].plot(adv, label=f'λ={lam}', alpha=0.8, linewidth=2)
    axes[1].plot(ret, label=f'λ={lam}', alpha=0.8, linewidth=2)

axes[0].set_ylabel('Advantage $\\hat{A}_t^{GAE}$')
axes[0].set_title('GAE Advantages for Different λ')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('Return Estimate')
axes[1].set_title('Return Estimates for Different λ')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Variance comparison
n_episodes = 500
var_by_lambda = {}
for lam in np.arange(0, 1.05, 0.1):
    advs = []
    for _ in range(n_episodes):
        r = np.random.randn(30) * 0.1 + 0.5
        v = np.cumsum(np.random.randn(30) * 0.05) + 5
        a, _ = compute_gae(r, v, lam=lam)
        advs.append(np.var(a))
    var_by_lambda[round(lam, 1)] = np.mean(advs)

plt.figure(figsize=(8, 5))
plt.plot(list(var_by_lambda.keys()), list(var_by_lambda.values()), 'o-', linewidth=2)
plt.xlabel('λ')
plt.ylabel('Average Variance of Advantages')
plt.title('GAE: Variance vs λ (Bias-Variance Tradeoff)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""),
md("""## Summary

| Concept | Formula |
|---------|---------|
| **Advantage** | $A^\\pi(s,a) = Q^\\pi(s,a) - V^\\pi(s)$ |
| **GAE** | $\\hat{A}_t = \\sum_{l=0}^\\infty (\\gamma\\lambda)^l \\delta_{t+l}$ |
| **Backward recursion** | $\\hat{A}_t = \\delta_t + \\gamma\\lambda \\hat{A}_{t+1}$ |
| **λ=0** | TD advantage (low var, high bias) |
| **λ=1** | MC advantage (high var, low bias) |

**Next:** Notebook 14 — PPO
"""),
])

save("14_Proximal_Policy_Optimization_PPO", [
md("""# Notebook 14: Proximal Policy Optimization (PPO)

**Series:** Reinforcement Learning Basics (14/20)

---

## 1. The Core Problem

Policy gradient methods are sensitive to step size:
- **Too large** → policy collapses (catastrophic performance drop)
- **Too small** → painfully slow learning

**PPO** constrains how much the policy can change per update.

## 2. Probability Ratio

$$r_t(\\theta) = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{old}}(a_t|s_t)}$$

- $r_t = 1$ when new and old policies agree
- $r_t > 1$ means new policy gives higher probability to $a_t$

## 3. PPO-Clip Objective

$$L^{CLIP}(\\theta) = \\mathbb{E}_t \\left[ \\min\\Big( r_t(\\theta) \\hat{A}_t, \\; \\text{clip}(r_t(\\theta), 1-\\varepsilon, 1+\\varepsilon) \\hat{A}_t \\Big) \\right]$$

**How clipping works:**
- When $\\hat{A}_t > 0$ (good action): caps the benefit at $r_t = 1 + \\varepsilon$
- When $\\hat{A}_t < 0$ (bad action): caps the benefit at $r_t = 1 - \\varepsilon$

## 4. Full PPO Objective

$$L(\\theta) = L^{CLIP}(\\theta) - c_1 L^{VF}(\\theta) + c_2 H[\\pi_\\theta]$$

| Term | Purpose |
|------|---------|
| $L^{CLIP}$ | Clipped policy gradient |
| $L^{VF} = (V_\\theta(s_t) - V_t^{target})^2$ | Value function accuracy |
| $H[\\pi_\\theta]$ | Entropy bonus for exploration |

## 5. PPO Algorithm

1. Collect trajectories with current policy $\\pi_{\\theta_{old}}$
2. Compute advantages using GAE
3. For $K$ epochs:
   - Sample minibatches from collected data
   - Compute clipped objective and update $\\theta$
4. $\\theta_{old} \\leftarrow \\theta$, repeat
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# CartPole environment (reused)
class CartPole:
    def __init__(self):
        self.gravity = 9.8; self.masscart = 1.0; self.masspole = 0.1
        self.total_mass = 1.1; self.length = 0.5; self.polemass_length = 0.05
        self.force_mag = 10.0; self.tau = 0.02
        self.theta_threshold = 12*np.pi/180; self.x_threshold = 2.4
    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, 4); return self.state.copy()
    def step(self, action):
        x,xd,th,thd = self.state; f = self.force_mag if action==1 else -self.force_mag
        ct,st = np.cos(th),np.sin(th)
        tmp = (f+self.polemass_length*thd**2*st)/self.total_mass
        thacc = (self.gravity*st-ct*tmp)/(self.length*(4/3-self.masspole*ct**2/self.total_mass))
        xacc = tmp - self.polemass_length*thacc*ct/self.total_mass
        self.state = np.array([x+self.tau*xd,xd+self.tau*xacc,th+self.tau*thd,thd+self.tau*thacc])
        done = abs(self.state[0])>self.x_threshold or abs(self.state[2])>self.theta_threshold
        return self.state.copy(), 1.0 if not done else 0.0, done

def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = np.zeros(T)
    last_gae = 0
    for t in reversed(range(T)):
        next_val = 0 if t == T-1 else values[t+1]
        delta = rewards[t] + gamma * next_val - values[t]
        advantages[t] = last_gae = delta + gamma * lam * last_gae
    return advantages, advantages + np.array(values)

# ============================================
# PPO Implementation
# ============================================
class PPO:
    def __init__(self, lr_actor=0.003, lr_critic=0.01, clip_eps=0.2, gamma=0.99, lam=0.95, epochs=4):
        self.theta = np.zeros((4, 2))
        self.w = np.zeros(4)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs

    def get_action_and_prob(self, state):
        probs = softmax(state @ self.theta)
        action = np.random.choice(2, p=probs)
        return action, probs[action]

    def train(self, states, actions, old_probs, advantages, returns):
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            for i in range(len(states)):
                s, a = states[i], actions[i]
                probs = softmax(s @ self.theta)
                ratio = probs[a] / (old_probs[i] + 1e-8)

                # Clipped objective
                adv = advantages[i]
                surr1 = ratio * adv
                surr2 = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                policy_loss = -min(surr1, surr2)

                # Actor gradient (approximate)
                grad_log = -probs.copy()
                grad_log[a] += 1
                if surr1 < surr2:
                    self.theta -= self.lr_actor * policy_loss * np.outer(s, grad_log) / len(states)
                else:
                    clipped_grad = np.clip(ratio, 1-self.clip_eps, 1+self.clip_eps)
                    self.theta += self.lr_actor * adv * np.outer(s, grad_log) / len(states)

                # Critic update
                v_pred = s @ self.w
                critic_loss = returns[i] - v_pred
                self.w += self.lr_critic * critic_loss * s / len(states)

# Train PPO
env = CartPole()
ppo = PPO()
episode_rewards = []

for ep in range(800):
    state = env.reset()
    states, actions, rewards, old_probs = [], [], [], []

    for _ in range(500):
        action, prob = ppo.get_action_and_prob(state)
        next_state, reward, done = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        old_probs.append(prob)
        state = next_state
        if done: break

    values = [s @ ppo.w for s in states]
    advantages, returns = compute_gae(rewards, values, ppo.gamma, ppo.lam)

    ppo.train(states, actions, old_probs, advantages, returns)
    episode_rewards.append(sum(rewards))

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, alpha=0.3, color='steelblue')
w = 50
sm = np.convolve(episode_rewards, np.ones(w)/w, mode='valid')
plt.plot(range(w-1, len(episode_rewards)), sm, 'r-', linewidth=2, label=f'{w}-ep avg')
plt.xlabel('Episode'); plt.ylabel('Total Reward')
plt.title('PPO on CartPole (from scratch)')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.show()
print(f"Last 100 avg: {np.mean(episode_rewards[-100:]):.1f}")
"""),
md("""## Summary

| Component | Formula |
|-----------|---------|
| **Ratio** | $r_t(\\theta) = \\pi_\\theta(a_t|s_t) / \\pi_{\\theta_{old}}(a_t|s_t)$ |
| **PPO-Clip** | $\\min(r_t \\hat{A}_t, \\text{clip}(r_t, 1\\pm\\varepsilon) \\hat{A}_t)$ |
| **Full Objective** | $L^{CLIP} - c_1 L^{VF} + c_2 H$ |

**Next:** Notebook 15 — Deep Q-Networks
"""),
])

save("15_Deep_Q_Networks_DQN", [
md("""# Notebook 15: Deep Q-Networks (DQN)

**Series:** Reinforcement Learning Basics (15/20)

---

## 1. The Problem

Q-Learning with neural networks naively **diverges** due to:
1. **Correlated samples:** Sequential data is highly correlated
2. **Non-stationary targets:** Target changes as Q-network updates
3. **Function approximation + bootstrapping + off-policy** = The Deadly Triad

## 2. DQN Innovations

### Experience Replay
Store transitions $(s, a, r, s', done)$ in buffer $\\mathcal{D}$, sample random minibatches.
- Breaks temporal correlations
- Reuses data (sample efficiency)
- Each transition can be used in many updates

### Target Network
Use a **separate frozen network** $Q(s, a; \\theta^-)$ for computing targets:

$$y_i = r + \\gamma \\max_{a'} Q(s', a'; \\theta^-)$$

$\\theta^-$ is updated periodically: $\\theta^- \\leftarrow \\theta$ every $C$ steps.

## 3. DQN Loss

$$L(\\theta) = \\mathbb{E}_{(s,a,r,s') \\sim \\mathcal{D}} \\left[ \\Big( y_i - Q(s, a; \\theta) \\Big)^2 \\right]$$

## 4. Huber Loss (Smooth L1)

$$L_\\delta(a) = \\begin{cases} \\frac{1}{2}a^2 & \\text{if } |a| \\leq \\delta \\\\ \\delta(|a| - \\frac{1}{2}\\delta) & \\text{otherwise} \\end{cases}$$

More robust to outliers than MSE.
"""),
code("""import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# CartPole environment
class CartPole:
    def __init__(self):
        self.gravity=9.8;self.masscart=1.0;self.masspole=0.1;self.total_mass=1.1
        self.length=0.5;self.polemass_length=0.05;self.force_mag=10.0;self.tau=0.02
        self.theta_threshold=12*np.pi/180;self.x_threshold=2.4
    def reset(self):
        self.state=np.random.uniform(-0.05,0.05,4);return self.state.copy()
    def step(self,action):
        x,xd,th,thd=self.state;f=self.force_mag if action==1 else -self.force_mag
        ct,st=np.cos(th),np.sin(th)
        tmp=(f+self.polemass_length*thd**2*st)/self.total_mass
        thacc=(self.gravity*st-ct*tmp)/(self.length*(4/3-self.masspole*ct**2/self.total_mass))
        xacc=tmp-self.polemass_length*thacc*ct/self.total_mass
        self.state=np.array([x+self.tau*xd,xd+self.tau*xacc,th+self.tau*thd,thd+self.tau*thacc])
        done=abs(self.state[0])>self.x_threshold or abs(self.state[2])>self.theta_threshold
        return self.state.copy(),1.0 if not done else 0.0,done

# ============================================
# Simple DQN (no PyTorch — pure numpy 2-layer network)
# ============================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, sn, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(sn), np.array(d)
    def __len__(self):
        return len(self.buffer)

class SimpleQNetwork:
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2, lr=0.001):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
        self.lr = lr

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.h1 = np.maximum(0, self.z1)  # ReLU
        self.out = self.h1 @ self.W2 + self.b2
        return self.out

    def predict(self, state):
        return self.forward(state.reshape(1, -1))[0]

    def update(self, states, targets, actions):
        batch_size = len(states)
        # Forward
        out = self.forward(states)
        # Loss gradient: d/dout only for taken actions
        d_out = np.zeros_like(out)
        for i in range(batch_size):
            d_out[i, actions[i]] = 2 * (out[i, actions[i]] - targets[i]) / batch_size
        # Backward
        d_W2 = self.h1.T @ d_out
        d_b2 = d_out.sum(axis=0)
        d_h1 = d_out @ self.W2.T
        d_z1 = d_h1 * (self.z1 > 0)
        d_W1 = states.T @ d_z1
        d_b1 = d_z1.sum(axis=0)
        # Update
        self.W1 -= self.lr * d_W1; self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2; self.b2 -= self.lr * d_b2

    def copy_from(self, other):
        self.W1 = other.W1.copy(); self.b1 = other.b1.copy()
        self.W2 = other.W2.copy(); self.b2 = other.b2.copy()

# Train DQN
env = CartPole()
q_net = SimpleQNetwork(lr=0.001)
target_net = SimpleQNetwork()
target_net.copy_from(q_net)
buffer = ReplayBuffer(10000)
gamma = 0.99; epsilon = 1.0; eps_decay = 0.995; eps_min = 0.01
batch_size = 64; target_update = 10
episode_rewards = []

for ep in range(500):
    state = env.reset(); total_reward = 0
    for t in range(500):
        if np.random.random() < epsilon:
            action = np.random.randint(2)
        else:
            q_vals = q_net.predict(state)
            action = np.argmax(q_vals)
        next_state, reward, done = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state

        if len(buffer) >= batch_size:
            s, a, r, sn, d = buffer.sample(batch_size)
            q_next = target_net.forward(sn)
            targets = r + gamma * np.max(q_next, axis=1) * (1 - d)
            q_net.update(s, targets, a)

        if done: break

    epsilon = max(eps_min, epsilon * eps_decay)
    if ep % target_update == 0:
        target_net.copy_from(q_net)
    episode_rewards.append(total_reward)

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, alpha=0.3, color='steelblue')
w = 30
sm = np.convolve(episode_rewards, np.ones(w)/w, mode='valid')
plt.plot(range(w-1, len(episode_rewards)), sm, 'r-', linewidth=2, label=f'{w}-ep avg')
plt.xlabel('Episode'); plt.ylabel('Total Reward')
plt.title('DQN on CartPole (Pure NumPy Implementation)')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.show()
print(f"Last 50 avg: {np.mean(episode_rewards[-50:]):.1f}")
"""),
md("""## Summary

| Innovation | Purpose |
|-----------|---------|
| **Experience Replay** | Break correlations, reuse data |
| **Target Network** | Stable targets, prevent divergence |
| **Huber Loss** | Robustness to outliers |

**Next:** Notebook 16 — Double DQN & Dueling DQN
"""),
])

save("16_Double_DQN_and_Dueling_DQN", [
md("""# Notebook 16: Double DQN & Dueling DQN

**Series:** Reinforcement Learning Basics (16/20)

---

## 1. Double DQN

### Problem: Maximization Bias
DQN uses $\\max_a Q(s', a; \\theta^-)$ — the same network selects and evaluates. This **overestimates** Q-values.

### Solution: Decouple Selection and Evaluation

$$y = r + \\gamma Q\\Big(s', \\underbrace{\\arg\\max_{a'} Q(s', a'; \\theta)}_{\\text{online selects}};\\; \\underbrace{\\theta^-}_{\\text{target evaluates}}\\Big)$$

## 2. Dueling DQN

### Decompose Q into Value + Advantage

$$Q(s, a; \\theta) = V(s; \\theta_V) + A(s, a; \\theta_A)$$

**Network architecture:** Shared feature layers → two streams:
- **Value stream:** scalar $V(s)$
- **Advantage stream:** $A(s, a)$ for each action

### Identifiability Trick
$$Q(s,a) = V(s) + A(s,a) - \\frac{1}{|\\mathcal{A}|} \\sum_{a'} A(s, a')$$

**Why Dueling?** In many states, the value doesn't depend on which action you take. Dueling learns $V(s)$ more efficiently.
"""),
code("""import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# Reuse CartPole and ReplayBuffer from notebook 15
class CartPole:
    def __init__(self):
        self.gravity=9.8;self.masscart=1.0;self.masspole=0.1;self.total_mass=1.1
        self.length=0.5;self.polemass_length=0.05;self.force_mag=10.0;self.tau=0.02
        self.theta_threshold=12*np.pi/180;self.x_threshold=2.4
    def reset(self):
        self.state=np.random.uniform(-0.05,0.05,4);return self.state.copy()
    def step(self,action):
        x,xd,th,thd=self.state;f=self.force_mag if action==1 else -self.force_mag
        ct,st=np.cos(th),np.sin(th)
        tmp=(f+self.polemass_length*thd**2*st)/self.total_mass
        thacc=(self.gravity*st-ct*tmp)/(self.length*(4/3-self.masspole*ct**2/self.total_mass))
        xacc=tmp-self.polemass_length*thacc*ct/self.total_mass
        self.state=np.array([x+self.tau*xd,xd+self.tau*xacc,th+self.tau*thd,thd+self.tau*thacc])
        done=abs(self.state[0])>self.x_threshold or abs(self.state[2])>self.theta_threshold
        return self.state.copy(),1.0 if not done else 0.0,done

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, sn, d):
        self.buffer.append((s,a,r,sn,d))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,sn,d = zip(*batch)
        return np.array(s),np.array(a),np.array(r),np.array(sn),np.array(d)
    def __len__(self): return len(self.buffer)

class DQNAgent:
    \"\"\"Configurable: vanilla DQN, Double DQN, or Dueling DQN.\"\"\"
    def __init__(self, double=False, dueling=False, lr=0.001):
        self.double = double
        self.dueling = dueling
        h = 64
        self.W1 = np.random.randn(4, h)*0.1; self.b1 = np.zeros(h)
        if dueling:
            self.W_v = np.random.randn(h, 1)*0.1; self.b_v = np.zeros(1)
            self.W_a = np.random.randn(h, 2)*0.1; self.b_a = np.zeros(2)
        else:
            self.W2 = np.random.randn(h, 2)*0.1; self.b2 = np.zeros(2)
        self.lr = lr
        # Target copies
        self._copy_params()

    def _copy_params(self):
        self.tW1=self.W1.copy();self.tb1=self.b1.copy()
        if self.dueling:
            self.tW_v=self.W_v.copy();self.tb_v=self.b_v.copy()
            self.tW_a=self.W_a.copy();self.tb_a=self.b_a.copy()
        else:
            self.tW2=self.W2.copy();self.tb2=self.b2.copy()

    def _forward(self, x, target=False):
        W1,b1 = (self.tW1,self.tb1) if target else (self.W1,self.b1)
        h = np.maximum(0, x @ W1 + b1)
        if self.dueling:
            Wv,bv = (self.tW_v,self.tb_v) if target else (self.W_v,self.b_v)
            Wa,ba = (self.tW_a,self.tb_a) if target else (self.W_a,self.b_a)
            v = h @ Wv + bv
            a = h @ Wa + ba
            q = v + a - a.mean(axis=1, keepdims=True)
        else:
            W2,b2 = (self.tW2,self.tb2) if target else (self.W2,self.b2)
            q = h @ W2 + b2
        return q

    def predict(self, state):
        return self._forward(state.reshape(1,-1))[0]

    def get_targets(self, s, a, r, sn, d):
        gamma = 0.99
        if self.double:
            a_best = np.argmax(self._forward(sn, target=False), axis=1)
            q_target = self._forward(sn, target=True)
            q_next = q_target[np.arange(len(sn)), a_best]
        else:
            q_next = np.max(self._forward(sn, target=True), axis=1)
        return r + gamma * q_next * (1 - d)

def train_agent(agent_type, n_episodes=400):
    double = 'double' in agent_type
    dueling = 'dueling' in agent_type
    agent = DQNAgent(double=double, dueling=dueling)
    buf = ReplayBuffer(10000)
    eps = 1.0; rewards = []

    for ep in range(n_episodes):
        env = CartPole(); state = env.reset(); total = 0
        for _ in range(500):
            if np.random.random() < eps:
                action = np.random.randint(2)
            else:
                action = np.argmax(agent.predict(state))
            ns, r, done = env.step(action)
            buf.push(state, action, r, ns, done)
            total += r; state = ns
            if done: break
        eps = max(0.01, eps * 0.995)
        if ep % 10 == 0: agent._copy_params()
        rewards.append(total)
    return rewards

# Compare
np.random.seed(42)
r_dqn = train_agent('vanilla', 400)
np.random.seed(42)
r_ddqn = train_agent('double', 400)
np.random.seed(42)
r_duel = train_agent('dueling', 400)

plt.figure(figsize=(10, 5))
w = 30
for rewards, label, color in [(r_dqn,'DQN','blue'),(r_ddqn,'Double DQN','red'),(r_duel,'Dueling DQN','green')]:
    sm = np.convolve(rewards, np.ones(w)/w, mode='valid')
    plt.plot(range(w-1, len(rewards)), sm, linewidth=2, label=label, color=color)
plt.xlabel('Episode'); plt.ylabel(f'Reward ({w}-ep avg)')
plt.title('DQN vs Double DQN vs Dueling DQN')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.show()
"""),
md("""## Summary

| Variant | Key Change | Benefit |
|---------|-----------|---------|
| **Double DQN** | Online selects, target evaluates | Reduces overestimation |
| **Dueling DQN** | $Q = V + A$ decomposition | Better value estimation |

**Next:** Notebook 17 — Experience Replay
"""),
])

save("17_Experience_Replay", [
md("""# Notebook 17: Experience Replay & Prioritized ER

**Series:** Reinforcement Learning Basics (17/20)

---

## 1. Uniform Experience Replay

Store transitions $(s, a, r, s', done)$ in a circular buffer $\\mathcal{D}$ of fixed capacity $N$.

**Sample uniformly:** Each transition equally likely to be sampled.

**Benefits:**
- Breaks temporal correlations between consecutive samples
- Each transition can be reused many times (sample efficiency)
- Stabilizes training (reduces non-stationarity)

## 2. Prioritized Experience Replay (PER)

**Idea:** Some transitions are more "surprising" — sample them more often.

**Priority:** $p_i = |\\delta_i| + \\epsilon$ where $\\delta_i$ is TD error

**Sampling probability:**
$$P(i) = \\frac{p_i^\\alpha}{\\sum_k p_k^\\alpha}$$

- $\\alpha = 0$: uniform sampling
- $\\alpha = 1$: fully prioritized

### Importance Sampling Correction

Biased sampling → biased gradient → need correction:
$$w_i = \\left(\\frac{1}{N \\cdot P(i)}\\right)^\\beta$$

$\\beta$ is annealed from some initial value to 1 over training.

### SumTree Data Structure

Efficient $O(\\log N)$ sampling using a binary tree where each leaf stores a priority and internal nodes store the sum of their children.

## 3. Hindsight Experience Replay (HER)

For **sparse reward** environments:
- Episode failed to reach goal $g$
- Relabel: pretend the achieved state was the goal
- Failed experience → successful experience for a different goal

**Strategies:** final, future, episode, random
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# ============================================
# SumTree for Prioritized Replay
# ============================================
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 0.01
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        self.tree.add(self.max_priority ** self.alpha, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices, priorities, samples = [], [], []
        segment = self.tree.total() / batch_size

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            if data is not None:
                indices.append(idx)
                priorities.append(priority)
                samples.append(data)

        probs = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights /= weights.max()

        s, a, r, sn, d = zip(*samples)
        return np.array(s), np.array(a), np.array(r), np.array(sn), np.array(d), indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries

# Demo: Compare sampling distributions
buf_uniform = []
buf_per = PrioritizedReplayBuffer(capacity=1000)

# Add 1000 transitions with varying TD errors
td_errors = np.random.exponential(1.0, 1000)
for i in range(1000):
    data = (np.zeros(4), 0, 0, np.zeros(4), False)
    buf_uniform.append(data)
    buf_per.push(*data)
    buf_per.update_priorities([i + 999], [td_errors[i]])

# Sample and see distribution
n_samples = 10000
uniform_counts = np.zeros(1000)
per_counts = np.zeros(1000)

for _ in range(n_samples):
    idx = np.random.randint(1000)
    uniform_counts[idx] += 1

# For PER, track which indices get sampled
for _ in range(n_samples // 32):
    _, _, _, _, _, indices, _ = buf_per.sample(32)
    for idx in indices:
        data_idx = idx - 999
        if 0 <= data_idx < 1000:
            per_counts[data_idx] += 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sorted_idx = np.argsort(td_errors)
axes[0].bar(range(1000), uniform_counts[sorted_idx], alpha=0.7, width=1)
axes[0].set_title('Uniform Replay Sampling')
axes[0].set_xlabel('Transitions (sorted by TD error)')
axes[0].set_ylabel('Sample Count')

axes[1].bar(range(1000), per_counts[sorted_idx], alpha=0.7, width=1, color='red')
axes[1].set_title('Prioritized Replay Sampling')
axes[1].set_xlabel('Transitions (sorted by TD error)')
axes[1].set_ylabel('Sample Count')

plt.suptitle('Uniform vs Prioritized Experience Replay', fontsize=14)
plt.tight_layout()
plt.show()
"""),
md("""## Summary

| Method | Sampling | Correction |
|--------|----------|-----------|
| **Uniform ER** | $P(i) = 1/N$ | None needed |
| **PER** | $P(i) = p_i^\\alpha / \\sum p_k^\\alpha$ | IS weights $w_i = (NP(i))^{-\\beta}$ |
| **HER** | Relabel goals | None |

**Next:** Notebook 18 — Exploration vs Exploitation
"""),
])

save("18_Exploration_vs_Exploitation", [
md("""# Notebook 18: Exploration vs Exploitation

**Series:** Reinforcement Learning Basics (18/20)

---

## 1. The Dilemma

- **Exploit:** Choose the action with highest estimated value (use what you know)
- **Explore:** Try other actions to discover potentially better options

## 2. Undirected Methods

### ε-Greedy
$$a_t = \\begin{cases} \\arg\\max_a Q(a) & \\text{with probability } 1-\\varepsilon \\\\ \\text{random action} & \\text{with probability } \\varepsilon \\end{cases}$$

### Boltzmann (Softmax)
$$\\pi(a) = \\frac{\\exp(Q(a)/\\tau)}{\\sum_b \\exp(Q(b)/\\tau)}$$

$\\tau \\to 0$: greedy, $\\tau \\to \\infty$: uniform

## 3. Directed Methods

### UCB (Upper Confidence Bound)
$$A_t = \\arg\\max_a \\left[ Q_t(a) + c \\sqrt{\\frac{\\ln t}{N_t(a)}} \\right]$$

The bonus term is large for rarely-visited actions → optimism in the face of uncertainty.

### Thompson Sampling
1. Sample $\\hat{Q}(a) \\sim \\text{Posterior}(a)$
2. Act greedily: $a_t = \\arg\\max_a \\hat{Q}(a)$

For Bernoulli bandits: $\\hat{Q}(a) \\sim \\text{Beta}(\\alpha_a, \\beta_a)$

## 4. Intrinsic Motivation

Add **curiosity bonus** to reward:
$$r_t^{\\text{total}} = r_t^{\\text{extrinsic}} + \\eta \\cdot r_t^{\\text{intrinsic}}$$

Methods:
- **ICM:** Prediction error of a forward dynamics model
- **RND:** Error of a random network distillation
- **Count-based:** $r^+ = \\beta / \\sqrt{N(s)}$
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# ============================================
# 10-Armed Bandit Testbed
# ============================================
class Bandit:
    def __init__(self, k=10):
        self.k = k
        self.q_star = np.random.randn(k)  # true action values

    def pull(self, action):
        return np.random.randn() + self.q_star[action]

def run_epsilon_greedy(bandit, epsilon, n_steps=1000):
    Q = np.zeros(bandit.k)
    N = np.zeros(bandit.k)
    rewards = []
    optimal_actions = []
    best_action = np.argmax(bandit.q_star)

    for t in range(n_steps):
        if np.random.random() < epsilon:
            action = np.random.randint(bandit.k)
        else:
            action = np.argmax(Q)
        reward = bandit.pull(action)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        rewards.append(reward)
        optimal_actions.append(1 if action == best_action else 0)
    return rewards, optimal_actions

def run_ucb(bandit, c=2, n_steps=1000):
    Q = np.zeros(bandit.k)
    N = np.zeros(bandit.k)
    rewards = []; optimal_actions = []
    best_action = np.argmax(bandit.q_star)

    for t in range(n_steps):
        if 0 in N:
            action = np.argmin(N)
        else:
            ucb = Q + c * np.sqrt(np.log(t+1) / N)
            action = np.argmax(ucb)
        reward = bandit.pull(action)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        rewards.append(reward)
        optimal_actions.append(1 if action == best_action else 0)
    return rewards, optimal_actions

def run_thompson(bandit, n_steps=1000):
    alpha = np.ones(bandit.k)
    beta = np.ones(bandit.k)
    rewards = []; optimal_actions = []
    best_action = np.argmax(bandit.q_star)

    for t in range(n_steps):
        samples = np.random.beta(alpha, beta)
        action = np.argmax(samples)
        reward = bandit.pull(action)
        # Convert to 0/1 for Beta update
        success = 1 if reward > 0 else 0
        alpha[action] += success
        beta[action] += 1 - success
        rewards.append(reward)
        optimal_actions.append(1 if action == best_action else 0)
    return rewards, optimal_actions

# Run experiments
n_runs = 2000; n_steps = 1000
methods = {
    'ε=0 (greedy)': lambda b: run_epsilon_greedy(b, 0),
    'ε=0.01': lambda b: run_epsilon_greedy(b, 0.01),
    'ε=0.1': lambda b: run_epsilon_greedy(b, 0.1),
    'UCB c=2': lambda b: run_ucb(b, c=2),
    'Thompson': lambda b: run_thompson(b),
}

results = {name: {'rewards': np.zeros(n_steps), 'optimal': np.zeros(n_steps)} for name in methods}

for _ in range(n_runs):
    bandit = Bandit(10)
    for name, fn in methods.items():
        r, o = fn(bandit)
        results[name]['rewards'] += np.array(r)
        results[name]['optimal'] += np.array(o)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
colors = ['gray', 'blue', 'green', 'red', 'purple']
for i, (name, data) in enumerate(results.items()):
    axes[0].plot(data['rewards']/n_runs, label=name, color=colors[i], alpha=0.8)
    axes[1].plot(data['optimal']/n_runs*100, label=name, color=colors[i], alpha=0.8)

axes[0].set_ylabel('Average Reward'); axes[0].set_title('Average Reward over Time')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel('Steps'); axes[1].set_ylabel('% Optimal Action')
axes[1].set_title('Optimal Action Selection Rate')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""),
md("""## Summary

| Method | Exploration Mechanism |
|--------|---------------------|
| **ε-greedy** | Random action with probability ε |
| **Softmax** | Temperature-controlled randomization |
| **UCB** | Confidence-based optimism |
| **Thompson** | Posterior sampling |
| **Intrinsic motivation** | Curiosity/novelty bonus |

**Next:** Notebook 19 — Multi-Armed Bandits
"""),
])

save("19_Multi_Armed_Bandits", [
md("""# Notebook 19: Multi-Armed Bandits

**Series:** Reinforcement Learning Basics (19/20)

---

## 1. The Multi-Armed Bandit Problem

$K$ slot machines ("arms"), each with unknown reward distribution. Goal: maximize total reward over $T$ pulls.

**Formally:** Maximize $\\mathbb{E}\\left[\\sum_{t=1}^T R_t\\right]$

### Regret
$$L_T = T \\cdot \\mu^* - \\mathbb{E}\\left[\\sum_{t=1}^T R_t\\right]$$

where $\\mu^* = \\max_k \\mu_k$ is the best arm's mean reward.

**Goal:** Minimize regret = find the best arm quickly.

## 2. Action-Value Estimation

**Sample average:** $Q_t(a) = \\frac{\\sum_{i=1}^{t-1} R_i \\cdot \\mathbb{1}(A_i = a)}{N_t(a)}$

**Incremental update:** $Q_{n+1}(a) = Q_n(a) + \\frac{1}{n}[R_n - Q_n(a)]$

**Non-stationary (exponential recency):**
$$Q_{n+1}(a) = Q_n(a) + \\alpha[R_n - Q_n(a)]$$
$$= (1-\\alpha)^n Q_1 + \\sum_{i=1}^n \\alpha(1-\\alpha)^{n-i} R_i$$

## 3. Gradient Bandit Algorithm

Learn **preferences** $H_t(a)$ instead of values:

$$\\pi_t(a) = \\frac{e^{H_t(a)}}{\\sum_b e^{H_t(b)}}$$

**Update:**
$$H_{t+1}(a) = H_t(a) + \\alpha(R_t - \\bar{R}_t)(\\mathbb{1}(A_t = a) - \\pi_t(a))$$

where $\\bar{R}_t$ is the average reward baseline.

## 4. Contextual Bandits

State/context $s_t$ is observed before choosing action. Bridge between bandits and full RL.

$$a_t = \\arg\\max_a Q(s_t, a)$$

Applications: news recommendation, ad placement, clinical trials.
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Comprehensive Bandit Comparison
# ============================================
class KArmedBandit:
    def __init__(self, k=10, stationary=True):
        self.k = k
        self.q_star = np.random.randn(k)
        self.stationary = stationary

    def pull(self, action):
        if not self.stationary:
            self.q_star += np.random.randn(self.k) * 0.01
        return np.random.randn() + self.q_star[action]

    def optimal_action(self):
        return np.argmax(self.q_star)

def gradient_bandit(bandit, alpha=0.1, n_steps=1000):
    H = np.zeros(bandit.k)
    avg_reward = 0
    rewards = []; optimal = []

    for t in range(n_steps):
        probs = np.exp(H - np.max(H))
        probs /= probs.sum()
        action = np.random.choice(bandit.k, p=probs)
        reward = bandit.pull(action)
        avg_reward += (reward - avg_reward) / (t + 1)

        one_hot = np.zeros(bandit.k)
        one_hot[action] = 1
        H += alpha * (reward - avg_reward) * (one_hot - probs)

        rewards.append(reward)
        optimal.append(1 if action == bandit.optimal_action() else 0)
    return rewards, optimal

def optimistic_greedy(bandit, Q_init=5.0, alpha=0.1, n_steps=1000):
    Q = np.full(bandit.k, Q_init)
    N = np.zeros(bandit.k)
    rewards = []; optimal = []

    for t in range(n_steps):
        action = np.argmax(Q)
        reward = bandit.pull(action)
        N[action] += 1
        Q[action] += alpha * (reward - Q[action])
        rewards.append(reward)
        optimal.append(1 if action == bandit.optimal_action() else 0)
    return rewards, optimal

# Run all methods
n_runs = 2000; n_steps = 1000
all_methods = {
    'ε-greedy (ε=0.1)': lambda b: (lambda: __import__('__main__').run_eps(b, 0.1))(),
    'Optimistic (Q₀=5)': lambda b: optimistic_greedy(b, Q_init=5.0),
    'UCB (c=2)': lambda b: (lambda: __import__('__main__').run_ucb_b(b, 2))(),
    'Gradient (α=0.1)': lambda b: gradient_bandit(b, alpha=0.1),
}

# Inline helpers
def run_eps(b, eps):
    Q=np.zeros(b.k);N=np.zeros(b.k);rs=[];os=[]
    for t in range(1000):
        a=np.random.randint(b.k) if np.random.random()<eps else np.argmax(Q)
        r=b.pull(a);N[a]+=1;Q[a]+=(r-Q[a])/N[a];rs.append(r)
        os.append(1 if a==b.optimal_action() else 0)
    return rs,os

def run_ucb_b(b, c):
    Q=np.zeros(b.k);N=np.zeros(b.k);rs=[];os=[]
    for t in range(1000):
        if 0 in N: a=np.argmin(N)
        else: a=np.argmax(Q+c*np.sqrt(np.log(t+1)/N))
        r=b.pull(a);N[a]+=1;Q[a]+=(r-Q[a])/N[a];rs.append(r)
        os.append(1 if a==b.optimal_action() else 0)
    return rs,os

methods = {
    'ε-greedy (ε=0.1)': lambda b: run_eps(b, 0.1),
    'Optimistic (Q₀=5)': lambda b: optimistic_greedy(b, Q_init=5.0),
    'UCB (c=2)': lambda b: run_ucb_b(b, 2),
    'Gradient (α=0.1)': lambda b: gradient_bandit(b, alpha=0.1),
}

results = {n: {'rewards': np.zeros(n_steps), 'optimal': np.zeros(n_steps)} for n in methods}
for _ in range(n_runs):
    bandit = KArmedBandit(10)
    for name, fn in methods.items():
        r, o = fn(bandit)
        results[name]['rewards'] += np.array(r)
        results[name]['optimal'] += np.array(o)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
colors = ['blue', 'orange', 'red', 'green']
for i, (name, data) in enumerate(results.items()):
    axes[0].plot(data['rewards']/n_runs, label=name, color=colors[i], linewidth=1.5)
    axes[1].plot(data['optimal']/n_runs*100, label=name, color=colors[i], linewidth=1.5)

axes[0].set_ylabel('Average Reward'); axes[0].set_title('10-Armed Bandit Testbed')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel('Steps'); axes[1].set_ylabel('% Optimal Action')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Regret comparison
fig, ax = plt.subplots(figsize=(10, 5))
for i, (name, data) in enumerate(results.items()):
    avg_rewards = data['rewards'] / n_runs
    # Estimate optimal reward per step (approximately the max q_star mean)
    cumulative_regret = np.cumsum(1.5 - avg_rewards)  # rough optimal
    ax.plot(cumulative_regret, label=name, color=colors[i], linewidth=2)
ax.set_xlabel('Steps'); ax.set_ylabel('Cumulative Regret')
ax.set_title('Cumulative Regret Comparison')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""),
md("""## Summary

| Method | Regret Bound | Key Idea |
|--------|-------------|----------|
| **ε-greedy** | $O(\\varepsilon T)$ | Fixed exploration rate |
| **Optimistic** | Depends on init | Encourages initial exploration |
| **UCB** | $O(\\sqrt{KT \\ln T})$ | Confidence-based exploration |
| **Thompson** | $O(\\sqrt{KT \\ln T})$ | Posterior sampling |
| **Gradient** | — | Learn preferences via gradient |

**Next:** Notebook 20 — Model-Based RL
"""),
])

save("20_Model_Based_Reinforcement_Learning", [
md("""# Notebook 20: Model-Based Reinforcement Learning

**Series:** Reinforcement Learning Basics (20/20)

---

## 1. Model-Based vs Model-Free

| Aspect | Model-Free | Model-Based |
|--------|-----------|------------|
| **Learns** | Value/Policy directly | Environment model $\\hat{P}, \\hat{R}$ |
| **Sample efficiency** | Low | **High** |
| **Computation** | Low | Higher (planning) |
| **Model errors** | N/A | Can hurt if model is wrong |

## 2. Dyna Architecture (Sutton, 1991)

Three processes running simultaneously:
1. **Direct RL:** Learn from real experience (e.g., Q-learning update)
2. **Model Learning:** Update model $\\hat{P}(s'|s,a)$, $\\hat{R}(s,a)$ from real experience
3. **Planning:** Generate simulated experience from model, apply RL updates

### Dyna-Q Algorithm

```
For each real step:
  1. s, a → r, s' (real environment)
  2. Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]     # Direct RL
  3. Model(s,a) ← r, s'                                          # Model learning
  4. Repeat n times (planning):
       s_sim, a_sim ← random previously observed (s,a)
       r_sim, s'_sim ← Model(s_sim, a_sim)
       Q(s_sim, a_sim) ← Q + α[r_sim + γ max Q(s'_sim,.) - Q]  # Planning
```

## 3. Dyna-Q+

Adds exploration bonus for long-unvisited state-action pairs:

$$r_{\\text{bonus}} = \\kappa \\sqrt{\\tau(s, a)}$$

where $\\tau(s,a)$ = time steps since $(s,a)$ was last tried.

## 4. Monte Carlo Tree Search (MCTS)

Four phases per simulation:
1. **Selection:** Use UCB to traverse tree
2. **Expansion:** Add a new child node
3. **Simulation:** Random rollout to terminal
4. **Backpropagation:** Update statistics up the tree

Used in AlphaGo: neural network replaces random rollouts.

## 5. World Models

Learn a latent dynamics model:
$$z_{t+1} = f_\\theta(z_t, a_t)$$
$$\\hat{r}_{t+1} = g_\\theta(z_t, a_t)$$

Plan in latent space ("dreaming"). Examples: Dreamer, MBPO, PlaNet.
"""),
code("""import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Maze Environment
# ============================================
class Maze:
    def __init__(self):
        self.maze = np.zeros((6, 9))
        for r, c in [(1,2),(2,2),(3,2),(4,5),(0,7),(1,7),(2,7)]:
            self.maze[r, c] = 1
        self.start = (2, 0)
        self.goal = (0, 8)
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        r, c = self.state
        moves = [(-1,0),(0,1),(1,0),(0,-1)]
        dr, dc = moves[action]
        nr, nc = r+dr, c+dc
        if 0 <= nr < 6 and 0 <= nc < 9 and self.maze[nr, nc] == 0:
            self.state = (nr, nc)
        reward = 1.0 if self.state == self.goal else 0.0
        done = self.state == self.goal
        return self.state, reward, done

# ============================================
# Dyna-Q
# ============================================
def dyna_q(n_planning=5, n_episodes=50, alpha=0.1, gamma=0.95, epsilon=0.1):
    Q = {}
    model = {}  # model[(s,a)] = (r, s')
    steps_per_episode = []

    for ep in range(n_episodes):
        env = Maze()
        state = env.reset()
        steps = 0

        while True:
            steps += 1
            if state not in Q:
                Q[state] = np.zeros(4)

            # Epsilon-greedy
            if np.random.random() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done = env.step(action)
            if next_state not in Q:
                Q[next_state] = np.zeros(4)

            # Direct RL
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

            # Model learning
            model[(state, action)] = (reward, next_state)

            # Planning
            observed = list(model.keys())
            for _ in range(n_planning):
                s_sim, a_sim = observed[np.random.randint(len(observed))]
                r_sim, sn_sim = model[(s_sim, a_sim)]
                if s_sim not in Q: Q[s_sim] = np.zeros(4)
                if sn_sim not in Q: Q[sn_sim] = np.zeros(4)
                Q[s_sim][a_sim] += alpha * (r_sim + gamma * np.max(Q[sn_sim]) - Q[s_sim][a_sim])

            state = next_state
            if done or steps > 500:
                break

        steps_per_episode.append(steps)
    return steps_per_episode

# Compare different planning steps
np.random.seed(42)
results = {}
for n in [0, 5, 50]:
    all_steps = np.zeros(30)
    for _ in range(5):
        steps = dyna_q(n_planning=n, n_episodes=30)
        all_steps += np.array(steps)
    results[n] = all_steps / 5

plt.figure(figsize=(10, 6))
for n, steps in results.items():
    plt.plot(steps, linewidth=2, label=f'{n} planning steps')
plt.xlabel('Episode')
plt.ylabel('Steps per Episode')
plt.title('Dyna-Q: Effect of Planning Steps on Learning Speed')
plt.legend(fontsize=12)
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Key insight: More planning steps = faster learning!")
print(f"Episode 2 steps - 0 planning: {results[0][1]:.0f}, 5 planning: {results[5][1]:.0f}, 50 planning: {results[50][1]:.0f}")
"""),
md("""## Summary

| Method | Key Idea |
|--------|----------|
| **Dyna-Q** | Interleave real experience with simulated planning |
| **Dyna-Q+** | Bonus for unvisited states |
| **MCTS** | Tree search with UCB + rollouts |
| **World Models** | Learn latent dynamics, plan in dream space |

---

## Congratulations!

You've completed all 20 notebooks covering the foundations of Reinforcement Learning:

1. Introduction to RL
2. Markov Decision Processes
3. Bellman Equations
4. Dynamic Programming
5. Monte Carlo Methods
6. Temporal Difference Learning
7. SARSA
8. Q-Learning
9. N-Step Bootstrapping & Eligibility Traces
10. Function Approximation
11. Policy Gradient (REINFORCE)
12. Actor-Critic Methods
13. Advantage Functions & GAE
14. Proximal Policy Optimization (PPO)
15. Deep Q-Networks (DQN)
16. Double DQN & Dueling DQN
17. Experience Replay
18. Exploration vs Exploitation
19. Multi-Armed Bandits
20. Model-Based RL

**Recommended next steps:** Apply these to the advanced project notebooks in this repo!
"""),
])

print("\n✅ All 20 notebooks generated successfully!")
