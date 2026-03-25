# Project 08: Meta-RL (Learning to Learn)

## What is Meta-RL?

Meta-Reinforcement Learning is the idea of **training agents that can quickly adapt to new tasks**. Instead of training a separate agent from scratch for every new problem, Meta-RL produces an agent that has "learned how to learn" — it can pick up a brand-new task in just a handful of episodes.

Think of it like this: a regular RL agent is a student who memorizes answers for one specific exam. A Meta-RL agent is a student who has learned **how to study**, so they can ace any new exam after just a few minutes of preparation.

---

## The Key Idea: Learn the Learning Algorithm Itself

In standard RL, we optimize a policy for a single task. In Meta-RL, we optimize across a **distribution of tasks** so the agent internalizes a general learning strategy.

The training has two nested loops:

| Loop | What It Does | Analogy |
|------|-------------|---------|
| **Inner Loop** | Adapt to a specific task using a few gradient steps | Cramming for one exam |
| **Outer Loop** | Update the meta-parameters so inner-loop adaptation works better across all tasks | Getting better at studying in general |

---

## Core Algorithms

### 1. MAML (Model-Agnostic Meta-Learning)

MAML finds an **initialization** of the neural network weights such that a few gradient steps on any new task produce a good policy.

**How MAML works step by step:**

1. Start with meta-parameters θ (shared across all tasks).
2. **Inner loop** — For each task τ_i, take one (or a few) gradient steps:
   - θ'_i = θ - α · ∇_θ L_τi(θ)
3. **Outer loop** — Update θ using the loss *after* the inner-loop adaptation:
   - θ ← θ - β · Σ_i ∇_θ L_τi(θ'_i)

**The MAML formula (compact form):**

```
θ* = θ - α ∇_θ L_task(θ - α ∇_θ L_task(θ))
```

This is a "gradient through a gradient" — the outer update accounts for the fact that the inner update itself depends on θ.

### 2. RL² (RL-Squared)

RL² takes a different approach: it uses a **recurrent neural network** (like an LSTM) as the policy. The hidden state of the RNN effectively *becomes* the learning algorithm.

**How RL² works:**

1. The agent receives (state, action, reward, done) as input at each step.
2. The RNN hidden state accumulates experience from the current task.
3. Over an episode, the hidden state encodes a task-specific strategy.
4. Training across many tasks teaches the RNN to be a fast learner.

The beauty of RL² is that no explicit inner-loop gradient is needed — the RNN *implicitly* implements the adaptation through its recurrent dynamics.

---

## Few-Shot Adaptation

The hallmark of Meta-RL is **few-shot adaptation**: the ability to learn a new task from very few episodes (often just 1-5).

```
Regular RL:    Task → 10,000 episodes → Good policy
Meta-RL:       Task → 5 episodes → Good policy
               (after meta-training on many tasks)
```

This works because the meta-learner has already extracted the **structure shared across tasks**. It only needs a few samples to figure out which specific task it is facing.

---

## Why Does This Matter?

| Problem | How Meta-RL Helps |
|---------|-------------------|
| New environments | Adapt in minutes, not hours |
| Personalization | Quickly customize to a new user |
| Robotics | Learn new manipulation tasks from a few demonstrations |
| Game AI | Adapt to new opponents or rule changes on the fly |

---

## Mathematical Summary

Given a task distribution p(τ):

- **Inner update:** θ'_τ = θ - α ∇_θ L_τ(f_θ)
- **Meta-objective:** min_θ Σ_{τ~p(τ)} L_τ(f_{θ'_τ})
- **Outer update:** θ ← θ - β ∇_θ Σ_{τ~p(τ)} L_τ(f_{θ'_τ})

Where:
- α = inner learning rate (task-specific adaptation speed)
- β = outer (meta) learning rate
- L_τ = loss on task τ
- f_θ = model parameterized by θ

---

## How to Run the Interactive Demo

```bash
pip install streamlit numpy matplotlib
streamlit run app.py
```

The demo simulates Meta-RL on multi-armed bandit tasks, showing how a meta-learner adapts much faster than a regular learner when encountering new tasks.
