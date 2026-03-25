# Project 05: World Models

## What Are World Models?

Imagine you're learning to ride a bicycle. After a few tries, you start to build a **mental model** of how the bicycle behaves — if you lean left, you'll turn left; if you pedal harder, you'll go faster. You can even *imagine* what would happen if you tried something new without actually doing it. **World Models** in Reinforcement Learning work exactly the same way.

A World Model is a learned representation of how the environment works. Instead of only learning from real interactions (which can be slow and expensive), the agent learns a model of the environment's dynamics and then **plans inside its own imagination**.

---

## How It Works

### The Architecture

A World Model has three main components:

```
Real Observation → [Encoder] → Latent State (z_t)
                                    ↓
                    [Latent Dynamics Model] → Next State (z_{t+1})
                                    ↓
                    [Reward Predictor] → Predicted Reward (r_hat)
```

1. **Encoder**: Compresses high-dimensional observations (images, sensor data) into a compact latent representation `z_t`.
2. **Latent Dynamics Model**: Predicts how the latent state changes given an action — this is the "world model" itself.
3. **Reward Predictor**: Estimates the reward the agent will receive for a given state-action pair.

### Key Formulas

**State Transition (Dynamics Model):**
```
z_{t+1} = f_θ(z_t, a_t)
```
Given the current latent state `z_t` and action `a_t`, predict the next latent state `z_{t+1}`.

**Reward Prediction:**
```
r_hat = g_θ(z_t, a_t)
```
Given the current state and action, predict the expected reward.

**Model Loss (trained via supervised learning on real experience):**
```
L_dynamics = ||z_{t+1} - f_θ(z_t, a_t)||²
L_reward   = ||r_t - g_θ(z_t, a_t)||²
```

---

## Dyna-Style Planning

The **Dyna** architecture (Sutton, 1991) is the classic approach to combining model-free and model-based learning:

### Steps:

1. **Interact** with the real environment and collect experience `(s, a, r, s')`.
2. **Update the model**: Train the dynamics model `f_θ` and reward model `g_θ` on real experience.
3. **Plan in imagination**: Use the learned model to generate *simulated* experience:
   - Pick a previously visited state
   - Simulate several steps forward using the model
   - Update the policy/value function using these simulated transitions
4. **Repeat**: Each real step can be augmented with many simulated steps.

```
For each real step:
    1. Take action a in real environment → get (s, a, r, s')
    2. Update Q(s, a) with real experience
    3. Update model: f_θ, g_θ
    4. For k planning steps:
        - Sample s_sim from past states
        - Simulate: s'_sim = f_θ(s_sim, a_sim), r_sim = g_θ(s_sim, a_sim)
        - Update Q(s_sim, a_sim) with simulated experience
```

---

## Why World Models Matter

| Benefit | Explanation |
|---------|-------------|
| **Sample Efficiency** | Learn from fewer real interactions by generating simulated experience |
| **Planning** | Look ahead multiple steps before committing to an action |
| **Safety** | Test risky actions in imagination before trying them in reality |
| **Transfer** | A good model can be reused across different tasks in the same environment |

---

## Key Algorithms

### Dreamer (Hafner et al., 2020)
- Learns a world model in latent space
- Trains actor-critic entirely from imagined trajectories
- Achieves human-level performance on Atari from pixels

### PlaNet (Hafner et al., 2019)
- Planning with learned dynamics in latent space
- Uses cross-entropy method (CEM) for action selection
- No policy network — plans online at each step

### MBPO — Model-Based Policy Optimization (Janner et al., 2019)
- Uses short model-generated rollouts to augment real data
- Balances model accuracy with rollout length
- Achieves state-of-the-art sample efficiency on continuous control

---

## The Model Accuracy Challenge

The biggest risk with world models is **model error compounding**. Small prediction errors accumulate over long rollouts:

```
Step 1: small error ε
Step 2: error grows to ~2ε
Step k: error grows to ~kε (or worse, exponentially)
```

Solutions include:
- **Short rollouts** (MBPO uses 1-5 step rollouts)
- **Ensemble models** to estimate uncertainty
- **Adaptive rollout length** based on model confidence

---

## Running the Interactive Demo

```bash
pip install streamlit numpy matplotlib
streamlit run app.py
```

The demo lets you:
- Compare model-free vs model-based learning curves
- See how model prediction accuracy improves over training
- Adjust planning horizon and model update frequency
- Visualize planned vs actual trajectories
