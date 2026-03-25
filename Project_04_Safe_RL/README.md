# Project 04: Safe Reinforcement Learning

## What is Safe RL?

Safe Reinforcement Learning is about training AI agents that **maximize performance while respecting safety constraints**. A regular RL agent only cares about getting the highest reward — it might take dangerous shortcuts if they lead to more reward. A Safe RL agent, on the other hand, learns to avoid dangerous behaviors even if they would be rewarding.

Think of it like teaching someone to drive: you don't just want them to get to the destination quickly (maximize reward) — you also need them to follow traffic rules, avoid pedestrians, and stay within speed limits (satisfy safety constraints).

## Why Does Safety Matter?

In many real-world applications, an RL agent that ignores safety constraints could cause serious harm:

| Domain | Reward | Safety Constraint |
|--------|--------|-------------------|
| **Autonomous Driving** | Reach destination quickly | Don't hit pedestrians, stay in lane |
| **Robotics** | Complete task efficiently | Don't exceed joint torque limits |
| **Medical Devices** | Optimize treatment | Don't exceed safe dosage levels |
| **Power Grids** | Maximize efficiency | Keep voltage within safe range |
| **Trading Bots** | Maximize profit | Limit maximum drawdown / risk |

In all these cases, a single safety violation could be catastrophic — you can't just "learn from mistakes" when mistakes can be fatal.

## How It Works — Step by Step

### Step 1: Define a Constrained MDP (CMDP)

A regular MDP has states, actions, transitions, and rewards. A **Constrained MDP** adds one extra element — a **cost function** and a **cost budget**.

```
CMDP = (S, A, P, R, C, d)
```

- **S**: States (e.g., robot position)
- **A**: Actions (e.g., move directions)
- **P**: Transition probabilities
- **R**: Reward function (what we want to maximize)
- **C**: Cost function (measures safety violations — e.g., 1 if entering danger zone, 0 otherwise)
- **d**: Cost threshold / budget (maximum acceptable expected cost)

**The goal:**
```
Maximize   E[sum of rewards]
Subject to E[sum of costs] <= d
```

We want the highest reward possible, but the expected total cost must stay below the threshold `d`.

### Step 2: Convert to Unconstrained Problem Using Lagrangian Relaxation

Directly solving a constrained optimization problem is hard. The **Lagrangian method** converts it into an unconstrained problem by adding the constraint as a penalty term.

**The Lagrangian Formula:**

```
L(theta, lambda) = J(theta) - lambda * (J_c(theta) - d)
```

Where:
- **J(theta)**: Expected total reward under policy theta (we want this HIGH)
- **J_c(theta)**: Expected total cost under policy theta (we want this BELOW d)
- **lambda**: Lagrange multiplier (a learnable parameter that controls the penalty)
- **d**: Cost threshold (our safety budget)

**How to read this formula:**
- If `J_c(theta) > d` (policy is too unsafe), the penalty term `lambda * (J_c(theta) - d)` is positive, which **reduces** the Lagrangian. This pushes the policy to be safer.
- If `J_c(theta) < d` (policy is within budget), the penalty term is negative, which **increases** the Lagrangian. This allows the policy to take slightly more risk for more reward.
- **lambda** automatically adjusts: it **increases** when the policy is too unsafe (strengthening the penalty) and **decreases** when the policy is safe enough (relaxing the penalty).

### Step 3: Solve with Primal-Dual Optimization

We alternate between two updates:

**Primal update** (improve the policy):
```
theta <- theta + lr_theta * gradient of L with respect to theta
```
Make the policy better at getting reward while respecting the cost penalty.

**Dual update** (adjust the constraint strength):
```
lambda <- max(0, lambda + lr_lambda * (J_c(theta) - d))
```
If costs are too high, increase lambda (more penalty). If costs are within budget, decrease lambda (less penalty).

## Algorithms Explained

### 1. Unconstrained Q-Learning (Baseline)

Standard Q-learning that **ignores safety entirely**. It only maximizes reward.

```
Q(s, a) <- Q(s, a) + lr * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

This agent will happily walk through danger zones if it's the shortest path to the goal. It serves as a baseline to show why safety constraints are needed.

### 2. Lagrangian Q-Learning (Safe)

Modifies the reward signal to include a cost penalty:

```
Modified reward = r - lambda * c
```

Where:
- `r` is the original reward
- `c` is the cost (1 if in danger zone, 0 otherwise)
- `lambda` is dynamically adjusted based on constraint satisfaction

The Q-function then learns using this modified reward:
```
Q(s, a) <- Q(s, a) + lr * [(r - lambda * c) + gamma * max_a' Q(s', a') - Q(s, a)]
```

And lambda is updated after each episode:
```
lambda <- max(0, lambda + lr_lambda * (episode_cost - d))
```

This agent learns to **balance reward and safety**, finding paths that are efficient but avoid danger zones.

### 3. Constrained Policy Optimization (CPO)

CPO is a more advanced algorithm that provides **theoretical guarantees** on constraint satisfaction at every policy update. It works by:

1. Computing the maximum step size that keeps the policy within the constraint boundary
2. Projecting the policy update onto the feasible set if it would violate constraints

While CPO is more complex to implement, the Lagrangian approach captures its core intuition and works well in practice.

## Key Concepts

### Safety vs. Performance Trade-off

There is always a trade-off between safety and reward:
- **Very safe (high lambda):** Agent takes long detours to avoid all danger, gets less reward
- **Moderate safety:** Agent avoids most danger while still being efficient
- **No safety (lambda = 0):** Agent takes the shortest/highest-reward path regardless of danger

The cost threshold `d` controls where you want to be on this spectrum.

### Constraint Satisfaction

A policy "satisfies the constraint" if:
```
Expected total cost per episode <= d (cost threshold)
```

The Lagrangian method doesn't guarantee this at every single episode, but it ensures this holds **on average** over many episodes.

### Hard vs. Soft Constraints

- **Hard constraints:** Must NEVER be violated (e.g., "never crash into a wall"). These require special methods like safety shields or constrained action spaces.
- **Soft constraints:** Violations are acceptable occasionally as long as the average stays below a threshold. The Lagrangian approach handles soft constraints.

## Running the Interactive Demo

```bash
streamlit run app.py
```

The demo lets you:
- Build a grid world with customizable danger zones
- Watch an unconstrained agent walk through danger vs. a safe agent that avoids it
- Adjust the cost threshold to control how cautious the agent is
- Visualize paths, reward curves, cost curves, and constraint satisfaction
- See how lambda evolves during training

---

## Contributors & Domain Experts

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/mlnjsh">
        <img src="https://github.com/mlnjsh.png" width="80px;" alt="Milan Amrut Joshi"/><br />
        <sub><b>Milan Amrut Joshi</b></sub>
      </a><br />
      <sub>Project Author</sub>
    </td>
    <td align="center">
      <a href="https://github.com/jachiam">
        <img src="https://github.com/jachiam.png" width="80px;" alt="Joshua Achiam"/><br />
        <sub><b>Joshua Achiam</b></sub>
      </a><br />
      <sub>Creator of CPO (Constrained Policy Optimization), OpenAI</sub>
    </td>
    <td align="center">
      <a href="https://github.com/alexray">
        <img src="https://github.com/alexray.png" width="80px;" alt="Alex Ray"/><br />
        <sub><b>Alex Ray</b></sub>
      </a><br />
      <sub>Safe RL Researcher, OpenAI, Co-author of Safety Gym</sub>
    </td>
  </tr>
</table>
