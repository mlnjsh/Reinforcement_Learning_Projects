# Project 10: Sim-to-Real Transfer

## What is Sim-to-Real?

Sim-to-Real transfer is the practice of **training a policy in simulation and then deploying it in the real world**. Simulation is cheap, fast, and safe — you can run millions of episodes without breaking a physical robot. But simulation is never perfect, so a policy trained purely in simulation often fails when transferred to reality.

The central challenge is called the **Reality Gap**: the differences between simulated and real-world physics, sensing, and dynamics.

---

## The Reality Gap

| Simulation | Real World |
|-----------|------------|
| Perfect physics equations | Friction, wear, imperfections |
| No sensor noise | Noisy cameras, encoders, IMUs |
| Exact state measurements | Partial observability |
| Instant reset | Time-consuming resets |
| Deterministic (or controlled randomness) | Unpredictable disturbances |

A policy that works perfectly in simulation may completely fail in reality because it has **overfit to the simulator's specific dynamics**.

---

## Solution 1: Domain Randomization

The key idea: **if you train on a wide enough variety of simulated environments, the real world becomes just another variation**.

During training, randomly vary simulation parameters:
- Gravity, friction, mass
- Sensor noise levels
- Visual textures and lighting
- Actuator delays and damping

The policy learns to be **robust** to all these variations. When deployed in the real world (which has its own specific parameter values), the policy generalizes because it has already seen similar conditions.

### The Domain Randomization Objective

```
pi* = argmax_pi  E_{xi ~ P(xi)} [ J(pi, xi) ]
```

Where:
- pi = the policy
- xi = domain parameters (gravity, friction, noise, etc.)
- P(xi) = distribution over domain parameters (the randomization range)
- J(pi, xi) = expected return of policy pi in environment with parameters xi

The optimal policy maximizes **average performance across all possible domain configurations**.

---

## Solution 2: System Identification

Instead of randomizing, **measure or estimate the real-world parameters** and then configure the simulation to match.

Steps:
1. Collect data from the real system (e.g., drop an object, measure trajectories).
2. Fit simulation parameters (mass, friction, damping) to match real data.
3. Train the policy in the calibrated simulation.

```
xi* = argmin_xi || trajectory_real - trajectory_sim(xi) ||^2
```

This is more sample-efficient than domain randomization but requires real-world data collection.

---

## Comparing Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **Sim-only** | Fast, cheap | Brittle, reality gap |
| **Domain Randomization** | Robust, no real data needed | Conservative policy, wide training |
| **System Identification** | Accurate match | Needs real-world data |
| **Real-only** | No gap | Expensive, slow, dangerous |

---

## Applications

- **Robotics**: Train manipulation policies in simulation, deploy on physical arms
- **Autonomous Vehicles**: Train driving in simulated cities, transfer to real cars
- **Manufacturing**: Optimize assembly processes in digital twins
- **Drones**: Learn flight control in simulation, fly real quadrotors

---

## Key Formulas

### Transfer Performance Gap
```
Gap = J_real(pi_sim) - J_sim(pi_sim)
```
The difference between how well the policy performs in the real world vs simulation.

### Robustness (worst-case performance)
```
J_robust = min_{xi in Xi} J(pi, xi)
```
A robust policy maximizes its *worst-case* performance across all possible domain configurations.

---

## How to Run the Interactive Demo

```bash
pip install streamlit numpy matplotlib
streamlit run app.py
```

The demo simulates a CartPole-like balancing task with clean (sim) and noisy (real) dynamics. Compare sim-only training, domain-randomized training, and real training to see how domain randomization closes the reality gap.
