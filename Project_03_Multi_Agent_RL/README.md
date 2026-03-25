# Project 03: Multi-Agent Reinforcement Learning

## What is Multi-Agent RL?

Multi-Agent Reinforcement Learning (MARL) is the study of how **multiple agents** can learn to make decisions in a **shared environment**. Unlike single-agent RL where one agent learns alone, MARL involves several agents that observe, act, and learn simultaneously — and each agent's actions affect the others.

Think of it like a team sport: each player (agent) needs to figure out the best strategy, but what's "best" depends on what their teammates and opponents are doing.

## Types of Multi-Agent Interactions

### 1. Cooperative
All agents share the same goal and work together.

**Example:** A team of robots cleaning a warehouse — they all want to clean every aisle as fast as possible.

### 2. Competitive
Agents have opposing goals — one agent's gain is another's loss.

**Example:** Two players in a zero-sum game like chess or Go.

### 3. Mixed (Cooperative-Competitive)
Some agents cooperate while competing against others.

**Example:** Team-based games like football — players cooperate with teammates but compete against the opposing team.

## How It Works — Step by Step

### Step 1: Define the Multi-Agent Environment

Each agent `i` has:
- **Observation space**: What agent `i` can see (might be partial)
- **Action space**: What agent `i` can do
- **Reward function**: What agent `i` gets rewarded for

### Step 2: Each Agent Learns Independently (or Together)

At every time step:
1. Each agent observes the environment
2. Each agent picks an action (using its own policy)
3. All actions are executed simultaneously
4. The environment transitions to a new state
5. Each agent receives its own reward
6. Each agent updates its policy

### Step 3: Agents Adapt to Each Other

Over many episodes, agents learn to respond to each other's strategies, ideally converging to effective joint behavior.

## Algorithms Explained

### 1. Independent Q-Learning (IQL)

The simplest MARL approach — each agent runs its **own Q-learning algorithm** as if the other agents were just part of the environment.

**How it works:**
- Agent 1 maintains Q-table Q_1(s, a_1)
- Agent 2 maintains Q-table Q_2(s, a_2)
- Each agent updates its Q-values using standard Q-learning
- Each agent ignores the fact that other agents are also learning

```
Q_i(s, a_i) <- Q_i(s, a_i) + lr * [r_i + gamma * max_a' Q_i(s', a') - Q_i(s, a_i)]
```

**Pros:** Simple, scalable, no communication needed.

**Cons:** The environment appears non-stationary to each agent (because other agents keep changing their behavior), which violates Q-learning's assumptions and can cause instability.

### 2. QMIX

QMIX is a **cooperative** MARL algorithm that learns a **joint Q-function** by combining individual agent Q-values in a smart way.

**Key idea:** The total team Q-value is a monotonic mixing of individual Q-values:

```
Q_total = f(Q_1, Q_2, ..., Q_n)   where df/dQ_i >= 0 for all i
```

The monotonicity constraint ensures that if each agent acts to maximize its own Q-value, the joint action also maximizes the team Q-value. This makes decentralized execution possible — each agent can act using only its own Q-function.

### 3. MADDPG (Multi-Agent Deep Deterministic Policy Gradient)

MADDPG uses a "centralized training, decentralized execution" framework:

- **During training:** Each agent's critic sees the actions of ALL agents (centralized)
- **During execution:** Each agent's actor uses only its OWN observations (decentralized)

```
Critic_i: Q_i(s, a_1, a_2, ..., a_n)    <- sees everything during training
Actor_i:  pi_i(o_i)                       <- sees only own observation during execution
```

This allows agents to learn about each other during training, but still act independently at test time.

## Key Concepts

### Nash Equilibrium

A Nash Equilibrium is a state where **no agent can improve its reward by changing its own strategy alone**, assuming all other agents keep their strategies fixed.

```
For all agents i:
  J_i(pi_i*, pi_{-i}*) >= J_i(pi_i, pi_{-i}*)   for all pi_i
```

Where `pi_i*` is agent i's equilibrium policy and `pi_{-i}*` represents all other agents' equilibrium policies. In simpler terms: everyone is doing the best they can, given what everyone else is doing.

### Challenges in Multi-Agent RL

| Challenge | Description |
|-----------|-------------|
| **Non-Stationarity** | From each agent's perspective, the environment keeps changing because other agents are learning and changing their behavior |
| **Credit Assignment** | When the team succeeds or fails, which agent deserves credit or blame? |
| **Scalability** | Joint action space grows exponentially with number of agents |
| **Communication** | Should agents share information? How much? When? |
| **Partial Observability** | Each agent might only see part of the world |

### Coordination Strategies

- **Implicit coordination:** Agents learn to coordinate purely through shared rewards and repeated interaction
- **Explicit communication:** Agents send messages to each other (learned or hand-designed)
- **Role assignment:** Agents specialize in different sub-tasks

## The Predator-Prey Problem

This project implements a classic MARL benchmark: the **Predator-Prey** (or "pursuit") game.

- **Predators** (cooperative team): Must work together to surround and catch the prey
- **Prey**: Tries to escape the predators

This is a great testbed because:
- It requires **cooperation** between predators
- The prey creates **non-stationarity** (it learns to escape)
- Success requires solving the **credit assignment** problem
- It's visually intuitive and easy to understand

## Running the Interactive Demo

```bash
streamlit run app.py
```

The demo lets you:
- Watch predators learn to catch prey in a grid world
- Control grid size, number of agents, and training parameters
- Visualize agent trajectories and learned policies
- Track capture rate and cooperation metrics over training
