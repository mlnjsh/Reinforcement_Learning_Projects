# Project 07: Hierarchical Reinforcement Learning

## What Is Hierarchical RL?

Imagine you want to travel from New York to a restaurant in San Francisco. You don't plan every single footstep — you think hierarchically:
- **High level**: Fly to San Francisco → Take a taxi → Walk to restaurant
- **Low level**: Each sub-task (walking, driving) uses its own policy for moment-to-moment decisions

**Hierarchical RL** applies the same idea to artificial agents. Instead of learning a single flat policy that maps every state to every possible action, we break the problem into layers:

- **High-level policy** (the "Manager"): Chooses **subgoals** or **options** — "Go to Room 2", "Pick up the key"
- **Low-level policy** (the "Worker"): Executes primitive actions to achieve each subgoal — "move up", "move right"

---

## The Options Framework

The **Options Framework** (Sutton, Precup, Singh, 1999) formalizes temporal abstraction in RL.

An **option** `o` is defined by three components:

```
Option o = (I, π, β)

I = Initiation Set    → Which states can this option start in?
π = Intra-option Policy → What actions to take while the option runs?
β = Termination Condition → When does this option end?
```

### Example: Four-Room Grid World

| Option | Initiation | Policy | Termination |
|--------|-----------|--------|-------------|
| "Go to Room 2" | Any state in Room 1 | Navigate to doorway | Reach Room 2 |
| "Go to Room 3" | Any state in Room 2 | Navigate to doorway | Reach Room 3 |
| "Go to Goal" | Any state in Room 4 | Navigate to goal | Reach goal |

### Option Value Function:

```
V(s) = max_o [R(s, o) + γ^τ · V(s')]
```

Where:
- `o` is an option (temporal abstraction over multiple steps)
- `R(s, o)` is the cumulative reward while executing option `o` from state `s`
- `τ` is the number of time steps the option takes
- `γ^τ` discounts over the duration of the option

---

## Goal-Conditioned Policies

Instead of fixed options, we can use **goal-conditioned policies**:

```
π(a | s, g) — choose action a given state s and goal g
```

The high-level policy selects goals `g`, and the low-level policy executes `π(a|s,g)` to reach them.

**Hindsight relabeling**: Even if the agent fails to reach goal `g`, it reached *some* state `s'` — relabel that trajectory as if `s'` was the intended goal. This dramatically improves learning efficiency.

---

## Feudal Networks

The **Feudal RL** architecture (Dayan & Hinton, 1993; Vezhnevets et al., 2017) uses a Manager-Worker hierarchy:

```
┌─────────────────────────┐
│    MANAGER (High-Level)  │
│  Observes state s_t      │
│  Sets goal g_t           │
│  Operates at slower      │
│  timescale (every c steps)│
├─────────────────────────┤
│    WORKER (Low-Level)    │
│  Observes state s_t      │
│  Receives goal g_t       │
│  Outputs action a_t      │
│  Operates every step     │
└─────────────────────────┘
```

- **Manager reward**: Based on task completion
- **Worker reward**: Based on reaching the Manager's subgoal (intrinsic reward)

---

## Benefits of Hierarchical RL

| Benefit | Explanation |
|---------|-------------|
| **Temporal Abstraction** | Plan over long time horizons using fewer decisions |
| **Transfer Learning** | Low-level skills transfer across tasks |
| **Exploration** | Subgoals provide directed exploration instead of random actions |
| **Credit Assignment** | Easier to figure out what went wrong at which level |
| **Interpretability** | Subgoal sequences are human-readable plans |

---

## How Flat vs Hierarchical Compare

### Flat Q-Learning
- State space: Every cell in the grid
- Action space: Up, Down, Left, Right
- Challenge: Reward is sparse (only at the goal), so exploration is hard
- Steps to learn: O(|S| x |A|) updates needed

### Hierarchical Approach
- **High level**: Choose which room/subgoal to target next
- **Low level**: Navigate within a room (much smaller problem)
- Steps to learn: O(|rooms| x |subgoals|) + O(|room_size| x |A|)
- Much faster because each sub-problem is smaller

---

## Key Algorithms

1. **Option-Critic** (Bacon et al., 2017) — Learns options end-to-end with gradient descent
2. **HIRO** (Nachum et al., 2018) — Goal-conditioned hierarchical RL for continuous control
3. **HAM** (Parr & Russell, 1998) — Hierarchies of Abstract Machines
4. **MAXQ** (Dietterich, 2000) — Decomposes value function hierarchically

---

## Running the Interactive Demo

```bash
pip install streamlit numpy matplotlib
streamlit run app.py
```

The demo lets you:
- Explore a four-room grid world
- Compare flat Q-learning vs hierarchical approach
- See subgoal selection heatmaps
- Adjust grid size, room layout, and subgoal positions
