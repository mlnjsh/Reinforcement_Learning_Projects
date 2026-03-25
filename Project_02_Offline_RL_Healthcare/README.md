# Project 02: Offline Reinforcement Learning for Healthcare

## What is Offline RL?

Offline Reinforcement Learning (also called **Batch RL**) is a way for an AI agent to learn the best decision-making strategy using only a **fixed dataset** of past experiences. Unlike regular RL, the agent **never interacts with the real environment** during training. It simply looks at historical data — records of what happened when certain actions were taken in certain situations — and tries to figure out the best policy from that data alone.

Think of it like a medical student studying thousands of patient case files to learn treatment strategies, without actually treating any patients during their study period.

## Why is Offline RL Critical for Healthcare?

In healthcare, you **cannot experiment on patients** just to train an AI. You cannot randomly assign treatments to see what works best — that would be dangerous and unethical. But hospitals have **years of electronic health records** containing:

- What condition patients were in (states)
- What treatments doctors prescribed (actions)
- How patients responded (rewards/outcomes)

Offline RL lets us use this treasure trove of historical data to discover treatment policies that could be **better than what any individual doctor might choose**, all without putting a single patient at risk.

**Applications include:**
- Sepsis treatment optimization in ICUs
- Chemotherapy dosing for cancer patients
- Ventilator management for respiratory patients
- Chronic disease management (diabetes, heart failure)

## How It Works — Step by Step

### Step 1: Collect Offline Data

Gather a dataset of past patient interactions:

```
Dataset D = {(s_t, a_t, r_t, s_{t+1})} from historical records
```

Where:
- **s_t** = patient state (vitals like heart rate, blood pressure, lab values)
- **a_t** = treatment action (medication type, dosage level)
- **r_t** = reward (health outcome — improvement or deterioration)
- **s_{t+1}** = next patient state after treatment

### Step 2: Learn a Policy from Data

Use an offline RL algorithm to learn the best treatment strategy without ever interacting with real patients.

### Step 3: Evaluate and Validate

Test the learned policy using off-policy evaluation methods before any real-world deployment.

## Algorithms Explained

### 1. Behavior Cloning (BC)

The simplest approach — just **imitate what doctors did** in the historical data.

**How it works:**
- Look at each (state, action) pair in the data
- Train a model to predict: "Given this patient state, what action did doctors take?"
- Use supervised learning (like training a classifier)

**Pros:** Simple, stable, easy to implement.

**Cons:** Can only be as good as the average doctor in the data. If doctors made mistakes, BC will copy those mistakes too.

```
Policy_BC(s) = argmax_a  P(a | s)  learned from data
```

### 2. Conservative Q-Learning (CQL)

CQL is a smarter approach that tries to find a **better policy than what's in the data**, while being careful not to overestimate the value of actions it hasn't seen.

**The key problem it solves:**
Regular Q-learning can get fooled by "out-of-distribution" actions — treatments that were rarely or never tried in the data. The Q-function might assign these unknown actions a very high value simply because there's no data to correct the estimate. This is dangerous in healthcare — recommending an untested treatment could harm patients.

**CQL's solution:** Add a penalty that **pushes down Q-values for actions the learned policy likes** but that **weren't commonly seen in the data**.

**The CQL Formula:**

```
Q_CQL = Q_standard - alpha * E_pi[Q(s,a)] + alpha * E_data[Q(s,a)]
```

Breaking this down:
- **Q_standard**: The regular Q-value from Bellman updates
- **alpha * E_pi[Q(s,a)]**: Penalty — pushes DOWN Q-values for actions the new policy would choose (discourages novel actions)
- **alpha * E_data[Q(s,a)]**: Bonus — pushes UP Q-values for actions actually seen in the data (trusts real experience)
- **alpha**: Controls how conservative the algorithm is (higher = more conservative = stays closer to data)

The net effect: CQL learns Q-values that are **lower bounds** on the true values, making the policy conservative and safe.

### 3. Batch-Constrained Q-Learning (BCQ)

BCQ takes a different approach — instead of penalizing Q-values, it **restricts which actions the policy can even consider**.

**How it works:**
1. Train a generative model to learn which actions appeared in the data for each state
2. When choosing an action, only consider actions that the generative model says are "plausible" (i.e., similar to actions in the data)
3. Among those plausible actions, pick the one with the highest Q-value

```
Policy_BCQ(s) = argmax_a  Q(s,a)   where a is in {actions similar to data}
```

**Analogy:** Instead of letting the AI suggest any treatment imaginable, BCQ says "only suggest treatments that are similar to what real doctors have actually prescribed, then pick the best among those."

## Key Concepts

### The Distribution Shift Problem

The core challenge in offline RL: the data was collected under one policy (what doctors actually did), but we want to learn a **different, better** policy. This mismatch is called "distribution shift" and can cause the learned Q-values to be wildly inaccurate for unseen state-action pairs.

### Conservative vs. Aggressive Policies

| Approach | Risk | Potential Upside |
|----------|------|-----------------|
| Behavior Cloning | Low (copies doctors) | Limited (can't improve) |
| CQL (high alpha) | Low (very conservative) | Moderate |
| CQL (low alpha) | Moderate | Higher |
| Unconstrained Q-Learning | High (may suggest untested treatments) | Potentially highest, but unreliable |

## Running the Interactive Demo

```bash
streamlit run app.py
```

The demo lets you:
- Generate synthetic patient treatment data
- Compare Behavior Cloning vs CQL policies
- See how CQL penalizes out-of-distribution actions
- Adjust dataset size, CQL conservatism (alpha), and number of treatments
- Visualize Q-value distributions, reward curves, and policy comparisons
