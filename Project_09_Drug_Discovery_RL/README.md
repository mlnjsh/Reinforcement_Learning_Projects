# Project 09: Drug Discovery with RL

## What is RL for Drug Discovery?

Reinforcement Learning for Drug Discovery uses an RL agent to **design molecules** that have desirable pharmaceutical properties. Instead of a human chemist manually proposing molecules one by one, the agent learns to *build* molecules step by step, guided by rewards that measure how "drug-like" and effective the molecule is.

Think of it as a builder with molecular LEGO blocks: the agent picks which atom or bond to add next, and gets a score based on whether the resulting molecule would make a good drug.

---

## MDP Formulation

Drug design is framed as a Markov Decision Process:

| MDP Component | Drug Discovery Mapping |
|---------------|----------------------|
| **State** | The partial molecule built so far (atoms + bonds) |
| **Action** | Add an atom (C, N, O, S, ...) or a bond type |
| **Reward** | Score based on drug properties (higher = better drug) |
| **Terminal** | Molecule is complete (reached target size or stop action) |

---

## Key Properties to Optimize

### 1. LogP (Lipophilicity)
How well a molecule dissolves in fats vs water. Ideal range: -0.4 to 5.6.

```
LogP ≈ Σ (atom contributions) + Σ (bond corrections)
```

- Too low → can't cross cell membranes
- Too high → poor water solubility, accumulates in fat tissue

### 2. Molecular Weight (MW)
Total mass of the molecule in Daltons.

- Ideal: < 500 Da (Lipinski's rule)
- Bigger molecules have trouble entering cells

### 3. Drug-Likeness (QED)
Quantitative Estimate of Drug-likeness — a single score from 0 to 1 combining multiple properties.

```
QED = exp(1/n × Σ ln(desirability_i))
```

Where each desirability function maps a property to [0, 1].

### 4. Toxicity
A penalty for structural features known to be toxic (reactive groups, etc.).

---

## Multi-Objective Optimization

Real drug design balances **multiple conflicting objectives**:

```
R_total = w₁ · R_LogP + w₂ · R_MW + w₃ · R_toxicity + w₄ · R_druglikeness
```

Where w₁, w₂, w₃, w₄ are weights that reflect the relative importance of each property.

The set of solutions where no property can be improved without worsening another is called the **Pareto front**.

---

## Lipinski's Rule of Five

A quick filter for oral drug candidates. A good drug should satisfy:

| Rule | Threshold |
|------|-----------|
| Molecular Weight | ≤ 500 Da |
| LogP | ≤ 5 |
| H-bond donors | ≤ 5 |
| H-bond acceptors | ≤ 10 |

Molecules violating more than one rule are unlikely to be orally active drugs.

---

## Algorithm: Policy Gradient for Molecular Generation

1. **Policy network** outputs a probability distribution over possible next atoms/bonds.
2. **Sample** an action (atom choice) from the policy.
3. **Build** the molecule step by step.
4. **Score** the completed molecule on all property objectives.
5. **Update** the policy using REINFORCE:

```
∇_θ J(θ) = E[Σ_t ∇_θ log π(a_t|s_t) · R]
```

Where R is the multi-objective reward for the completed molecule.

---

## How to Run the Interactive Demo

```bash
pip install streamlit numpy matplotlib
streamlit run app.py
```

The demo simulates a simplified molecular builder where an RL agent learns to construct "molecules" (sequences of atoms) that satisfy target property constraints, with interactive control over objectives and weights.
