# Project 06: RLAIF (Reinforcement Learning from AI Feedback)

## What Is RLAIF?

**RLHF** (Reinforcement Learning from Human Feedback) changed how we train language models — instead of just predicting the next word, models learn what humans actually prefer. But there's a problem: **human feedback is expensive, slow, and inconsistent**.

**RLAIF** (Reinforcement Learning from AI Feedback) solves this by replacing human annotators with an AI model. Instead of asking thousands of humans "Which response is better?", you ask a capable AI model to judge.

---

## How It Differs from RLHF

| Aspect | RLHF | RLAIF |
|--------|------|-------|
| **Who provides feedback?** | Human annotators | AI model (e.g., GPT-4, Claude) |
| **Cost** | $$$$ (human labor) | $ (API calls) |
| **Speed** | Slow (hours to days) | Fast (seconds) |
| **Consistency** | Variable (humans disagree) | Consistent (same model = same biases) |
| **Scalability** | Limited by human workforce | Virtually unlimited |
| **Bias type** | Random noise + individual biases | Systematic model biases |

---

## The RLAIF Process

### Step-by-Step:

```
Step 1: GENERATE
    → Model produces multiple candidate responses for each prompt

Step 2: AI RANKS
    → An AI judge evaluates and ranks the responses
    → "Response A is better than Response B because..."

Step 3: TRAIN REWARD MODEL
    → Use AI preference labels to train a reward model
    → reward_model(prompt, response) → score

Step 4: PPO OPTIMIZATION
    → Fine-tune the policy model using PPO with the learned reward
    → Maximize: E[reward_model(prompt, π(prompt))] - β · KL(π || π_ref)
```

### The Constitutional AI Approach (Anthropic, 2022)

Constitutional AI takes RLAIF further with **self-critique**:

1. **Generate**: Model produces a response.
2. **Critique**: The same model critiques its own response against a set of principles (the "constitution").
3. **Revise**: Model generates a revised response addressing the critique.
4. **Repeat**: Multiple rounds of critique-revise.
5. **Train**: Use the final revised responses as training signal.

```
Constitution Principles (examples):
  - "Please choose the response that is most helpful"
  - "Please choose the response that is least harmful"
  - "Please choose the response that is most honest"
```

---

## Key Formulas

**Reward Model Training:**
```
L_reward = -E[log σ(r_θ(x, y_w) - r_θ(x, y_l))]
```
Where `y_w` is the preferred response and `y_l` is the less preferred one (as judged by AI).

**PPO Objective with KL Penalty:**
```
J(π) = E[r_θ(x, y)] - β · KL[π(y|x) || π_ref(y|x)]
```
- Maximize reward while staying close to the reference policy.
- `β` controls the trade-off between reward and staying on-distribution.

**AI Judge Agreement:**
```
Agreement Rate = (# times AI and human agree) / (total comparisons)
```
Research shows AI judges agree with humans 80-90% of the time — comparable to human-human agreement.

---

## Advantages of RLAIF

1. **Scalability**: Generate millions of preference labels without human labor.
2. **Consistency**: No inter-annotator disagreement (though systematic biases exist).
3. **Cost**: Orders of magnitude cheaper than human annotation.
4. **Speed**: Label thousands of examples per minute.
5. **Iterative**: Easy to re-label data as the AI judge improves.

## Limitations

1. **Systematic Bias**: AI judges have blind spots (verbosity bias, sycophancy).
2. **Circular Training**: Risk of "model collapse" if training on own outputs.
3. **Ceiling Effect**: AI judge quality limits final model quality.

---

## Key Papers

- **Bai et al. (2022)** — "Constitutional AI: Harmlessness from AI Feedback" (Anthropic)
- **Lee et al. (2023)** — "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback" (Google)
- **Dubois et al. (2024)** — "Alpaca Farm: A Simulation Framework for Methods that Learn from Human Feedback"

---

## Running the Interactive Demo

```bash
pip install streamlit numpy matplotlib
streamlit run app.py
```

The demo lets you:
- Compare RLHF (noisy human labels) vs RLAIF (AI model labels)
- Adjust AI judge accuracy and human noise levels
- See how both approaches converge to a final policy
- Explore label agreement rates and reward curves

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
      <a href="https://github.com/lvwerra">
        <img src="https://github.com/lvwerra.png" width="80px;" alt="Leandro von Werra"/><br />
        <sub><b>Leandro von Werra</b></sub>
      </a><br />
      <sub>TRL Library, Hugging Face RL Team</sub>
    </td>
    <td align="center">
      <a href="https://github.com/edbeeching">
        <img src="https://github.com/edbeeching.png" width="80px;" alt="Edward Beeching"/><br />
        <sub><b>Edward Beeching</b></sub>
      </a><br />
      <sub>Hugging Face RL Researcher</sub>
    </td>
  </tr>
</table>
