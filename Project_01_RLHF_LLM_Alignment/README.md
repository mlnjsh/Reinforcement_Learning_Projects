# Project 01: RLHF — Reinforcement Learning from Human Feedback for LLM Alignment

## What is this Project About?

This project shows how **Reinforcement Learning from Human Feedback (RLHF)** is used to make language models (LLMs) behave the way humans want. Instead of just predicting the next word, we train a model to give helpful, harmless, and honest responses.

**Real-World Use:** This is exactly how ChatGPT, Claude, and other AI assistants are trained. After pre-training on text data, they use RLHF to align with human preferences.

## How Does RLHF Work? (Simple Explanation)

Think of it like training a dog:
1. **Pre-training** = The dog learns basic commands (the LLM learns language)
2. **Reward Model** = You decide what "good behavior" looks like (humans rank responses)
3. **RL Fine-tuning** = The dog practices and gets treats for good behavior (PPO optimizes the LLM)

## The 3 Steps of RLHF

### Step 1: Supervised Fine-Tuning (SFT)
- Take a pre-trained language model
- Fine-tune it on high-quality human-written examples
- **Input:** "What is gravity?" → **Output:** A clear, helpful explanation
- This gives us a decent starting model

### Step 2: Train a Reward Model
- Show the SFT model a prompt
- Generate multiple responses
- **Humans rank the responses** from best to worst
- Train a separate neural network (Reward Model) to predict these rankings
- **Formula:** The reward model learns: R(prompt, response) → score

### Step 3: PPO Fine-Tuning
- Use the reward model as a "judge"
- The LLM generates responses
- The reward model scores them
- **PPO (Proximal Policy Optimization)** updates the LLM to get higher scores
- **KL penalty** prevents the model from changing too much from the SFT model

## Key Algorithms

### PPO (Proximal Policy Optimization)
```
L_CLIP = min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)
```
- `r(θ)` = ratio of new policy to old policy probabilities
- `A` = advantage (how much better this response is than average)
- `ε` = clipping range (usually 0.2) — prevents too-large updates

### KL Divergence Penalty
```
Loss = -Reward + β * KL(π_new || π_sft)
```
- Keeps the new policy close to the SFT model
- Prevents "reward hacking" (finding loopholes in the reward model)

### Bradley-Terry Model (for Reward Model)
```
P(response_A > response_B) = σ(R(A) - R(B))
```
- Converts pairwise rankings into a reward function
- σ = sigmoid function

## What Our Streamlit App Does

1. **Simulates the RLHF pipeline** with a simple text generation model
2. **Visualize reward model training** — see how the model learns human preferences
3. **Interactive PPO training** — watch the policy improve in real-time
4. **Compare outputs** before and after RLHF alignment
5. **KL divergence tracking** — see the balance between improvement and stability

## Project Structure

```
Project_01_RLHF_LLM_Alignment/
├── README.md              ← You are here
├── app.py                 ← Streamlit interactive app
├── rlhf_notebook.ipynb    ← Detailed notebook with code
└── requirements.txt       ← Dependencies
```

## How to Run

```bash
pip install streamlit numpy matplotlib
cd Project_01_RLHF_LLM_Alignment
streamlit run app.py
```

## References
- Ouyang et al., "Training language models to follow instructions with human feedback" (2022)
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Christiano et al., "Deep RL from Human Preferences" (2017)
