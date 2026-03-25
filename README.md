<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-Only-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Apps-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/github/stars/mlnjsh/Reinforcement_Learning_Projects?style=for-the-badge&color=gold" />
  <img src="https://img.shields.io/github/license/mlnjsh/Reinforcement_Learning_Projects?style=for-the-badge" />
</p>

<h1 align="center">Reinforcement Learning from Scratch</h1>

<p align="center">
  <strong>20 Theory Notebooks + 10 Real-World Projects + 10 Interactive Streamlit Apps</strong><br>
  <em>Every algorithm implemented from scratch using only NumPy and Matplotlib. No black boxes.</em>
</p>

<p align="center">
  <a href="#-notebooks">Notebooks</a> &bull;
  <a href="#-projects">Projects</a> &bull;
  <a href="#-quick-start">Quick Start</a> &bull;
  <a href="#-streamlit-apps">Streamlit Apps</a> &bull;
  <a href="#-learning-path">Learning Path</a>
</p>

---

## Why This Repo?

Most RL resources either drown you in theory or hand you a black-box library. **This repo does neither.**

- Every algorithm is **implemented from scratch** -- you see every gradient, every update, every Q-value
- Every concept comes with **LaTeX math** and **plain-English explanations** side by side
- Every project has an **interactive Streamlit app** you can play with in your browser
- **Zero dependency on OpenAI Gym, Stable Baselines, or any RL library** -- just NumPy

> *"What I cannot create, I do not understand."* -- Richard Feynman

---

## What's Inside

### 20 Theory Notebooks

Build your RL foundation from the ground up. Each notebook contains **detailed math**, **step-by-step derivations**, and **working code** with visualizations.

| # | Topic | Key Concepts |
|:-:|-------|-------------|
| 01 | **Introduction to RL** | Agent-Environment loop, Return, Discount factor |
| 02 | **Markov Decision Processes** | Markov property, MDP tuple, V/Q functions |
| 03 | **Bellman Equations** | Expectation & Optimality equations, Matrix form |
| 04 | **Dynamic Programming** | Policy Evaluation, Policy Iteration, Value Iteration |
| 05 | **Monte Carlo Methods** | First-Visit MC, Importance Sampling |
| 06 | **Temporal Difference Learning** | TD(0), Bias-Variance tradeoff, TD vs MC |
| 07 | **SARSA** | On-policy TD control, Expected SARSA |
| 08 | **Q-Learning** | Off-policy TD, Double Q-Learning |
| 09 | **N-Step & Eligibility Traces** | N-step returns, TD(lambda), Backward view |
| 10 | **Function Approximation** | Linear FA, Semi-gradient TD, Tile Coding |
| 11 | **Policy Gradient (REINFORCE)** | Policy Gradient Theorem, Baseline |
| 12 | **Actor-Critic** | TD error as advantage, A2C |
| 13 | **GAE** | Generalized Advantage Estimation |
| 14 | **PPO** | Clipped objective, KL penalty |
| 15 | **DQN** | Experience Replay, Target Networks |
| 16 | **Double & Dueling DQN** | Overestimation fix, V+A decomposition |
| 17 | **Experience Replay** | Uniform, Prioritized (SumTree), HER |
| 18 | **Exploration vs Exploitation** | UCB, Thompson Sampling, Intrinsic Motivation |
| 19 | **Multi-Armed Bandits** | Regret, Gradient Bandit, Contextual Bandits |
| 20 | **Model-Based RL** | Dyna-Q, MCTS, World Models |

### 10 Real-World Projects

Each project applies RL to a real domain with a **detailed notebook**, **interactive Streamlit app**, and **comprehensive documentation**.

| # | Project | Domain | Key Algorithm | App |
|:-:|---------|--------|--------------|:---:|
| 01 | **RLHF for LLM Alignment** | AI Safety | PPO + Bradley-Terry Reward Model | [Launch](#run-any-app) |
| 02 | **Offline RL for Healthcare** | Medicine | Conservative Q-Learning (CQL) | [Launch](#run-any-app) |
| 03 | **Multi-Agent RL** | Robotics | Independent Q-Learning, Predator-Prey | [Launch](#run-any-app) |
| 04 | **Safe RL** | Autonomous Systems | Lagrangian Constrained MDP | [Launch](#run-any-app) |
| 05 | **World Models** | Planning | Dyna-Q, Learned Dynamics | [Launch](#run-any-app) |
| 06 | **RLAIF** | AI Safety | AI Feedback vs Human Feedback | [Launch](#run-any-app) |
| 07 | **Hierarchical RL** | Navigation | Options Framework, Four Rooms | [Launch](#run-any-app) |
| 08 | **Meta-RL** | Few-Shot Learning | MAML, Learning to Learn | [Launch](#run-any-app) |
| 09 | **Drug Discovery** | Pharma | Multi-Objective Policy Gradient | [Launch](#run-any-app) |
| 10 | **Sim-to-Real Transfer** | Robotics | Domain Randomization | [Launch](#run-any-app) |

---

## Quick Start

### Prerequisites

```bash
pip install numpy matplotlib streamlit jupyter
```

That's it. No complex dependencies.

### Run Any Notebook

```bash
cd Notebooks
jupyter notebook
# Open any of the 20 notebooks
```

### Run Any App

```bash
# Example: RLHF Demo
cd Project_01_RLHF_LLM_Alignment
streamlit run app.py

# Example: Drug Discovery
cd Project_09_Drug_Discovery_RL
streamlit run app.py
```

### Run All Notebooks Programmatically

```bash
cd Notebooks
jupyter nbconvert --to notebook --execute --inplace *.ipynb
```

---

## Learning Path

Not sure where to start? Follow this path:

```
                    START HERE
                        |
                        v
            +-------------------------+
            |  01-04: Foundations      |
            |  MDPs, Bellman, DP      |
            +-------------------------+
                        |
              +---------+---------+
              v                   v
    +----------------+   +----------------+
    | 05-08: Tabular |   | 18-19: Bandits |
    | MC, TD, SARSA  |   | Exploration    |
    | Q-Learning     |   +----------------+
    +----------------+
              |
              v
    +-------------------+
    | 09-10: Scaling Up |
    | N-Step, Func Approx|
    +-------------------+
              |
     +--------+--------+
     v                  v
+-----------+   +-------------+
| 11-14:    |   | 15-17:      |
| Policy    |   | Value-Based |
| Gradient  |   | DQN Family  |
| REINFORCE |   | Replay      |
| AC, PPO   |   +-------------+
+-----------+
     |
     v
+-----------------------------------+
|     20: Model-Based RL            |
+-----------------------------------+
              |
              v
+-----------------------------------+
|   PROJECTS: Pick your interest!   |
|                                   |
|   AI Safety --> 01 (RLHF), 06    |
|   Healthcare --> 02, 09           |
|   Robotics --> 03, 04, 10        |
|   Planning --> 05, 07            |
|   Meta-Learning --> 08           |
+-----------------------------------+
```

---

## Project Deep Dives

<details>
<summary><strong>Project 01: RLHF for LLM Alignment</strong></summary>

Train language models to follow human preferences using the same pipeline behind ChatGPT and Claude.

**The 3-Step Pipeline:**
1. **Supervised Fine-Tuning (SFT)** -- Train on human demonstrations
2. **Reward Model** -- Learn preferences from human rankings (Bradley-Terry model)
3. **PPO Fine-Tuning** -- Optimize policy with KL-constrained PPO

**Key Equation:**

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t, 1\pm\epsilon)\hat{A}_t\right)\right]$$

</details>

<details>
<summary><strong>Project 02: Offline RL for Healthcare</strong></summary>

Learn optimal treatment policies from patient records without experimenting on real patients.

**Why Offline RL?** You can't reset a patient. Offline RL learns from fixed datasets.

**Key Algorithm:** Conservative Q-Learning (CQL) penalizes out-of-distribution actions:

$$Q_{CQL} = Q - \alpha \cdot \mathbb{E}_\pi[Q(s,a)] + \alpha \cdot \mathbb{E}_{data}[Q(s,a)]$$

</details>

<details>
<summary><strong>Project 03: Multi-Agent RL</strong></summary>

Multiple agents learning simultaneously in a predator-prey environment.

**Challenge:** Each agent's environment is non-stationary because other agents are also learning.

</details>

<details>
<summary><strong>Project 04: Safe RL</strong></summary>

Learn policies that maximize reward while satisfying safety constraints.

**Lagrangian Approach:**

$$L(\theta, \lambda) = J(\theta) - \lambda(J_c(\theta) - d)$$

</details>

<details>
<summary><strong>Project 05: World Models</strong></summary>

Learn the environment dynamics and plan in imagination.

**Key Idea:** $z_{t+1} = f_\theta(z_t, a_t)$ -- predict next state from current state and action.

</details>

<details>
<summary><strong>Project 06: RLAIF (AI Feedback)</strong></summary>

Replace expensive human feedback with AI-generated feedback. Compare convergence and quality.

</details>

<details>
<summary><strong>Project 07: Hierarchical RL</strong></summary>

Break complex tasks into subtasks using the Options Framework in a Four Rooms environment.

</details>

<details>
<summary><strong>Project 08: Meta-RL (Learning to Learn)</strong></summary>

Train agents that can adapt to new tasks in just a few episodes using MAML.

**MAML Update:**

$$\theta^* = \theta - \alpha \nabla_\theta \mathcal{L}_{task}(\theta - \alpha \nabla_\theta \mathcal{L}_{task}(\theta))$$

</details>

<details>
<summary><strong>Project 09: Drug Discovery</strong></summary>

Use RL to design molecules that satisfy multiple drug properties (LogP, toxicity, drug-likeness).

</details>

<details>
<summary><strong>Project 10: Sim-to-Real Transfer</strong></summary>

Train in simulation, deploy in the real world using Domain Randomization.

</details>

---

## Environments Built from Scratch

No OpenAI Gym dependency. Every environment is hand-crafted:

| Environment | Used In | States | Actions |
|-------------|---------|--------|---------|
| GridWorld (4x4) | Notebooks 01-04 | 16 discrete | 4 (up/right/down/left) |
| Blackjack | Notebook 05 | Player sum x Dealer | Hit / Stick |
| Random Walk | Notebook 06, 09 | 5-19 states | Left / Right |
| Cliff Walking | Notebook 07 | 4x12 grid | 4 directions |
| Mountain Car | Notebook 10 | Position x Velocity | 3 (reverse/neutral/forward) |
| CartPole | Notebooks 11-16 | 4D continuous | 2 (left/right) |
| 10-Armed Bandit | Notebooks 18-19 | None | K arms |
| Maze (6x9) | Notebook 20 | 54 cells | 4 directions |
| Predator-Prey | Project 03 | Grid positions | 5 (4 dirs + stay) |
| Four Rooms | Project 07 | Multi-room grid | 4 directions |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Core Math | NumPy |
| Visualization | Matplotlib |
| Interactive Apps | Streamlit |
| Notebooks | Jupyter |
| Language | Python 3.10+ |

**Philosophy:** Zero abstraction layers. When you read `Q[state][action] += alpha * td_error`, that's exactly what's happening. No hidden magic.

---

## Contributors & Domain Experts

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/mlnjsh">
        <img src="https://github.com/mlnjsh.png" width="100px;" alt="Milan Amrut Joshi"/><br />
        <sub><b>Milan Amrut Joshi</b></sub>
      </a><br />
      <sub>Project Lead & Core Author</sub><br />
      <sub>All 20 notebooks, 10 projects, Streamlit apps</sub>
    </td>
    <td align="center">
      <a href="https://github.com/araffin">
        <img src="https://github.com/araffin.png" width="100px;" alt="Antonin Raffin"/><br />
        <sub><b>Antonin Raffin</b></sub>
      </a><br />
      <sub>RL Expert & Advisor</sub><br />
      <sub>Creator of Stable-Baselines3, DLR Robotics</sub>
    </td>
    <td align="center">
      <a href="https://github.com/vwxyzjn">
        <img src="https://github.com/vwxyzjn.png" width="100px;" alt="Costa Huang"/><br />
        <sub><b>Costa Huang</b></sub>
      </a><br />
      <sub>RL Expert & Advisor</sub><br />
      <sub>Creator of CleanRL, Hugging Face RL Team</sub>
    </td>
  </tr>
</table>

### Domain Experts by Project

<table>
  <tr>
    <th>Project</th>
    <th>Expert 1</th>
    <th>Expert 2</th>
  </tr>
  <tr>
    <td><strong>01 — RLHF LLM Alignment</strong></td>
    <td align="center">
      <a href="https://github.com/joschu">
        <img src="https://github.com/joschu.png" width="60px;" alt="John Schulman"/><br />
        <sub><b>John Schulman</b></sub>
      </a><br />
      <sub>Co-creator of PPO, TRPO & RLHF</sub>
    </td>
    <td align="center">
      <a href="https://github.com/lvwerra">
        <img src="https://github.com/lvwerra.png" width="60px;" alt="Leandro von Werra"/><br />
        <sub><b>Leandro von Werra</b></sub>
      </a><br />
      <sub>Creator of TRL, Hugging Face</sub>
    </td>
  </tr>
  <tr>
    <td><strong>02 — Offline RL Healthcare</strong></td>
    <td align="center">
      <a href="https://github.com/aviralkumar2907">
        <img src="https://github.com/aviralkumar2907.png" width="60px;" alt="Aviral Kumar"/><br />
        <sub><b>Aviral Kumar</b></sub>
      </a><br />
      <sub>Creator of CQL, UC Berkeley</sub>
    </td>
    <td align="center">
      <a href="https://github.com/justinjfu">
        <img src="https://github.com/justinjfu.png" width="60px;" alt="Justin Fu"/><br />
        <sub><b>Justin Fu</b></sub>
      </a><br />
      <sub>D4RL Benchmark Creator, UC Berkeley</sub>
    </td>
  </tr>
  <tr>
    <td><strong>03 — Multi-Agent RL</strong></td>
    <td align="center">
      <a href="https://github.com/shariqiqbal2810">
        <img src="https://github.com/shariqiqbal2810.png" width="60px;" alt="Shariq Iqbal"/><br />
        <sub><b>Shariq Iqbal</b></sub>
      </a><br />
      <sub>MARL Researcher, USC</sub>
    </td>
    <td align="center">
      <a href="https://github.com/schroederdewitt">
        <img src="https://github.com/schroederdewitt.png" width="60px;" alt="Christian Schroeder de Witt"/><br />
        <sub><b>Christian Schroeder de Witt</b></sub>
      </a><br />
      <sub>QMIX Co-author, Oxford</sub>
    </td>
  </tr>
  <tr>
    <td><strong>04 — Safe RL</strong></td>
    <td align="center">
      <a href="https://github.com/jachiam">
        <img src="https://github.com/jachiam.png" width="60px;" alt="Joshua Achiam"/><br />
        <sub><b>Joshua Achiam</b></sub>
      </a><br />
      <sub>Creator of CPO, OpenAI</sub>
    </td>
    <td align="center">
      <a href="https://github.com/alexray">
        <img src="https://github.com/alexray.png" width="60px;" alt="Alex Ray"/><br />
        <sub><b>Alex Ray</b></sub>
      </a><br />
      <sub>Safety Gym Co-author, OpenAI</sub>
    </td>
  </tr>
  <tr>
    <td><strong>05 — World Models</strong></td>
    <td align="center">
      <a href="https://github.com/hardmaru">
        <img src="https://github.com/hardmaru.png" width="60px;" alt="David Ha"/><br />
        <sub><b>David Ha</b></sub>
      </a><br />
      <sub>World Models Paper, Google Brain</sub>
    </td>
    <td align="center">
      <a href="https://github.com/danijar">
        <img src="https://github.com/danijar.png" width="60px;" alt="Danijar Hafner"/><br />
        <sub><b>Danijar Hafner</b></sub>
      </a><br />
      <sub>Dreamer/V2/V3, Google DeepMind</sub>
    </td>
  </tr>
  <tr>
    <td><strong>06 — RLAIF</strong></td>
    <td align="center">
      <a href="https://github.com/lvwerra">
        <img src="https://github.com/lvwerra.png" width="60px;" alt="Leandro von Werra"/><br />
        <sub><b>Leandro von Werra</b></sub>
      </a><br />
      <sub>TRL Library, Hugging Face</sub>
    </td>
    <td align="center">
      <a href="https://github.com/edbeeching">
        <img src="https://github.com/edbeeching.png" width="60px;" alt="Edward Beeching"/><br />
        <sub><b>Edward Beeching</b></sub>
      </a><br />
      <sub>HF RL Researcher</sub>
    </td>
  </tr>
  <tr>
    <td><strong>07 — Hierarchical RL</strong></td>
    <td align="center">
      <a href="https://github.com/ofirnachum">
        <img src="https://github.com/ofirnachum.png" width="60px;" alt="Ofir Nachum"/><br />
        <sub><b>Ofir Nachum</b></sub>
      </a><br />
      <sub>Hierarchical RL, Google Brain</sub>
    </td>
    <td align="center">
      <a href="https://github.com/pierrelux">
        <img src="https://github.com/pierrelux.png" width="60px;" alt="Pierre-Luc Bacon"/><br />
        <sub><b>Pierre-Luc Bacon</b></sub>
      </a><br />
      <sub>Options Framework, Mila</sub>
    </td>
  </tr>
  <tr>
    <td><strong>08 — Meta-RL</strong></td>
    <td align="center">
      <a href="https://github.com/cbfinn">
        <img src="https://github.com/cbfinn.png" width="60px;" alt="Chelsea Finn"/><br />
        <sub><b>Chelsea Finn</b></sub>
      </a><br />
      <sub>Creator of MAML, Stanford</sub>
    </td>
    <td align="center">
      <a href="https://github.com/katerakelly">
        <img src="https://github.com/katerakelly.png" width="60px;" alt="Kate Rakelly"/><br />
        <sub><b>Kate Rakelly</b></sub>
      </a><br />
      <sub>PEARL Meta-RL, UC Berkeley</sub>
    </td>
  </tr>
  <tr>
    <td><strong>09 — Drug Discovery RL</strong></td>
    <td align="center">
      <a href="https://github.com/wengong-jin">
        <img src="https://github.com/wengong-jin.png" width="60px;" alt="Wengong Jin"/><br />
        <sub><b>Wengong Jin</b></sub>
      </a><br />
      <sub>Molecular Generation, MIT</sub>
    </td>
    <td align="center">
      <a href="https://github.com/rbharath">
        <img src="https://github.com/rbharath.png" width="60px;" alt="Bharath Ramsundar"/><br />
        <sub><b>Bharath Ramsundar</b></sub>
      </a><br />
      <sub>Creator of DeepChem</sub>
    </td>
  </tr>
  <tr>
    <td><strong>10 — Sim-to-Real Transfer</strong></td>
    <td align="center">
      <a href="https://github.com/josht">
        <img src="https://github.com/josht.png" width="60px;" alt="Josh Tobin"/><br />
        <sub><b>Josh Tobin</b></sub>
      </a><br />
      <sub>Domain Randomization, OpenAI</sub>
    </td>
    <td align="center">
      <a href="https://github.com/xbpeng">
        <img src="https://github.com/xbpeng.png" width="60px;" alt="Xue Bin Peng"/><br />
        <sub><b>Xue Bin Peng</b></sub>
      </a><br />
      <sub>Sim-to-Real Transfer, UC Berkeley</sub>
    </td>
  </tr>
</table>

---

## Contributing

Found a bug? Want to add a project? PRs welcome!

1. Fork the repo
2. Create your branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes
4. Push and open a PR

---

## References

| Resource | Author |
|----------|--------|
| *Reinforcement Learning: An Introduction* | Sutton & Barto (2018) |
| *Deep RL Course* | Hugging Face |
| *Spinning Up in Deep RL* | OpenAI |
| *Training LLMs to Follow Instructions with Human Feedback* | Ouyang et al. (2022) |
| *Proximal Policy Optimization Algorithms* | Schulman et al. (2017) |
| *Playing Atari with Deep Reinforcement Learning* | Mnih et al. (2013) |

---

<p align="center">
  <strong>If you find this useful, please give it a star!</strong><br>
  It helps others discover this resource.
</p>

<p align="center">
  Made with determination by <a href="https://github.com/mlnjsh">Milan Amrut Joshi</a>
</p>
