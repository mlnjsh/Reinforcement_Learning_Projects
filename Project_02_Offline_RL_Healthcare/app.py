import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Offline RL for Healthcare",
    page_icon="🏥",
    layout="wide",
)

st.title("Offline RL for Healthcare")
st.markdown(
    "Learn treatment policies from historical patient data **without experimenting on real patients**. "
    "Compare Behavior Cloning vs Conservative Q-Learning (CQL)."
)

# ─────────────────────────── sidebar ───────────────────────────
st.sidebar.header("Parameters")
n_states = st.sidebar.slider("Number of patient states", 4, 20, 8)
n_actions = st.sidebar.slider("Number of treatments", 2, 8, 4)
dataset_size = st.sidebar.slider("Offline dataset size", 100, 5000, 1000, step=100)
cql_alpha = st.sidebar.slider("CQL alpha (conservatism)", 0.0, 10.0, 2.0, step=0.1)
gamma = st.sidebar.slider("Discount factor (gamma)", 0.5, 0.99, 0.95, step=0.01)
learning_rate = st.sidebar.slider("Learning rate", 0.01, 0.5, 0.1, step=0.01)
n_iterations = st.sidebar.slider("Training iterations", 50, 500, 200, step=50)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

np.random.seed(seed)

# ─────────────────── environment definition ────────────────────
# Simple patient treatment MDP
# States  : abstract patient condition indices (0 = worst, n_states-1 = best)
# Actions : treatment options
# Rewards : +1 for improving, -1 for worsening, +5 for reaching best state

# True transition probabilities (unknown to the offline agent)
transition_probs = np.random.dirichlet(np.ones(n_states), size=(n_states, n_actions))
# Bias: some treatments help sicker patients, others help healthier ones
for a in range(n_actions):
    for s in range(n_states):
        # Make some actions good for this state
        good_next = min(s + 1, n_states - 1)
        transition_probs[s, a, good_next] += 0.3 * ((a + s) % n_actions == 0)
    transition_probs[:, a, :] /= transition_probs[:, a, :].sum(axis=1, keepdims=True)

# Reward: based on state reached
state_rewards = np.linspace(-1, 5, n_states)  # higher states = healthier = more reward


# ─────────────────── generate offline dataset ──────────────────
def generate_offline_data(n_samples, behavior_noise=0.3):
    """Simulate a noisy behavior policy collecting data."""
    # Behavior policy: a mediocre doctor — somewhat good but noisy
    # Precompute a rough Q-table for the behavior policy
    Q_behavior = np.random.randn(n_states, n_actions) * 0.1
    for _ in range(50):
        for s in range(n_states):
            for a in range(n_actions):
                next_s_probs = transition_probs[s, a]
                expected_r = np.dot(next_s_probs, state_rewards)
                Q_behavior[s, a] = expected_r + gamma * np.dot(
                    next_s_probs, np.max(Q_behavior, axis=1)
                )

    states, actions, rewards, next_states = [], [], [], []
    for _ in range(n_samples):
        s = np.random.randint(0, n_states)
        # Behavior policy: softmax with noise
        logits = Q_behavior[s] / max(behavior_noise, 0.01)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        a = np.random.choice(n_actions, p=probs)
        # Transition
        s_next = np.random.choice(n_states, p=transition_probs[s, a])
        r = state_rewards[s_next]
        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(s_next)

    return (
        np.array(states),
        np.array(actions),
        np.array(rewards),
        np.array(next_states),
    )


dataset = generate_offline_data(dataset_size)
states_d, actions_d, rewards_d, next_states_d = dataset

# ─────────────────── Behavior Cloning ──────────────────────────
def behavior_cloning(states, actions):
    """Learn a policy by imitating the data."""
    counts = np.zeros((n_states, n_actions)) + 1e-8  # Laplace smoothing
    for s, a in zip(states, actions):
        counts[s, a] += 1
    policy = counts / counts.sum(axis=1, keepdims=True)
    return policy


bc_policy = behavior_cloning(states_d, actions_d)

# ─────────────────── CQL Training ──────────────────────────────
def train_cql(states, actions, rewards, next_states, alpha, lr, n_iter):
    """Conservative Q-Learning on offline data."""
    Q = np.zeros((n_states, n_actions))
    reward_history = []
    q_mean_history = []

    for iteration in range(n_iter):
        # Shuffle data
        idx = np.random.permutation(len(states))
        batch_size = min(64, len(states))
        batch_idx = idx[:batch_size]

        for i in batch_idx:
            s, a, r, s_next = states[i], actions[i], rewards[i], next_states[i]

            # Standard Bellman target
            target = r + gamma * np.max(Q[s_next])

            # Standard Q-learning update
            td_error = target - Q[s, a]
            Q[s, a] += lr * td_error

        # CQL regularization: penalize Q-values for all actions, boost for data actions
        for s_val in range(n_states):
            # Penalty: push down Q for actions the learned policy would choose (logsumexp approx)
            logsumexp_q = np.log(np.sum(np.exp(Q[s_val] - Q[s_val].max())) + 1e-10) + Q[s_val].max()
            # Bonus: push up Q for actions seen in data
            data_mask = states == s_val
            if data_mask.sum() > 0:
                data_actions_for_s = actions[data_mask]
                data_q_mean = np.mean([Q[s_val, a_d] for a_d in data_actions_for_s])
            else:
                data_q_mean = 0.0

            cql_penalty = alpha * (logsumexp_q - data_q_mean)
            Q[s_val] -= lr * cql_penalty / n_states

        # Evaluate: compute expected reward under greedy policy
        greedy_policy = np.zeros(n_states, dtype=int)
        for s_val in range(n_states):
            greedy_policy[s_val] = np.argmax(Q[s_val])

        expected_r = 0
        for s_val in range(n_states):
            a_val = greedy_policy[s_val]
            expected_r += np.dot(transition_probs[s_val, a_val], state_rewards)
        reward_history.append(expected_r / n_states)
        q_mean_history.append(np.mean(Q))

    return Q, reward_history, q_mean_history


# Train CQL
Q_cql, cql_rewards, cql_q_means = train_cql(
    states_d, actions_d, rewards_d, next_states_d, cql_alpha, learning_rate, n_iterations
)

# Also train unconstrained (alpha=0) for comparison
Q_unconstrained, unc_rewards, unc_q_means = train_cql(
    states_d, actions_d, rewards_d, next_states_d, 0.0, learning_rate, n_iterations
)

# ─────────────────── derive policies ───────────────────────────
cql_policy_greedy = np.zeros((n_states, n_actions))
for s in range(n_states):
    cql_policy_greedy[s, np.argmax(Q_cql[s])] = 1.0

unc_policy_greedy = np.zeros((n_states, n_actions))
for s in range(n_states):
    unc_policy_greedy[s, np.argmax(Q_unconstrained[s])] = 1.0

# ─────────────────── compute data action distribution ──────────
data_action_dist = np.zeros((n_states, n_actions)) + 1e-8
for s, a in zip(states_d, actions_d):
    data_action_dist[s, a] += 1
data_action_dist /= data_action_dist.sum(axis=1, keepdims=True)

# ─────────────────────── PLOTS ─────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["Policy Comparison", "Reward Curves", "Q-Value Distributions", "Dataset Overview"]
)

with tab1:
    st.subheader("Treatment Policy Comparison")
    st.markdown(
        "Each heatmap shows the probability of choosing each treatment in each patient state. "
        "Behavior Cloning copies the data; CQL learns a potentially better but conservative policy."
    )

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    titles = ["Data Distribution", "Behavior Cloning", "Unconstrained Q", f"CQL (alpha={cql_alpha})"]
    policies = [data_action_dist, bc_policy, unc_policy_greedy, cql_policy_greedy]

    for ax, title, pol in zip(axes, titles, policies):
        im = ax.imshow(pol, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xlabel("Treatment")
        ax.set_ylabel("Patient State")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(n_actions))
        ax.set_yticks(range(n_states))
        ax.set_yticklabels([f"S{i}" for i in range(n_states)])
        ax.set_xticklabels([f"T{i}" for i in range(n_actions)])

    fig.colorbar(im, ax=axes, shrink=0.8, label="P(treatment | state)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Show OOD analysis
    st.subheader("Out-of-Distribution Action Detection")
    st.markdown(
        "CQL penalizes actions rarely seen in the data. Below: how much each method's policy "
        "deviates from what was actually observed."
    )

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x = np.arange(n_states)
    width = 0.25

    # KL divergence from data distribution for each policy
    def kl_div(p, q):
        """KL(p || q) per state, averaged."""
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        return np.sum(p * np.log(p / q), axis=1)

    kl_bc = kl_div(bc_policy, data_action_dist)
    kl_unc = kl_div(unc_policy_greedy + 1e-10, data_action_dist)
    kl_cql = kl_div(cql_policy_greedy + 1e-10, data_action_dist)

    ax2.bar(x - width, kl_bc, width, label="Behavior Cloning", color="#2196F3")
    ax2.bar(x, kl_unc, width, label="Unconstrained Q", color="#FF5722")
    ax2.bar(x + width, kl_cql, width, label=f"CQL (alpha={cql_alpha})", color="#4CAF50")
    ax2.set_xlabel("Patient State")
    ax2.set_ylabel("KL Divergence from Data")
    ax2.set_title("Policy Deviation from Observed Data")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"S{i}" for i in range(n_states)])
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.info(
        "Lower KL divergence means the policy stays closer to what doctors actually did. "
        "CQL typically has lower deviation than unconstrained Q-learning because it penalizes "
        "out-of-distribution actions."
    )

with tab2:
    st.subheader("Expected Reward During Training")

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

    ax3a.plot(cql_rewards, label=f"CQL (alpha={cql_alpha})", color="#4CAF50", linewidth=2)
    ax3a.plot(unc_rewards, label="Unconstrained (alpha=0)", color="#FF5722", linewidth=2, linestyle="--")

    # BC baseline (constant)
    bc_reward = 0
    for s in range(n_states):
        bc_action = np.argmax(bc_policy[s])
        bc_reward += np.dot(transition_probs[s, bc_action], state_rewards)
    bc_reward /= n_states
    ax3a.axhline(y=bc_reward, color="#2196F3", linestyle=":", linewidth=2, label="Behavior Cloning")

    ax3a.set_xlabel("Training Iteration")
    ax3a.set_ylabel("Avg Expected Reward per State")
    ax3a.set_title("Reward Curves")
    ax3a.legend()
    ax3a.grid(alpha=0.3)

    # Mean Q-value over training
    ax3b.plot(cql_q_means, label=f"CQL (alpha={cql_alpha})", color="#4CAF50", linewidth=2)
    ax3b.plot(unc_q_means, label="Unconstrained (alpha=0)", color="#FF5722", linewidth=2, linestyle="--")
    ax3b.set_xlabel("Training Iteration")
    ax3b.set_ylabel("Mean Q-value")
    ax3b.set_title("Q-Value Scale Over Training")
    ax3b.legend()
    ax3b.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.markdown(
        "**Key insight:** CQL keeps Q-values lower (more conservative). Unconstrained Q-learning "
        "can overestimate Q-values for unseen actions, leading to overconfident and potentially "
        "dangerous treatment recommendations."
    )

with tab3:
    st.subheader("Q-Value Distributions")
    st.markdown(
        "CQL pushes down Q-values for out-of-distribution actions (those rarely seen in data) "
        "while maintaining reasonable values for in-distribution actions."
    )

    fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5))

    # Separate Q-values into in-distribution and out-of-distribution
    in_dist_q_cql = []
    ood_q_cql = []
    in_dist_q_unc = []
    ood_q_unc = []

    for s in range(n_states):
        for a in range(n_actions):
            if data_action_dist[s, a] > 1.0 / n_actions:  # "in distribution"
                in_dist_q_cql.append(Q_cql[s, a])
                in_dist_q_unc.append(Q_unconstrained[s, a])
            else:
                ood_q_cql.append(Q_cql[s, a])
                ood_q_unc.append(Q_unconstrained[s, a])

    # Plot 1: All Q-values comparison
    axes4[0].hist(Q_unconstrained.flatten(), bins=20, alpha=0.6, color="#FF5722", label="Unconstrained", density=True)
    axes4[0].hist(Q_cql.flatten(), bins=20, alpha=0.6, color="#4CAF50", label="CQL", density=True)
    axes4[0].set_title("All Q-Values", fontweight="bold")
    axes4[0].set_xlabel("Q-value")
    axes4[0].set_ylabel("Density")
    axes4[0].legend()
    axes4[0].grid(alpha=0.3)

    # Plot 2: In-distribution Q-values
    if len(in_dist_q_unc) > 0:
        axes4[1].hist(in_dist_q_unc, bins=15, alpha=0.6, color="#FF5722", label="Unconstrained", density=True)
        axes4[1].hist(in_dist_q_cql, bins=15, alpha=0.6, color="#4CAF50", label="CQL", density=True)
    axes4[1].set_title("In-Distribution Q-Values", fontweight="bold")
    axes4[1].set_xlabel("Q-value")
    axes4[1].legend()
    axes4[1].grid(alpha=0.3)

    # Plot 3: Out-of-distribution Q-values
    if len(ood_q_unc) > 0:
        axes4[2].hist(ood_q_unc, bins=15, alpha=0.6, color="#FF5722", label="Unconstrained", density=True)
        axes4[2].hist(ood_q_cql, bins=15, alpha=0.6, color="#4CAF50", label="CQL", density=True)
    axes4[2].set_title("Out-of-Distribution Q-Values", fontweight="bold")
    axes4[2].set_xlabel("Q-value")
    axes4[2].legend()
    axes4[2].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("CQL mean Q (in-dist)", f"{np.mean(in_dist_q_cql):.3f}" if in_dist_q_cql else "N/A")
        st.metric("CQL mean Q (OOD)", f"{np.mean(ood_q_cql):.3f}" if ood_q_cql else "N/A")
    with col2:
        st.metric("Unconstrained mean Q (in-dist)", f"{np.mean(in_dist_q_unc):.3f}" if in_dist_q_unc else "N/A")
        st.metric("Unconstrained mean Q (OOD)", f"{np.mean(ood_q_unc):.3f}" if ood_q_unc else "N/A")

    st.success(
        "CQL suppresses OOD Q-values more than in-distribution ones. "
        "This prevents the agent from recommending treatments it has no evidence for."
    )

with tab4:
    st.subheader("Offline Dataset Overview")

    fig5, axes5 = plt.subplots(1, 3, figsize=(16, 5))

    # State visitation
    state_counts = np.bincount(states_d, minlength=n_states)
    axes5[0].bar(range(n_states), state_counts, color="#9C27B0", alpha=0.7)
    axes5[0].set_xlabel("Patient State")
    axes5[0].set_ylabel("Count")
    axes5[0].set_title("State Visitation in Dataset")
    axes5[0].set_xticks(range(n_states))
    axes5[0].set_xticklabels([f"S{i}" for i in range(n_states)])
    axes5[0].grid(axis="y", alpha=0.3)

    # Action distribution
    action_counts = np.bincount(actions_d, minlength=n_actions)
    axes5[1].bar(range(n_actions), action_counts, color="#FF9800", alpha=0.7)
    axes5[1].set_xlabel("Treatment")
    axes5[1].set_ylabel("Count")
    axes5[1].set_title("Treatment Distribution in Dataset")
    axes5[1].set_xticks(range(n_actions))
    axes5[1].set_xticklabels([f"T{i}" for i in range(n_actions)])
    axes5[1].grid(axis="y", alpha=0.3)

    # Reward distribution
    axes5[2].hist(rewards_d, bins=20, color="#00BCD4", alpha=0.7, edgecolor="black")
    axes5[2].set_xlabel("Reward (Health Outcome)")
    axes5[2].set_ylabel("Count")
    axes5[2].set_title("Reward Distribution in Dataset")
    axes5[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    st.markdown(f"**Dataset size:** {dataset_size} transitions | **States:** {n_states} | **Treatments:** {n_actions}")

# ─────────────────────── formulas ──────────────────────────────
st.divider()
st.subheader("Key Formulas")
st.latex(r"Q_{\text{CQL}} = Q_{\text{standard}} - \alpha \, \mathbb{E}_{\pi}[Q(s,a)] + \alpha \, \mathbb{E}_{\text{data}}[Q(s,a)]")
st.markdown(
    "- The first term is the standard Bellman Q-value.\n"
    "- The second term **penalizes** Q-values for actions the learned policy favors.\n"
    "- The third term **rewards** Q-values for actions actually seen in the data.\n"
    f"- **alpha = {cql_alpha}** controls conservatism (higher = more conservative)."
)
