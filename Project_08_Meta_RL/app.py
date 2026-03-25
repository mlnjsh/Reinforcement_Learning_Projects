import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Meta-RL: Learning to Learn", layout="wide")

st.title("Meta-RL: Learning to Learn")
st.markdown("""
This demo simulates **Model-Agnostic Meta-Learning (MAML)** on multi-armed bandit tasks.
A **meta-learner** trains across many tasks so it can adapt to new tasks in very few steps,
while a **regular learner** starts from scratch each time.
""")

# ─── Sidebar controls ───
st.sidebar.header("Configuration")
num_arms = st.sidebar.slider("Arms per bandit", 2, 10, 5)
num_meta_tasks = st.sidebar.slider("Meta-training tasks", 5, 50, 20)
num_test_tasks = st.sidebar.slider("Test tasks to evaluate", 3, 15, 5)
inner_steps = st.sidebar.slider("Inner-loop adaptation steps", 1, 30, 10)
meta_lr = st.sidebar.slider("Meta learning rate (β)", 0.01, 1.0, 0.3, 0.01)
inner_lr = st.sidebar.slider("Inner learning rate (α)", 0.01, 1.0, 0.5, 0.01)
meta_epochs = st.sidebar.slider("Meta-training epochs", 5, 100, 30)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

np.random.seed(seed)


# ─── Helper: softmax policy ───
def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def sample_action(logits):
    probs = softmax(logits)
    return np.random.choice(len(probs), p=probs)


def bandit_reward(true_probs, action):
    return 1.0 if np.random.rand() < true_probs[action] else 0.0


# ─── Generate task distribution ───
def generate_tasks(n_tasks, n_arms):
    """Each task is a bandit with random reward probabilities."""
    return [np.random.dirichlet(np.ones(n_arms)) * np.random.uniform(0.3, 1.0) for _ in range(n_tasks)]


# ─── Inner loop: adapt logits to a single task ───
def inner_loop_adapt(init_logits, task_probs, steps, lr, track=False):
    """Gradient-ascent on expected reward via REINFORCE on a bandit."""
    logits = init_logits.copy()
    rewards_history = []
    cumulative = 0.0
    for step in range(steps):
        probs = softmax(logits)
        action = sample_action(logits)
        r = bandit_reward(task_probs, action)
        cumulative += r
        rewards_history.append(cumulative / (step + 1))
        # Policy gradient: ∇ log π(a) * R
        grad = -probs.copy()
        grad[action] += 1.0
        logits += lr * grad * r
    if track:
        return logits, rewards_history
    return logits


# ─── MAML meta-training ───
def maml_train(meta_tasks, n_arms, inner_steps, inner_lr, meta_lr, epochs):
    """Train meta-parameters θ across a distribution of bandit tasks."""
    theta = np.zeros(n_arms)
    meta_loss_history = []
    for epoch in range(epochs):
        meta_grad = np.zeros(n_arms)
        epoch_reward = 0.0
        for task_probs in meta_tasks:
            # Inner loop: adapt
            adapted = inner_loop_adapt(theta, task_probs, inner_steps, inner_lr)
            # Evaluate adapted policy
            probs = softmax(adapted)
            action = sample_action(adapted)
            r = bandit_reward(task_probs, action)
            epoch_reward += r
            # Outer gradient (simplified: gradient of post-adaptation performance w.r.t. θ)
            grad = -probs.copy()
            grad[action] += 1.0
            meta_grad += grad * r
        meta_grad /= len(meta_tasks)
        theta += meta_lr * meta_grad
        meta_loss_history.append(epoch_reward / len(meta_tasks))
    return theta, meta_loss_history


# ─── Run everything ───
if st.sidebar.button("Run Meta-RL Experiment", type="primary"):
    with st.spinner("Meta-training in progress..."):
        meta_tasks = generate_tasks(num_meta_tasks, num_arms)
        test_tasks = generate_tasks(num_test_tasks, num_arms)

        # Meta-train
        theta_meta, meta_loss_curve = maml_train(
            meta_tasks, num_arms, inner_steps, inner_lr, meta_lr, meta_epochs
        )

    # ─── Evaluate on test tasks ───
    meta_curves = []
    regular_curves = []
    for task_probs in test_tasks:
        _, meta_rewards = inner_loop_adapt(theta_meta, task_probs, inner_steps, inner_lr, track=True)
        _, regular_rewards = inner_loop_adapt(np.zeros(num_arms), task_probs, inner_steps, inner_lr, track=True)
        meta_curves.append(meta_rewards)
        regular_curves.append(regular_rewards)

    meta_curves = np.array(meta_curves)
    regular_curves = np.array(regular_curves)

    # ─── Row 1: Meta-training curve + Task distribution ───
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Meta-Training Progress")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(meta_loss_curve, color="#2196F3", linewidth=2)
        ax1.set_xlabel("Meta-Epoch")
        ax1.set_ylabel("Avg Post-Adaptation Reward")
        ax1.set_title("Outer Loop: Meta-Training Curve")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.subheader("Task Distribution (Test Tasks)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        task_data = np.array([t for t in test_tasks])
        im = ax2.imshow(task_data, aspect="auto", cmap="YlOrRd")
        ax2.set_xlabel("Arm Index")
        ax2.set_ylabel("Task Index")
        ax2.set_title("Reward Probabilities per Task")
        plt.colorbar(im, ax=ax2, label="P(reward)")
        st.pyplot(fig2)
        plt.close(fig2)

    # ─── Row 2: Meta vs Regular comparison ───
    st.subheader("Adaptation Speed: Meta-Learner vs Regular Learner")
    col3, col4 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        steps_x = np.arange(1, inner_steps + 1)
        meta_mean = meta_curves.mean(axis=0)
        regular_mean = regular_curves.mean(axis=0)
        meta_std = meta_curves.std(axis=0)
        regular_std = regular_curves.std(axis=0)

        ax3.plot(steps_x, meta_mean, label="Meta-Learner (MAML)", color="#4CAF50", linewidth=2.5)
        ax3.fill_between(steps_x, meta_mean - meta_std, meta_mean + meta_std, alpha=0.2, color="#4CAF50")
        ax3.plot(steps_x, regular_mean, label="Regular Learner", color="#F44336", linewidth=2.5, linestyle="--")
        ax3.fill_between(steps_x, regular_mean - regular_std, regular_mean + regular_std, alpha=0.2, color="#F44336")
        ax3.set_xlabel("Inner-Loop Steps")
        ax3.set_ylabel("Avg Cumulative Reward")
        ax3.set_title("Adaptation Curves (Mean ± Std)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        plt.close(fig3)

    with col4:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        final_meta = meta_curves[:, -1]
        final_regular = regular_curves[:, -1]
        x_pos = np.arange(num_test_tasks)
        width = 0.35
        ax4.bar(x_pos - width / 2, final_meta, width, label="Meta-Learner", color="#4CAF50", alpha=0.8)
        ax4.bar(x_pos + width / 2, final_regular, width, label="Regular Learner", color="#F44336", alpha=0.8)
        ax4.set_xlabel("Test Task")
        ax4.set_ylabel("Final Avg Reward")
        ax4.set_title("Per-Task Final Performance")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f"Task {i}" for i in range(num_test_tasks)], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis="y")
        st.pyplot(fig4)
        plt.close(fig4)

    # ─── Row 3: Per-task adaptation detail ───
    st.subheader("Per-Task Adaptation Detail")
    fig5, axes = plt.subplots(1, min(num_test_tasks, 5), figsize=(4 * min(num_test_tasks, 5), 3.5), squeeze=False)
    for i in range(min(num_test_tasks, 5)):
        ax = axes[0][i]
        ax.plot(meta_curves[i], color="#4CAF50", linewidth=2, label="Meta")
        ax.plot(regular_curves[i], color="#F44336", linewidth=2, linestyle="--", label="Regular")
        ax.set_title(f"Task {i}")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Avg Reward")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    # ─── Summary metrics ───
    st.subheader("Summary Metrics")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Meta-Learner Final Reward", f"{meta_curves[:, -1].mean():.3f}")
    mc2.metric("Regular Learner Final Reward", f"{regular_curves[:, -1].mean():.3f}")
    improvement = (meta_curves[:, -1].mean() - regular_curves[:, -1].mean())
    mc3.metric("Meta Advantage", f"+{improvement:.3f}" if improvement > 0 else f"{improvement:.3f}")

    st.info(
        "The meta-learner starts from a learned initialization (θ*) that allows rapid "
        "adaptation, while the regular learner starts from zeros every time."
    )
else:
    st.info("Configure parameters in the sidebar and click **Run Meta-RL Experiment** to start.")
    st.markdown("""
    ### How This Demo Works

    1. **Task Distribution**: We create many multi-armed bandit tasks, each with different reward probabilities.
    2. **Meta-Training (Outer Loop)**: MAML finds an initialization θ* that enables fast adaptation.
    3. **Adaptation (Inner Loop)**: On a new test task, both meta and regular learners take gradient steps.
    4. **Comparison**: The meta-learner adapts much faster because its starting point is already "primed" for learning.

    **Key insight**: MAML does not learn a single good policy — it learns a good **starting point** for learning.
    """)
