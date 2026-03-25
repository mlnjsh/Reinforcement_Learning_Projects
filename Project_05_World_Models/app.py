import streamlit as st
st.set_page_config(page_title="World Models", page_icon="🌍", layout="wide")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# Simple Grid-World Environment
# ─────────────────────────────────────────────
class SimpleGridWorld:
    """A stochastic grid world with a goal and optional obstacles."""
    def __init__(self, size=6, n_obstacles=3, seed=42):
        self.size = size
        self.rng = np.random.RandomState(seed)
        self.goal = (size - 1, size - 1)
        self.obstacles = set()
        while len(self.obstacles) < n_obstacles:
            obs = (self.rng.randint(0, size), self.rng.randint(0, size))
            if obs != (0, 0) and obs != self.goal:
                self.obstacles.add(obs)
        self.state = (0, 0)

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        """Actions: 0=up, 1=right, 2=down, 3=left"""
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        nx = max(0, min(self.size - 1, self.state[0] + dx))
        ny = max(0, min(self.size - 1, self.state[1] + dy))
        new_state = (nx, ny)
        if new_state in self.obstacles:
            new_state = self.state  # blocked
        self.state = new_state
        if self.state == self.goal:
            return self.state, 10.0, True
        return self.state, -0.1, False

    def state_to_idx(self, s):
        return s[0] * self.size + s[1]


# ─────────────────────────────────────────────
# Learned World Model
# ─────────────────────────────────────────────
class LearnedModel:
    """Tabular dynamics + reward model learned from experience."""
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        # Transition counts: T[s, a, s'] = count
        self.T_counts = np.zeros((n_states, n_actions, n_states))
        self.R_sum = np.zeros((n_states, n_actions))
        self.R_count = np.zeros((n_states, n_actions))

    def update(self, s, a, r, s_next):
        self.T_counts[s, a, s_next] += 1
        self.R_sum[s, a] += r
        self.R_count[s, a] += 1

    def predict(self, s, a, rng):
        total = self.T_counts[s, a].sum()
        if total == 0:
            return rng.randint(self.n_states), 0.0
        probs = self.T_counts[s, a] / total
        s_next = rng.choice(self.n_states, p=probs)
        r_hat = self.R_sum[s, a] / max(self.R_count[s, a], 1)
        return s_next, r_hat

    def prediction_error(self, real_transitions):
        """Compute average dynamics prediction error on a batch."""
        if len(real_transitions) == 0:
            return 1.0
        errors = 0
        for (s, a, r, s_next) in real_transitions:
            total = self.T_counts[s, a].sum()
            if total == 0:
                errors += 1
                continue
            probs = self.T_counts[s, a] / total
            predicted_s = np.argmax(probs)
            if predicted_s != s_next:
                errors += 1
        return errors / len(real_transitions)


# ─────────────────────────────────────────────
# Q-Learning Agent (Model-Free)
# ─────────────────────────────────────────────
def run_model_free(env_size, n_obstacles, n_episodes, seed):
    rng = np.random.RandomState(seed)
    env = SimpleGridWorld(env_size, n_obstacles, seed)
    n_states = env_size * env_size
    n_actions = 4
    Q = np.zeros((n_states, n_actions))
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.3

    episode_rewards = []
    for ep in range(n_episodes):
        s = env.reset()
        s_idx = env.state_to_idx(s)
        total_r = 0
        for _ in range(200):
            if rng.random() < epsilon:
                a = rng.randint(n_actions)
            else:
                a = np.argmax(Q[s_idx])
            s_next, r, done = env.step(a)
            s_next_idx = env.state_to_idx(s_next)
            Q[s_idx, a] += alpha * (r + gamma * np.max(Q[s_next_idx]) - Q[s_idx, a])
            s_idx = s_next_idx
            total_r += r
            if done:
                break
        epsilon = max(0.05, epsilon * 0.995)
        episode_rewards.append(total_r)
    return episode_rewards, Q


# ─────────────────────────────────────────────
# Dyna-Q Agent (Model-Based)
# ─────────────────────────────────────────────
def run_model_based(env_size, n_obstacles, n_episodes, planning_steps, seed):
    rng = np.random.RandomState(seed)
    env = SimpleGridWorld(env_size, n_obstacles, seed)
    n_states = env_size * env_size
    n_actions = 4
    Q = np.zeros((n_states, n_actions))
    model = LearnedModel(n_states, n_actions)
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.3

    episode_rewards = []
    model_errors = []
    visited = []  # store (s, a) pairs we've seen
    all_transitions = []

    for ep in range(n_episodes):
        s = env.reset()
        s_idx = env.state_to_idx(s)
        total_r = 0
        ep_transitions = []

        for _ in range(200):
            if rng.random() < epsilon:
                a = rng.randint(n_actions)
            else:
                a = np.argmax(Q[s_idx])
            s_next, r, done = env.step(a)
            s_next_idx = env.state_to_idx(s_next)

            # Real Q-update
            Q[s_idx, a] += alpha * (r + gamma * np.max(Q[s_next_idx]) - Q[s_idx, a])

            # Update model
            model.update(s_idx, a, r, s_next_idx)
            visited.append((s_idx, a))
            ep_transitions.append((s_idx, a, r, s_next_idx))

            # Planning: Dyna-style simulated updates
            for _ in range(planning_steps):
                idx = rng.randint(len(visited))
                sim_s, sim_a = visited[idx]
                sim_s_next, sim_r = model.predict(sim_s, sim_a, rng)
                Q[sim_s, sim_a] += alpha * (sim_r + gamma * np.max(Q[sim_s_next]) - Q[sim_s, sim_a])

            s_idx = s_next_idx
            total_r += r
            if done:
                break

        epsilon = max(0.05, epsilon * 0.995)
        episode_rewards.append(total_r)
        all_transitions.extend(ep_transitions)

        # Compute model error on recent transitions
        recent = all_transitions[-200:] if len(all_transitions) > 200 else all_transitions
        model_errors.append(model.prediction_error(recent))

    return episode_rewards, model_errors, Q, model


# ─────────────────────────────────────────────
# Extract trajectory from Q-table
# ─────────────────────────────────────────────
def extract_trajectory(Q, env_size, n_obstacles, seed, use_model=False, model=None):
    env = SimpleGridWorld(env_size, n_obstacles, seed)
    rng = np.random.RandomState(seed + 999)
    s = env.reset()
    traj = [s]
    for _ in range(100):
        s_idx = env.state_to_idx(s)
        a = np.argmax(Q[s_idx])
        s_next, r, done = env.step(a)
        traj.append(s_next)
        s = s_next
        if done:
            break
    return traj


def extract_planned_trajectory(model, Q, env_size, seed):
    """Generate a trajectory using the learned model (imagination)."""
    rng = np.random.RandomState(seed + 777)
    s_idx = 0  # start
    size = env_size
    traj = [(0, 0)]
    for _ in range(100):
        a = np.argmax(Q[s_idx])
        s_next_idx, r = model.predict(s_idx, a, rng)
        x, y = s_next_idx // size, s_next_idx % size
        traj.append((x, y))
        if (x, y) == (size - 1, size - 1):
            break
        s_idx = s_next_idx
    return traj


# ─────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────
st.title("World Models: Learning Environment Dynamics")
st.markdown("""
**World Models** learn how the environment works (transitions + rewards) and use that
knowledge to *plan in imagination* — generating simulated experience to speed up learning.
This demo compares **model-free Q-learning** vs **Dyna-Q (model-based)** on a grid world.
""")

st.sidebar.header("Configuration")
env_size = st.sidebar.slider("Grid Size", 4, 10, 6)
n_obstacles = st.sidebar.slider("Number of Obstacles", 0, 8, 3)
n_episodes = st.sidebar.slider("Training Episodes", 50, 500, 200, step=50)
planning_steps = st.sidebar.slider("Planning Steps per Real Step", 0, 20, 5)
seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)

if st.sidebar.button("Run Experiment", type="primary"):
    with st.spinner("Training model-free agent..."):
        mf_rewards, mf_Q = run_model_free(env_size, n_obstacles, n_episodes, seed)

    with st.spinner("Training model-based agent..."):
        mb_rewards, mb_errors, mb_Q, learned_model = run_model_based(
            env_size, n_obstacles, n_episodes, planning_steps, seed
        )

    # ── Smoothing helper ──
    def smooth(data, window=15):
        if len(data) < window:
            return data
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode='valid')

    # ── Tab layout ──
    tab1, tab2, tab3 = st.tabs([
        "Learning Curves", "Model Accuracy", "Trajectories"
    ])

    # ──────── TAB 1: Learning Curves ────────
    with tab1:
        st.subheader("Model-Free vs Model-Based Learning Curves")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(smooth(mf_rewards), label="Model-Free (Q-Learning)", color="#e74c3c", linewidth=2)
        ax.plot(smooth(mb_rewards), label=f"Model-Based (Dyna-Q, k={planning_steps})", color="#2ecc71", linewidth=2)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Episode Reward (smoothed)", fontsize=12)
        ax.set_title("Learning Curves: Real vs Imagined Experience", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model-Free Final Avg Reward", f"{np.mean(mf_rewards[-20:]):.2f}")
        with col2:
            st.metric("Model-Based Final Avg Reward", f"{np.mean(mb_rewards[-20:]):.2f}")

        st.info(f"""
        **Insight:** With {planning_steps} planning steps per real step, the model-based
        agent effectively gets {planning_steps + 1}x more learning updates per real interaction.
        This dramatically improves sample efficiency.
        """)

    # ──────── TAB 2: Model Accuracy ────────
    with tab2:
        st.subheader("Learned Model Prediction Accuracy Over Training")
        fig, ax = plt.subplots(figsize=(10, 5))
        accuracy = [1.0 - e for e in mb_errors]
        ax.plot(smooth(accuracy, 10), color="#3498db", linewidth=2)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Model Accuracy", fontsize=12)
        ax.set_title("Dynamics Model: Prediction Accuracy Over Training", fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect accuracy')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        The model starts knowing nothing (low accuracy) and gradually learns the environment
        dynamics from real experience. As accuracy improves, the planning steps become more
        valuable — the agent learns from increasingly realistic "imagined" experience.
        """)

    # ──────── TAB 3: Trajectories ────────
    with tab3:
        st.subheader("Planned vs Actual Trajectories")

        real_traj_mf = extract_trajectory(mf_Q, env_size, n_obstacles, seed)
        real_traj_mb = extract_trajectory(mb_Q, env_size, n_obstacles, seed)
        planned_traj = extract_planned_trajectory(learned_model, mb_Q, env_size, seed)

        env_ref = SimpleGridWorld(env_size, n_obstacles, seed)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        titles = ["Model-Free Path", "Model-Based Path", "Imagined Path (from model)"]
        trajs = [real_traj_mf, real_traj_mb, planned_traj]
        colors = ["#e74c3c", "#2ecc71", "#9b59b6"]

        for idx, (ax, title, traj, color) in enumerate(zip(axes, titles, trajs, colors)):
            # Draw grid
            for i in range(env_size + 1):
                ax.axhline(y=i, color='gray', linewidth=0.5)
                ax.axvline(x=i, color='gray', linewidth=0.5)

            # Draw obstacles
            for (ox, oy) in env_ref.obstacles:
                rect = mpatches.Rectangle((ox, oy), 1, 1, color='#2c3e50', alpha=0.7)
                ax.add_patch(rect)

            # Draw goal
            rect = mpatches.Rectangle((env_size-1, env_size-1), 1, 1, color='#f1c40f', alpha=0.6)
            ax.add_patch(rect)
            ax.text(env_size - 0.5, env_size - 0.5, 'G', ha='center', va='center',
                    fontsize=14, fontweight='bold')

            # Draw start
            rect = mpatches.Rectangle((0, 0), 1, 1, color='#3498db', alpha=0.4)
            ax.add_patch(rect)
            ax.text(0.5, 0.5, 'S', ha='center', va='center', fontsize=14, fontweight='bold')

            # Draw trajectory
            if len(traj) > 1:
                xs = [p[0] + 0.5 for p in traj]
                ys = [p[1] + 0.5 for p in traj]
                ax.plot(xs, ys, color=color, linewidth=2.5, alpha=0.8, marker='o',
                        markersize=3, zorder=5)

            ax.set_xlim(0, env_size)
            ax.set_ylim(0, env_size)
            ax.set_aspect('equal')
            ax.set_title(f"{title}\n({len(traj)-1} steps)", fontsize=12)
            ax.invert_yaxis()

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        - **Model-Free Path**: Learned purely from real experience.
        - **Model-Based Path**: Learned with help from simulated experience (Dyna planning).
        - **Imagined Path**: What the learned model *predicts* would happen — this is what the
          agent "sees" when it plans. As the model improves, imagined paths match reality more closely.
        """)

else:
    st.markdown("---")
    st.markdown("### How This Demo Works")
    st.markdown("""
    1. **Model-Free Agent** (Q-Learning): Learns only from real environment interactions.
    2. **Model-Based Agent** (Dyna-Q): Also learns a dynamics model and uses it for planning.

    **Dyna-Q Planning Loop:**
    ```
    For each real step:
        1. Take real action → observe (s, a, r, s')
        2. Update Q-table with real experience
        3. Update dynamics model
        4. For k planning steps:
           - Pick a random previously-visited state
           - Simulate a transition using the learned model
           - Update Q-table with simulated experience
    ```

    Adjust the sidebar parameters and click **Run Experiment** to compare both approaches.
    """)

    st.markdown("### Key Concepts")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Dynamics Model**")
        st.latex(r"z_{t+1} = f_\theta(z_t, a_t)")
        st.markdown("Predicts next state given current state and action.")
    with col2:
        st.markdown("**Reward Model**")
        st.latex(r"\hat{r} = g_\theta(z_t, a_t)")
        st.markdown("Predicts reward for a state-action pair.")
    with col3:
        st.markdown("**Planning Benefit**")
        st.latex(r"\text{Updates} = 1 + k")
        st.markdown("Each real step yields 1 + k learning updates.")
