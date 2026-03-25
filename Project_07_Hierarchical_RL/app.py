import streamlit as st
st.set_page_config(page_title="Hierarchical RL", page_icon="🏗️", layout="wide")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────
# Four-Room Grid World
# ─────────────────────────────────────────────
class FourRoomGrid:
    """
    Grid world divided into rooms by walls with doorways.
    Layout (example 12x12):

    Room 0 (top-left)  | Room 1 (top-right)
    ─────── door ───────|─────── door ──────
    Room 2 (bottom-left)| Room 3 (bottom-right)
    """
    def __init__(self, size=12):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)  # 0=free, 1=wall
        self.mid = size // 2

        # Build walls
        for i in range(size):
            self.grid[self.mid, i] = 1  # horizontal wall
            self.grid[i, self.mid] = 1  # vertical wall

        # Create doorways (2-cell wide openings)
        door_offsets = [size // 4, 3 * size // 4]
        # Horizontal wall doors (connect top-bottom)
        self.grid[self.mid, door_offsets[0]] = 0
        self.grid[self.mid, door_offsets[1]] = 0
        # Vertical wall doors (connect left-right)
        self.grid[door_offsets[0], self.mid] = 0
        self.grid[door_offsets[1], self.mid] = 0

        self.doors = [
            (self.mid, door_offsets[0]),  # door between room 0 and 2
            (self.mid, door_offsets[1]),  # door between room 1 and 3
            (door_offsets[0], self.mid),  # door between room 0 and 1
            (door_offsets[1], self.mid),  # door between room 2 and 3
        ]

        self.start = (1, 1)
        self.goal = (size - 2, size - 2)
        self.state = self.start

    def get_room(self, pos):
        r, c = pos
        if r < self.mid and c < self.mid:
            return 0
        elif r < self.mid and c >= self.mid:
            return 1
        elif r >= self.mid and c < self.mid:
            return 2
        else:
            return 3

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        """Actions: 0=up, 1=right, 2=down, 3=left"""
        dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        nr = max(0, min(self.size - 1, self.state[0] + dr))
        nc = max(0, min(self.size - 1, self.state[1] + dc))
        if self.grid[nr, nc] == 1:
            nr, nc = self.state  # blocked by wall
        self.state = (nr, nc)
        if self.state == self.goal:
            return self.state, 10.0, True
        return self.state, -0.1, False

    def state_idx(self, s):
        return s[0] * self.size + s[1]

    def idx_to_state(self, idx):
        return (idx // self.size, idx % self.size)


# ─────────────────────────────────────────────
# Flat Q-Learning
# ─────────────────────────────────────────────
def flat_q_learning(grid_size, n_episodes, seed):
    rng = np.random.RandomState(seed)
    env = FourRoomGrid(grid_size)
    n_states = grid_size * grid_size
    n_actions = 4
    Q = np.zeros((n_states, n_actions))
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.4
    visit_counts = np.zeros((grid_size, grid_size))

    episode_rewards = []
    episode_steps = []

    for ep in range(n_episodes):
        s = env.reset()
        total_r = 0
        steps = 0
        for _ in range(500):
            s_idx = env.state_idx(s)
            visit_counts[s[0], s[1]] += 1
            if rng.random() < epsilon:
                a = rng.randint(n_actions)
            else:
                a = np.argmax(Q[s_idx])
            s_next, r, done = env.step(a)
            s_next_idx = env.state_idx(s_next)
            Q[s_idx, a] += alpha * (r + gamma * np.max(Q[s_next_idx]) - Q[s_idx, a])
            s = s_next
            total_r += r
            steps += 1
            if done:
                break
        epsilon = max(0.05, epsilon * 0.997)
        episode_rewards.append(total_r)
        episode_steps.append(steps)

    return episode_rewards, episode_steps, Q, visit_counts, env


# ─────────────────────────────────────────────
# Hierarchical Q-Learning (Options / Subgoals)
# ─────────────────────────────────────────────
def hierarchical_q_learning(grid_size, n_episodes, seed):
    rng = np.random.RandomState(seed)
    env = FourRoomGrid(grid_size)
    n_states = grid_size * grid_size
    n_actions = 4

    # Subgoals: the four doorways + the final goal
    subgoals = list(env.doors) + [env.goal]
    n_subgoals = len(subgoals)

    # High-level Q: for each room, pick which subgoal to go to
    Q_high = np.zeros((4, n_subgoals))  # 4 rooms
    # Low-level Q: for each (state, subgoal), pick action
    Q_low = np.zeros((n_states, n_subgoals, n_actions))

    alpha_h = 0.1
    alpha_l = 0.15
    gamma = 0.95
    epsilon_h = 0.3
    epsilon_l = 0.4

    visit_counts = np.zeros((grid_size, grid_size))
    subgoal_counts = np.zeros((grid_size, grid_size))
    episode_rewards = []
    episode_steps = []

    for ep in range(n_episodes):
        s = env.reset()
        total_r = 0
        steps = 0

        for _ in range(10):  # max 10 subgoal selections per episode
            room = env.get_room(s)

            # High-level: pick subgoal
            if rng.random() < epsilon_h:
                sg_idx = rng.randint(n_subgoals)
            else:
                sg_idx = np.argmax(Q_high[room])

            subgoal = subgoals[sg_idx]
            subgoal_counts[subgoal[0], subgoal[1]] += 1
            sg_reward = 0
            sg_steps = 0

            # Low-level: navigate to subgoal
            for _ in range(150):
                s_idx = env.state_idx(s)
                visit_counts[s[0], s[1]] += 1

                if rng.random() < epsilon_l:
                    a = rng.randint(n_actions)
                else:
                    a = np.argmax(Q_low[s_idx, sg_idx])

                s_next, r, done = env.step(a)
                s_next_idx = env.state_idx(s_next)

                # Intrinsic reward for reaching subgoal
                intrinsic_r = 5.0 if s_next == subgoal else -0.1

                # Low-level Q-update (toward subgoal)
                Q_low[s_idx, sg_idx, a] += alpha_l * (
                    intrinsic_r + gamma * np.max(Q_low[s_next_idx, sg_idx]) - Q_low[s_idx, sg_idx, a]
                )

                s = s_next
                sg_reward += r
                sg_steps += 1
                steps += 1
                total_r += r

                if done or s == subgoal:
                    break

            # High-level Q-update
            new_room = env.get_room(s)
            Q_high[room, sg_idx] += alpha_h * (
                sg_reward + (gamma ** sg_steps) * np.max(Q_high[new_room]) - Q_high[room, sg_idx]
            )

            if done or steps > 500:
                break

        epsilon_h = max(0.05, epsilon_h * 0.997)
        epsilon_l = max(0.05, epsilon_l * 0.997)
        episode_rewards.append(total_r)
        episode_steps.append(steps)

    return episode_rewards, episode_steps, Q_high, Q_low, visit_counts, subgoal_counts, env, subgoals


# ─────────────────────────────────────────────
# Extract greedy path
# ─────────────────────────────────────────────
def extract_flat_path(Q, env):
    s = env.reset()
    path = [s]
    for _ in range(300):
        s_idx = env.state_idx(s)
        a = np.argmax(Q[s_idx])
        s_next, r, done = env.step(a)
        path.append(s_next)
        s = s_next
        if done:
            break
    return path


def extract_hier_path(Q_high, Q_low, env, subgoals):
    s = env.reset()
    path = [s]
    chosen_subgoals = []
    for _ in range(10):
        room = env.get_room(s)
        sg_idx = np.argmax(Q_high[room])
        subgoal = subgoals[sg_idx]
        chosen_subgoals.append(subgoal)

        for _ in range(150):
            s_idx = env.state_idx(s)
            a = np.argmax(Q_low[s_idx, sg_idx])
            s_next, r, done = env.step(a)
            path.append(s_next)
            s = s_next
            if done or s == subgoal:
                break
        if done:
            break
    return path, chosen_subgoals


# ─────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────
def draw_grid(ax, env, title, path=None, path_color='#e74c3c', subgoals_highlight=None,
              heatmap=None):
    size = env.size

    if heatmap is not None:
        cmap = LinearSegmentedColormap.from_list('custom', ['#ffffff', '#3498db', '#2c3e50'])
        hm = heatmap.copy()
        hm[env.grid == 1] = 0
        ax.imshow(hm, cmap=cmap, alpha=0.6, origin='upper')

    # Draw walls
    for r in range(size):
        for c in range(size):
            if env.grid[r, c] == 1:
                rect = mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, color='#2c3e50')
                ax.add_patch(rect)

    # Draw grid lines
    for i in range(size + 1):
        ax.axhline(y=i - 0.5, color='gray', linewidth=0.3)
        ax.axvline(x=i - 0.5, color='gray', linewidth=0.3)

    # Goal
    gr, gc = env.goal
    rect = mpatches.Rectangle((gc - 0.5, gr - 0.5), 1, 1, color='#f1c40f', alpha=0.7)
    ax.add_patch(rect)
    ax.text(gc, gr, 'G', ha='center', va='center', fontsize=10, fontweight='bold')

    # Start
    sr, sc = env.start
    rect = mpatches.Rectangle((sc - 0.5, sr - 0.5), 1, 1, color='#3498db', alpha=0.5)
    ax.add_patch(rect)
    ax.text(sc, sr, 'S', ha='center', va='center', fontsize=10, fontweight='bold')

    # Doors
    for (dr, dc) in env.doors:
        rect = mpatches.Rectangle((dc - 0.5, dr - 0.5), 1, 1, color='#e67e22', alpha=0.4)
        ax.add_patch(rect)

    # Subgoals highlighted
    if subgoals_highlight:
        for i, (sr_, sc_) in enumerate(subgoals_highlight):
            ax.plot(sc_, sr_, 's', color='#9b59b6', markersize=12, markeredgecolor='white',
                    markeredgewidth=2, zorder=10)
            ax.text(sc_ + 0.3, sr_ - 0.3, f'SG{i+1}', fontsize=7, color='#9b59b6',
                    fontweight='bold')

    # Path
    if path and len(path) > 1:
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        ax.plot(cols, rows, color=path_color, linewidth=2, alpha=0.8, zorder=5)
        ax.plot(cols[0], rows[0], 'o', color=path_color, markersize=6, zorder=6)
        ax.plot(cols[-1], rows[-1], '*', color=path_color, markersize=10, zorder=6)

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


# ─────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────
st.title("Hierarchical RL: Breaking Problems into Subgoals")
st.markdown("""
**Hierarchical RL** decomposes complex tasks into a hierarchy of subtasks.
A **high-level policy** picks subgoals (which room/doorway to target), while a
**low-level policy** handles primitive navigation. This demo compares flat Q-learning
against a hierarchical approach in a four-room grid world.
""")

st.sidebar.header("Configuration")
grid_size = st.sidebar.select_slider("Grid Size", options=[8, 10, 12, 14, 16], value=12)
n_episodes = st.sidebar.slider("Training Episodes", 100, 1000, 400, step=100)
seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)

if st.sidebar.button("Run Experiment", type="primary"):

    with st.spinner("Training flat Q-learning agent..."):
        flat_rewards, flat_steps, flat_Q, flat_visits, flat_env = flat_q_learning(
            grid_size, n_episodes, seed
        )

    with st.spinner("Training hierarchical agent..."):
        (hier_rewards, hier_steps, Q_high, Q_low, hier_visits,
         sg_counts, hier_env, subgoals) = hierarchical_q_learning(
            grid_size, n_episodes, seed
        )

    def smooth(data, window=20):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    tab1, tab2, tab3, tab4 = st.tabs([
        "Learning Curves", "Paths", "Subgoal Heatmap", "Exploration"
    ])

    # ──────── TAB 1: Learning Curves ────────
    with tab1:
        st.subheader("Flat vs Hierarchical Learning")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(smooth(flat_rewards), label="Flat Q-Learning", color="#e74c3c", linewidth=2)
        ax.plot(smooth(hier_rewards), label="Hierarchical RL", color="#2ecc71", linewidth=2)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Episode Reward", fontsize=12)
        ax.set_title("Episode Reward (smoothed)", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(smooth(flat_steps), label="Flat Q-Learning", color="#e74c3c", linewidth=2)
        ax.plot(smooth(hier_steps), label="Hierarchical RL", color="#2ecc71", linewidth=2)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Steps to Goal", fontsize=12)
        ax.set_title("Steps per Episode (smoothed)", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Flat — Final Avg Steps", f"{np.mean(flat_steps[-20:]):.0f}")
            st.metric("Flat — Final Avg Reward", f"{np.mean(flat_rewards[-20:]):.1f}")
        with col2:
            st.metric("Hierarchical — Final Avg Steps", f"{np.mean(hier_steps[-20:]):.0f}")
            st.metric("Hierarchical — Final Avg Reward", f"{np.mean(hier_rewards[-20:]):.1f}")

    # ──────── TAB 2: Paths ────────
    with tab2:
        st.subheader("Learned Paths: Flat vs Hierarchical")

        flat_path = extract_flat_path(flat_Q, flat_env)
        hier_path, chosen_sgs = extract_hier_path(Q_high, Q_low, hier_env, subgoals)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        draw_grid(axes[0], flat_env, f"Flat Q-Learning ({len(flat_path)-1} steps)",
                  path=flat_path, path_color='#e74c3c')

        draw_grid(axes[1], hier_env,
                  f"Hierarchical ({len(hier_path)-1} steps)",
                  path=hier_path, path_color='#2ecc71',
                  subgoals_highlight=chosen_sgs)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        if chosen_sgs:
            st.markdown("**Chosen subgoal sequence:** " +
                        " → ".join([f"({r},{c})" for r, c in chosen_sgs]) +
                        f" → Goal {hier_env.goal}")

    # ──────── TAB 3: Subgoal Heatmap ────────
    with tab3:
        st.subheader("Subgoal Selection Frequency")

        fig, ax = plt.subplots(figsize=(8, 7))
        draw_grid(ax, hier_env, "Subgoal Selection Heatmap", heatmap=sg_counts)

        # Annotate counts on doorways and goal
        for (r, c) in subgoals:
            count = int(sg_counts[r, c])
            if count > 0:
                ax.text(c, r, str(count), ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#9b59b6', alpha=0.8))

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        The heatmap shows how often the high-level policy selected each subgoal during training.
        Doorways that lie on the optimal path from start to goal should have the highest counts.
        The hierarchical agent learns to chain subgoals efficiently:
        **Start → nearest doorway → cross rooms → Goal**.
        """)

    # ──────── TAB 4: Exploration Comparison ────────
    with tab4:
        st.subheader("Exploration Patterns")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        draw_grid(axes[0], flat_env, "Flat Agent — State Visit Counts",
                  heatmap=flat_visits)
        draw_grid(axes[1], hier_env, "Hierarchical Agent — State Visit Counts",
                  heatmap=hier_visits)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        **Key observation:** The flat agent tends to spread exploration uniformly
        (or get stuck near the start), while the hierarchical agent focuses exploration
        along corridors between subgoals, leading to more efficient coverage of the
        relevant state space.

        **Why hierarchy helps exploration:**
        - Subgoals act as "stepping stones" that guide the agent through the environment
        - Instead of random walks, the agent makes directed progress toward intermediate targets
        - Each room becomes a smaller, easier sub-problem to solve
        """)

else:
    st.markdown("---")
    st.markdown("### How This Demo Works")
    st.markdown("""
    The environment is a **four-room grid world** divided by walls with doorways.
    The agent must navigate from the top-left corner to the bottom-right goal.

    **Two approaches are compared:**

    | | Flat Q-Learning | Hierarchical RL |
    |---|---|---|
    | **Decisions** | One action per step | Manager picks subgoal, Worker picks actions |
    | **Reward** | Only at final goal | Intrinsic reward for reaching subgoals |
    | **Exploration** | Random until goal found | Directed toward subgoals |
    | **Scalability** | Degrades with grid size | Scales well (sub-problems stay small) |
    """)

    st.markdown("### Key Concepts")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Options Framework**")
        st.latex(r"o = (I, \pi, \beta)")
        st.markdown("An option has an initiation set, a policy, and a termination condition.")
    with col2:
        st.markdown("**Hierarchical Value Function**")
        st.latex(r"V(s) = \max_o \left[ R(s,o) + \gamma^\tau V(s') \right]")
        st.markdown("Value includes temporal abstraction: discount over option duration.")
