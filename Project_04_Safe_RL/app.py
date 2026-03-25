import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

st.set_page_config(
    page_title="Safe Reinforcement Learning",
    page_icon="🛡️",
    layout="wide",
)

st.title("Safe Reinforcement Learning")
st.markdown(
    "Compare an **unconstrained** agent (maximizes reward, ignores danger) vs a **safe** agent "
    "(Lagrangian method: maximizes reward subject to safety constraints) in a grid world with danger zones."
)

# ─────────────────────────── sidebar ───────────────────────────
st.sidebar.header("Parameters")
grid_size = st.sidebar.slider("Grid size", 4, 10, 6)
cost_threshold = st.sidebar.slider("Cost threshold (d)", 0.0, 5.0, 1.0, step=0.1)
lambda_init = st.sidebar.slider("Initial lambda", 0.0, 5.0, 1.0, step=0.1)
lambda_lr = st.sidebar.slider("Lambda learning rate", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
learning_rate = st.sidebar.slider("Q-learning rate", 0.05, 0.5, 0.2, step=0.05)
gamma = st.sidebar.slider("Discount factor", 0.8, 0.99, 0.95, step=0.01)
n_episodes = st.sidebar.slider("Training episodes", 200, 5000, 1500, step=100)
epsilon_start = st.sidebar.slider("Epsilon (start)", 0.5, 1.0, 1.0, step=0.05)
epsilon_end = st.sidebar.slider("Epsilon (end)", 0.01, 0.2, 0.05, step=0.01)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

st.sidebar.header("Danger Zone Placement")
danger_mode = st.sidebar.selectbox("Danger pattern", ["Diagonal", "Center block", "Random", "Border corridor"])
danger_fraction = st.sidebar.slider("Danger fraction (for random)", 0.05, 0.5, 0.2, step=0.05)

np.random.seed(seed)

# ─────────────────── environment ───────────────────────────────
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right
ACTION_NAMES = ["Up", "Down", "Left", "Right"]
N_ACTIONS = 4

start = (0, 0)
goal = (grid_size - 1, grid_size - 1)

# Build danger map
danger_map = np.zeros((grid_size, grid_size), dtype=bool)

if danger_mode == "Diagonal":
    for i in range(grid_size):
        if (i, i) != start and (i, i) != goal:
            danger_map[i, i] = True
        if i + 1 < grid_size and (i, i + 1) != start and (i, i + 1) != goal:
            danger_map[i, i + 1] = True
elif danger_mode == "Center block":
    c = grid_size // 2
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            r, col = c + dr, c + dc
            if 0 <= r < grid_size and 0 <= col < grid_size and (r, col) != start and (r, col) != goal:
                danger_map[r, col] = True
elif danger_mode == "Random":
    n_danger = max(1, int(danger_fraction * grid_size * grid_size))
    all_cells = [(r, c) for r in range(grid_size) for c in range(grid_size)
                 if (r, c) != start and (r, c) != goal]
    chosen = np.random.choice(len(all_cells), min(n_danger, len(all_cells)), replace=False)
    for idx in chosen:
        danger_map[all_cells[idx]] = True
elif danger_mode == "Border corridor":
    # Danger along the direct path
    for i in range(1, grid_size - 1):
        if (grid_size - 1, i) != goal:
            danger_map[grid_size - 1, i] = True
        if (i, grid_size - 1) != goal:
            danger_map[i, grid_size - 1] = True


def state_idx(pos):
    return pos[0] * grid_size + pos[1]


def idx_to_pos(idx):
    return (idx // grid_size, idx % grid_size)


def step(pos, action):
    delta = ACTIONS[action]
    new_r = max(0, min(grid_size - 1, pos[0] + delta[0]))
    new_c = max(0, min(grid_size - 1, pos[1] + delta[1]))
    new_pos = (new_r, new_c)

    # Reward
    if new_pos == goal:
        reward = 10.0
    else:
        reward = -0.1  # step penalty

    # Cost
    cost = 1.0 if danger_map[new_pos] else 0.0

    done = new_pos == goal
    return new_pos, reward, cost, done


N_STATES = grid_size * grid_size

# ─────────────────── training ──────────────────────────────────


def train_agent(use_lagrangian=False):
    """Train a Q-learning agent, optionally with Lagrangian safety."""
    Q = np.zeros((N_STATES, N_ACTIONS))
    lam = lambda_init if use_lagrangian else 0.0

    reward_history = []
    cost_history = []
    lambda_history = []

    for ep in range(n_episodes):
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * ep / max(n_episodes - 1, 1)
        pos = start
        ep_reward = 0.0
        ep_cost = 0.0

        for t in range(grid_size * grid_size * 3):  # max steps
            s = state_idx(pos)

            # Epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(N_ACTIONS)
            else:
                action = int(np.argmax(Q[s]))

            new_pos, reward, cost, done = step(pos, action)
            s_next = state_idx(new_pos)

            # Modified reward for Lagrangian
            modified_reward = reward - lam * cost if use_lagrangian else reward

            # Q-update
            if done:
                target = modified_reward
            else:
                target = modified_reward + gamma * np.max(Q[s_next])

            Q[s, action] += learning_rate * (target - Q[s, action])

            ep_reward += reward
            ep_cost += cost
            pos = new_pos

            if done:
                break

        # Update lambda (dual variable)
        if use_lagrangian:
            lam = max(0.0, lam + lambda_lr * (ep_cost - cost_threshold))

        reward_history.append(ep_reward)
        cost_history.append(ep_cost)
        lambda_history.append(lam)

    return Q, reward_history, cost_history, lambda_history


progress_text = st.empty()
progress_text.text("Training unconstrained agent...")
Q_unsafe, rewards_unsafe, costs_unsafe, _ = train_agent(use_lagrangian=False)

progress_text.text("Training safe (Lagrangian) agent...")
Q_safe, rewards_safe, costs_safe, lambda_hist = train_agent(use_lagrangian=True)

progress_text.text("Training complete!")

# ─────────────────── extract policies and paths ────────────────


def extract_path(Q):
    """Follow greedy policy from start to goal."""
    pos = start
    path = [pos]
    visited = set()
    visited.add(pos)
    for _ in range(N_STATES * 2):
        s = state_idx(pos)
        action = int(np.argmax(Q[s]))
        new_pos, _, _, done = step(pos, action)
        path.append(new_pos)
        if done or new_pos in visited:
            break
        visited.add(new_pos)
        pos = new_pos
    return path


path_unsafe = extract_path(Q_unsafe)
path_safe = extract_path(Q_safe)

# ─────────────────────── PLOTS ─────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["Path Visualization", "Reward & Cost Curves", "Constraint Satisfaction", "Policy Maps"]
)

with tab1:
    st.subheader("Learned Paths: Unconstrained vs Safe Agent")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, path, Q, title, color in [
        (ax1, path_unsafe, Q_unsafe, "Unconstrained Agent", "#FF5722"),
        (ax2, path_safe, Q_safe, "Safe Agent (Lagrangian)", "#4CAF50"),
    ]:
        # Draw grid cells
        for r in range(grid_size):
            for c in range(grid_size):
                if danger_map[r, c]:
                    rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#FFCDD2", zorder=0)
                    ax.add_patch(rect)
                    ax.text(c, r, "X", ha="center", va="center", fontsize=10, color="#D32F2F", fontweight="bold")

        # Grid lines
        for i in range(grid_size + 1):
            ax.axhline(y=i - 0.5, color="gray", linewidth=0.5)
            ax.axvline(x=i - 0.5, color="gray", linewidth=0.5)

        # Start and goal
        ax.plot(start[1], start[0], "s", color="#2196F3", markersize=20, zorder=5, label="Start")
        ax.plot(goal[1], goal[0], "*", color="#FFD700", markersize=25, markeredgecolor="black",
                markeredgewidth=1, zorder=5, label="Goal")

        # Path
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        ax.plot(cols, rows, "-o", color=color, linewidth=3, markersize=6, zorder=3, alpha=0.8)

        # Count danger cells visited
        danger_count = sum(1 for p in path if danger_map[p])
        reached_goal = path[-1] == goal

        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nDanger cells hit: {danger_count} | Goal reached: {reached_goal}",
                      fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    c1, c2 = st.columns(2)
    danger_unsafe = sum(1 for p in path_unsafe if danger_map[p])
    danger_safe = sum(1 for p in path_safe if danger_map[p])
    c1.metric("Unconstrained: danger visits", danger_unsafe)
    c2.metric("Safe agent: danger visits", danger_safe)

    if danger_safe < danger_unsafe:
        st.success("The safe agent successfully avoids more danger zones while still reaching the goal.")
    elif danger_safe == danger_unsafe == 0:
        st.info("Both agents avoid danger zones. Try increasing danger density or changing the pattern.")
    else:
        st.warning(
            "Try increasing training episodes or adjusting lambda / cost threshold for better safety behavior."
        )

with tab2:
    st.subheader("Reward and Cost During Training")

    window = max(n_episodes // 30, 10)

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    # Rewards
    r_unsafe_ma = np.convolve(rewards_unsafe, np.ones(window) / window, mode="valid")
    r_safe_ma = np.convolve(rewards_safe, np.ones(window) / window, mode="valid")
    axes2[0].plot(r_unsafe_ma, color="#FF5722", linewidth=2, label="Unconstrained")
    axes2[0].plot(r_safe_ma, color="#4CAF50", linewidth=2, label="Safe (Lagrangian)")
    axes2[0].set_xlabel("Episode")
    axes2[0].set_ylabel("Episode Reward")
    axes2[0].set_title("Reward Curves")
    axes2[0].legend()
    axes2[0].grid(alpha=0.3)

    # Costs
    c_unsafe_ma = np.convolve(costs_unsafe, np.ones(window) / window, mode="valid")
    c_safe_ma = np.convolve(costs_safe, np.ones(window) / window, mode="valid")
    axes2[1].plot(c_unsafe_ma, color="#FF5722", linewidth=2, label="Unconstrained")
    axes2[1].plot(c_safe_ma, color="#4CAF50", linewidth=2, label="Safe (Lagrangian)")
    axes2[1].axhline(y=cost_threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold d={cost_threshold}")
    axes2[1].set_xlabel("Episode")
    axes2[1].set_ylabel("Episode Cost")
    axes2[1].set_title("Cost Curves")
    axes2[1].legend()
    axes2[1].grid(alpha=0.3)

    # Lambda evolution
    axes2[2].plot(lambda_hist, color="#9C27B0", linewidth=2)
    axes2[2].set_xlabel("Episode")
    axes2[2].set_ylabel("Lambda")
    axes2[2].set_title("Lambda (Lagrange Multiplier) Evolution")
    axes2[2].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown(
        "**Lambda adapts automatically:** it increases when costs exceed the threshold (pushing the agent "
        "to be safer) and decreases when costs are within budget (allowing more risk for more reward)."
    )

with tab3:
    st.subheader("Constraint Satisfaction Analysis")

    # Compute cumulative constraint violation
    window_eval = max(n_episodes // 10, 20)
    bins_unsafe = [np.mean(costs_unsafe[i:i + window_eval]) for i in range(0, n_episodes - window_eval + 1, window_eval)]
    bins_safe = [np.mean(costs_safe[i:i + window_eval]) for i in range(0, n_episodes - window_eval + 1, window_eval)]

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(bins_unsafe))
    width = 0.35
    ax3a.bar(x - width / 2, bins_unsafe, width, color="#FF5722", alpha=0.7, label="Unconstrained")
    ax3a.bar(x + width / 2, bins_safe, width, color="#4CAF50", alpha=0.7, label="Safe")
    ax3a.axhline(y=cost_threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold d={cost_threshold}")
    ax3a.set_xlabel("Training Phase")
    ax3a.set_ylabel("Average Episode Cost")
    ax3a.set_title("Cost by Training Phase")
    ax3a.legend()
    ax3a.grid(axis="y", alpha=0.3)

    # Scatter: reward vs cost (final portion of training)
    final_n = min(200, n_episodes)
    ax3b.scatter(costs_unsafe[-final_n:], rewards_unsafe[-final_n:],
                 alpha=0.3, color="#FF5722", s=15, label="Unconstrained")
    ax3b.scatter(costs_safe[-final_n:], rewards_safe[-final_n:],
                 alpha=0.3, color="#4CAF50", s=15, label="Safe")
    ax3b.axvline(x=cost_threshold, color="black", linestyle="--", linewidth=2, label=f"d={cost_threshold}")
    ax3b.set_xlabel("Episode Cost")
    ax3b.set_ylabel("Episode Reward")
    ax3b.set_title("Reward vs. Cost Trade-off (last episodes)")
    ax3b.legend()
    ax3b.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # Summary metrics
    final_cost_unsafe = np.mean(costs_unsafe[-window_eval:])
    final_cost_safe = np.mean(costs_safe[-window_eval:])
    final_reward_unsafe = np.mean(rewards_unsafe[-window_eval:])
    final_reward_safe = np.mean(rewards_safe[-window_eval:])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unconstrained Avg Cost", f"{final_cost_unsafe:.2f}")
    c2.metric("Safe Avg Cost", f"{final_cost_safe:.2f}")
    c3.metric("Unconstrained Avg Reward", f"{final_reward_unsafe:.2f}")
    c4.metric("Safe Avg Reward", f"{final_reward_safe:.2f}")

    if final_cost_safe <= cost_threshold:
        st.success(f"Safe agent satisfies the constraint! Average cost {final_cost_safe:.2f} <= threshold {cost_threshold}")
    else:
        st.warning(
            f"Safe agent's cost ({final_cost_safe:.2f}) is above threshold ({cost_threshold}). "
            "Try more episodes, higher lambda_lr, or a more lenient threshold."
        )

with tab4:
    st.subheader("Learned Policy Maps")
    st.markdown("Arrows show the preferred action at each cell. Red-shaded cells are danger zones.")

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))

    arrow_map = {0: (0, -0.35), 1: (0, 0.35), 2: (-0.35, 0), 3: (0.35, 0)}

    for ax, Q, title, color in [
        (ax4a, Q_unsafe, "Unconstrained Policy", "#FF5722"),
        (ax4b, Q_safe, "Safe Policy (Lagrangian)", "#4CAF50"),
    ]:
        # Background
        for r in range(grid_size):
            for c in range(grid_size):
                if danger_map[r, c]:
                    rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#FFCDD2", zorder=0)
                    ax.add_patch(rect)

        # Grid lines
        for i in range(grid_size + 1):
            ax.axhline(y=i - 0.5, color="gray", linewidth=0.5)
            ax.axvline(x=i - 0.5, color="gray", linewidth=0.5)

        # Arrows
        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) == goal:
                    continue
                s = state_idx((r, c))
                best_a = int(np.argmax(Q[s]))
                dx, dy = arrow_map[best_a]
                ax.arrow(c, r, dx, dy, head_width=0.15, head_length=0.08,
                         fc=color, ec=color, alpha=0.8, zorder=2)

        ax.plot(start[1], start[0], "s", color="#2196F3", markersize=18, zorder=5)
        ax.plot(goal[1], goal[0], "*", color="#FFD700", markersize=22, markeredgecolor="black",
                markeredgewidth=1, zorder=5)

        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    st.info(
        "Notice how the safe agent's arrows tend to route around danger zones, "
        "while the unconstrained agent may cut through them for a shorter path."
    )

# ─────────────────────── formulas ──────────────────────────────
st.divider()
st.subheader("Key Formulas")

st.latex(r"\text{CMDP Goal: } \max_\theta J(\theta) \quad \text{s.t.} \quad J_c(\theta) \leq d")
st.latex(r"L(\theta, \lambda) = J(\theta) - \lambda \left( J_c(\theta) - d \right)")
st.latex(r"\lambda \leftarrow \max\!\left(0,\; \lambda + \eta_\lambda \left( J_c(\theta) - d \right)\right)")

st.markdown(
    f"- **J(theta):** Expected total reward (we maximize this)\n"
    f"- **J_c(theta):** Expected total cost (must stay below **d = {cost_threshold}**)\n"
    f"- **lambda:** Lagrange multiplier (auto-adjusts; final value = {lambda_hist[-1]:.3f})\n"
    f"- Modified Q-update uses reward **r - lambda * c** instead of just **r**"
)
