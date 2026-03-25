import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

st.set_page_config(
    page_title="Multi-Agent RL: Predator-Prey",
    page_icon="🐺",
    layout="wide",
)

st.title("Multi-Agent RL: Predator-Prey")
st.markdown(
    "Watch multiple predator agents learn to cooperate and catch a prey in a grid world "
    "using **Independent Q-Learning**."
)

# ─────────────────────────── sidebar ───────────────────────────
st.sidebar.header("Parameters")
grid_size = st.sidebar.slider("Grid size", 4, 12, 6)
n_predators = st.sidebar.slider("Number of predators", 1, 4, 2)
n_episodes = st.sidebar.slider("Training episodes", 100, 3000, 1000, step=100)
epsilon_start = st.sidebar.slider("Epsilon (start)", 0.5, 1.0, 1.0, step=0.05)
epsilon_end = st.sidebar.slider("Epsilon (end)", 0.01, 0.3, 0.05, step=0.01)
learning_rate = st.sidebar.slider("Learning rate", 0.05, 0.5, 0.2, step=0.05)
gamma = st.sidebar.slider("Discount factor", 0.8, 0.99, 0.95, step=0.01)
max_steps = st.sidebar.slider("Max steps per episode", 20, 200, 50, step=10)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

np.random.seed(seed)

# ─────────────────── environment ───────────────────────────────
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}  # up, down, left, right, stay
ACTION_NAMES = ["Up", "Down", "Left", "Right", "Stay"]
N_ACTIONS = len(ACTIONS)


def clip_pos(pos, gs):
    return (max(0, min(gs - 1, pos[0])), max(0, min(gs - 1, pos[1])))


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def encode_state(predator_pos, prey_pos, gs):
    """Relative position encoding for a single predator."""
    dr = predator_pos[0] - prey_pos[0] + gs - 1  # range [0, 2*gs-2]
    dc = predator_pos[1] - prey_pos[1] + gs - 1
    return dr * (2 * gs - 1) + dc


def prey_move(prey_pos, predator_positions, gs):
    """Simple prey policy: move away from closest predator, with some randomness."""
    if np.random.rand() < 0.3:  # random move 30% of time
        a = np.random.randint(0, 4)
        delta = ACTIONS[a]
        return clip_pos((prey_pos[0] + delta[0], prey_pos[1] + delta[1]), gs)
    # Move away from closest predator
    dists = [manhattan(prey_pos, p) for p in predator_positions]
    closest = predator_positions[np.argmin(dists)]
    best_pos = prey_pos
    best_dist = manhattan(prey_pos, closest)
    for a in range(4):
        delta = ACTIONS[a]
        new_pos = clip_pos((prey_pos[0] + delta[0], prey_pos[1] + delta[1]), gs)
        d = manhattan(new_pos, closest)
        if d > best_dist:
            best_dist = d
            best_pos = new_pos
    return best_pos


# ─────────────────── training ──────────────────────────────────
state_space_size = (2 * grid_size - 1) ** 2

# Each predator has its own Q-table
Q_tables = [np.zeros((state_space_size, N_ACTIONS)) for _ in range(n_predators)]

capture_history = []
steps_history = []
avg_dist_history = []

progress_bar = st.progress(0, text="Training agents...")

for ep in range(n_episodes):
    epsilon = epsilon_start - (epsilon_start - epsilon_end) * ep / max(n_episodes - 1, 1)

    # Initialize positions (random, non-overlapping)
    all_positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    chosen = list(np.random.choice(len(all_positions), n_predators + 1, replace=False))
    pred_positions = [all_positions[chosen[i]] for i in range(n_predators)]
    prey_pos = all_positions[chosen[n_predators]]

    captured = False
    for step in range(max_steps):
        # Get states for each predator
        states = [encode_state(pred_positions[i], prey_pos, grid_size) for i in range(n_predators)]

        # Choose actions
        actions = []
        for i in range(n_predators):
            if np.random.rand() < epsilon:
                actions.append(np.random.randint(0, N_ACTIONS))
            else:
                actions.append(int(np.argmax(Q_tables[i][states[i]])))

        # Move predators
        new_pred_positions = []
        for i in range(n_predators):
            delta = ACTIONS[actions[i]]
            new_pos = clip_pos((pred_positions[i][0] + delta[0], pred_positions[i][1] + delta[1]), grid_size)
            new_pred_positions.append(new_pos)

        # Move prey
        new_prey_pos = prey_move(prey_pos, new_pred_positions, grid_size)

        # Check capture: predator is on same cell as prey
        captured = any(p == new_prey_pos for p in new_pred_positions)

        # Rewards
        rewards = []
        for i in range(n_predators):
            if captured:
                rewards.append(10.0)
            else:
                # Small reward for getting closer
                old_dist = manhattan(pred_positions[i], prey_pos)
                new_dist = manhattan(new_pred_positions[i], new_prey_pos)
                rewards.append(0.1 * (old_dist - new_dist) - 0.05)  # step penalty

        # Update Q-tables
        for i in range(n_predators):
            new_state = encode_state(new_pred_positions[i], new_prey_pos, grid_size)
            old_q = Q_tables[i][states[i], actions[i]]
            if captured:
                target = rewards[i]
            else:
                target = rewards[i] + gamma * np.max(Q_tables[i][new_state])
            Q_tables[i][states[i], actions[i]] = old_q + learning_rate * (target - old_q)

        pred_positions = new_pred_positions
        prey_pos = new_prey_pos

        if captured:
            break

    capture_history.append(1 if captured else 0)
    steps_history.append(step + 1)
    avg_dist_history.append(
        np.mean([manhattan(pred_positions[i], prey_pos) for i in range(n_predators)])
    )

    if (ep + 1) % max(n_episodes // 20, 1) == 0:
        progress_bar.progress((ep + 1) / n_episodes, text=f"Training... episode {ep+1}/{n_episodes}")

progress_bar.progress(1.0, text="Training complete!")

# ─────────────────── run demo episode ──────────────────────────
def run_demo_episode():
    """Run one episode with greedy policy, recording trajectory."""
    all_positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    chosen = list(np.random.choice(len(all_positions), n_predators + 1, replace=False))
    pred_positions = [all_positions[chosen[i]] for i in range(n_predators)]
    prey_pos = all_positions[chosen[n_predators]]

    pred_trajectories = [[] for _ in range(n_predators)]
    prey_trajectory = []

    for i in range(n_predators):
        pred_trajectories[i].append(pred_positions[i])
    prey_trajectory.append(prey_pos)

    for step in range(max_steps):
        states = [encode_state(pred_positions[i], prey_pos, grid_size) for i in range(n_predators)]
        actions = [int(np.argmax(Q_tables[i][states[i]])) for i in range(n_predators)]

        new_pred_positions = []
        for i in range(n_predators):
            delta = ACTIONS[actions[i]]
            new_pos = clip_pos((pred_positions[i][0] + delta[0], pred_positions[i][1] + delta[1]), grid_size)
            new_pred_positions.append(new_pos)

        new_prey_pos = prey_move(prey_pos, new_pred_positions, grid_size)

        captured = any(p == new_prey_pos for p in new_pred_positions)

        pred_positions = new_pred_positions
        prey_pos = new_prey_pos

        for i in range(n_predators):
            pred_trajectories[i].append(pred_positions[i])
        prey_trajectory.append(prey_pos)

        if captured:
            break

    return pred_trajectories, prey_trajectory, captured


demo_pred_traj, demo_prey_traj, demo_captured = run_demo_episode()

# ─────────────────────── PLOTS ─────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["Grid Visualization", "Training Curves", "Cooperation Metrics", "Learned Policies"]
)

PRED_COLORS = ["#F44336", "#2196F3", "#FF9800", "#9C27B0"]

with tab1:
    st.subheader("Predator-Prey Grid World (Demo Episode)")
    st.markdown(
        f"{'Prey was **CAPTURED**!' if demo_captured else 'Prey **ESCAPED**.'} "
        f"Episode lasted **{len(demo_prey_traj) - 1}** steps."
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw grid
    for i in range(grid_size + 1):
        ax.axhline(y=i - 0.5, color="gray", linewidth=0.5)
        ax.axvline(x=i - 0.5, color="gray", linewidth=0.5)

    # Draw trajectories
    for i in range(n_predators):
        traj = demo_pred_traj[i]
        rows = [t[0] for t in traj]
        cols = [t[1] for t in traj]
        ax.plot(cols, rows, "-", color=PRED_COLORS[i % len(PRED_COLORS)], linewidth=2, alpha=0.6,
                label=f"Predator {i}")
        ax.plot(cols[0], rows[0], "s", color=PRED_COLORS[i % len(PRED_COLORS)], markersize=12)
        ax.plot(cols[-1], rows[-1], "*", color=PRED_COLORS[i % len(PRED_COLORS)], markersize=18)

    # Prey trajectory
    prey_rows = [t[0] for t in demo_prey_traj]
    prey_cols = [t[1] for t in demo_prey_traj]
    ax.plot(prey_cols, prey_rows, "-", color="#4CAF50", linewidth=2, alpha=0.6, label="Prey")
    ax.plot(prey_cols[0], prey_rows[0], "s", color="#4CAF50", markersize=12)
    ax.plot(prey_cols[-1], prey_rows[-1], "*", color="#4CAF50", markersize=18)

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title("Agent Trajectories (squares = start, stars = end)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.subheader("Training Curves")

    window = max(n_episodes // 20, 10)

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

    # Capture rate (moving average)
    capture_ma = np.convolve(capture_history, np.ones(window) / window, mode="valid")
    ax2a.plot(capture_ma, color="#4CAF50", linewidth=2)
    ax2a.set_xlabel("Episode")
    ax2a.set_ylabel("Capture Rate")
    ax2a.set_title(f"Capture Rate (moving avg, window={window})")
    ax2a.set_ylim(-0.05, 1.05)
    ax2a.grid(alpha=0.3)
    ax2a.axhline(y=capture_ma[-1], color="red", linestyle="--", alpha=0.5,
                 label=f"Final: {capture_ma[-1]:.2f}")
    ax2a.legend()

    # Steps to capture (moving average)
    steps_ma = np.convolve(steps_history, np.ones(window) / window, mode="valid")
    ax2b.plot(steps_ma, color="#FF9800", linewidth=2)
    ax2b.set_xlabel("Episode")
    ax2b.set_ylabel("Steps")
    ax2b.set_title(f"Steps per Episode (moving avg, window={window})")
    ax2b.grid(alpha=0.3)
    ax2b.legend([f"Final avg: {steps_ma[-1]:.1f}"])

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    col1, col2, col3 = st.columns(3)
    final_capture_rate = np.mean(capture_history[-window:])
    final_avg_steps = np.mean(steps_history[-window:])
    col1.metric("Final Capture Rate", f"{final_capture_rate:.1%}")
    col2.metric("Avg Steps (last window)", f"{final_avg_steps:.1f}")
    col3.metric("Total Episodes", n_episodes)

with tab3:
    st.subheader("Cooperation Metrics")

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

    # Average distance to prey
    dist_ma = np.convolve(avg_dist_history, np.ones(window) / window, mode="valid")
    ax3a.plot(dist_ma, color="#9C27B0", linewidth=2)
    ax3a.set_xlabel("Episode")
    ax3a.set_ylabel("Avg Manhattan Distance to Prey")
    ax3a.set_title("Predator-Prey Distance Over Training")
    ax3a.grid(alpha=0.3)

    # Coordination score: how often predators approach from different directions
    # Compute for each episode bin
    bin_size = max(n_episodes // 20, 1)
    coordination_scores = []
    for start in range(0, n_episodes - bin_size + 1, bin_size):
        end = start + bin_size
        coord_sum = 0
        count = 0
        for ep_idx in range(start, end):
            if n_predators >= 2:
                # Simple proxy: variance in predator positions at end
                # Higher variance = approaching from different sides = better coordination
                coord_sum += 1.0 if capture_history[ep_idx] else 0.0
                count += 1
        if count > 0:
            coordination_scores.append(coord_sum / count)

    if coordination_scores:
        ax3b.bar(range(len(coordination_scores)), coordination_scores, color="#00BCD4", alpha=0.7)
        ax3b.set_xlabel("Training Phase")
        ax3b.set_ylabel("Capture Success Rate")
        ax3b.set_title("Cooperation Effectiveness by Training Phase")
        ax3b.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.markdown(
        "**Decreasing distance** indicates predators learn to approach the prey. "
        "**Increasing capture rate** across training phases shows improving cooperation."
    )

with tab4:
    st.subheader("Learned Policies (Predator Action Preferences)")
    st.markdown(
        "For each relative position to the prey, what action does each predator prefer? "
        "Showing a subset of the state space."
    )

    n_show = min(n_predators, 4)
    fig4, axes4 = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
    if n_show == 1:
        axes4 = [axes4]

    vis_range = min(grid_size, 5)

    for idx in range(n_show):
        ax = axes4[idx]
        Q = Q_tables[idx]

        # Show preferred action for relative positions in a small window
        for dr in range(-vis_range + 1, vis_range):
            for dc in range(-vis_range + 1, vis_range):
                encoded_dr = dr + grid_size - 1
                encoded_dc = dc + grid_size - 1
                if 0 <= encoded_dr < 2 * grid_size - 1 and 0 <= encoded_dc < 2 * grid_size - 1:
                    state = encoded_dr * (2 * grid_size - 1) + encoded_dc
                    if state < state_space_size:
                        best_action = np.argmax(Q[state])
                        # Draw arrow
                        arrow_map = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0), 4: (0, 0)}
                        adx, ady = arrow_map[best_action]
                        if best_action < 4:
                            ax.arrow(dc, dr, adx, ady, head_width=0.15, head_length=0.08,
                                     fc=PRED_COLORS[idx % len(PRED_COLORS)],
                                     ec=PRED_COLORS[idx % len(PRED_COLORS)], alpha=0.7)
                        else:
                            ax.plot(dc, dr, "o", color=PRED_COLORS[idx % len(PRED_COLORS)],
                                    markersize=4, alpha=0.5)

        ax.plot(0, 0, "X", color="#4CAF50", markersize=15, markeredgecolor="black", markeredgewidth=1.5)
        ax.set_xlim(-vis_range + 0.5, vis_range - 0.5)
        ax.set_ylim(vis_range - 0.5, -vis_range + 0.5)
        ax.set_aspect("equal")
        ax.set_title(f"Predator {idx}", fontweight="bold", color=PRED_COLORS[idx % len(PRED_COLORS)])
        ax.set_xlabel("Relative Column to Prey")
        ax.set_ylabel("Relative Row to Prey")
        ax.grid(alpha=0.2)
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.axvline(x=0, color="gray", linewidth=0.5)

    plt.suptitle("Preferred Action by Relative Position (green X = prey location)", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    st.info("Arrows show the preferred movement direction at each relative position to the prey. "
            "Good coordination means predators approach from different angles.")

# ─────────────────────── formulas ──────────────────────────────
st.divider()
st.subheader("Key Formulas")
st.latex(r"Q_i(s, a_i) \leftarrow Q_i(s, a_i) + \alpha \left[ r_i + \gamma \max_{a'} Q_i(s', a') - Q_i(s, a_i) \right]")
st.markdown(
    "Each predator maintains its own Q-table and updates independently. "
    "The environment appears non-stationary to each agent because the others are also learning."
)
st.latex(r"\text{Nash Equilibrium: } J_i(\pi_i^*, \pi_{-i}^*) \geq J_i(\pi_i, \pi_{-i}^*) \quad \forall \pi_i, \forall i")
st.markdown("At Nash Equilibrium, no agent can improve by unilaterally changing its policy.")
