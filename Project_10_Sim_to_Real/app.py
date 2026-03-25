import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sim-to-Real Transfer", layout="wide")

st.title("Sim-to-Real Transfer")
st.markdown("""
This demo simulates a **CartPole-like balancing task** comparing three training strategies:
**Sim-Only**, **Domain Randomized**, and **Real-World** training, and evaluates how each transfers to the noisy real world.
""")

# ─── Sidebar ───
st.sidebar.header("Environment Parameters")
real_noise = st.sidebar.slider("Real-world noise level", 0.0, 2.0, 0.5, 0.05)
real_gravity_offset = st.sidebar.slider("Real gravity offset", -2.0, 2.0, 0.3, 0.1)
real_friction = st.sidebar.slider("Real friction factor", 0.5, 2.0, 1.2, 0.1)

st.sidebar.header("Domain Randomization")
rand_noise_range = st.sidebar.slider("Randomization: noise range", 0.0, 3.0, 1.0, 0.1)
rand_gravity_range = st.sidebar.slider("Randomization: gravity range", 0.0, 3.0, 1.5, 0.1)
rand_friction_range = st.sidebar.slider("Randomization: friction range", 0.0, 1.5, 0.5, 0.1)
num_sim_variants = st.sidebar.slider("Number of sim variants", 3, 30, 10)

st.sidebar.header("Training")
train_episodes = st.sidebar.slider("Training episodes", 50, 500, 200, 50)
eval_episodes = st.sidebar.slider("Evaluation episodes", 10, 100, 30)
episode_length = st.sidebar.slider("Steps per episode", 20, 200, 100, 10)
learning_rate = st.sidebar.slider("Learning rate", 0.001, 0.1, 0.02, 0.001)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

np.random.seed(seed)

# ─── Simplified CartPole-like dynamics ───
STATE_DIM = 4  # (x, x_dot, theta, theta_dot)
ACTION_DIM = 2  # left, right


def step_dynamics(state, action, noise=0.0, gravity_offset=0.0, friction=1.0):
    """Simplified cartpole-like step."""
    x, x_dot, theta, theta_dot = state
    force = 1.0 if action == 1 else -1.0

    gravity = 9.8 + gravity_offset
    cart_mass = 1.0
    pole_mass = 0.1
    pole_length = 0.5
    dt = 0.02

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    total_mass = cart_mass + pole_mass

    # Simplified equations of motion
    temp = (force + pole_mass * pole_length * theta_dot ** 2 * sin_t) / total_mass
    theta_acc = (gravity * sin_t - cos_t * temp) / (
        pole_length * (4.0 / 3.0 - pole_mass * cos_t ** 2 / total_mass)
    )
    x_acc = temp - pole_mass * pole_length * theta_acc * cos_t / total_mass

    # Apply friction
    x_dot *= friction
    theta_dot *= friction

    # Integrate
    x += dt * x_dot
    x_dot += dt * x_acc
    theta += dt * theta_dot
    theta_dot += dt * theta_acc

    # Add noise
    if noise > 0:
        x += np.random.normal(0, noise * 0.01)
        x_dot += np.random.normal(0, noise * 0.01)
        theta += np.random.normal(0, noise * 0.005)
        theta_dot += np.random.normal(0, noise * 0.005)

    return np.array([x, x_dot, theta, theta_dot])


def is_done(state):
    x, _, theta, _ = state
    return abs(x) > 2.4 or abs(theta) > 0.21


def get_reward(state):
    if is_done(state):
        return 0.0
    return 1.0


# ─── Linear policy ───
def select_action(state, weights):
    logits = state @ weights
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    return np.random.choice(ACTION_DIM, p=probs), probs


def run_episode(weights, noise=0.0, gravity_offset=0.0, friction=1.0, max_steps=100):
    """Run one episode and return (total_reward, trajectory_data)."""
    state = np.array([0.0, 0.0, np.random.uniform(-0.05, 0.05), 0.0])
    total_reward = 0.0
    states, actions, rewards = [], [], []

    for _ in range(max_steps):
        action, probs = select_action(state, weights)
        r = get_reward(state)
        states.append(state.copy())
        actions.append(action)
        rewards.append(r)
        total_reward += r

        state = step_dynamics(state, action, noise, gravity_offset, friction)
        if is_done(state):
            break

    return total_reward, states, actions, rewards


def train_policy(episodes, lr, noise=0.0, gravity_offset=0.0, friction=1.0,
                 max_steps=100, domain_random=False, rand_params=None):
    """Train using REINFORCE with optional domain randomization."""
    weights = np.random.randn(STATE_DIM, ACTION_DIM) * 0.01
    reward_history = []

    for ep in range(episodes):
        # Domain randomization: sample env parameters
        if domain_random and rand_params is not None:
            ep_noise = np.random.uniform(0, rand_params["noise_range"])
            ep_grav = np.random.uniform(-rand_params["gravity_range"], rand_params["gravity_range"])
            ep_fric = np.random.uniform(max(0.5, 1.0 - rand_params["friction_range"]),
                                        1.0 + rand_params["friction_range"])
        else:
            ep_noise = noise
            ep_grav = gravity_offset
            ep_fric = friction

        total_reward, states, actions, rewards = run_episode(
            weights, ep_noise, ep_grav, ep_fric, max_steps
        )
        reward_history.append(total_reward)

        # REINFORCE update
        if len(states) > 0:
            returns = np.zeros(len(rewards))
            G = 0
            for t in reversed(range(len(rewards))):
                G = rewards[t] + 0.99 * G
                returns[t] = G
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            for t in range(len(states)):
                state_t = states[t]
                logits = state_t @ weights
                probs = np.exp(logits - np.max(logits))
                probs /= probs.sum()
                grad = -probs
                grad[actions[t]] += 1.0
                weights += lr * np.outer(state_t, grad) * returns[t]

    return weights, reward_history


def evaluate_policy(weights, episodes, noise, gravity_offset, friction, max_steps):
    """Evaluate a trained policy and return list of episode rewards."""
    rewards = []
    for _ in range(episodes):
        r, _, _, _ = run_episode(weights, noise, gravity_offset, friction, max_steps)
        rewards.append(r)
    return rewards


# ─── Run experiment ───
if st.sidebar.button("Run Sim-to-Real Experiment", type="primary"):
    progress = st.progress(0, text="Training Sim-Only policy...")

    # 1. Sim-only training (clean dynamics)
    weights_sim, hist_sim = train_policy(
        train_episodes, learning_rate, noise=0.0, gravity_offset=0.0, friction=1.0,
        max_steps=episode_length
    )
    progress.progress(33, text="Training Domain-Randomized policy...")

    # 2. Domain-randomized training
    rand_params = {
        "noise_range": rand_noise_range,
        "gravity_range": rand_gravity_range,
        "friction_range": rand_friction_range,
    }
    weights_dr, hist_dr = train_policy(
        train_episodes, learning_rate, domain_random=True, rand_params=rand_params,
        max_steps=episode_length
    )
    progress.progress(66, text="Training Real-World policy...")

    # 3. Real-world training (expensive but accurate)
    weights_real, hist_real = train_policy(
        train_episodes, learning_rate, noise=real_noise,
        gravity_offset=real_gravity_offset, friction=real_friction,
        max_steps=episode_length
    )
    progress.progress(90, text="Evaluating all policies...")

    # ─── Evaluate all three in REAL environment ───
    eval_sim = evaluate_policy(weights_sim, eval_episodes, real_noise, real_gravity_offset, real_friction, episode_length)
    eval_dr = evaluate_policy(weights_dr, eval_episodes, real_noise, real_gravity_offset, real_friction, episode_length)
    eval_real = evaluate_policy(weights_real, eval_episodes, real_noise, real_gravity_offset, real_friction, episode_length)

    # Also evaluate in sim (for gap measurement)
    eval_sim_in_sim = evaluate_policy(weights_sim, eval_episodes, 0.0, 0.0, 1.0, episode_length)

    progress.empty()
    st.success("Experiment complete!")

    # ─── Summary metrics ───
    st.subheader("Transfer Performance Summary")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Sim-Only (in Real)", f"{np.mean(eval_sim):.1f}", delta=f"gap: {np.mean(eval_sim_in_sim) - np.mean(eval_sim):.1f}")
    mc2.metric("Domain Randomized (in Real)", f"{np.mean(eval_dr):.1f}")
    mc3.metric("Real-Trained (in Real)", f"{np.mean(eval_real):.1f}")
    gap_closed = (np.mean(eval_dr) - np.mean(eval_sim)) / (np.mean(eval_real) - np.mean(eval_sim) + 1e-8) * 100
    mc4.metric("Reality Gap Closed by DR", f"{min(max(gap_closed, 0), 100):.0f}%")

    # ─── Row 1: Training curves + Transfer comparison ───
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Curves")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        window = max(1, train_episodes // 15)
        for hist, label, color in [
            (hist_sim, "Sim-Only", "#2196F3"),
            (hist_dr, "Domain Randomized", "#4CAF50"),
            (hist_real, "Real-World", "#FF9800"),
        ]:
            smoothed = np.convolve(hist, np.ones(window) / window, mode="valid")
            ax1.plot(smoothed, label=label, color=color, linewidth=2)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Episode Reward")
        ax1.set_title("Training Progress (in respective environments)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.subheader("Transfer Performance (Evaluated in Real World)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        data = [eval_sim, eval_dr, eval_real]
        labels = ["Sim-Only", "Domain\nRandomized", "Real-Trained"]
        colors = ["#BBDEFB", "#C8E6C9", "#FFE0B2"]
        bp = ax2.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
        means = [np.mean(d) for d in data]
        ax2.plot([1, 2, 3], means, "ro-", markersize=8, label="Mean", zorder=5)
        ax2.set_ylabel("Episode Reward (in Real World)")
        ax2.set_title("Transfer Performance Gap")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
        st.pyplot(fig2)
        plt.close(fig2)

    # ─── Row 2: Robustness curves + Domain parameter distributions ───
    st.subheader("Robustness Analysis")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Performance vs Noise Level**")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        noise_levels = np.linspace(0, 2.0, 15)
        perf_sim = []
        perf_dr = []
        perf_real = []
        for nl in noise_levels:
            r_s = evaluate_policy(weights_sim, 10, nl, real_gravity_offset, real_friction, episode_length)
            r_d = evaluate_policy(weights_dr, 10, nl, real_gravity_offset, real_friction, episode_length)
            r_r = evaluate_policy(weights_real, 10, nl, real_gravity_offset, real_friction, episode_length)
            perf_sim.append(np.mean(r_s))
            perf_dr.append(np.mean(r_d))
            perf_real.append(np.mean(r_r))

        ax3.plot(noise_levels, perf_sim, "o-", color="#2196F3", label="Sim-Only", linewidth=2)
        ax3.plot(noise_levels, perf_dr, "s-", color="#4CAF50", label="Domain Randomized", linewidth=2)
        ax3.plot(noise_levels, perf_real, "^-", color="#FF9800", label="Real-Trained", linewidth=2)
        ax3.axvline(real_noise, color="red", linestyle="--", alpha=0.5, label=f"Real noise = {real_noise}")
        ax3.set_xlabel("Noise Level")
        ax3.set_ylabel("Avg Episode Reward")
        ax3.set_title("Robustness Curve (varying noise)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        plt.close(fig3)

    with col4:
        st.markdown("**Domain Randomization Parameter Distributions**")
        fig4, axes4 = plt.subplots(1, 3, figsize=(6, 4))

        # Noise distribution
        noise_samples = np.random.uniform(0, rand_noise_range, 500)
        axes4[0].hist(noise_samples, bins=20, color="#81D4FA", edgecolor="black", alpha=0.8)
        axes4[0].axvline(real_noise, color="red", linewidth=2, linestyle="--", label="Real")
        axes4[0].set_title("Noise", fontsize=9)
        axes4[0].legend(fontsize=7)

        # Gravity distribution
        grav_samples = np.random.uniform(-rand_gravity_range, rand_gravity_range, 500)
        axes4[1].hist(grav_samples, bins=20, color="#A5D6A7", edgecolor="black", alpha=0.8)
        axes4[1].axvline(real_gravity_offset, color="red", linewidth=2, linestyle="--", label="Real")
        axes4[1].set_title("Gravity Offset", fontsize=9)
        axes4[1].legend(fontsize=7)

        # Friction distribution
        fric_lo = max(0.5, 1.0 - rand_friction_range)
        fric_hi = 1.0 + rand_friction_range
        fric_samples = np.random.uniform(fric_lo, fric_hi, 500)
        axes4[2].hist(fric_samples, bins=20, color="#FFCC80", edgecolor="black", alpha=0.8)
        axes4[2].axvline(real_friction, color="red", linewidth=2, linestyle="--", label="Real")
        axes4[2].set_title("Friction", fontsize=9)
        axes4[2].legend(fontsize=7)

        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    # ─── Row 3: Gravity robustness + detailed gap analysis ───
    st.subheader("Detailed Transfer Analysis")
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("**Performance vs Gravity Offset**")
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        grav_levels = np.linspace(-2.0, 2.0, 15)
        perf_sim_g = []
        perf_dr_g = []
        for gl in grav_levels:
            r_s = evaluate_policy(weights_sim, 10, real_noise, gl, real_friction, episode_length)
            r_d = evaluate_policy(weights_dr, 10, real_noise, gl, real_friction, episode_length)
            perf_sim_g.append(np.mean(r_s))
            perf_dr_g.append(np.mean(r_d))
        ax5.plot(grav_levels, perf_sim_g, "o-", color="#2196F3", label="Sim-Only", linewidth=2)
        ax5.plot(grav_levels, perf_dr_g, "s-", color="#4CAF50", label="Domain Randomized", linewidth=2)
        ax5.axvline(real_gravity_offset, color="red", linestyle="--", alpha=0.5, label=f"Real = {real_gravity_offset}")
        ax5.set_xlabel("Gravity Offset")
        ax5.set_ylabel("Avg Episode Reward")
        ax5.set_title("Robustness to Gravity Variation")
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        st.pyplot(fig5)
        plt.close(fig5)

    with col6:
        st.markdown("**Sim vs Real Performance for Each Policy**")
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        sim_perfs = [
            np.mean(evaluate_policy(weights_sim, eval_episodes, 0, 0, 1.0, episode_length)),
            np.mean(evaluate_policy(weights_dr, eval_episodes, 0, 0, 1.0, episode_length)),
            np.mean(evaluate_policy(weights_real, eval_episodes, 0, 0, 1.0, episode_length)),
        ]
        real_perfs = [np.mean(eval_sim), np.mean(eval_dr), np.mean(eval_real)]
        x_pos = np.arange(3)
        width = 0.3
        ax6.bar(x_pos - width / 2, sim_perfs, width, label="In Simulation", color="#90CAF9", edgecolor="black")
        ax6.bar(x_pos + width / 2, real_perfs, width, label="In Real World", color="#EF9A9A", edgecolor="black")
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(["Sim-Only", "Domain\nRandomized", "Real-Trained"])
        ax6.set_ylabel("Avg Episode Reward")
        ax6.set_title("Reality Gap: Sim vs Real Performance")
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis="y")
        st.pyplot(fig6)
        plt.close(fig6)

    st.info(
        "**Key takeaway**: Domain Randomization bridges the reality gap by exposing the policy to "
        "diverse simulation parameters during training, producing a robust policy that transfers well "
        "without ever needing real-world training data."
    )

else:
    st.info("Configure parameters in the sidebar and click **Run Sim-to-Real Experiment** to start.")
    st.markdown("""
    ### How This Demo Works

    1. **CartPole-like Task**: Balance a pole on a cart. The "sim" has clean dynamics; the "real" has noise, gravity offset, and friction changes.
    2. **Three Policies**:
       - **Sim-Only**: Trained in clean simulation (no noise, default physics)
       - **Domain Randomized**: Trained on many simulation variants with randomized parameters
       - **Real-Trained**: Trained directly in the noisy real environment (gold standard)
    3. **Evaluation**: All policies are tested in the "real" environment to measure transfer.
    4. **Robustness Curves**: Show how each policy degrades as environmental parameters change.

    **Domain Randomization** aims to close the gap between Sim-Only and Real-Trained without needing any real-world data.
    """)
