import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

st.set_page_config(page_title="Drug Discovery with RL", layout="wide")

st.title("Drug Discovery with Reinforcement Learning")
st.markdown("""
This demo simulates an RL agent that learns to **build molecules** (simplified as sequences of atoms)
to satisfy pharmaceutical property targets — balancing LogP, molecular weight, and toxicity.
""")

# ─── Atom definitions ───
ATOMS = {
    "C": {"weight": 12.0, "logp_contrib": 0.5, "toxicity": 0.0, "hbd": 0, "hba": 0, "symbol": "C"},
    "N": {"weight": 14.0, "logp_contrib": -1.0, "toxicity": 0.1, "hbd": 1, "hba": 1, "symbol": "N"},
    "O": {"weight": 16.0, "logp_contrib": -1.2, "toxicity": 0.05, "hbd": 1, "hba": 2, "symbol": "O"},
    "S": {"weight": 32.0, "logp_contrib": 0.8, "toxicity": 0.2, "hbd": 0, "hba": 0, "symbol": "S"},
    "F": {"weight": 19.0, "logp_contrib": 0.4, "toxicity": 0.15, "hbd": 0, "hba": 1, "symbol": "F"},
}
ATOM_NAMES = list(ATOMS.keys())
NUM_ATOMS = len(ATOM_NAMES)

# ─── Sidebar controls ───
st.sidebar.header("Molecule Targets")
target_logp = st.sidebar.slider("Target LogP", -2.0, 6.0, 2.5, 0.1)
target_mw = st.sidebar.slider("Target Molecular Weight (Da)", 50.0, 500.0, 250.0, 10.0)
mol_length = st.sidebar.slider("Molecule length (atoms)", 3, 20, 10)

st.sidebar.header("Reward Weights")
w_logp = st.sidebar.slider("Weight: LogP", 0.0, 2.0, 1.0, 0.1)
w_mw = st.sidebar.slider("Weight: MW", 0.0, 2.0, 1.0, 0.1)
w_tox = st.sidebar.slider("Weight: Toxicity penalty", 0.0, 2.0, 0.5, 0.1)
w_lipinski = st.sidebar.slider("Weight: Lipinski bonus", 0.0, 2.0, 0.8, 0.1)

st.sidebar.header("Training")
num_episodes = st.sidebar.slider("Training episodes", 50, 1000, 300, 50)
learning_rate = st.sidebar.slider("Learning rate", 0.001, 0.1, 0.02, 0.001)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

np.random.seed(seed)


# ─── Molecule property calculators ───
def calc_properties(molecule):
    """Calculate drug properties for a molecule (list of atom names)."""
    mw = sum(ATOMS[a]["weight"] for a in molecule)
    logp = sum(ATOMS[a]["logp_contrib"] for a in molecule)
    toxicity = sum(ATOMS[a]["toxicity"] for a in molecule) / len(molecule)
    hbd = sum(ATOMS[a]["hbd"] for a in molecule)
    hba = sum(ATOMS[a]["hba"] for a in molecule)
    return {"mw": mw, "logp": logp, "toxicity": toxicity, "hbd": hbd, "hba": hba}


def lipinski_score(props):
    """Count how many of Lipinski's rules are satisfied (0-4)."""
    score = 0
    if props["mw"] <= 500:
        score += 1
    if props["logp"] <= 5:
        score += 1
    if props["hbd"] <= 5:
        score += 1
    if props["hba"] <= 10:
        score += 1
    return score / 4.0


def compute_reward(molecule, target_logp, target_mw, w_logp, w_mw, w_tox, w_lipinski):
    props = calc_properties(molecule)
    r_logp = -abs(props["logp"] - target_logp) / max(abs(target_logp), 1.0)
    r_mw = -abs(props["mw"] - target_mw) / max(target_mw, 1.0)
    r_tox = -props["toxicity"]
    r_lip = lipinski_score(props)
    reward = w_logp * r_logp + w_mw * r_mw + w_tox * r_tox + w_lipinski * r_lip
    return reward, props


# ─── Softmax policy ───
def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ─── Training ───
if st.sidebar.button("Train Molecular Agent", type="primary"):
    # Policy: position-dependent atom preferences (mol_length x NUM_ATOMS)
    policy_logits = np.zeros((mol_length, NUM_ATOMS))

    reward_history = []
    logp_history = []
    mw_history = []
    tox_history = []
    lipinski_history = []
    best_reward = -np.inf
    best_molecule = None
    all_molecules = []

    progress = st.progress(0, text="Training...")

    for ep in range(num_episodes):
        # Generate molecule
        molecule = []
        log_probs = []
        for pos in range(mol_length):
            probs = softmax(policy_logits[pos])
            action = np.random.choice(NUM_ATOMS, p=probs)
            log_probs.append(np.log(probs[action] + 1e-10))
            molecule.append(ATOM_NAMES[action])

        reward, props = compute_reward(molecule, target_logp, target_mw, w_logp, w_mw, w_tox, w_lipinski)

        # REINFORCE update
        for pos in range(mol_length):
            probs = softmax(policy_logits[pos])
            action_idx = ATOM_NAMES.index(molecule[pos])
            grad = -probs.copy()
            grad[action_idx] += 1.0
            policy_logits[pos] += learning_rate * grad * reward

        reward_history.append(reward)
        logp_history.append(props["logp"])
        mw_history.append(props["mw"])
        tox_history.append(props["toxicity"])
        lipinski_history.append(lipinski_score(props))
        all_molecules.append((reward, props["logp"], props["mw"], props["toxicity"], molecule))

        if reward > best_reward:
            best_reward = reward
            best_molecule = molecule[:]

        if (ep + 1) % 20 == 0:
            progress.progress((ep + 1) / num_episodes, text=f"Episode {ep+1}/{num_episodes}")

    progress.empty()
    st.success(f"Training complete! Best reward: {best_reward:.4f}")

    # ─── Best molecule display ───
    st.subheader("Best Molecule Found")
    best_props = calc_properties(best_molecule)
    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        mol_str = " — ".join(best_molecule)
        st.code(mol_str, language=None)
    with col_m2:
        st.metric("LogP", f"{best_props['logp']:.2f}", delta=f"target: {target_logp:.1f}")
        st.metric("MW", f"{best_props['mw']:.1f} Da", delta=f"target: {target_mw:.0f}")
        st.metric("Lipinski", f"{lipinski_score(best_props)*4:.0f}/4")

    # ─── Row 1: Training curves ───
    st.subheader("Training Progress")
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        window = max(1, num_episodes // 20)
        smoothed = np.convolve(reward_history, np.ones(window) / window, mode="valid")
        ax1.plot(reward_history, alpha=0.3, color="#90CAF9")
        ax1.plot(np.arange(window - 1, len(reward_history)), smoothed, color="#1565C0", linewidth=2, label="Smoothed")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title("Total Reward Over Training")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
        ax2a.plot(logp_history, alpha=0.4, color="#66BB6A")
        ax2a.axhline(target_logp, color="red", linestyle="--", label=f"Target ({target_logp})")
        ax2a.set_ylabel("LogP")
        ax2a.legend(fontsize=8)
        ax2a.grid(True, alpha=0.3)

        ax2b.plot(mw_history, alpha=0.4, color="#FFA726")
        ax2b.axhline(target_mw, color="red", linestyle="--", label=f"Target ({target_mw:.0f})")
        ax2b.set_xlabel("Episode")
        ax2b.set_ylabel("MW (Da)")
        ax2b.legend(fontsize=8)
        ax2b.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ─── Row 2: Property distributions + Pareto front ───
    st.subheader("Property Analysis")
    col3, col4 = st.columns(2)

    with col3:
        fig3, axes3 = plt.subplots(2, 2, figsize=(6, 5))
        axes3[0, 0].hist(logp_history, bins=30, color="#4CAF50", alpha=0.7, edgecolor="black")
        axes3[0, 0].axvline(target_logp, color="red", linestyle="--", linewidth=2)
        axes3[0, 0].set_title("LogP Distribution")

        axes3[0, 1].hist(mw_history, bins=30, color="#FF9800", alpha=0.7, edgecolor="black")
        axes3[0, 1].axvline(target_mw, color="red", linestyle="--", linewidth=2)
        axes3[0, 1].set_title("MW Distribution")

        axes3[1, 0].hist(tox_history, bins=30, color="#F44336", alpha=0.7, edgecolor="black")
        axes3[1, 0].set_title("Toxicity Distribution")

        axes3[1, 1].hist(lipinski_history, bins=5, color="#2196F3", alpha=0.7, edgecolor="black")
        axes3[1, 1].set_title("Lipinski Score")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col4:
        st.markdown("**Pareto Front: LogP Accuracy vs MW Accuracy**")
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        logp_err = [abs(m[1] - target_logp) for m in all_molecules]
        mw_err = [abs(m[2] - target_mw) for m in all_molecules]
        rewards_c = [m[0] for m in all_molecules]
        sc = ax4.scatter(logp_err, mw_err, c=rewards_c, cmap="RdYlGn", alpha=0.5, s=15, edgecolors="none")
        plt.colorbar(sc, ax=ax4, label="Reward")
        ax4.set_xlabel("|LogP - Target|")
        ax4.set_ylabel("|MW - Target|")
        ax4.set_title("Pareto Front (lower-left = better)")

        # Highlight Pareto-optimal points
        combined = list(zip(logp_err, mw_err))
        pareto_idx = []
        for i, (le, me) in enumerate(combined):
            dominated = False
            for j, (le2, me2) in enumerate(combined):
                if i != j and le2 <= le and me2 <= me and (le2 < le or me2 < me):
                    dominated = True
                    break
            if not dominated:
                pareto_idx.append(i)
        if pareto_idx:
            pareto_x = [logp_err[i] for i in pareto_idx]
            pareto_y = [mw_err[i] for i in pareto_idx]
            order = np.argsort(pareto_x)
            ax4.plot(np.array(pareto_x)[order], np.array(pareto_y)[order],
                     "r-o", markersize=5, linewidth=2, label="Pareto front", zorder=5)
            ax4.legend()
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
        plt.close(fig4)

    # ─── Row 3: Molecule quality over training ───
    st.subheader("Molecule Quality Over Training")
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    quarter = max(1, num_episodes // 4)
    phases = ["First 25%", "25-50%", "50-75%", "Last 25%"]
    phase_rewards = [
        reward_history[:quarter],
        reward_history[quarter:2*quarter],
        reward_history[2*quarter:3*quarter],
        reward_history[3*quarter:],
    ]
    bp = ax5.boxplot(phase_rewards, labels=phases, patch_artist=True)
    colors = ["#FFCDD2", "#FFE0B2", "#C8E6C9", "#B3E5FC"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax5.set_ylabel("Reward")
    ax5.set_title("Reward Distribution by Training Phase")
    ax5.grid(True, alpha=0.3, axis="y")
    st.pyplot(fig5)
    plt.close(fig5)

    # ─── Learned policy heatmap ───
    st.subheader("Learned Policy (Atom Preferences by Position)")
    fig6, ax6 = plt.subplots(figsize=(10, 3))
    probs_matrix = softmax(policy_logits)
    im = ax6.imshow(probs_matrix.T, aspect="auto", cmap="Blues")
    ax6.set_xlabel("Position in Molecule")
    ax6.set_ylabel("Atom Type")
    ax6.set_yticks(range(NUM_ATOMS))
    ax6.set_yticklabels(ATOM_NAMES)
    ax6.set_title("P(atom | position) — Learned Generation Policy")
    plt.colorbar(im, ax=ax6)
    st.pyplot(fig6)
    plt.close(fig6)

else:
    st.info("Configure parameters in the sidebar and click **Train Molecular Agent** to start.")
    st.markdown("""
    ### How This Demo Works

    1. **Simplified Molecules**: Each molecule is a sequence of atoms (C, N, O, S, F), each contributing to drug properties.
    2. **Properties Computed**: LogP (lipophilicity), molecular weight, toxicity, and Lipinski compliance.
    3. **Multi-Objective Reward**: Weighted combination of how close each property is to the target.
    4. **Policy Gradient**: The agent learns which atoms to place at each position using REINFORCE.
    5. **Pareto Front**: Visualizes the trade-off between competing objectives.

    **Adjust the reward weights** to see how the agent prioritizes different drug properties.
    """)
