"""
Project 01: RLHF — Reinforcement Learning from Human Feedback
Interactive Streamlit App demonstrating the RLHF pipeline.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

st.set_page_config(page_title="RLHF Demo", layout="wide")
st.title("🤖 RLHF: Reinforcement Learning from Human Feedback")
st.markdown("**Simulating how language models are aligned with human preferences**")

# --- Sidebar ---
st.sidebar.header("⚙️ RLHF Parameters")
n_responses = st.sidebar.slider("Vocabulary Size (tokens)", 5, 20, 10)
n_prompts = st.sidebar.slider("Number of Prompts", 10, 100, 50)
n_comparisons = st.sidebar.slider("Human Comparisons per Prompt", 2, 10, 5)
ppo_epochs = st.sidebar.slider("PPO Training Epochs", 10, 200, 100)
lr = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
kl_coeff = st.sidebar.slider("KL Penalty (β)", 0.0, 1.0, 0.1)
clip_eps = st.sidebar.slider("PPO Clip ε", 0.05, 0.5, 0.2)

tab1, tab2, tab3, tab4 = st.tabs([
    "📖 How RLHF Works", "🏆 Reward Model", "🎯 PPO Training", "📊 Results"
])

# --- Tab 1: Explanation ---
with tab1:
    st.header("The 3 Steps of RLHF")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Step 1: SFT")
        st.markdown("""
        **Supervised Fine-Tuning**
        - Start with pre-trained LLM
        - Fine-tune on human demonstrations
        - Creates a "decent" starting policy
        """)
    with col2:
        st.subheader("Step 2: Reward Model")
        st.markdown("""
        **Learn Human Preferences**
        - Generate pairs of responses
        - Humans rank: which is better?
        - Train model to predict rankings
        - `P(A>B) = σ(R(A) - R(B))`
        """)
    with col3:
        st.subheader("Step 3: PPO")
        st.markdown("""
        **RL Fine-Tuning**
        - LLM generates responses
        - Reward model scores them
        - PPO updates LLM for higher rewards
        - KL penalty prevents drift
        """)

    st.markdown("---")
    st.latex(r"L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]")
    st.latex(r"\text{Final Loss} = -R_{\text{reward model}} + \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{SFT}})")

# --- Simulate RLHF Pipeline ---
np.random.seed(42)

# True "human preference" scores (hidden ground truth)
true_quality = np.random.randn(n_responses)
true_quality = (true_quality - true_quality.min()) / (true_quality.max() - true_quality.min())

# SFT policy: slightly informed but noisy
sft_logits = true_quality * 0.3 + np.random.randn(n_responses) * 0.5
sft_probs = np.exp(sft_logits) / np.sum(np.exp(sft_logits))

# --- Tab 2: Reward Model Training ---
with tab2:
    st.header("Training the Reward Model")
    st.markdown("We simulate human preference comparisons and train a reward model using the **Bradley-Terry model**.")

    # Generate comparison data
    reward_weights = np.random.randn(n_responses) * 0.1
    reward_losses = []

    for epoch in range(100):
        epoch_loss = 0
        for _ in range(n_comparisons * n_prompts):
            # Pick two random responses
            i, j = np.random.choice(n_responses, 2, replace=False)
            # Human prefers the one with higher true quality (with noise)
            p_prefer_i = 1 / (1 + np.exp(-(true_quality[i] - true_quality[j]) * 3))
            human_prefers_i = np.random.random() < p_prefer_i

            # Bradley-Terry loss
            r_diff = reward_weights[i] - reward_weights[j]
            pred_prob = 1 / (1 + np.exp(-r_diff))
            if human_prefers_i:
                loss = -np.log(pred_prob + 1e-8)
                reward_weights[i] += lr * (1 - pred_prob)
                reward_weights[j] -= lr * (1 - pred_prob)
            else:
                loss = -np.log(1 - pred_prob + 1e-8)
                reward_weights[i] -= lr * pred_prob
                reward_weights[j] += lr * pred_prob
            epoch_loss += loss
        reward_losses.append(epoch_loss / (n_comparisons * n_prompts))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(reward_losses, color='crimson', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Bradley-Terry Loss')
        ax.set_title('Reward Model Training Loss')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        correlation = np.corrcoef(true_quality, reward_weights)[0, 1]
        ax.scatter(true_quality, reward_weights, alpha=0.7, s=100, c='steelblue', edgecolors='navy')
        ax.set_xlabel('True Human Quality Score')
        ax.set_ylabel('Learned Reward')
        ax.set_title(f'Reward Model Accuracy (r={correlation:.3f})')
        ax.plot([0, 1], [min(reward_weights), max(reward_weights)], 'r--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    st.success(f"Reward model correlation with true preferences: **{correlation:.3f}**")

# --- Tab 3: PPO Training ---
with tab3:
    st.header("PPO Fine-Tuning")
    st.markdown("Using the learned reward model to optimize the language model policy via PPO with KL penalty.")

    # PPO training
    policy_logits = sft_logits.copy()
    sft_log_probs = np.log(sft_probs + 1e-8)

    reward_history = []
    kl_history = []
    entropy_history = []
    policy_history = []

    for epoch in range(ppo_epochs):
        # Current policy
        probs = np.exp(policy_logits) / np.sum(np.exp(policy_logits))
        log_probs = np.log(probs + 1e-8)

        # Sample actions and compute rewards
        actions = np.random.choice(n_responses, size=32, p=probs)
        rewards = reward_weights[actions]

        # KL divergence
        kl = np.sum(probs * (log_probs - sft_log_probs))

        # Entropy
        entropy = -np.sum(probs * log_probs)

        # PPO update (simplified)
        for a in actions:
            advantage = reward_weights[a] - np.sum(probs * reward_weights)
            old_prob = probs[a]

            # Policy gradient with clipping
            ratio = probs[a] / (old_prob + 1e-8)
            clipped_ratio = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
            pg_loss = -min(ratio * advantage, clipped_ratio * advantage)

            # KL penalty
            kl_penalty = kl_coeff * (log_probs[a] - sft_log_probs[a])

            # Update
            policy_logits[a] += lr * (advantage - kl_penalty)

        reward_history.append(np.mean(rewards))
        kl_history.append(kl)
        entropy_history.append(entropy)
        policy_history.append(probs.copy())

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(reward_history, color='green', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Mean Reward')
    axes[0].set_title('Reward During PPO Training')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(kl_history, color='orange', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('KL Divergence')
    axes[1].set_title(f'KL(π || π_SFT), β={kl_coeff}')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(entropy_history, color='purple', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Entropy')
    axes[2].set_title('Policy Entropy')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# --- Tab 4: Results ---
with tab4:
    st.header("Before vs After RLHF")

    final_probs = np.exp(policy_logits) / np.sum(np.exp(policy_logits))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(n_responses)
        width = 0.25
        ax.bar(x - width, sft_probs, width, label='SFT Policy', color='skyblue', edgecolor='navy')
        ax.bar(x, final_probs, width, label='RLHF Policy', color='salmon', edgecolor='darkred')
        ax.bar(x + width, true_quality / true_quality.sum(), width, label='True Quality', color='lightgreen', edgecolor='darkgreen')
        ax.set_xlabel('Response Token')
        ax.set_ylabel('Probability')
        ax.set_title('Policy Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close()

    with col2:
        # Quality improvement
        sft_expected = np.sum(sft_probs * true_quality)
        rlhf_expected = np.sum(final_probs * true_quality)
        random_expected = np.mean(true_quality)

        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(['Random', 'SFT', 'RLHF'], [random_expected, sft_expected, rlhf_expected],
                       color=['gray', 'skyblue', 'salmon'], edgecolor='black')
        ax.set_ylabel('Expected Quality Score')
        ax.set_title('Expected Response Quality')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, [random_expected, sft_expected, rlhf_expected]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    improvement = ((rlhf_expected - sft_expected) / sft_expected) * 100
    st.metric("Quality Improvement (RLHF vs SFT)", f"{improvement:+.1f}%")

    st.markdown("---")
    st.markdown("""
    ### Key Takeaways
    - **SFT** gives a reasonable starting point but doesn't optimize for quality
    - **Reward Model** learns to approximate human preferences from comparisons
    - **PPO + KL penalty** improves the policy while staying close to SFT
    - The **KL coefficient β** controls the explore-exploit tradeoff
    """)
