import streamlit as st
st.set_page_config(page_title="RLAIF", page_icon="🤖", layout="wide")

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Simulated Preference Environment
# ─────────────────────────────────────────────
# We simulate a scenario where a "policy" produces responses of varying quality.
# Quality is a scalar from 0 to 1. The goal is to learn a reward model from
# preference labels (either from humans or AI) and use it to improve the policy.

def true_quality(response_features):
    """
    True quality function (unknown to the agent).
    response_features: array of shape (n_features,) in [0, 1]
    Quality = weighted combination emphasizing helpfulness and honesty.
    """
    weights = np.array([0.4, 0.3, 0.2, 0.1])  # helpfulness, honesty, harmlessness, style
    return np.dot(response_features[:len(weights)], weights)


def human_preference(q1, q2, noise_level):
    """
    Human labels: correct preference + random noise.
    Returns 1 if response 1 preferred, 0 otherwise.
    """
    rng = np.random.random()
    correct = 1 if q1 > q2 else 0
    if rng < noise_level:
        return 1 - correct  # flip with probability = noise_level
    return correct


def ai_preference(q1, q2, features1, features2, ai_accuracy, verbosity_bias):
    """
    AI judge labels: generally accurate but with systematic verbosity bias.
    The AI judge slightly prefers longer/more verbose responses regardless of quality.
    """
    # Simulate verbosity as the last feature
    verbosity_diff = features1[3] - features2[3]  # style feature as proxy for verbosity
    ai_score1 = q1 + verbosity_bias * features1[3]
    ai_score2 = q2 + verbosity_bias * features2[3]

    correct = 1 if q1 > q2 else 0
    ai_choice = 1 if ai_score1 > ai_score2 else 0

    # AI has base accuracy; when it errs, it follows its biased scoring
    if np.random.random() < ai_accuracy:
        return ai_choice
    else:
        return 1 - ai_choice


def train_reward_model(preferences, features_pairs, n_features, lr=0.05, epochs=30):
    """
    Train a simple linear reward model from preference pairs.
    Uses logistic regression on preference pairs.
    """
    weights = np.zeros(n_features)

    for epoch in range(epochs):
        for (f1, f2), pref in zip(features_pairs, preferences):
            # Score difference
            score1 = np.dot(weights, f1)
            score2 = np.dot(weights, f2)
            diff = score1 - score2

            # Sigmoid
            prob = 1.0 / (1.0 + np.exp(-np.clip(diff, -10, 10)))

            # Gradient of log-likelihood
            if pref == 1:
                grad = (1 - prob) * (f1 - f2)
            else:
                grad = -prob * (f1 - f2)

            weights += lr * grad

    return weights


def simulate_training(n_comparisons, n_training_steps, human_noise, ai_accuracy,
                      verbosity_bias, seed):
    """Run full RLHF vs RLAIF simulation."""
    rng = np.random.RandomState(seed)
    n_features = 4

    # ── Phase 1: Collect preference data ──
    human_prefs = []
    ai_prefs = []
    feature_pairs = []
    agreements = []

    for _ in range(n_comparisons):
        f1 = rng.random(n_features)
        f2 = rng.random(n_features)
        q1 = true_quality(f1)
        q2 = true_quality(f2)

        h_pref = human_preference(q1, q2, human_noise)
        a_pref = ai_preference(q1, q2, f1, f2, ai_accuracy, verbosity_bias)

        human_prefs.append(h_pref)
        ai_prefs.append(a_pref)
        feature_pairs.append((f1, f2))
        agreements.append(1 if h_pref == a_pref else 0)

    # ── Phase 2: Train reward models ──
    human_reward_weights = train_reward_model(human_prefs, feature_pairs, n_features)
    ai_reward_weights = train_reward_model(ai_prefs, feature_pairs, n_features)

    # ── Phase 3: Policy optimization (simplified PPO) ──
    # Policy = Gaussian over feature space, we optimize the mean
    human_policy_mean = rng.random(n_features) * 0.5
    ai_policy_mean = rng.random(n_features) * 0.5
    baseline_policy_mean = human_policy_mean.copy()

    human_rewards_history = []
    ai_rewards_history = []
    human_true_quality_history = []
    ai_true_quality_history = []

    policy_lr = 0.02
    policy_std = 0.15

    for step in range(n_training_steps):
        # Sample responses from each policy
        batch_size = 20
        # Human-feedback policy
        h_samples = rng.normal(human_policy_mean, policy_std, (batch_size, n_features))
        h_samples = np.clip(h_samples, 0, 1)
        h_learned_rewards = h_samples @ human_reward_weights
        h_true_qualities = np.array([true_quality(s) for s in h_samples])

        # AI-feedback policy
        a_samples = rng.normal(ai_policy_mean, policy_std, (batch_size, n_features))
        a_samples = np.clip(a_samples, 0, 1)
        a_learned_rewards = a_samples @ ai_reward_weights
        a_true_qualities = np.array([true_quality(s) for s in a_samples])

        # REINFORCE-style update
        h_advantages = h_learned_rewards - np.mean(h_learned_rewards)
        a_advantages = a_learned_rewards - np.mean(a_learned_rewards)

        for i in range(batch_size):
            h_grad = (h_samples[i] - human_policy_mean) / (policy_std ** 2)
            human_policy_mean += policy_lr * h_advantages[i] * h_grad

            a_grad = (a_samples[i] - ai_policy_mean) / (policy_std ** 2)
            ai_policy_mean += policy_lr * a_advantages[i] * a_grad

        human_policy_mean = np.clip(human_policy_mean, 0, 1)
        ai_policy_mean = np.clip(ai_policy_mean, 0, 1)

        human_rewards_history.append(np.mean(h_learned_rewards))
        ai_rewards_history.append(np.mean(a_learned_rewards))
        human_true_quality_history.append(np.mean(h_true_qualities))
        ai_true_quality_history.append(np.mean(a_true_qualities))

    # ── Compute policy distributions at end ──
    n_eval = 500
    h_final = rng.normal(human_policy_mean, policy_std, (n_eval, n_features))
    h_final = np.clip(h_final, 0, 1)
    a_final = rng.normal(ai_policy_mean, policy_std, (n_eval, n_features))
    a_final = np.clip(a_final, 0, 1)

    h_final_q = [true_quality(s) for s in h_final]
    a_final_q = [true_quality(s) for s in a_final]

    # Running agreement rate
    window = max(1, n_comparisons // 20)
    running_agreement = []
    for i in range(0, len(agreements), window):
        chunk = agreements[i:i+window]
        running_agreement.append(np.mean(chunk))

    return {
        'human_rewards': human_rewards_history,
        'ai_rewards': ai_rewards_history,
        'human_true_quality': human_true_quality_history,
        'ai_true_quality': ai_true_quality_history,
        'agreements': running_agreement,
        'overall_agreement': np.mean(agreements),
        'h_final_quality': h_final_q,
        'a_final_quality': a_final_q,
        'human_reward_weights': human_reward_weights,
        'ai_reward_weights': ai_reward_weights,
        'true_weights': np.array([0.4, 0.3, 0.2, 0.1]),
    }


# ─────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────
st.title("RLAIF: RL from AI Feedback vs Human Feedback")
st.markdown("""
**RLAIF** replaces expensive human annotators with an AI judge to provide preference labels.
This demo simulates both approaches and compares their effectiveness at training a policy
to produce high-quality responses.
""")

st.sidebar.header("Configuration")
n_comparisons = st.sidebar.slider("Preference Comparisons", 100, 2000, 500, step=100)
n_training_steps = st.sidebar.slider("Policy Training Steps", 20, 300, 100, step=20)
human_noise = st.sidebar.slider("Human Label Noise", 0.0, 0.5, 0.15, step=0.05,
                                 help="Probability a human annotator flips their preference")
ai_accuracy = st.sidebar.slider("AI Judge Accuracy", 0.5, 1.0, 0.85, step=0.05,
                                 help="How often the AI judge makes the correct preference call")
verbosity_bias = st.sidebar.slider("AI Verbosity Bias", 0.0, 0.5, 0.15, step=0.05,
                                    help="How much the AI judge favors verbose responses")
seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)

if st.sidebar.button("Run Experiment", type="primary"):
    with st.spinner("Running RLHF vs RLAIF simulation..."):
        results = simulate_training(
            n_comparisons, n_training_steps, human_noise, ai_accuracy,
            verbosity_bias, seed
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Reward Curves", "True Quality", "Label Agreement", "Policy Analysis"
    ])

    # ──────── TAB 1: Reward Curves ────────
    with tab1:
        st.subheader("Learned Reward During Training")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(results['human_rewards'], label="RLHF (Human Feedback)", color="#e74c3c",
                linewidth=2, alpha=0.8)
        ax.plot(results['ai_rewards'], label="RLAIF (AI Feedback)", color="#2ecc71",
                linewidth=2, alpha=0.8)
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Learned Reward", fontsize=12)
        ax.set_title("Reward Model Score Over Training", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        These curves show the **learned reward** (from each reward model) over training.
        Note: a higher learned reward doesn't always mean better true quality —
        the reward model may have biases.
        """)

    # ──────── TAB 2: True Quality ────────
    with tab2:
        st.subheader("True Response Quality During Training")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(results['human_true_quality'], label="RLHF Policy", color="#e74c3c",
                linewidth=2, alpha=0.8)
        ax.plot(results['ai_true_quality'], label="RLAIF Policy", color="#2ecc71",
                linewidth=2, alpha=0.8)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("True Quality", fontsize=12)
        ax.set_title("Actual Response Quality (Ground Truth)", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        col1, col2 = st.columns(2)
        with col1:
            final_h = np.mean(results['human_true_quality'][-10:])
            st.metric("RLHF Final Quality", f"{final_h:.3f}")
        with col2:
            final_a = np.mean(results['ai_true_quality'][-10:])
            st.metric("RLAIF Final Quality", f"{final_a:.3f}")

        diff = final_a - final_h
        if abs(diff) < 0.02:
            st.success("Both approaches achieve similar true quality.")
        elif diff > 0:
            st.info(f"RLAIF produces {diff:.3f} higher true quality than RLHF.")
        else:
            st.info(f"RLHF produces {-diff:.3f} higher true quality than RLAIF.")

    # ──────── TAB 3: Label Agreement ────────
    with tab3:
        st.subheader("Human-AI Label Agreement")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Running agreement
        ax = axes[0]
        ax.plot(results['agreements'], color="#9b59b6", linewidth=2, marker='o', markersize=4)
        ax.axhline(y=results['overall_agreement'], color='gray', linestyle='--', alpha=0.7,
                    label=f"Overall: {results['overall_agreement']:.1%}")
        ax.set_xlabel("Batch", fontsize=12)
        ax.set_ylabel("Agreement Rate", fontsize=12)
        ax.set_title("Human-AI Agreement Over Comparisons", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Reward weight comparison
        ax = axes[1]
        features = ['Helpful', 'Honest', 'Harmless', 'Style']
        x = np.arange(len(features))
        width = 0.25
        bars1 = ax.bar(x - width, results['true_weights'], width, label='True Weights',
                        color='#3498db', alpha=0.8)
        bars2 = ax.bar(x, results['human_reward_weights'] / (np.sum(np.abs(results['human_reward_weights'])) + 1e-8),
                        width, label='RLHF Learned', color='#e74c3c', alpha=0.8)
        bars3 = ax.bar(x + width, results['ai_reward_weights'] / (np.sum(np.abs(results['ai_reward_weights'])) + 1e-8),
                        width, label='RLAIF Learned', color='#2ecc71', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(features, fontsize=10)
        ax.set_ylabel("Normalized Weight", fontsize=12)
        ax.set_title("Reward Model Weights", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(f"""
        **Overall Agreement Rate: {results['overall_agreement']:.1%}**

        - Human noise level: {human_noise:.0%} (random errors)
        - AI accuracy: {ai_accuracy:.0%} (systematic, with verbosity bias of {verbosity_bias})

        The right chart shows how well each reward model learned the true quality weights.
        Note how the AI model may over-weight "Style" due to its verbosity bias.
        """)

    # ──────── TAB 4: Policy Distributions ────────
    with tab4:
        st.subheader("Final Policy Quality Distributions")

        fig, ax = plt.subplots(figsize=(10, 5))
        bins = np.linspace(0, 1, 30)
        ax.hist(results['h_final_quality'], bins=bins, alpha=0.6, label='RLHF Policy',
                color='#e74c3c', density=True, edgecolor='white')
        ax.hist(results['a_final_quality'], bins=bins, alpha=0.6, label='RLAIF Policy',
                color='#2ecc71', density=True, edgecolor='white')
        ax.axvline(x=np.mean(results['h_final_quality']), color='#e74c3c', linestyle='--',
                    linewidth=2, label=f'RLHF mean: {np.mean(results["h_final_quality"]):.3f}')
        ax.axvline(x=np.mean(results['a_final_quality']), color='#2ecc71', linestyle='--',
                    linewidth=2, label=f'RLAIF mean: {np.mean(results["a_final_quality"]):.3f}')
        ax.set_xlabel("True Response Quality", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Distribution of Response Quality from Final Policies", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        This shows the distribution of **true quality** for responses sampled from each
        final policy. A rightward shift means the policy learned to produce better responses.

        **Key takeaways:**
        - RLHF has random noise but no systematic bias → generally learns correct priorities
        - RLAIF is more consistent but may develop blind spots from the AI judge's biases
        - With a good AI judge (high accuracy, low bias), RLAIF can match or exceed RLHF
        """)

else:
    st.markdown("---")
    st.markdown("### How This Demo Works")
    st.markdown("""
    We simulate a simplified version of the RLHF vs RLAIF pipeline:

    1. **Response Generation**: Responses are represented as feature vectors
       (helpfulness, honesty, harmlessness, style).
    2. **Preference Collection**: Pairs of responses are shown to either a human
       (noisy but unbiased) or AI judge (accurate but with verbosity bias).
    3. **Reward Model Training**: A linear reward model is trained from preference labels.
    4. **Policy Optimization**: A policy is optimized using REINFORCE to maximize the
       learned reward.

    The simulation reveals how **noise type matters**: humans make random errors,
    while AI judges make systematic ones.
    """)

    st.markdown("### Key Concepts")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**RLHF Pipeline**")
        st.markdown("""
        ```
        Generate → Human Ranks → Train Reward → PPO
        ```
        - Pro: Unbiased (on average)
        - Con: Expensive, noisy, slow
        """)
    with col2:
        st.markdown("**RLAIF Pipeline**")
        st.markdown("""
        ```
        Generate → AI Ranks → Train Reward → PPO
        ```
        - Pro: Cheap, fast, consistent
        - Con: Systematic biases
        """)
