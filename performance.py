import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from scipy.stats import ttest_rel
import networkx as nx
import tikzplotlib

toxicity_levels = [1, 2, 3, 4]
pd_l1_levels = [1, 2, 3]
pfs_status = [1, 2]
actions = ["Chemo", "Immuno", "ChemoImmuno"]
num_cycles = 10

states = [(t, p, f) for t in toxicity_levels for p in pd_l1_levels for f in pfs_status]
state_index = {state: i for i, state in enumerate(states)}

tox_burdens = {1: 0, 2: 0.2, 3: 0.4, 4: 0}
progression_penalty = 0.5
os_values = {
    "Chemo": 11.3,
    "Immuno_1": 16.4,
    "Immuno_2": 18,
    "Immuno_3": 21.22,
    "ChemoImmuno": 19.89
}

final_cycle_tox_weight = 0.5  
final_cycle_reward_weight = 0.5 
apply_switch_penalty = False
switch_penalty_weight = 0.05
max_burden = max(tox_burdens.values()) + progression_penalty
max_os = max(os_values.values())

# Transition Probabilties
toxicity_probs = {
    "Chemo": {
        1: {1: {1: 0.562, 2: 0.438}, 2: {3: 0.51, 4: 0.49}, 3: {3: 0.511, 4: 0.489}, 4: {3: 0.15, 4: 0.85}},
        2: {1: {1: 0.808, 2: 0.192}, 2: {3: 0.51, 4: 0.49}, 3: {3: 0.511, 4: 0.489}, 4: {3: 0.15, 4: 0.85}},
        3: {1: {1: 0.9051, 2: 0.0949}, 2: {3: 0.56, 4: 0.44}, 3: {3: 0.724, 4: 0.276}, 4: {3: 0.15, 4: 0.85}},
        4: {1: {1: 0.9684, 2: 0.0316}, 2: {3: 0.426, 4: 0.574}, 3: {3: 0.557, 4: 0.443}, 4: {3: 0.15, 4: 0.85}},
    },
    "Immuno": {
        1: {1: {1: 0.9607, 2: 0.0393}, 2: {3: 0.462, 4: 0.538}, 3: {3: 0.8, 4: 0.2}, 4: {3: 0.15, 4: 0.85}},
        2: {1: {1: 0.8914, 2: 0.1086}, 2: {3: 0.462, 4: 0.538}, 3: {3: 0.8, 4: 0.2}, 4: {3: 0.15, 4: 0.85}},
        3: {1: {1: 0.9325, 2: 0.0675}, 2: {3: 0.462, 4: 0.538}, 3: {3: 0.8, 4: 0.2}, 4: {3: 0.15, 4: 0.85}},
        4: {1: {1: 0.9677, 2: 0.0323}, 2: {3: 0.462, 4: 0.538}, 3: {3: 0.8, 4: 0.2}, 4: {3: 0.15, 4: 0.85}},
        5: {1: {1: 0.9849, 2: 0.0151}, 2: {3: 0.462, 4: 0.538}, 3: {3: 0.8, 4: 0.2}, 4: {3: 0.15, 4: 0.85}},
        6: {1: {1: 0.9928, 2: 0.0072}, 2: {3: 0.462, 4: 0.538}, 3: {3: 0.8, 4: 0.2}, 4: {3: 0.15, 4: 0.85}},
        7: {1: {1: 0.9964, 2: 0.0036}, 2: {3: 0.462, 4: 0.538}, 3: {3: 0.8, 4: 0.2}, 4: {3: 0.15, 4: 0.85}}, 
        8: {1: {1: 0.9982, 2: 0.0018}, 2: {3: 0.462, 4: 0.538}, 3: {3: 0.8, 4: 0.2}, 4: {3: 0.15, 4: 0.85}},
        9: {1: {1: 0.999, 2: 0.001}, 2: {3: 0.462, 4: 0.538}, 3: {3: 0.8, 4: 0.2}, 4: {3: 0.15, 4: 0.85}},
        10: {1: {1: 0.9995, 2: 0.0005}, 2: {3: 0.462, 4: 0.538}, 3: {3: 0.8, 4: 0.2}, 4: {3: 0.15, 4: 0.85}},
    },
    "ChemoImmuno": {
        c: {1: {1: 0.862, 2: 0.138}, 2: {3: 0.592, 4: 0.408}, 3: {3: 0.8, 4: 0.2}, 4: {3: 0.15, 4: 0.85}} for c in range(1, 11)
    }
}

pd_l1_transitions = {
    c: {
        1: {1: 0.599, 2: 0.401},
        2: {2: 0.751, 3: 0.249},
        3: {3: 1.0}
    } for c in range(1, 5)
}
for c in range(5, 11):
    pd_l1_transitions[c] = {s: {s: 1.0} for s in pd_l1_levels}

pfs_probs = {
    "Chemo": [0.9568, 0.9315, 0.9217, 0.9172, 0.9063, 0.8973, 0.8896, 0.8828, 0.8766, 0.8710],
    "ChemoImmuno": [0.9835, 0.9670, 0.9555, 0.9459, 0.9375, 0.9298, 0.9228, 0.9163, 0.9101, 0.9043],
    "Immuno": {
        1: [0.8906, 0.9037, 0.9081, 0.9108, 0.9128, 0.9144, 0.9157, 0.9168, 0.9177, 0.9185],
        2: [0.9071, 0.9157, 0.9187, 0.9205, 0.9219, 0.9230, 0.9239, 0.9246, 0.9253, 0.9258],
        3: [0.9331, 0.9317, 0.9311, 0.9307, 0.9305, 0.9303, 0.9301, 0.9299, 0.9298, 0.9297]
    }
}

# Cost Matrix
def build_cost_matrix():
    C = {a: np.zeros(len(states)) for a in actions}
    for a in actions:
        for s_idx, (tox, pd_l1, pfs) in enumerate(states):
            burden = tox_burdens[tox] + (progression_penalty if pfs == 2 else 0)
            os = get_terminal_os_reward(a, pd_l1, pfs)
            # Normalize both burden and OS before computing cost
            normalized_burden = burden / max_burden
            normalized_os = os / max_os
            C[a][s_idx] = normalized_burden / normalized_os
    return C

# Rewards 
def get_terminal_os_reward(action: str, pd_l1: int, pfs: int) -> float:
    if action == "Immuno":
        return os_values[f"Immuno_{pd_l1}"]
    elif action == "Chemo":
        return os_values["Chemo"]
    elif action == "ChemoImmuno":
        return os_values["ChemoImmuno"]
    return 1.0

# Transition Matrix 
def build_transition_matrix(cycle, prev_action, current_action):
    T = {a: np.zeros((len(states), len(states))) for a in actions}
    rel_cycle = 1 if prev_action != current_action else cycle
    rel_cycle = min(rel_cycle, 10)
    pd_rel_cycle = rel_cycle  # using your defined pd_l1_transitions for this cycle
    
    for a in actions:
        for s_idx, (tox, pd, pfs) in enumerate(states):
            # Get toxicity transition probabilities (for the appropriate relative cycle)
            tox_probs = toxicity_probs[a].get(min(rel_cycle, 4), toxicity_probs[a][4])[tox]
            # For Chemo, use the PD-L1 transitions; for other actions assume PD-L1 doesn't change.
            if a == "Chemo":
                pd_probs = pd_l1_transitions[pd_rel_cycle][pd]
            else:
                pd_probs = {pd: 1.0}
            # Now, for PFS, enforce that if pfs==2, it stays at 2.
            if pfs == 2:
                pfs_probs_cycle = {2: 1.0}
            else:
                if a != "Immuno":
                    pfs_surv = pfs_probs[a][cycle - 1]
                else:
                    pfs_surv = pfs_probs["Immuno"][pd][cycle - 1]
                pfs_probs_cycle = {1: pfs_surv, 2: 1 - pfs_surv}
            
            # Combine the probabilities to update the transition matrix.
            for new_tox, p_tox in tox_probs.items():
                for new_pd, p_pd in pd_probs.items():
                    for new_pfs, p_pfs in pfs_probs_cycle.items():
                        next_state = (new_tox, new_pd, new_pfs)
                        s_next_idx = state_index[next_state]
                        T[a][s_idx, s_next_idx] += p_tox * p_pd * p_pfs
    return T


# Value Iteration 
def value_iteration(horizon=10, discount=1.0, cost_matrix=None, build_transition_matrix=None,
                    get_terminal_os_reward=None, states=None, actions=None):
    V = np.zeros((horizon + 1, len(states)))  # Expected cumulative cost
    policy = np.empty((horizon, len(states)), dtype=object)
    prev_actions = [None] * len(states)

    for t in reversed(range(horizon)):
        for s_idx, state in enumerate(states):
            tox, pd_l1, pfs = state
            action_costs = []
            for a in actions:
                # Get transition probabilities for cycle (t+1)
                T = build_transition_matrix(t + 1, prev_actions[s_idx], a)
                # Immediate cost from the cost matrix (e.g., normalized burden/OS)
                immediate_cost = cost_matrix[a][s_idx]
            
                # If we're at the final cycle, add the final adjustment.
                if t == horizon - 1:
                    # Calculate the normalized toxicity burden for this state.
                    burden = tox_burdens[tox] + (progression_penalty if pfs == 2 else 0)
                    normalized_tox = burden / max_burden
                    normalized_os = get_terminal_os_reward(a, pd_l1, pfs) / max_os
                
                    # In cost minimization: higher toxicity increases cost, higher OS reduces cost.
                    final_adjustment = final_cycle_tox_weight * normalized_tox - final_cycle_reward_weight * normalized_os
                    total_cost = immediate_cost + final_adjustment
            
                else:
                    # For non-final cycles, add the discounted expected future cost.
                    total_cost = immediate_cost + discount * np.dot(T[a][s_idx, :], V[t + 1])
            
                action_costs.append(total_cost)
        
            # Select the action that minimizes total cost.
            best_action_idx = np.argmin(action_costs)
            V[t][s_idx] = action_costs[best_action_idx]
            policy[t][s_idx] = actions[best_action_idx]
            prev_actions[s_idx] = actions[best_action_idx]

    return policy, V[0]

# Evaluate Fixed Policies
def evaluate_fixed_policy(fixed_action, horizon=10, discount=1.0, cost_matrix=None):
    V = np.zeros((horizon + 1, len(states)))
    for t in reversed(range(horizon)):
        T = build_transition_matrix(t + 1, fixed_action, fixed_action)
        for s_idx, state in enumerate(states):
            tox, pd_l1, pfs = state
            immediate_cost = cost_matrix[fixed_action][s_idx]

            if t == horizon - 1:
                burden = tox_burdens[tox] + (progression_penalty if pfs == 2 else 0)
                normalized_tox = burden / max_burden
                normalized_os = get_terminal_os_reward(fixed_action, pd_l1, pfs) / max_os
                final_adjustment = final_cycle_tox_weight * normalized_tox - final_cycle_reward_weight * normalized_os
                total_cost = immediate_cost + final_adjustment
            else:
                total_cost = immediate_cost + discount * np.dot(T[fixed_action][s_idx, :], V[t + 1])
            V[t][s_idx] = total_cost
    return V[0]

def format_policy_table(policy, V0, chemo_costs, immuno_costs, combo_costs):
    data = []
    for state in states:
        idx = state_index[state]
        row = [
            str(state),
            *[policy[t][idx] for t in range(num_cycles)],
            round(V0[idx], 3),
            round(chemo_costs[idx], 3),
            round(immuno_costs[idx], 3),
            round(combo_costs[idx], 3)
        ]
        data.append(row)
    cols = ["State"] + [f"d*_{i+1}" for i in range(1, 11)] + [
        "v0_opt", "v0_chemo", "v0_immuno", "v0_combo"
    ]
    return pd.DataFrame(data, columns=cols)

cost_matrix = build_cost_matrix()
policy, V0 = value_iteration(cost_matrix=cost_matrix)
chemo_costs = evaluate_fixed_policy("Chemo", cost_matrix=cost_matrix)
immuno_costs = evaluate_fixed_policy("Immuno", cost_matrix=cost_matrix)
combo_costs = evaluate_fixed_policy("ChemoImmuno", cost_matrix=cost_matrix)

policy_table = format_policy_table(policy, V0, chemo_costs, immuno_costs, combo_costs)
policy_table.to_csv("final_policy_comparison.csv", index=False)


# Visualization
# Bar Graph
delta_chemo = chemo_costs - V0
delta_immuno = immuno_costs - V0
delta_combo = combo_costs - V0

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

bar_colors = {
    "Optimal": "#2FA39E",
    "Chemo": "#0077B6",
    "Immuno": "#D4A017",
    "Chemoimmuno": "#C93047"
}

avg_costs = {
    "Optimal": np.mean(V0),
    "Chemo": np.mean(chemo_costs),
    "Immuno": np.mean(immuno_costs),
    "Chemoimmuno": np.mean(combo_costs)
}

labels = list(avg_costs.keys())
values = list(avg_costs.values())
colors = [bar_colors[key] for key in labels]

bars = axs[0].bar(labels, values, color=colors)
axs[0].set_title("Average Cost by Strategy")
axs[0].set_ylabel("Average Cost")
axs[0].tick_params(axis='x', rotation=0)

for bar in bars:
    height = bar.get_height()
    offset = -0.15 if height > 7 else -0.2
    axs[0].text(
        bar.get_x() + bar.get_width()/2,
        height + offset,
        f"{height:.2f}",
        ha='center', va='top', fontsize=10
    )

axs[0].set_ylim(0, max(values) + 0.5)

# Box Plot
data = [V0, chemo_costs, immuno_costs, combo_costs]
box_colors = ["#2FA39E","#0077B6","#D4A017","#C93047"]

box = axs[1].boxplot(
    data,
    patch_artist=True,
    labels=["Optimal", "Chemo", "Immuno", "Chemoimmuno"],
    boxprops=dict(color='black'),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
    medianprops=dict(color='black')
)

for patch in box['boxes']:
    patch.set_facecolor('white')

for median, color in zip(box['medians'], box_colors):
    median.set_color(color)
    median.set_linewidth(2)

axs[1].set_title("Cost Distribution per Strategy")
axs[1].set_ylabel("Cost")
axs[1].yaxis.set_label_coords(-0.05, 0.5)


# Delta Histogram
colors = {
    "Chemo": "#0077B6",     # Mid blue
    "Immuno": "#D4A017",    # Warm goldenrod
    "Chemoimmuno": "#C93047"      # Deep red
}

axs[2].hist(delta_chemo, bins=15, alpha=0.6, label="Chemo - Optimal", color=colors["Chemo"])
axs[2].hist(delta_immuno, bins=15, alpha=0.6, label="Immuno - Optimal", color=colors["Immuno"])
axs[2].hist(delta_combo, bins=15, alpha=0.6, label="Chemoimmuno - Optimal", color=colors["Chemoimmuno"])

axs[2].axvline(0, color='black', linestyle='--', linewidth=1.2)

axs[2].set_title("Cost Difference from Optimal Policy")
axs[2].set_xlabel(r"$\Delta$ Cost")
axs[2].set_ylabel("Number of States")
axs[2].legend()

plt.tight_layout()
plt.subplots_adjust(wspace=0.15)
tikzplotlib.save("performance.tex")
plt.show()

# Paired t-tests among all strategies 
cost_dict = {
    "Optimal": V0,
    "Chemo": chemo_costs,
    "Immuno": immuno_costs,
    "Chemoimmuno": combo_costs
}

pairs = [
    ("Optimal", "Chemo"),
    ("Optimal", "Immuno"),
    ("Optimal", "Chemoimmuno"),
    ("Chemo",   "Immuno"),
    ("Chemo",   "Chemoimmuno"),
    ("Immuno",  "Chemoimmuno")
]

print("\nPaired t-tests for all strategy pairs:")
for (s1, s2) in pairs:
    t_stat, p_val = ttest_rel(cost_dict[s1], cost_dict[s2])
    print(f"  {s1} vs. {s2}: t-stat={t_stat:.3f}, p-value={p_val:.3g}")

# Spaghetti Plot for Paired Cost Data
conditions = ['Optimal', 'Chemo', 'Immuno', 'Chemoimmuno']
x_positions = np.arange(len(conditions))  # [0, 1, 2, 3]

# Combine your cost arrays into a 2D array of shape (n_states, 4).
# Each row corresponds to one state, and the columns correspond to the four strategies.
paired_data = np.vstack([V0, chemo_costs, immuno_costs, combo_costs]).T

# Create the spaghetti plot.
plt.figure(figsize=(8, 6))
for i in range(paired_data.shape[0]):
    plt.plot(x_positions, paired_data[i, :], marker='o', linestyle='-', alpha=0.7)
    
plt.xticks(x_positions, conditions)
plt.ylabel("Cost")
plt.title("Paired Cost Comparisons Across Strategies")
plt.grid(True)
plt.tight_layout()
tikzplotlib.save("spagetti.tex")
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Compute the paired differences (for example, Optimal - Combo)
diff_opt_combo = V0 - combo_costs  # or use combo_costs - V0 if you prefer a different sign

# Create a figure with two subplots: a histogram and a box plot
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

combo_color = "#C93047"
combo_fill = "#F2C7CC"

# Histogram
axs[0].hist(diff_opt_combo, bins=15, alpha=0.75, color=combo_color, edgecolor='black')
axs[0].axvline(0, color='black', linestyle='--', linewidth=1.2)
axs[0].set_title("Cost Differences (Optimal - Chemoimmuno)")
axs[0].set_xlabel("Difference in Cost")
axs[0].set_ylabel("Frequency")

# Boxplot
box = axs[1].boxplot(diff_opt_combo, patch_artist=True)
axs[1].axhline(0, color='black', linestyle='--', linewidth=1.2)
axs[1].set_title("Cost Differences (Optimal - Chemoimmuno)")
axs[1].set_ylabel("Difference in Cost")
axs[1].set_xticklabels(["Optimal - Chemoimmuno"])

# Colors
for patch in box['boxes']:
    patch.set_facecolor(combo_fill)
    patch.set_edgecolor(combo_color)
for median in box['medians']:
    median.set_color(combo_color)
for whisker in box['whiskers']:
    whisker.set_color(combo_color)
for cap in box['caps']:
    cap.set_color(combo_color)
for flier in box['fliers']:
    flier.set(marker='o', color=combo_color, alpha=0.5)

plt.tight_layout()
tikzplotlib.save("combocomp.tex")
plt.show()

from scipy.stats import ttest_rel

# Run paired t-test for Optimal vs Combo
t_stat, p_val = ttest_rel(V0, combo_costs)

print("Paired t-test for Optimal vs Chemoimmunotherapy:")
print("t-statistic =", t_stat)
print("p-value =", p_val)
