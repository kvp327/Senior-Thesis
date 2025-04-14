import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict

my_colors = ["#FF1F5B", "#FFC61E", "#009ADE"]

# Set a global random seed for reproducibility
np.random.seed(42)

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



# Simulation Function
# For each cycle, records: cycle number, pre-transition state, action, post-transition state
def simulate_trajectory_and_cost_updated(policy, build_transition_matrix, states, state_index, init_state, horizon=10):
    current_state = init_state
    prev_action = None
    total_cost = 0.0
    trajectory = []  # will store (cycle, pre_state, action, post_state)
    for t in range(horizon):
        s_idx = state_index[current_state]
        action = policy[t][s_idx]
        T = build_transition_matrix(t + 1, prev_action, action)
        p_next = T[action][s_idx, :]
        immediate_cost = cost_matrix[action][s_idx]
        if apply_switch_penalty and prev_action is not None and prev_action != action:
            immediate_cost += switch_penalty_weight
        if t == horizon - 1:
            tox, pd_l1, pfs = current_state
            burden = tox_burdens[tox] + (progression_penalty if pfs == 2 else 0)
            normalized_tox = burden / max_burden
            normalized_os = get_terminal_os_reward(action, pd_l1, pfs) / max_os
            final_adjustment = final_cycle_tox_weight * normalized_tox - final_cycle_reward_weight * normalized_os
            immediate_cost += final_adjustment
        total_cost += immediate_cost
        next_idx = np.random.choice(len(states), p=p_next)
        next_state = states[next_idx]
        trajectory.append((t + 1, current_state, action, next_state))
        prev_action = action
        current_state = next_state
    return trajectory, total_cost, current_state

# Helper Functions to Obtain & Analyze Simulation Results
def get_simulation_results_for_state(policy, build_transition_matrix, states, state_index, init_state, horizon=10, N=100):
    """Run N simulations for a given initial state and return trajectories, final states and costs."""
    trajectories = []
    final_states = []
    costs = []
    for _ in range(N):
        traj, cost, final_state = simulate_trajectory_and_cost_updated(
            policy, build_transition_matrix, states, state_index, init_state, horizon
        )
        trajectories.append(traj)
        final_states.append(final_state)
        costs.append(cost)
    return trajectories, final_states, costs

def analyze_final_state_distribution_from_results(final_states, N=100):
    counter = Counter(final_states)
    total = sum(counter.values())
    print("\n--- Final State Frequency Distribution ---")
    for state, count in counter.most_common():
        status = "Progressed" if state[2] == 2 else "Stable"
        print(f"Final State: {state} | Count: {count} | Frequency: {count/total:.2%} | Status: {status}")

def analyze_most_likely_state_per_cycle_from_results(trajectories, horizon=10, N=100):
    # Use the post-transition state from each cycle.
    cycle_states = {cycle: [] for cycle in range(1, horizon + 1)}
    for traj in trajectories:
        for (cycle, pre_state, action, post_state) in traj:
            cycle_states[cycle].append(post_state)
    print("\n--- Most Likely Post-Transition State Per Cycle ---")
    for cycle in range(1, horizon + 1):
        mode_state, count = Counter(cycle_states[cycle]).most_common(1)[0]
        progression_flag = "YES" if mode_state[2] == 2 else "NO"
        print(f"Cycle {cycle:2d} | Modal post-state: {mode_state} | Count: {count}/{N} | Progression: {progression_flag}")

def get_mode_action_trajectory(initial_state, policy, N=100):
    sequences = []
    for _ in range(N):
        traj, _, _ = simulate_trajectory_and_cost_updated(
            policy, build_transition_matrix, states, state_index, initial_state, horizon=num_cycles
        )
        actions_seq = [step[2] for step in traj]
        sequences.append(tuple(actions_seq))
    most_common = Counter(sequences).most_common(1)[0][0]
    return [action_to_code[a] for a in most_common]



# Compute Policy, Run Simulations, Build Summary & Plot
if __name__ == "__main__":
    # Compute the optimal policy and expected costs.
    cost_matrix = build_cost_matrix()
    policy, V0 = value_iteration(
        horizon=num_cycles,
        discount=1.0,
        cost_matrix=cost_matrix,
        build_transition_matrix=build_transition_matrix,
        get_terminal_os_reward=get_terminal_os_reward,
        states=states,
        actions=actions
    )

    # Build summary table for all initial states by running a single set of simulations per state.
    N = 100  # number of simulations per state
    summary = []
    # We'll also store simulation results in a dictionary for later analysis.
    results_dict = {}
    for init_state in states:
        trajectories, final_states_list, costs = get_simulation_results_for_state(
            policy, build_transition_matrix, states, state_index, init_state, horizon=num_cycles, N=N
        )
        results_dict[init_state] = (trajectories, final_states_list, costs)
        first_actions = [traj[0][2] for traj in trajectories]      # first action taken in each trajectory
        final_actions = [traj[-1][2] for traj in trajectories]       # last action taken
        avg_cost = np.mean(costs)
        most_common_first = Counter(first_actions).most_common(1)[0][0]
        most_common_final = Counter(final_actions).most_common(1)[0][0]
        most_common_final_state = Counter(final_states_list).most_common(1)[0][0]
        expected_cost = V0[state_index[init_state]]
        summary.append({
            "Initial State": init_state,
            "Most Likely 1st Action": most_common_first,
            "Most Likely Final Action": most_common_final,
            "Most Likely Final State": most_common_final_state,
            "Empirical Cost": round(avg_cost, 3),
            "Expected - Empirical": round(expected_cost, 3) - round(avg_cost, 3)
        })
    df_summary = pd.DataFrame(summary)
    df_summary.sort_values(by=["Initial State"], inplace=True)
    df_summary.reset_index(drop=True, inplace=True)
    print(df_summary)
    df_summary.to_csv("multiple_sims_summary.csv", index=False)

    # Plot stochastic trajectories for filtered states (where toxicity=1 and pfs=1).
    cycles = list(range(1, num_cycles + 1))  # x-axis for plotting
    filtered_states = [s for s in states if s[0] == 1 and s[2] == 1]
    action_to_code = {a: i for i, a in enumerate(actions)}
    stochastic_N = 100
    plt.figure(figsize=(8, 5))
    x_vals = np.arange(1, num_cycles + 1)
    for i, s in enumerate(filtered_states):
        traj_sequences = []
        for _ in range(stochastic_N):
            traj, _, _ = simulate_trajectory_and_cost_updated(
                policy, build_transition_matrix, states, state_index, s, horizon=num_cycles
            )
            action_seq = tuple(step[2] for step in traj)
            traj_sequences.append(action_seq)
        mode_traj, count = Counter(traj_sequences).most_common(1)[0]
        y_actions = [action_to_code[action] for action in mode_traj]
        line_styles = ['-', '--', ':']
        markers = ['o', 's', '^']
        plt.plot(x_vals, y_actions, marker=markers[i],
                 linestyle=line_styles[i],
                 color=my_colors[i],
                 alpha=0.8,
                 label=str(s))
    plt.xlabel("Cycle")
    plt.title("Stochastic Action Trajectories by Initial State", fontsize=14, weight='bold')
    plt.xlabel("Cycle", fontsize=12)
    plt.ylabel("Treatment", fontsize=12)
    a1 = get_mode_action_trajectory((1, 1, 1), policy)
    a2 = get_mode_action_trajectory((1, 2, 1), policy)
    a3 = get_mode_action_trajectory((1, 3, 1), policy)
    plt.plot(cycles, a1, label='(1, 1, 1)', color='crimson', linewidth=2.5, marker='o')
    plt.plot(cycles, a2, label='(1, 2, 1)', color='goldenrod', linewidth=2.5, marker='s', linestyle='--')
    plt.plot(cycles, a3, label='(1, 3, 1)', color='steelblue', linewidth=2.5, marker='^', linestyle=':')

    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Detailed Analysis for Selected Initial States ---
    for init_state in [(1, 1, 1), (1, 2, 1), (1, 3, 1)]:
        print(f"\n===== Analyzing Initial State: {init_state} =====")
        # Use the same simulation results from results_dict
        trajectories, final_states_results, _ = results_dict[init_state]
        analyze_final_state_distribution_from_results(final_states_results, N=N)
        analyze_most_likely_state_per_cycle_from_results(trajectories, horizon=num_cycles, N=N)