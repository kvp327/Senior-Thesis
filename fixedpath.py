import numpy as np
import pandas as pd
from typing import Dict

import numpy as np
import pandas as pd
from typing import Dict

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

# Optimal policy and value function
cost_matrix = build_cost_matrix()
policy, V0 = value_iteration(
    cost_matrix=cost_matrix,
    build_transition_matrix=build_transition_matrix,
    get_terminal_os_reward=get_terminal_os_reward,
    states=states,
    actions=actions
)

def simulate_fixed_path_cost(
    path: list,
    cost_matrix: Dict[str, np.ndarray],
    initial_state: tuple = (1, 1, 1),
    discount=1.0,
    simulate_transitions=True
):
    state = initial_state
    total_cost = 0.0
    prev_action = None

    for t, action in enumerate(path):
        cycle = t + 1
        s_idx = state_index[state]

        immediate_cost = cost_matrix[action][s_idx]
        if cycle == num_cycles:
            tox, pd, pfs = state
            burden = tox_burdens[tox] + (progression_penalty if pfs == 2 else 0)
            norm_tox = burden / max_burden
            norm_os = get_terminal_os_reward(action, pd, pfs) / max_os
            final_adjustment = final_cycle_tox_weight * norm_tox - final_cycle_reward_weight * norm_os
            immediate_cost += final_adjustment
        total_cost += (discount ** (cycle - 1)) * immediate_cost

        if cycle < num_cycles:
            T = build_transition_matrix(cycle, prev_action, action)
            probs = T[action][s_idx]
            next_state_idx = np.random.choice(len(states), p=probs) if simulate_transitions else np.argmax(probs)
            state = states[next_state_idx]

        prev_action = action

    return total_cost


cost_matrix = build_cost_matrix()
# Define the fixed paths - change depending on which one we want to look at
fixed_path = ["Immuno" if i % 2 == 0 else "ChemoImmuno" for i in range(num_cycles)]
sequential_path = ["Chemo"] * 3 + ["Immuno"] * 7
alternating_path = ["Immuno" if i % 2 == 0 else "ChemoImmuno" for i in range(num_cycles)]
alternating_path2 = ["ChemoImmuno" if i % 2 == 0 else "Immuno" for i in range(num_cycles)]
np.random.seed(42)

fixed_results = []
optimal_results = []

for initial_state in states:
    costs = []
    for _ in range(100): 
        cost = simulate_fixed_path_cost(
            [policy[t][state_index[initial_state]] for t in range(num_cycles)],
            cost_matrix=cost_matrix,
            initial_state=initial_state,
            simulate_transitions=True  
        )
        costs.append(cost)
    avg_cost = round(np.mean(costs), 4)
    optimal_results.append((initial_state, avg_cost))


for initial_state in states:
    costs = []
    for _ in range(100): 
        cost = simulate_fixed_path_cost(
            fixed_path,
            cost_matrix=cost_matrix,
            initial_state=initial_state,
            simulate_transitions=True  
        )
        costs.append(cost)
    avg_cost = round(np.mean(costs), 4)
    fixed_results.append((initial_state, avg_cost))

df_seq = pd.DataFrame(fixed_results, columns=["Initial State", "Total Cost"])
df_seq.to_csv("fixedpath.csv", index=False)

realized_comparison = []

for i, state in enumerate(states):
    opt_cost = optimal_results[i][1]
    fix_cost = fixed_results[i][1]
    diff = round(fix_cost - opt_cost, 4)
    better = "Fixed" if fix_cost < opt_cost else "Optimal"
    realized_comparison.append((state, round(fix_cost, 4), round(opt_cost, 4), diff, better))

df_realized_compare = pd.DataFrame(
    realized_comparison,
    columns=["Initial State", "Fized Cost", "Optimal Realized Cost", "Difference", "Better"]
)

df_realized_compare.to_csv("realizedfixedvsopt.csv", index=False)

from collections import Counter, defaultdict

def simulate_trajectory(policy_lookup, initial_state, simulate_transitions=True):
    state = initial_state
    trajectory = [state]
    actions_taken = []
    prev_action = None

    for t in range(num_cycles):
        if isinstance(policy_lookup, list):  # forced path
            action = policy_lookup[t]
        else:  # optimal policy
            s_idx = state_index[state]
            action = policy_lookup[t][s_idx]
        actions_taken.append(action)

        if t == num_cycles - 1:
            break  # No transition after final cycle

        cycle = t + 1
        T = build_transition_matrix(cycle, prev_action, action)
        s_idx = state_index[state]
        probs = T[action][s_idx]
        next_state_idx = np.random.choice(len(states), p=probs) if simulate_transitions else np.argmax(probs)
        state = states[next_state_idx]
        trajectory.append(state)
        prev_action = action

    return trajectory, actions_taken

def run_simulations(policy_or_path, label, N=100):
    final_state_counter = Counter()
    trajectory_counter = Counter()

    for init_state in states:
        for _ in range(N):
            traj, actions = simulate_trajectory(policy_or_path, init_state)
            final_state_counter[traj[-1]] += 1
            trajectory_counter[tuple(actions)] += 1

    return {
        "label": label,
        "final_states": final_state_counter,
        "trajectories": trajectory_counter
    }

N = 100  # Simulations per initial state

results_opt = run_simulations(policy, "Optimal Policy", N=N)
results_fix = run_simulations(fixed_path, "Fixed Path", N=N)

def summarize_final_states(final_state_counter):
    pfs = defaultdict(int)
    tox = defaultdict(int)
    pd_l1 = defaultdict(int)

    for (tox_level, pd_l1_level, pfs_status), count in final_state_counter.items():
        pfs[pfs_status] += count
        tox[tox_level] += count
        pd_l1[pd_l1_level] += count

    return dict(pfs), dict(tox), dict(pd_l1)

pfs_opt, tox_opt, pd_opt = summarize_final_states(results_opt["final_states"])
pfs_fix, tox_fix, pd_fix = summarize_final_states(results_fix["final_states"])

print("\nPFS Final Distribution:")
print("Optimal:", dict(sorted(pfs_opt.items())))
print("Fixed:", dict(sorted(pfs_fix.items())))


print("\nToxicity Final Distribution:")
print("Optimal:", dict(sorted(tox_opt.items())))
print("Fixed::", dict(sorted(tox_fix.items())))


print("\nPD-L1 Final Distribution:")
print("Optimal:", dict(sorted(pd_opt.items())))
print("Fixed::", dict(sorted(pd_fix.items())))

def top_trajectories(traj_counter, top_k=5):
    return traj_counter.most_common(top_k)

print("\nTop Trajectories - Optimal Policy:")
for fix, count in top_trajectories(results_opt["trajectories"]):
    print(f"{fix}: {count}")

print("\nTop Trajectories - Fixed:")
for fix, count in top_trajectories(results_fix["trajectories"]):
    print(f"{fix}: {count}")

from collections import defaultdict, Counter

# Simulate most common final states under fixed path
num_simulations = 100
final_state_fixed = defaultdict(list)

for initial_state in states:
    for _ in range(num_simulations):
        state = initial_state
        prev_action = None
        for t, action in enumerate(fixed_path):
            cycle = t + 1
            s_idx = state_index[state]
            T = build_transition_matrix(cycle, prev_action, action)
            probs = T[action][s_idx]
            next_state_idx = np.random.choice(len(states), p=probs)
            state = states[next_state_idx]
            prev_action = action
        final_state_fixed[initial_state].append(state)

# Analyze the most common final state for each initial state
summary_data = []
for init_state, final_list in final_state_fixed.items():
    final_counts = Counter(final_list)
    most_common_final, freq = final_counts.most_common(1)[0]
    init_idx = state_index[init_state]
    optimal_cost = V0[init_idx]
    fixed_cost = [x[1] for x in fixed_results if x[0] == init_state][0]
    cost_diff = round(fixed_cost - optimal_cost, 4)
    
    summary_data.append((
        init_state,
        most_common_final,
        freq,
        round(optimal_cost, 3),
        round(fixed_cost, 3),
        cost_diff
    ))

df_summary = pd.DataFrame(summary_data, columns=[
    "Initial State", "Most Common Final State (Fixed)", "Frequency",
    "Optimal Cost", "Fixed Cost", "Cost Difference (Fix - Opt)"
])




