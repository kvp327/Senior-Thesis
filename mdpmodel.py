# SOLVE THE MDP!
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


def format_policy_table(policy, V0):
    data = []
    for state in states:
        idx = state_index[state]
        row = [str(state)] + [policy[t][idx] for t in range(num_cycles)] + [round(V0[idx], 2)]
        data.append(row)
    cols = ["State"] + [f"d*_{i}" for i in range(1, 11)] + ["v_0"]
    return pd.DataFrame(data, columns=cols)

# Run
cost_matrix = build_cost_matrix()
policy, V0 = value_iteration(
    cost_matrix=cost_matrix,
    build_transition_matrix=build_transition_matrix,
    get_terminal_os_reward=get_terminal_os_reward,
    states=states,
    actions=actions
)
policy_table = format_policy_table(policy, V0)
policy_table.to_csv("final_policy_table.csv", index=False)
print("Saved as 'final_policy_table.csv'")

