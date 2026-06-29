import numpy as np

class GridWorldMDP:
    def __init__(self):
        self.num_states = 9
        self.terminal_state = 8
        self.actions = [0, 1, 2, 3] # 0:Up, 1:Down, 2:Left, 3:Right
        self.action_effects = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.action_names = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    
    def step(self, state, action):
        if state == self.terminal_state:
            return state, 0, True
        row, col = state // 3, state % 3
        d_row, d_col = self.action_effects[action]
        next_row = max(0, min(2, row + d_row))
        next_col = max(0, min(2, col + d_col))
        next_state = next_row * 3 + next_col
        return next_state, -1, (next_state == self.terminal_state)

def print_policy(policy, env):
    grid = []
    for s in range(env.num_states):
        if s == env.terminal_state:
            grid.append("G") # Goal
        else:
            grid.append(env.action_names[policy[s]])
    print(np.array(grid).reshape(3, 3))
    
# --- Initialization ---
env = GridWorldMDP()
gamma = 1.0
theta = 1e-6 # Convergence threshold

# =====================================================================
# PROJECT 1: VALUE ITERATION
# =====================================================================
print("--- Running Value Iteration ---")
V_val = np.zeros(env.num_states)

while True:
    delta = 0
    for s in range(env.num_states):
        if s == env.terminal_state: continue
        v = V_val[s]
        
        # Calculate Q(s,a) for all actions
        q_values = []
        for a in env.actions:
            next_s, reward, _ = env.step(s, a)
            q_values.append(reward + gamma * V_val[next_s])
            
        # Value Iteration Update: V(s) = max_a Q(s,a)
        V_val[s] = max(q_values)
        delta = max(delta, abs(v - V_val[s]))
    if delta < theta: break

# Extract Optimal Policy from Value Iteration
optimal_policy_vi = np.zeros(env.num_states, dtype=int)
for s in range(env.num_states):
    if s == env.terminal_state: continue
    q_values = [env.step(s, a)[1] + gamma * V_val[env.step(s, a)[0]] for a in env.actions]
    optimal_policy_vi[s] = np.argmax(q_values)

print_policy(optimal_policy_vi, env)

# =====================================================================
# PROJECT 2: POLICY ITERATION
# =====================================================================
print("\n--- Running Policy Iteration ---")
V_pi = np.zeros(env.num_states)
policy = np.zeros(env.num_states, dtype=int) # Start with all actions facing Up (0)

while True:
    # Phase 1: Policy Evaluation
    while True:
        delta = 0
        for s in range(env.num_states):
            if s == env.terminal_state: continue
            v = V_pi[s]
            chosen_action = policy[s]
            next_s, reward, _ = env.step(s, chosen_action)
            V_pi[s] = reward + gamma * V_pi[next_s]
            delta = max(delta, abs(v - V_pi[s]))
        if delta < theta: break