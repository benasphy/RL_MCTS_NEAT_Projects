import numpy as np

class GridWorldMDP:
    def __init__(self):
        self.num_states = 9
        self.terminal_state = 8
        self.actions = [0, 1, 2, 3] # Up, Down, Left, Right
        self.action_effects = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    
    def step(self, state, action):
        if state == self.terminal_state:
            return state, 0, True
        row, col = state // 3, state % 3
        d_row, d_col = self.action_effects[action]
        next_row = max(0, min(2, row + d_row))
        next_col = max(0, min(2, col + d_col))
        next_state = next_row * 3 + next_col
        return next_state, -1, (next_state == self.terminal_state)

# --- Hyperparameters ---
gamma = 0.95
alpha = 0.1
epsilon = 0.1
lam = 0.8  # Lambda trace decay parameter
episodes = 300

env = GridWorldMDP()
Q = np.zeros((env.num_states, len(env.actions)))

def epsilon_greedy(state, Q_table, eps):
    if np.random.rand() < eps:
        return np.random.choice(len(Q_table[state]))
    else:
        return np.argmax(Q_table[state])

print(f"--- Training SARSA(λ) with λ={lam} ---")