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