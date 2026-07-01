import numpy as np

class EnvironmentBlackBox:
    """The GridWorld from before, but the agent cannot look inside its rules."""
    def __init__(self):
        self.num_states = 9
        self.terminal_state = 8
        self.actions = [0, 1, 2, 3] # Up, Down, Left, Right
        self.action_effects = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    
    def reset(self):
        # Always start at state 0 for evaluations, or random for general tracking
        return 0
    def step(self, state, action):
        if state == self.terminal_state:
            return state, 0, True
        row, col = state // 3, state % 3
        d_row, d_col = self.action_effects[action]
        next_row = max(0, min(2, row + d_row))
        next_col = max(0, min(2, col + d_col))
        next_state = next_row * 3 + next_col
        return next_state, -1, (next_state == self.terminal_state)

# =====================================================================
# PROJECT 1: FIRST-VISIT MONTE CARLO PREDICTION
# =====================================================================
print("--- Running First-Visit MC Prediction (5000 Episodes) ---")
env = EnvironmentBlackBox()
V_estimates = np.zeros(env.num_states)
state_returns_count = np.zeros(env.num_states)

for episode in range(5000):
    state = env.reset()
    trajectory = []
    done = False
    
    # 1. Generate a trajectory using a random uniform policy
    while not done:
        action = np.random.choice(env.actions)
        next_state, reward, done = env.step(state, action)
        trajectory.append((state, action, reward))
        state = next_state