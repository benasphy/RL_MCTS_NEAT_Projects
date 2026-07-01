import numpy as np

class EnvironmentBlackBox:
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

# Shared Hyperparameters
gamma = 1.0
alpha = 0.1  # Learning rate
epsilon = 0.1
episodes = 2000
env = EnvironmentBlackBox()

# =====================================================================
# PROJECT 1: TD(0) PREDICTION
# =====================================================================
print("--- Running TD(0) Prediction ---")
V = np.zeros(env.num_states)

for episode in range(episodes):
    state = 0  # Start at cell 0
    done = False
    
    while not done:
        action = np.random.choice(env.actions) # Random Policy
        next_state, reward, done = env.step(state, action)
        
        # TD Target and Update
        td_target = reward + gamma * V[next_state]
        V[state] += alpha * (td_target - V[state])
        
        state = next_state

print("Converged V(s) via TD(0):")
print(np.round(V.reshape(3, 3), 1))