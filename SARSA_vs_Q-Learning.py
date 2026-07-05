import numpy as np

class CliffWalkingMDP:
    def __init__(self):
        # 2x4 grid = 8 states
        # 0(Start)  1        2        3
        # 4(Cliff)  5(Cliff)  6(Cliff)  7(Goal)
        self.num_states = 8
        self.start_state = 0
        self.goal_state = 7
        self.cliff_states = [4, 5, 6]
        self.actions = [0, 1, 2, 3] # Up, Down, Left, Right
        self.action_effects = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    
    def step(self, state, action):
        if state == self.goal_state:
            return state, 0, True
            
        row, col = state // 4, state % 4
        d_row, d_col = self.action_effects[action]
        next_row = max(0, min(1, row + d_row))
        next_col = max(0, min(3, col + d_col))
        next_state = next_row * 4 + next_col
        
        if next_state in self.cliff_states:
            return self.start_state, -100, False # Penalty and reset to start
            
        if next_state == self.goal_state:
            return next_state, 0, True
            
        return next_state, -1, False

# Hyperparameters
gamma = 1.0
alpha = 0.1
epsilon = 0.25 # High exploration rate to emphasize the policy split
episodes = 3000

env = CliffWalkingMDP()

# =====================================================================
# ALGORITHM 1: Q-LEARNING
# =====================================================================
Q_val = np.zeros((env.num_states, len(env.actions)))

for episode in range(episodes):
    state = env.start_state
    done = False
    
    while not done:
        # Action selection using Behavior Policy (epsilon-greedy)
        if np.random.rand() < epsilon:
            action = np.random.choice(env.actions)
        else:
            action = np.random.choice(np.flatnonzero(Q_val[state] == Q_val[state].max()))
            
        next_state, reward, done = env.step(state, action)
        
        # Q-Learning Target: Uses max over next actions (Target Policy is purely greedy)
        td_target = reward + gamma * np.max(Q_val[next_state])
        Q_val[state, action] += alpha * (td_target - Q_val[state, action])
        
        state = next_state
        
# Display Paths
def extract_path(Q_table, name):
    state = env.start_state
    path = [state]
    steps = 0
    while state != env.goal_state and steps < 10:
        action = np.argmax(Q_table[state])
        state, _, _ = env.step(state, action)
        path.append(state)
        steps += 1
    print(f"{name} Optimal Route: {' -> '.join(map(str, path))}")

print("--- The Results ---")
extract_path(Q_val, "Q-Learning")
extract_path(Q_sarsa, "SARSA")