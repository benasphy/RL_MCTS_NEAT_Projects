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