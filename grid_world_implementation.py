import numpy as np

class GridWorldMDP:
    def __init__(self):
        # 3x3 Grid represented as linear indices:
        # 0  1  2
        # 3  4  5
        # 6  7  8 (Terminal)
        self.num_states = 9
        self.terminal_state = 8
        
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.actions = [0, 1, 2, 3]
        self.action_effects = {
            0: (-1, 0),  # Up: change in (row, col)
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
    
    def get_state_coord(self, state):
        return state // 3, state % 3
    
    def get_state_index(self, row, col):
        return row * 3 + col
    
    def step(self, state, action):
        """
        Returns a tuple of (next_state, reward, is_terminal)
        """
        if state == self.terminal_state:
            return state, 0, True
            
        row, col = self.get_state_coord(state)
        d_row, d_col = self.action_effects[action]
        
        # Compute target position
        next_row = max(0, min(2, row + d_row))
        next_col = max(0, min(2, col + d_col))
        next_state = self.get_state_index(next_row, next_col)
        
        # Reward structure
        reward = -1
        is_terminal = (next_state == self.terminal_state)
        
        return next_state, reward, is_terminal