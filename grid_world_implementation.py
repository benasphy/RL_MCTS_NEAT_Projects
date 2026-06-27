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