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