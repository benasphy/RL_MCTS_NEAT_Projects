import numpy as np

class ContinuousLineEnv:
    """A truly continuous 1D environment."""
    def __init__(self):
        self.state = 0.0
        self.goal = 1.0
    
    def reset(self):
        self.state = 0.0
        return self.state