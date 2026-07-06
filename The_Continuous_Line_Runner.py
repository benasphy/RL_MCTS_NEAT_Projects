import numpy as np

class ContinuousLineEnv:
    """A truly continuous 1D environment."""
    def __init__(self):
        self.state = 0.0
        self.goal = 1.0
    
    def reset(self):
        self.state = 0.0
        return self.state
    
    def step(self, action):
        # Action is a forward push with a bit of noise
        step_size = np.random.uniform(0.05, 0.15)
        self.state = min(self.goal, self.state + step_size)
        
        reward = -1  # Every second spent costs time
        done = (self.state >= self.goal)
        return self.state, reward, done