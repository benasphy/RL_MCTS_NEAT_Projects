import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class DiscreteLineEnv:
    def __init__(self):
        self.state = 0.0
        self.goal = 1.0
    
    def reset(self):
        self.state = 0.0
        return np.array([self.state], dtype=np.float32)
    
    def step(self, action):
        if action == 1: # Move
            self.state += np.random.uniform(0.1, 0.2)
        self.state = min(self.goal, self.state)
        reward = -1.0 if self.state < self.goal else 0.0
        return np.array([self.state], dtype=np.float32), reward, (self.state >= self.goal)