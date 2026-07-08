import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ContinuousLineEnv:
    def __init__(self):
        self.state = 0.0
        self.goal = 1.0
    
    def reset(self):
        self.state = 0.0
        return np.array([self.state], dtype=np.float32)
    def step(self):
        step_size = np.random.uniform(0.05, 0.15)
        self.state = min(self.goal, self.state + step_size)
        reward = -1.0
        done = (self.state >= self.goal)
        return np.array([self.state], dtype=np.float32), reward, done