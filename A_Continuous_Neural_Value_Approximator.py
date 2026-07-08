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
    
class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        # Takes 1 raw continuous state float, passes through hidden layers, outputs 1 value
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)