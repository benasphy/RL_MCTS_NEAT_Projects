import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- Hyperparameters ---
GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 100  # Update target network every 100 steps
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
EPISODES = 300

class DiscreteLineEnv:
    def __init__(self):
        self.state = 0.0
        self.goal = 1.0
    
    def reset(self):
        self.state = 0.0
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        if action == 1: # Move forward
            self.state += np.random.uniform(0.05, 0.15)
        else: # Stay still
            self.state += 0.0
            
        self.state = min(self.goal, self.state)
        reward = -1.0 if self.state < self.goal else 0.0
        done = (self.state >= self.goal)
        return np.array([self.state], dtype=np.float32), reward, done

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Outputs Q-values for 2 actions: [Stay, Move]
        )
    def forward(self, x):
        return self.net(x)