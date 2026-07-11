import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- Hyperparameters ---
GAMMA = 0.95
LR = 0.0005
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 100
EPSILON = 0.1
EPISODES = 100

class DiscreteLineEnv:
    def __init__(self):
        self.state = 0.0
        self.goal = 1.0
    
    def reset(self):
        self.state = 0.0
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        if action == 1: self.state += np.random.uniform(0.05, 0.15)
        self.state = min(self.goal, self.state)
        reward = -1.0 if self.state < self.goal else 0.0
        return np.array([self.state], dtype=np.float32), reward, (self.state >= self.goal)

# =====================================================================
# DUELING ARCHITECTURE NETWORK
# =====================================================================
class DuelingQNetwork(nn.Module):
    def __init__(self):
        super(DuelingQNetwork, self).__init__()
        # Shared feature extractor base
        self.feature_network = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU()
        )
        
        # Stream 1: State Value V(s) Head
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Stream 2: Action Advantage A(s,a) Head
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # 2 actions: [Stay, Move]
        )
    
    def forward(self, x):
        features = self.feature_network(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Dueling Combine Rule: Q(s,a) = V(s) + (A(s,a) - Mean(A))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values