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
        # Action 0 is 'Stay', keeping the state identical
        self.state = min(self.goal, self.state)
        reward = -1.0 if self.state < self.goal else 0.0
        return np.array([self.state], dtype=np.float32), reward, (self.state >= self.goal)

# =====================================================================
# THE REINFORCE NETWORK
# =====================================================================
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # Outputs probabilities for 2 discrete actions: [Stay, Move]
        self.policy_head = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1) # Guarantees outputs sum to a valid 1.0 probability distribution
        )
    def forward(self, x):
        return self.policy_head(x)

# --- Initialization ---
env = DiscreteLineEnv()
policy = PolicyNetwork()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

GAMMA = 0.99
EPISODES = 400

print("--- Training REINFORCE Policy Gradient Agent ---")