import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# =====================================================================
# THE LEARNED WORLD MODEL (Transition/Dynamics Network)
# =====================================================================
class DynamicsModel(nn.Module):
    def __init__(self):
        super(DynamicsModel, self).__init__()
        # Input: state (1) + action (1)
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Outputs predicted next-state
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

# =====================================================================
# DYNA MODEL-BASED PLANNING ENGINE
# =====================================================================
class DynaAgent:
    def __init__(self, state_dim=1, action_dim=2):
        self.q_table = np.zeros((11, action_dim)) # Tabular states 0 to 10
        self.dynamics_model = DynamicsModel()
        self.model_optimizer = optim.Adam(self.dynamics_model.parameters(), lr=0.01)
        self.real_memory = [] # Store real world experiences
    
    def get_action(self, state_idx, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, 1)
        return np.argmax(self.q_table[state_idx])
    
    def discretize(self, state):
        # Scale continuous state [0.0, 1.0] to table index [0, 10]
        return int(np.clip(state * 10, 0, 10))