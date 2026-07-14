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