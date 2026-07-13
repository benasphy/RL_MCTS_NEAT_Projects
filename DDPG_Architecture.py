import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# DDPG ACTOR & CRITIC NETWORKS
# =====================================================================
class DDPGActor(nn.Module):
    def __init__(self, state_dim=3, action_dim=1):
        super(DDPGActor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Scales output bound strictly to [-1.0, 1.0] for safe control
        )
    def forward(self, state):
        return self.network(state)