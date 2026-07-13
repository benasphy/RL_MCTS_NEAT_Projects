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

class DDPGCritic(nn.Module):
    def __init__(self, state_dim=3, action_dim=1):
        super(DDPGCritic, self).__init__()
        # The Critic takes BOTH state and action as inputs combined
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 1) # Outputs a single scalar Q(s,a)
        
    def forward(self, state, action):
        # Concatenate state and action tensors along the feature dimension
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)