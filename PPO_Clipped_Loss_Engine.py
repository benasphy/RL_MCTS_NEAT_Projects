import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOPolicy(nn.Module):
    def __init__(self):
        super(PPOPolicy, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(2, 32), # Simulating a 2D state input
            nn.ReLU(),
            nn.Linear(32, 2), # 2 discrete actions
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.actor(x)