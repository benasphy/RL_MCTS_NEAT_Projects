import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# =====================================================================
# SQUASHED GAUSSIAN ACTOR WITH REPARAMETERIZATION TRICK
# =====================================================================
class SACActor(nn.Module):
    def __init__(self, state_dim=3, action_dim=1, log_std_min=-20, log_std_max=2):
        super(SACActor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(64, action_dim)
        self.log_std_head = nn.Linear(64, action_dim)
    
    def forward(self, state):
        features = self.shared(state)
        mean = self.mean_head(features)
        
        # Clip log_std to maintain numerical stability in exponentials
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mean, std