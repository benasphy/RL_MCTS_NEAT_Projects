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