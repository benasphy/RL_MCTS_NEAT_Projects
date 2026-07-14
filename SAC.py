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

    def sample_action(self, state):
        mean, std = self.forward(state)
        
        # 1. Reparameterization Trick: sample from standard normal noise
        normal = torch.randn_like(mean)
        raw_action = mean + std * normal
        
        # 2. Squash action to [-1.0, 1.0] bound
        action = torch.tanh(raw_action)
        
        # 3. Calculate squashed log probability with the Jacobian correction
        # log_prob = log_prob(raw_action) - sum(log(1 - tanh^2(raw_action)))
        log_prob_normal = -0.5 * (((raw_action - mean) / std).pow(2) + 2 * torch.log(std) + np.log(2 * np.pi))
        log_prob = log_prob_normal - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

# =====================================================================
# ADAPTIVE TEMPERATURE TUNING ENGINE
# =====================================================================
# Dimensions: 1 continuous action
action_dim = 1
target_entropy = -float(action_dim) # Common heuristic: -dim(A)

# Setup dual temperature parameters
log_alpha = torch.zeros(1, requires_grad=True)
alpha_optimizer = optim.Adam([log_alpha], lr=0.01)

actor = SACActor(state_dim=3, action_dim=1)
simulated_state = torch.randn(1, 3)

print("--- Initializing Soft Actor-Critic Components ---")