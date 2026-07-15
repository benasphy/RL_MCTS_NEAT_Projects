import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# =====================================================================
# DECENTRALIZED ACTOR NETWORK
# =====================================================================
class DecentralizedActor(nn.Module):
    def __init__(self, local_obs_dim=1, action_dim=2):
        super(DecentralizedActor, self).__init__()
        # Each agent only sees its own observation during execution
        self.net = nn.Sequential(
            nn.Linear(local_obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, obs):
        return self.net(obs)

# =====================================================================
# CENTRALIZED CRITIC NETWORK (Used only during training)
# =====================================================================
class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim=2, joint_action_dim=2):
        super(CentralizedCritic, self).__init__()
        # Ingests BOTH agents' states and actions to evaluate global quality
        self.net = nn.Sequential(
            nn.Linear(global_state_dim + joint_action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Outputs a single joint state-action value Q(S, a1, a2)
        )
        
    def forward(self, global_state, joint_actions):
        x = torch.cat([global_state, joint_actions], dim=-1)
        return self.net(x)