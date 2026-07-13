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

# --- Ornstein-Uhlenbeck Exploration Noise Pattern ---
class OUNoise:
    def __init__(self, action_dim=1, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = torch.ones(action_dim) * mu

    def sample(self):
        # Continuous mean reverting random walk simulation
        dx = self.theta * (self.mu - self.state) + self.sigma * torch.randn(len(self.state))
        self.state += dx
        return self.state

# =====================================================================
# TRAINING PIPELINE DEMO
# =====================================================================
# Instantiate live networks
actor = DDPGActor()
critic = DDPGCritic()

# Instantiate target networks
target_actor = DDPGActor()
target_critic = DDPGCritic()

# Force initial weights to match perfectly
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

ou_noise = OUNoise()
TAU = 0.005 # Polyak soft-update scale coefficient

print("--- Initializing Continuous DDPG Framework ---")