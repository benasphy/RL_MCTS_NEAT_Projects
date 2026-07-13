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

# Simulate collecting a transition state from environment
simulated_state = torch.randn(1, 3) # Batch size of 1, 3 state features

# 1. Action Selection with Exploration Noise
actor.eval()
with torch.no_grad():
    raw_action = actor(simulated_state)
    noise = ou_noise.sample()
    exploration_action = raw_action + noise

print(f"Raw Deterministic Action Output: {raw_action.item():.4f}")
print(f"Exploration Action (With OU Noise): {exploration_action.item():.4f}")

# 2. Critic Evaluation & Polyak Target Synchronization Check
# Mutate online weights artificially to simulate an optimization update step
with torch.no_grad():
    critic.fc2.weight.add_(0.5)
    actor.network[2].weight.add_(0.5)

# Execute Polyak Soft Target Updates across both network pairs
def soft_update(online_model, target_model, tau):
    for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
        target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

soft_update(actor, target_actor, TAU)
soft_update(critic, target_critic, TAU)

print("\n--- Target Synchronization Log ---")
print(f"Online Critic Weight Element Sample: {critic.fc2.weight[0,0].item():.4f}")
print(f"Target Critic Weight Element Sample: {target_critic.fc2.weight[0,0].item():.4f}")