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
    

# --- Hyperparameters ---
EPSILON = 0.2  # Clipping boundary limit (1 - eps to 1 + eps)
LR = 0.001

# --- Initialization ---
policy_net = PPOPolicy()
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

# 1. Simulate a batch of collected transitions
# 3 samples: states, actions taken, and their calculated GAE advantages
simulated_states = torch.randn(3, 2)
simulated_actions = torch.tensor([1, 0, 1])
simulated_advantages = torch.tensor([2.5, -1.2, 4.0], dtype=torch.float32)

# 2. Extract and fix the Old Log Probabilities (simulating the behavior data)
with torch.no_grad():
    probs_old = policy_net(simulated_states)
    dist_old = Categorical(probs_old)
    old_log_probs = dist_old.log_prob(simulated_actions)

print("--- Starting PPO Clip Step Optimization ---")
print(f"Old Action Probabilities: {probs_old.numpy()[:, 1]}")