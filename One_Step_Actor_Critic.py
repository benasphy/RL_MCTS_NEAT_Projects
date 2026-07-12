import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class DiscreteLineEnv:
    def __init__(self):
        self.state = 0.0
        self.goal = 1.0
    
    def reset(self):
        self.state = 0.0
        return np.array([self.state], dtype=np.float32)
    
    def step(self, action):
        if action == 1: # Move
            self.state += np.random.uniform(0.1, 0.2)
        self.state = min(self.goal, self.state)
        reward = -1.0 if self.state < self.goal else 0.0
        return np.array([self.state], dtype=np.float32), reward, (self.state >= self.goal)

# =====================================================================
# UNIFIED ACTOR-CRITIC NETWORK
# =====================================================================
class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super(ActorCriticNetwork, self).__init__()
        # Shared feature representation layer
        self.shared_trunk = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU()
        )
        # Head 1: The Actor (Outputs action probabilities)
        self.actor_head = nn.Sequential(
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
        # Head 2: The Critic (Outputs a single state value estimation V(s))
        self.critic_head = nn.Linear(64, 1)
        
    def forward(self, x):
        features = self.shared_trunk(x)
        action_probs = self.actor_head(features)
        state_value = self.critic_head(features)
        return action_probs, state_value

# --- Initialization ---
env = DiscreteLineEnv()
ac_model = ActorCriticNetwork()
# A single optimizer adjusts both actor and critic parameters simultaneously
optimizer = optim.Adam(ac_model.parameters(), lr=0.005)

GAMMA = 0.99
EPISODES = 300

print("--- Training Online One-Step Actor-Critic Agent ---")