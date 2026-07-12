import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.fc(x)

# --- Verification Setup ---
simulated_state = torch.tensor([[0.5]], dtype=torch.float32)

# Instantiate the baseline (old) policy and a candidate (new) policy
policy_old = PolicyNetwork()
policy_new = PolicyNetwork()

# Force the new policy to match the old policy's parameters initially
policy_new.load_state_dict(policy_old.state_dict())

print("--- Initializing Trust Region Mechanics ---")

# Extract the probability distributions for a specific state
probs_old = policy_old(simulated_state).detach()
probs_new = policy_new(simulated_state)

print(f"Old Policy Outputs: {probs_old.numpy()[0]}")
print(f"New Policy Outputs: {probs_new.detach().numpy()[0]}")