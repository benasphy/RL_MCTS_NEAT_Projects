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

# =====================================================================
# IMPORTANCE SAMPLING RATIO & KL CALCULATION
# =====================================================================
def calculate_kl_and_ratio(p_old, p_new, action_taken):
    dist_old = Categorical(p_old)
    dist_new = Categorical(p_new)
    
    # 1. Importance Sampling Ratio: r(theta) = pi_new(a|s) / pi_old(a|s)
    ratio = torch.exp(dist_new.log_prob(action_taken) - dist_old.log_prob(action_taken))
    
    # 2. Analytical KL Divergence between the two distributions
    # KL(P || Q) = Sum( P(x) * log( P(x) / Q(x) ) )
    kl_div = torch.sum(p_old * torch.log(p_old / p_new), dim=-1)
    
    return ratio, kl_div

# Simulate taking action 1
simulated_action = torch.tensor([1])

ratio, kl = calculate_kl_and_ratio(probs_old, probs_new, simulated_action)
print(f"Initial Setup -> Ratio: {ratio.item():.4f} | KL Divergence: {kl.item():.6f}")

# --- Simulating a Large, Dangerous Gradient Step ---
print("\n[Warning] Simulating an unconstrained large weight modification...")
with torch.no_grad():
    # Artificially alter the weights of the new policy's final layer significantly
    policy_new.fc[2].weight.add_(1.8)

# Re-evaluate distributions post modification
probs_new_mutated = policy_new(simulated_state)
ratio_mutated, kl_mutated = calculate_kl_and_ratio(probs_old, probs_new_mutated, simulated_action)

print(f"Post-Mutation -> Mutated Policy Outputs: {probs_new_mutated.detach().numpy()[0]}")
print(f"Post-Mutation -> Ratio: {ratio_mutated.item():.4f} | KL Divergence: {kl_mutated.item():.6f}")

# Enforce a conceptual Trust Region Boundary check
DELTA = 0.01
if kl_mutated.item() > DELTA:
    print(f"--> [REJECT UPDATE] KL Divergence ({kl_mutated.item():.4f}) exceeded the Trust Region bound of {DELTA}!")
else:
    print("--> [ACCEPT UPDATE] Update is safe within the trust region.")