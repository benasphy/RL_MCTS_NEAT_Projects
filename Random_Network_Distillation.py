import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# RND NETWORK ARCHITECTURE
# =====================================================================
class RNDTarget(nn.Module):
    """Fixed random target network mapping inputs to a 10-dim embedding."""
    def __init__(self, state_dim=1, latent_dim=10):
        super(RNDTarget, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        # Freeze weights immediately so this acts as a static, deterministic function
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.fc(x)

class RNDPredictor(nn.Module):
    """Trainable network designed to match the target network's output."""
    def __init__(self, state_dim=1, latent_dim=10):
        super(RNDPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, x):
        return self.fc(x)

# =====================================================================
# SIMULATED RND TRAINING ENGINE
# =====================================================================
# Dimensions: 1D State Space (e.g., position in a hallway)
target_net = RNDTarget()
predictor_net = RNDPredictor()
optimizer = optim.Adam(predictor_net.parameters(), lr=0.01)

# Helper function to get RND Intrinsic Reward (Prediction Error)
def get_intrinsic_reward(state):
    with torch.no_grad():
        target_out = target_net(state)
        pred_out = predictor_net(state)
        # Compute squared L2 norm per sample
        error = torch.sum((pred_out - target_out) ** 2, dim=-1)
    return error.item()

# 1. Evaluate familiarity on two distinct states
familiar_state = torch.tensor([[1.0]], dtype=torch.float32) # State we will visit frequently
novel_state = torch.tensor([[9.0]], dtype=torch.float32)    # State we will keep hidden

print("--- Initial Evaluation (Both States are Unexplored) ---")
print(f"Familiar State Curiosity Bonus: {get_intrinsic_reward(familiar_state):.4f}")
print(f"Novel State Curiosity Bonus:    {get_intrinsic_reward(novel_state):.4f}")

# 2. Simulate training: The agent visits the "familiar_state" repeatedly
print("\n[Training] Agent spends time exploring state 1.0...")
predictor_net.train()

for epoch in range(100):
    # Pass familiar_state through both networks
    target_out = target_net(familiar_state)
    pred_out = predictor_net(familiar_state)
    
    loss = torch.mean((pred_out - target_out) ** 2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 3. Evaluate curiosity values post training
print("\n--- Post-Training Evaluation ---")
bonus_familiar = get_intrinsic_reward(familiar_state)
bonus_novel = get_intrinsic_reward(novel_state)

print(f"Familiar State (1.0) Curiosity Bonus: {bonus_familiar:.6f}")
print(f"Novel State (9.0) Curiosity Bonus:    {bonus_novel:.4f}")

if bonus_novel > (bonus_familiar * 10):
    print("\n--> [Success] The agent got bored of state 1.0. It is now highly motivated to seek out state 9.0!")