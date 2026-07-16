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