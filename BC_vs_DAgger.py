import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# A simple controller mapping 1D position drift to corrective steering force
class StudentPolicy(nn.Module):
    def __init__(self):
        super(StudentPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Continuous corrective action
        )
    def forward(self, x):
        return self.fc(x)

# Perfect Expert Policy: Always strives to keep position at exactly 0.0
def expert_policy(position):
    return -2.0 * position  # Simple proportional controller

# --- 1. Generative Expert Demonstrations (Ideal Path) ---
# The expert always starts perfectly centered, so state is always 0.0
expert_states = torch.zeros(100, 1)
expert_actions = torch.zeros(100, 1) # Action is also 0.0 (no steering needed)

# --- 2. Classical Behavioral Cloning ---
student_bc = StudentPolicy()
optimizer_bc = optim.Adam(student_bc.parameters(), lr=0.01)