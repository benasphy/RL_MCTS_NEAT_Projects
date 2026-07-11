import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniRainbowNetwork(nn.Module):
    def __init__(self, num_actions=2, num_atoms=51):
        super(MiniRainbowNetwork, self).__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        
        # 1. Base Feature Extraction
        self.feature_backbone = nn.Sequential(
            nn.Linear(4, 64), # Simulating a 4D state input
            nn.ReLU()
        )
        
        # 2. Dueling Value Stream: Predicts distribution atoms for V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_atoms) # Outputs a value across the 51 atoms
        )
        
        # 3. Dueling Advantage Stream: Predicts distribution atoms for A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions * num_atoms) # Outputs atoms for each action
        )
    
    def forward(self, state):
        features = self.feature_backbone(state)
        
        # Reshape and extract streams
        v_atoms = self.value_stream(features).view(-1, 1, self.num_atoms)
        a_atoms = self.advantage_stream(features).view(-1, self.num_actions, self.num_atoms)
        
        # Dueling Combination Rule adjusted for Distributional Tensors
        # Q_atoms(s,a) = V_atoms(s) + (A_atoms(s,a) - Mean(A_atoms))
        q_atoms = v_atoms + (a_atoms - a_atoms.mean(dim=1, keepdim=True))
        
        # Apply Softmax across the atoms dimension to convert outputs to clean probabilities
        dist_probabilities = F.softmax(q_atoms, dim=-1)
        return dist_probabilities # Shape: [Batch, Actions, Atoms]

# --- Functional Demonstration ---
# Let's say our support points (Z) span from V_min = -10 to V_max = 10
v_min, v_max, num_atoms = -10.0, 10.0, 51
z_support = torch.linspace(v_min, v_max, num_atoms)

model = MiniRainbowNetwork(num_actions=2, num_atoms=num_atoms)
simulated_state = torch.randn(1, 4) # 1 sample batch, 4 state dimensions

# Run forward pass to get categorical probability distributions
action_distributions = model(simulated_state)

# Calculate the expected Q-value out of the distribution by taking the inner product with Z
expected_q_values = torch.sum(action_distributions * z_support, dim=-1)

print("--- Mini-Rainbow Output Check ---")
print("Raw Probability Distribution Shape (Batch, Actions, Atoms):", list(action_distributions.shape))
print("Calculated Expected Q-Values for Actions [0, 1]:", expected_q_values.detach().numpy()[0])