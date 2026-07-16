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

# Train student on expert's ideal data
for epoch in range(100):
    predictions = student_bc(expert_states)
    loss = nn.MSELoss()(predictions, expert_actions)
    optimizer_bc.zero_grad()
    loss.backward()
    optimizer_bc.step()

# --- Simulation of BC deployment with small environment noise ---
print("--- Deploying Passive Behavioral Cloning ---")
position = 0.5 # A gust of wind knocks the agent off course
print(f"Initial Drift: {position:.4f}")

for step in range(3):
    action = student_bc(torch.tensor([[position]], dtype=torch.float32)).item()
    # Apply action with environment dynamics + noise
    position = position + action + np.random.normal(0.05, 0.02)
    print(f"Step {step+1} | Applied Action: {action: .4f} | Position: {position:.4f}")

# --- 3. DAgger Loop (Interactive Correction) ---
print("\n--- Starting DAgger Dataset Aggregation ---")
student_dagger = StudentPolicy()
optimizer_dagger = optim.Adam(student_dagger.parameters(), lr=0.01)

# Start with the initial expert dataset
aggregated_states = expert_states.clone()
aggregated_actions = expert_actions.clone()

for dagger_round in range(3):
    # Train policy on all collected data so far
    for epoch in range(50):
        predictions = student_dagger(aggregated_states)
        loss = nn.MSELoss()(predictions, aggregated_actions)
        optimizer_dagger.zero_grad()
        loss.backward()
        optimizer_dagger.step()
    
    # Execute current policy and collect encountered states (with noise)
    new_states = []
    current_pos = 0.5 # Start displaced
    for _ in range(5):
        new_states.append([current_pos])
        action = student_dagger(torch.tensor([[current_pos]], dtype=torch.float32)).item()
        current_pos = current_pos + action + np.random.normal(0.05, 0.02)
    
    # Query the expert for corrective actions at these newly visited states
    new_states_tensor = torch.tensor(new_states, dtype=torch.float32)
    expert_corrective_actions = expert_policy(new_states_tensor)
    
    # Aggregate!
    aggregated_states = torch.cat([aggregated_states, new_states_tensor], dim=0)
    aggregated_actions = torch.cat([aggregated_actions, expert_corrective_actions], dim=0)
    print(f"DAgger Round {dagger_round+1} complete. Total dataset size: {len(aggregated_states)}")