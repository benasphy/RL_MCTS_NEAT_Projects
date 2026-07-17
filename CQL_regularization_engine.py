import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_dim=2, action_dim=3):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim) # Outputs Q-value for each discrete action
        )
    def forward(self, state):
        return self.fc(state)

# --- Initialize Network and Optimizer ---
q_net = QNetwork()
optimizer = optim.Adam(q_net.parameters(), lr=0.01)

# Simulating Offline Dataset Batch
# We have a batch of 1 state and the action actually taken in the dataset was action 0
state = torch.randn(1, 2)
dataset_action = torch.tensor([0], dtype=torch.long)
target_q_value = torch.tensor([2.5], dtype=torch.float32) # Standard TD Target

# CQL Pessimism Multiplier (alpha)
ALPHA = 1.0

# 1. Before update: Evaluate raw Q-values
q_net.eval()
with torch.no_grad():
    initial_q = q_net(state)
print("--- Pre-Update Q-Values ---")
print(f"Dataset Action (0) Q-Value: {initial_q[0, 0].item():.4f}")
print(f"Unseen Action (1) Q-Value:  {initial_q[0, 1].item():.4f}")
print(f"Unseen Action (2) Q-Value:  {initial_q[0, 2].item():.4f}")

# 2. Conservative Q-Learning Step
q_net.train()

# Standard Temporal Difference Loss (MSE between predicted Q and target Q)
all_q_values = q_net(state)
pred_dataset_q = all_q_values.gather(1, dataset_action.unsqueeze(1)).squeeze(1)
td_loss = nn.MSELoss()(pred_dataset_q, target_q_value)

# CQL Penalty Calculation
# logsumexp over all actions acts as the pull-down term
logsumexp_q = torch.logsumexp(all_q_values, dim=-1)
# Mean of the dataset Q-value acts as the push-up term
dataset_q_mean = pred_dataset_q.mean()

cql_loss_penalty = ALPHA * (logsumexp_q - dataset_q_mean).mean()

# Combined Loss
total_loss = td_loss + cql_loss_penalty

optimizer.zero_grad()
total_loss.backward()
optimizer.step()

# 3. Post-Update Evaluation
q_net.eval()
with torch.no_grad():
    updated_q = q_net(state)
print("\n--- Post-CQL-Update Q-Values ---")
print(f"Dataset Action (0) Q-Value: {updated_q[0, 0].item():.4f} (Targeted via TD)")
print(f"Unseen Action (1) Q-Value:  {updated_q[0, 1].item():.4f} (Suppressed)")
print(f"Unseen Action (2) Q-Value:  {updated_q[0, 2].item():.4f} (Suppressed)")