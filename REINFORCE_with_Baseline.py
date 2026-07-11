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
        # Action 0 is 'Stay', keeping the state identical
        self.state = min(self.goal, self.state)
        reward = -1.0 if self.state < self.goal else 0.0
        return np.array([self.state], dtype=np.float32), reward, (self.state >= self.goal)

# =====================================================================
# THE REINFORCE NETWORK
# =====================================================================
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # Outputs probabilities for 2 discrete actions: [Stay, Move]
        self.policy_head = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1) # Guarantees outputs sum to a valid 1.0 probability distribution
        )
    def forward(self, x):
        return self.policy_head(x)

# --- Initialization ---
env = DiscreteLineEnv()
policy = PolicyNetwork()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

GAMMA = 0.99
EPISODES = 400

print("--- Training REINFORCE Policy Gradient Agent ---")

for episode in range(EPISODES):
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False
    
    # 1. Collect a full Monte Carlo episode trajectory
    while not done:
        state_t = torch.from_numpy(state).float()
        probs = policy(state_t)
        
        # Create a PyTorch categorical distribution to sample cleanly from probabilities
        dist = Categorical(probs)
        action = dist.sample().item()
        
        next_state, reward, done = env.step(action)
        
        states.append(state_t)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        
    # 2. Compute discounted returns (G_t) backwards
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    
    # Simple Running Baseline for Variance Reduction (Mean of current trajectory returns)
    baseline = returns.mean()
    advantages = returns - baseline
    
    # 3. Calculate Policy Gradient Loss Pass
    policy_loss = []
    for state_t, action, advantage in zip(states, actions, advantages):
        probs = policy(state_t)
        dist = Categorical(probs)
        
        # Calculate log pi(a|s)
        log_prob = dist.log_prob(torch.tensor(action))
        
        # Policy Gradient standard minimization loss term (Negative sign for Gradient *Ascent*)
        policy_loss.append(-log_prob * advantage)
    # 4. Backpropagation Update
    optimizer.zero_grad()
    # Sum up individual transition losses and update network weights
    torch.stack(policy_loss).sum().backward()
    optimizer.step()

print("Training Complete.\n")

# --- Inference Check ---
policy.eval()
print("--- Directly Optimized Action Probabilities Profile ---")
with torch.no_grad():
    for test_pos in [0.1, 0.5, 0.9]:
        st = torch.tensor([test_pos], dtype=torch.float32)
        probs = policy(st).numpy()
        print(f"Continuous State: {test_pos:<3} -> Probabilities: [Stay: {probs[0]:.2f}, Move: {probs[1]:.2f}]")