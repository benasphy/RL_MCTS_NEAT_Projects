import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# =====================================================================
# DECENTRALIZED ACTOR NETWORK
# =====================================================================
class DecentralizedActor(nn.Module):
    def __init__(self, local_obs_dim=1, action_dim=2):
        super(DecentralizedActor, self).__init__()
        # Each agent only sees its own observation during execution
        self.net = nn.Sequential(
            nn.Linear(local_obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, obs):
        return self.net(obs)

# =====================================================================
# CENTRALIZED CRITIC NETWORK (Used only during training)
# =====================================================================
class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim=2, joint_action_dim=2):
        super(CentralizedCritic, self).__init__()
        # Ingests BOTH agents' states and actions to evaluate global quality
        self.net = nn.Sequential(
            nn.Linear(global_state_dim + joint_action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Outputs a single joint state-action value Q(S, a1, a2)
        )
        
    def forward(self, global_state, joint_actions):
        x = torch.cat([global_state, joint_actions], dim=-1)
        return self.net(x)

# =====================================================================
# CTDE MARL TRAINING LOOP
# =====================================================================
# Initialize 2 independent actors (Decentralized)
actor1 = DecentralizedActor()
actor2 = DecentralizedActor()

# Initialize 1 centralized critic (Centralized)
central_critic = CentralizedCritic()

# Optimizers
actor_opt = optim.Adam(list(actor1.parameters()) + list(actor2.parameters()), lr=0.01)
critic_opt = optim.Adam(central_critic.parameters(), lr=0.01)

# Simulating Global State & Local Observations
# Global state: positions of both agents
# Local observations: each agent only observes its own position
agent1_obs = torch.tensor([[0.2]], dtype=torch.float32)
agent2_obs = torch.tensor([[0.8]], dtype=torch.float32)
global_state = torch.cat([agent1_obs, agent2_obs], dim=-1) # shape: [1, 2]

print("--- Starting CTDE Training Optimization ---")

# 1. Forward Pass - Decentralized Actor Execution
probs1 = actor1(agent1_obs)
probs2 = actor2(agent2_obs)

dist1, dist2 = Categorical(probs1), Categorical(probs2)
action1, action2 = dist1.sample(), dist2.sample()

log_prob1 = dist1.log_prob(action1)
log_prob2 = dist2.log_prob(action2)

print(f"Agent 1 Action Choice: {action1.item()} (probs: {probs1.detach().numpy()[0]})")
print(f"Agent 2 Action Choice: {action2.item()} (probs: {probs2.detach().numpy()[0]})")

# 2. Centralized Evaluation
# Format joint actions as continuous floats for the critic input
joint_actions = torch.tensor([[float(action1), float(action2)]], dtype=torch.float32)

# Simulate a cooperative reward (e.g., they successfully moved closer to each other)
cooperative_reward = torch.tensor([[1.0]], dtype=torch.float32)

# Calculate target value (using simulated next-state target value = 0.5 for demonstration)
simulated_next_v = torch.tensor([[0.5]], dtype=torch.float32)
y_target = cooperative_reward + 0.99 * simulated_next_v

# Critic updates
q_value = central_critic(global_state, joint_actions)
critic_loss = nn.MSELoss()(q_value, y_target)

critic_opt.zero_grad()
critic_loss.backward()
critic_opt.step()

# 3. Policy Update utilizing the Centralized Critic's baseline
# Compute the advantage using the joint Q value
advantage = (y_target - q_value).detach()

# Update both actors based on the unified centralized signal
actor_loss = -(log_prob1 + log_prob2) * advantage
actor_opt.zero_grad()
actor_loss.backward()
actor_opt.step()

print("\n--- Training Step Successful ---")
print(f"Joint Q-Value Evaluation: {q_value.item():.4f}")
print(f"Calculated Critic Loss:    {critic_loss.item():.4f}")