import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

# --- Hyperparameters ---
NUM_ENVS = 4          # Number of parallel environments running concurrently
N_STEPS = 5           # Multi-step rollout horizon (N-step TD)
GAMMA = 0.99
LR = 0.003
TOTAL_UPDATES = 200

class ParallelDiscreteLineEnv:
    """ Manages multiple independent environments vectorized via NumPy """
    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.states = np.zeros((num_envs, 1), dtype=np.float32)
        self.goal = 1.0
    
    def reset(self):
        self.states = np.zeros((self.num_envs, 1), dtype=np.float32)
        return np.copy(self.states)

    def step(self, actions):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        
        for i in range(self.num_envs):
            if actions[i] == 1: # Move
                self.states[i] += np.random.uniform(0.08, 0.18)
            self.states[i] = min(self.goal, self.states[i])
            
            if self.states[i] >= self.goal:
                rewards[i] = 0.0
                dones[i] = True
                self.states[i] = 0.0 # Auto-reset env independently
            else:
                rewards[i] = -1.0
                dones[i] = False
                
        return np.copy(self.states), rewards, dones

class A2CNetwork(nn.Module):
    def __init__(self):
        super(A2CNetwork, self).__init__()
        self.trunk = nn.Sequential(nn.Linear(1, 64), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(64, 2), nn.Softmax(dim=-1))
        self.critic = nn.Linear(64, 1)
        
    def forward(self, x):
        features = self.trunk(x)
        return self.actor(features), self.critic(features)

# --- Initialization ---
envs = ParallelDiscreteLineEnv(num_envs=NUM_ENVS)
model = A2CNetwork()
optimizer = optim.Adam(model.parameters(), lr=LR)

states = envs.reset()
print(f"--- Training Vectorized A2C Engine across {NUM_ENVS} Parallel Envs ---")

for update in range(TOTAL_UPDATES):
    # Storage for N-step rollouts across all parallel environments
    mb_states, mb_actions, mb_rewards, mb_dones, mb_log_probs = [], [], [], [], []
    
    # 1. Collect N-step trajectories across parallel channels
    for _ in range(N_STEPS):
        states_t = torch.from_numpy(states)
        probs, values = model(states_t)
        
        dist = Categorical(probs)
        actions = dist.sample()
        
        next_states, rewards, dones = envs.step(actions.numpy())
        
        mb_states.append(states_t)
        mb_actions.append(actions)
        mb_rewards.append(torch.from_numpy(rewards))
        mb_dones.append(torch.from_numpy(dones.astype(np.float32)))
        mb_log_probs.append(dist.log_prob(actions))
        
        states = next_states

    # 2. Compute Bootstrap Values for the final horizon state
    with torch.no_grad():
        _, next_values = model(torch.from_numpy(states))
        next_values = next_values.squeeze(-1)
        
    # 3. Calculate N-step targets moving backwards
    mb_returns = torch.zeros((N_STEPS, NUM_ENVS))
    g_return = next_values
    for t in reversed(range(N_STEPS)):
        # If the environment reset at this step, set next value to 0
        g_return = mb_rewards[t] + GAMMA * g_return * (1 - mb_dones[t])
        mb_returns[t] = g_return

    # Flatten collected lists into long batch tensors for optimization pass
    states_batch = torch.cat(mb_states, dim=0)
    actions_batch = torch.cat(mb_actions, dim=0)
    returns_batch = mb_returns.view(-1, 1)
    log_probs_batch = torch.cat(mb_log_probs, dim=0).view(-1, 1)
    
    # 4. Compute Loss and Advantages
    _, values_batch = model(states_batch)
    advantages_batch = returns_batch - values_batch
    
    critic_loss = advantages_batch.pow(2).mean()
    actor_loss = -(log_probs_batch * advantages_batch.detach()).mean()
    
    # Total A2C Loss
    total_loss = actor_loss + 0.5 * critic_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

print("Training Complete.\n")

# --- Inference Check ---
model.eval()
print("--- Unified A2C Evaluator Pass ---")
with torch.no_grad():
    for test_pos in [0.1, 0.9]:
        st = torch.tensor([[test_pos]], dtype=torch.float32)
        probs, val = model(st)
        print(f"Position: {test_pos:<3} | Critic V(s): {val.item():.2f} | Actor [Stay, Move]: {probs.numpy()[0]}")