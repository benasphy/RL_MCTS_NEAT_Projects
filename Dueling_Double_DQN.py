import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- Hyperparameters ---
GAMMA = 0.95
LR = 0.0005
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 100
EPSILON = 0.1
EPISODES = 100

class DiscreteLineEnv:
    def __init__(self):
        self.state = 0.0
        self.goal = 1.0
    
    def reset(self):
        self.state = 0.0
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        if action == 1: self.state += np.random.uniform(0.05, 0.15)
        self.state = min(self.goal, self.state)
        reward = -1.0 if self.state < self.goal else 0.0
        return np.array([self.state], dtype=np.float32), reward, (self.state >= self.goal)

# =====================================================================
# DUELING ARCHITECTURE NETWORK
# =====================================================================
class DuelingQNetwork(nn.Module):
    def __init__(self):
        super(DuelingQNetwork, self).__init__()
        # Shared feature extractor base
        self.feature_network = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU()
        )
        
        # Stream 1: State Value V(s) Head
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Stream 2: Action Advantage A(s,a) Head
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # 2 actions: [Stay, Move]
        )
    
    def forward(self, x):
        features = self.feature_network(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Dueling Combine Rule: Q(s,a) = V(s) + (A(s,a) - Mean(A))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

# --- Initialization ---
env = DiscreteLineEnv()
online_net = DuelingQNetwork()
target_net = DuelingQNetwork()
target_net.load_state_dict(online_net.state_dict())

optimizer = optim.Adam(online_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()
replay_buffer = deque(maxlen=MEMORY_SIZE)
total_steps = 0

print("--- Training Dueling Double-DQN Engine ---")

for episode in range(EPISODES):
    state = env.reset()
    done = False
    
    while not done:
        total_steps += 1
        
        # Action selection (epsilon-greedy)
        if random.random() < EPSILON:
            action = random.choice([0, 1])
        else:
            state_t = torch.from_numpy(state).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(online_net(state_t)).item()
                
        next_state, reward, done = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        
        if len(replay_buffer) > BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states_t = torch.from_numpy(np.array(states))
            actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
            rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states_t = torch.from_numpy(np.array(next_states))
            dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
            
            # Current Q-values from live online network
            current_q = online_net(states_t).gather(1, actions_t)
            
            # =====================================================================
            # DOUBLE DQN TARGET LOGIC
            # =====================================================================
            with torch.no_grad():
                # Step 1: Use ONLINE network to determine the BEST action in the next state
                best_actions_online = online_net(next_states_t).argmax(dim=1, keepdim=True)
                
                # Step 2: Use TARGET network to evaluate the value of that chosen action
                target_q_next = target_net(next_states_t).gather(1, best_actions_online)
                
                # Step 3: Compute double-decoupled TD Target
                target_q = rewards_t + (GAMMA * target_q_next * (1 - dones_t))
                
            loss = loss_fn(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if total_steps % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

print("Training Complete.\n")