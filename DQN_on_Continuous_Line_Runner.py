import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- Hyperparameters ---
GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 100  # Update target network every 100 steps
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
EPISODES = 300

class DiscreteLineEnv:
    def __init__(self):
        self.state = 0.0
        self.goal = 1.0
    
    def reset(self):
        self.state = 0.0
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        if action == 1: # Move forward
            self.state += np.random.uniform(0.05, 0.15)
        else: # Stay still
            self.state += 0.0
            
        self.state = min(self.goal, self.state)
        reward = -1.0 if self.state < self.goal else 0.0
        done = (self.state >= self.goal)
        return np.array([self.state], dtype=np.float32), reward, done

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Outputs Q-values for 2 actions: [Stay, Move]
        )
    def forward(self, x):
        return self.net(x)

# --- Components Initialization ---
env = DiscreteLineEnv()
online_net = QNetwork()
target_net = QNetwork()
target_net.load_state_dict(online_net.state_dict()) # Synchronize initially

optimizer = optim.Adam(online_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()
replay_buffer = deque(maxlen=MEMORY_SIZE)

epsilon = EPSILON_START
total_steps = 0

print("--- Training Deep Q-Network (DQN) ---")

for episode in range(EPISODES):
    state = env.reset()
    done = False
    
    while not done:
        total_steps += 1
        
        # 1. Epsilon-Greedy Action Selection
        if random.random() < epsilon:
            action = random.choice([0, 1])
        else:
            state_t = torch.from_numpy(state).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(online_net(state_t)).item()
                
        # 2. Environment Step
        next_state, reward, done = env.step(action)
        
        # 3. Store Transition in Replay Buffer
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        
        # 4. Optimize Online Network if Buffer has enough data
        if len(replay_buffer) > BATCH_SIZE:
            # Sample random mini-batch (Breaks auto-correlation!)
            batch = random.sample(replay_buffer, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors
            states_t = torch.from_numpy(np.array(states))
            actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
            rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states_t = torch.from_numpy(np.array(next_states))
            dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
            
            # Gather Q(s, a) for chosen actions from the online network
            current_q = online_net(states_t).gather(1, actions_t)
            
            # Calculate stable TD Targets using the Target Network
            with torch.no_grad():
                max_next_q = target_net(next_states_t).max(1)[0].unsqueeze(1)
                target_q = rewards_t + (GAMMA * max_next_q * (1 - dones_t))
                
            # Compute Loss and Optimize
            loss = loss_fn(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            online_net.utils = torch.nn.utils.clip_grad_norm_(online_net.parameters(), 1.0)
            optimizer.step()
            
        # 5. Periodically Update Target Network Weights
        if total_steps % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())
            
    # Decay exploration rate
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

print("Training Complete.\n")

# --- Verification Evaluation ---
online_net.eval()
print("--- Learned Action-Value (Q) Profiles ---")
with torch.no_grad():
    for test_pos in [0.0, 0.5, 0.85]:
        st = torch.tensor([[test_pos]], dtype=torch.float32)
        q_vals = online_net(st).numpy()[0]
        best_act = "MOVE" if np.argmax(q_vals) == 1 else "STAY"
        print(f"Position: {test_pos:<4} -> Q-Values: [Stay: {q_vals[0]:.2f}, Move: {q_vals[1]:.2f}] -> Decision: {best_act}")