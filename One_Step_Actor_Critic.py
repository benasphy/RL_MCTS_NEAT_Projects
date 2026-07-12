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
        self.state = min(self.goal, self.state)
        reward = -1.0 if self.state < self.goal else 0.0
        return np.array([self.state], dtype=np.float32), reward, (self.state >= self.goal)

# =====================================================================
# UNIFIED ACTOR-CRITIC NETWORK
# =====================================================================
class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super(ActorCriticNetwork, self).__init__()
        # Shared feature representation layer
        self.shared_trunk = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU()
        )
        # Head 1: The Actor (Outputs action probabilities)
        self.actor_head = nn.Sequential(
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
        # Head 2: The Critic (Outputs a single state value estimation V(s))
        self.critic_head = nn.Linear(64, 1)
        
    def forward(self, x):
        features = self.shared_trunk(x)
        action_probs = self.actor_head(features)
        state_value = self.critic_head(features)
        return action_probs, state_value

# --- Initialization ---
env = DiscreteLineEnv()
ac_model = ActorCriticNetwork()
# A single optimizer adjusts both actor and critic parameters simultaneously
optimizer = optim.Adam(ac_model.parameters(), lr=0.005)

GAMMA = 0.99
EPISODES = 300

print("--- Training Online One-Step Actor-Critic Agent ---")

for episode in range(EPISODES):
    state = env.reset()
    done = False
    
    while not done:
        state_tensor = torch.from_numpy(state).float()
        
        # 1. Forward Pass: Get policy distribution and state value estimation simultaneously
        action_probs, v_state = ac_model(state_tensor)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # 2. Interact with the environment
        next_state, reward, done = env.step(action.item())
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        
        # 3. Calculate TD Target and TD Error (Advantage)
        if done:
            td_target = reward_tensor
        else:
            next_state_tensor = torch.from_numpy(next_state).float()
            # Detach the next state value prediction to isolate gradients
            with torch.no_grad():
                _, v_next = ac_model(next_state_tensor)
            td_target = reward_tensor + GAMMA * v_next
            
        td_error = td_target - v_state # This is our calculated Advantage
        
        # 4. Define Composite Loss
        # Critic Loss: Standard Mean Squared Error targeting the TD value
        critic_loss = td_error.pow(2)
        
        # Actor Loss: Negative log-probability scaled by the Critic's TD error
        log_prob = dist.log_prob(action)
        actor_loss = -log_prob * td_error.detach()
        
        # Total combined loss
        total_loss = actor_loss + 0.5 * critic_loss
        
        # 5. Optimize Entire Architecture
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        state = next_state

print("Training Complete.\n")

# --- Inference Check ---
ac_model.eval()
print("--- Actor-Critic Learned Profiles ---")
with torch.no_grad():
    for test_pos in [0.2, 0.8]:
        st = torch.tensor([test_pos], dtype=torch.float32)
        probs, val = ac_model(st)
        print(f"Position: {test_pos:<3} | Critic V(s): {val.item():.2f} | Actor Probs: [Stay: {probs.numpy()[0]:.2f}, Move: {probs.numpy()[1]:.2f}]")