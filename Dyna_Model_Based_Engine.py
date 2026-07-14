import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# =====================================================================
# THE LEARNED WORLD MODEL (Transition/Dynamics Network)
# =====================================================================
class DynamicsModel(nn.Module):
    def __init__(self):
        super(DynamicsModel, self).__init__()
        # Input: state (1) + action (1)
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Outputs predicted next-state
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

# =====================================================================
# DYNA MODEL-BASED PLANNING ENGINE
# =====================================================================
class DynaAgent:
    def __init__(self, state_dim=1, action_dim=2):
        self.q_table = np.zeros((11, action_dim)) # Tabular states 0 to 10
        self.dynamics_model = DynamicsModel()
        self.model_optimizer = optim.Adam(self.dynamics_model.parameters(), lr=0.01)
        self.real_memory = [] # Store real world experiences
    
    def get_action(self, state_idx, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, 1)
        return np.argmax(self.q_table[state_idx])
    
    def discretize(self, state):
        # Scale continuous state [0.0, 1.0] to table index [0, 10]
        return int(np.clip(state * 10, 0, 10))
    
    def train_dynamics(self, batch_size=8):
        if len(self.real_memory) < batch_size:
            return
        
        # Sample mini-batch of real experiences
        batch = random.sample(self.real_memory, batch_size)
        states = torch.tensor([[b[0]] for b in batch], dtype=torch.float32)
        actions = torch.tensor([[b[1]] for b in batch], dtype=torch.float32)
        next_states = torch.tensor([[b[2]] for b in batch], dtype=torch.float32)
        
        # Train model to predict the next state
        pred_next_states = self.dynamics_model(states, actions)
        loss = nn.MSELoss()(pred_next_states, next_states)
        
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        return loss.item()
    
    def dyna_planning(self, num_planning_steps=10, gamma=0.9):
        """ The Imagination Step: Update Q-values using the learned model """
        if len(self.real_memory) < 5:
            return
            
        self.dynamics_model.eval()
        with torch.no_grad():
            for _ in range(num_planning_steps):
                # 1. Sample a state that we have actually visited in the past
                real_transition = random.choice(self.real_memory)
                sim_state_val = real_transition[0]
                sim_state_idx = self.discretize(sim_state_val)
                
                # 2. Select a random hypothetical action
                sim_action = random.randint(0, 1)
                
                # 3. Query the World Model for the next state
                state_tensor = torch.tensor([[sim_state_val]], dtype=torch.float32)
                action_tensor = torch.tensor([[float(sim_action)]], dtype=torch.float32)
                sim_next_state_val = self.dynamics_model(state_tensor, action_tensor).item()
                sim_next_state_val = np.clip(sim_next_state_val, 0.0, 1.0)
                sim_next_idx = self.discretize(sim_next_state_val)
                
                # 4. Compute reward analytically (assuming known reward function for simplicity)
                sim_reward = -1.0 if sim_next_state_val < 1.0 else 0.0
                
                # 5. Update the Q-table using simulated transition (Planning)
                best_next_q = np.max(self.q_table[sim_next_idx])
                td_target = sim_reward + gamma * best_next_q
                self.q_table[sim_state_idx, sim_action] += 0.1 * (td_target - self.q_table[sim_state_idx, sim_action])

# =====================================================================
# SIMULATED ENVIRONMENT & TRAINING LOOP
# =====================================================================
class SimpleEnv:
    def __init__(self): self.state = 0.0
    def step(self, action):
        if action == 1: # Move action
            self.state += 0.2
        self.state = min(1.0, self.state)
        reward = -1.0 if self.state < 1.0 else 0.0
        return self.state, reward, (self.state >= 1.0)

env = SimpleEnv()
agent = DynaAgent()

print("--- Training Dyna Model-Based Agent ---")
for episode in range(25):
    state = 0.0
    done = False
    
    while not done:
        state_idx = agent.discretize(state)
        action = agent.get_action(state_idx)
        
        # Step in the real environment
        next_state, reward, done = env.step(action)
        next_state_idx = agent.discretize(next_state)
        
        # Direct Q-Learning update (Model-Free update)
        best_next_q = np.max(agent.q_table[next_state_idx])
        td_target = reward + 0.9 * best_next_q
        agent.q_table[state_idx, action] += 0.1 * (td_target - agent.q_table[state_idx, action])
        
        # Save to memory and train the world model
        agent.real_memory.append((state, action, next_state))
        model_loss = agent.train_dynamics()
        
        # Dyna Planning: Run 15 background imaginations using the model
        agent.dyna_planning(num_planning_steps=15)
        
        state = next_state
    env.state = 0.0 # reset

print("Training Complete.\n")

# Evaluate dynamics prediction accuracy
test_state = torch.tensor([[0.4]], dtype=torch.float32)
test_action = torch.tensor([[1.0]], dtype=torch.float32)
predicted_next = agent.dynamics_model(test_state, test_action).item()
print("--- World Model Diagnostic Pass ---")
print(f"Input State: 0.4 | Action: Move (1.0)")
print(f"True Next State: 0.6000")
print(f"Predicted Next State: {predicted_next:.4f}")