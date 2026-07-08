import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ContinuousLineEnv:
    def __init__(self):
        self.state = 0.0
        self.goal = 1.0
    
    def reset(self):
        self.state = 0.0
        return np.array([self.state], dtype=np.float32)
    def step(self):
        step_size = np.random.uniform(0.05, 0.15)
        self.state = min(self.goal, self.state + step_size)
        reward = -1.0
        done = (self.state >= self.goal)
        return np.array([self.state], dtype=np.float32), reward, done
    
class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        # Takes 1 raw continuous state float, passes through hidden layers, outputs 1 value
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)
env = ContinuousLineEnv()
model = ValueNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

gamma = 1.0
episodes = 600

print("--- Training Custom PyTorch Semi-Gradient TD(0) Loop ---")

for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Convert state array to PyTorch Tensor
        state_tensor = torch.from_numpy(state)
        
        # 1. Forward Pass: Predict current value V(S_t)
        v_pred = model(state_tensor)
        
        # 2. Interact with the environment
        next_state, reward, done = env.step()
        
        # 3. Calculate the TD Target
        if done:
            # Terminal state value is exactly 0
            td_target = torch.tensor([reward], dtype=torch.float32)
        else:
            next_state_tensor = torch.from_numpy(next_state)
            
            # CRITICAL: We use .detach() here to prevent gradients from flowing into the target calculation
            with torch.no_grad():
                v_next = model(next_state_tensor)
            td_target = reward + gamma * v_next
            
        # 4. Compute MSE Loss between current prediction and detached target
        loss = loss_fn(v_pred, td_target)
        
        # 5. PyTorch Backpropagation Optimization Routine
        optimizer.zero_grad() # Clear out old residual gradients
        loss.backward()       # Compute new gradients via chain rule
        
        # Optional but highly recommended in Deep RL: Clip gradients to avoid exploding adjustments
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()      # Update the internal neural network weights
        
        state = next_state

print("\nTraining Complete.")