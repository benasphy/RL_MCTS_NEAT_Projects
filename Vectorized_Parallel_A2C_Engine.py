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