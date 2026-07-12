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