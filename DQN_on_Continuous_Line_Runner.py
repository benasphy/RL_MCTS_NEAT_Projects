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