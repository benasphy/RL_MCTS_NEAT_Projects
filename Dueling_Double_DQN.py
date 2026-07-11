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