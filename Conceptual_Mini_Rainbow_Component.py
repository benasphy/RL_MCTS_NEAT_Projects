import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniRainbowNetwork(nn.Module):
    def __init__(self, num_actions=2, num_atoms=51):
        super(MiniRainbowNetwork, self).__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        
        # 1. Base Feature Extraction
        self.feature_backbone = nn.Sequential(
            nn.Linear(4, 64), # Simulating a 4D state input
            nn.ReLU()
        )
        
        # 2. Dueling Value Stream: Predicts distribution atoms for V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_atoms) # Outputs a value across the 51 atoms
        )