import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniRainbowNetwork(nn.Module):
    def __init__(self, num_actions=2, num_atoms=51):
        super(MiniRainbowNetwork, self).__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms