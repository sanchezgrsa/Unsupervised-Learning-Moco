import torch.nn as nn
import torch
import torch.nn.functional as F
torch.cuda.empty_cache()

import torch
import torch.nn.functional as F
from torch import nn

# define the network class
class LinNet(nn.Module):
    def __init__(self):
        # call constructor from superclass
        super().__init__()
        
        # define network layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        


    def forward(self, x, target):
        # define forward pass

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.log_softmax(x, dim=1)

        return x




