import torch
import torch.nn as nn
import torch.nn.functional as F


class Invincea(nn.Module):
    def __init__(self):
        super(Invincea, self).__init__()
        self.d1 = nn.Linear(1025, 1025)
        self.d2 = nn.Linear(1025, 1025)
        self.d3 = nn.Linear(1025, 1)

    def forward(self, x):
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        return torch.sigmoid(self.d3(x))