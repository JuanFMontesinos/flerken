import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FiLM']


class FiLM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bias = nn.Linear(in_channels, out_channels)
        self.scale = nn.Linear(in_channels, out_channels)

    def forward(self, x, c):
        scale, bias = self.feat(c)
        return scale, bias
