import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import SAGEConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import global_mean_pool, global_add_pool
import os
import numpy as np                                             
import pickle

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 2)
        self.bn1 = BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = self.conv2(h, edge_index)

        h = global_add_pool(h, batch)
        
        h = self.lin1(h)
        h = self.bn1(h)
        h = h.relu()
        h = self.lin2(h)
        h = F.dropout(h, p=0.5, training=self.training)

        return h