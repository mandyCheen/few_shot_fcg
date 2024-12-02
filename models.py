import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import SAGEConv, GCNConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import global_mean_pool, global_add_pool
import os
import numpy as np                                             
import pickle
import torch.nn as nn

class GCN(torch.nn.Module):
    def __init__(self, dim_in: int, dim_h: int, dim_out: int, num_layers: int, projection: bool = False):
        super().__init__()
        self.num_layers = num_layers
        
        # GNN layers
        self.gcn_convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # First layer
        self.gcn_convs.append(GCNConv(dim_in, dim_h))
        self.norms.append(BatchNorm1d(dim_h))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.gcn_convs.append(GCNConv(dim_h, dim_h))
            self.norms.append(BatchNorm1d(dim_h))

        if projection:
            self.output_proj = nn.Sequential(
                Linear(dim_h, dim_out),
                BatchNorm1d(dim_out),
                nn.ReLU()
            )
        else:
            self.output_proj = None

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        device = x.device
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        
        h = x
        for i in range(self.num_layers):
            h = self.gcn_convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            
        h = global_add_pool(h, batch)

        if self.output_proj is not None:
            h = self.output_proj(h)

        return h

class GraphSAGE(torch.nn.Module):
    def __init__(self, dim_in: int, dim_h: int, dim_out: int, num_layers: int, projection: bool = False):
        super().__init__()
        self.num_layers = num_layers
        
        # GNN layers
        self.sage_convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # First layer
        self.sage_convs.append(SAGEConv(dim_in, dim_h))
        self.norms.append(BatchNorm1d(dim_h))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.sage_convs.append(SAGEConv(dim_h, dim_h))
            self.norms.append(BatchNorm1d(dim_h))

        if projection:
            self.output_proj = nn.Sequential(
                Linear(dim_h, dim_out),
                BatchNorm1d(dim_out),
                nn.ReLU()
            )
        else:
            self.output_proj = None

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        device = x.device
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        
        h = x
        for i in range(self.num_layers):
            h = self.sage_convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            
        h = global_add_pool(h, batch)

        if self.output_proj is not None:
            h = self.output_proj(h)

        return h

class GraphClassifier(torch.nn.Module):
    def __init__(self, backbone_dim_in: int, backbone_dim_h: int, backbone_dim_out: int, 
                 backbone_num_layers: int, num_classes: int, backbone_type: str = 'sage',
                 projection: bool = False, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        
        # Choose backbone based on type
        if backbone_type.lower() == 'sage':
            self.backbone = GraphSAGE(
                dim_in=backbone_dim_in,
                dim_h=backbone_dim_h,
                dim_out=backbone_dim_out,
                num_layers=backbone_num_layers,
                projection=projection
            )
        elif backbone_type.lower() == 'gcn':
            self.backbone = GCN(
                dim_in=backbone_dim_in,
                dim_h=backbone_dim_h,
                dim_out=backbone_dim_out,
                num_layers=backbone_num_layers,
                projection=projection
            )
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

        self.classifier = nn.Sequential(
            Linear(backbone_dim_h, hidden_dim),
            BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        h = self.backbone(data)
        out = self.classifier(h)
        return out
    
    def get_embeddings(self, data):
        h = self.backbone(data)
        return h