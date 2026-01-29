# models/gnn.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class DynamicsLearner(nn.Module):
    def __init__(self, state_dim, n_heads=8, n_layers=3, hidden_dim=256):
        super(DynamicsLearner, self).__init__()
        self.state_dim = state_dim
        
        self.gat_layers = nn.ModuleList()
        # Input layer
        self.gat_layers.append(GATConv(state_dim * 2, hidden_dim, heads=n_heads, concat=True))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim * n_heads, hidden_dim, heads=n_heads, concat=True))
            
        # Output layer
        self.gat_layers.append(GATConv(hidden_dim * n_heads, state_dim * 2, heads=1, concat=False))
        
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, track_states, edge_index):
        # track_states: [N_tracks, state_dim * 2] (mean + flattened covariance)
        x = track_states
        
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = self.relu(x)
                
        # 输出为状态的残差 (delta_mean, delta_cov)
        delta_state = x
        
        return track_states + delta_state