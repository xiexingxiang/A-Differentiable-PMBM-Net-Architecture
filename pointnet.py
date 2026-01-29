# models/pointnet.py
import torch
import torch.nn as nn
from torch_geometric.nn import fps, radius, global_max_pool
from torch.nn import Sequential, Linear, ReLU

# Set Abstraction Layer (PointNet++ 的核心模块)
class SALayer(nn.Module):
    def __init__(self, in_channels, mlp_channels, radius_val, n_sample):
        super(SALayer, self).__init__()
        self.radius = radius_val
        self.n_sample = n_sample
        
        # --- START OF THE FIX ---
        # Correctly build the MLP layer by layer
        mlp_modules = []
    
        # The first layer's input size is special: it includes point features + relative position (3)
        # If there are no input features (first layer), in_channels will be 0.
        last_channel = in_channels + 3
    
        for out_channel in mlp_channels:
            mlp_modules.append(Linear(last_channel, out_channel))
            mlp_modules.append(ReLU())
            mlp_modules.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel # Update for the next layer's input
        
        self.mlp = Sequential(*mlp_modules)
        # --- END OF THE FIX ---
        
        self.max_pool = global_max_pool

    def forward(self, x, pos, batch):
        # Farthest Point Sampling
        idx = fps(pos, batch, ratio=0.5)
        row, col = radius(pos, pos[idx], self.radius, batch, batch[idx], max_num_neighbors=self.n_sample)
    
        # Grouping
        pos_grouped = pos[col] - pos[idx][row] # Normalize positions
    
        # --- START OF THE FIX ---
        # Feature Extraction logic is now conditional
        if x is not None:
            # For subsequent layers, group the input features and concatenate
            x_grouped = x[col]
            features = torch.cat([x_grouped, pos_grouped], dim=1)
        else:
            # For the first layer (where x is None), use only relative positions as features
            features = pos_grouped
        
        features = self.mlp(features)
        
        # Max Pooling
        pooled_features = self.max_pool(features, row)
        
        return pooled_features, pos[idx], batch[idx]

# PointNet++ Backbone
class PointNet2(nn.Module):
    def __init__(self):
        super(PointNet2, self).__init__()
        # 根据 nuScenes 的点云密度和论文设置，定义多层 Set Abstraction
        self.sa1 = SALayer(0, [64, 64, 128], radius_val=0.2, n_sample=32)
        self.sa2 = SALayer(128, [128, 128, 256], radius_val=0.4, n_sample=64)
        self.sa3 = SALayer(256, [256, 512, 1024], radius_val=0.8, n_sample=128)

    def forward(self, data):
        # data 应包含 pos (点坐标) 和 batch (点所属的样本)
        pos, batch = data.pos, data.batch
        
        # Layer 1
        x1, pos1, batch1 = self.sa1(None, pos, batch)
        # Layer 2
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)
        # Layer 3
        x3, pos3, batch3 = self.sa3(x2, pos2, batch2)
        
        # 返回最后一层的特征和下采样后的点位置
        return x3, pos3, batch3