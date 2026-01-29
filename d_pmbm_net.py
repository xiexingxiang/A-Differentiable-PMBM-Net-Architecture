# models/d_pmbm_net.py (最终修正版)
import torch
import torch.nn as nn
from .pointnet import PointNet2
from .gnn import DynamicsLearner
from utils.sinkhorn import sinkhorn
from torch_geometric.data import Data

class D_PMBM_Net(nn.Module):
    def __init__(self, state_dim=6, existence_threshold=0.0001, dynamics_type='gnn', predict_covariance=False):
       
        super(D_PMBM_Net, self).__init__()
        self.state_dim = state_dim
        self.existence_threshold = existence_threshold
        self.predict_covariance = predict_covariance

        # 1. Sensor Learner (传感器学习器)
        self.sensor_learner_backbone = PointNet2() # PointNet++ 作为骨干网络

        # 2. Detection Head (检测头)
        # 根据是否预测协方差来确定输出维度
        output_dim = self.state_dim + 1  # 状态均值 + 存在概率 r
        if self.predict_covariance:
            # L 是协方差矩阵Cholesky分解后的下三角矩阵元素
            # 对于一个6x6的对称矩阵，有 6+5+4+3+2+1=21 个独立元素
            num_cov_elements = self.state_dim * (self.state_dim + 1) // 2
            output_dim += num_cov_elements
            
        self.detection_head = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, output_dim)
        )




        # 3. Dynamics Learner (动力学学习器)
        self.dynamics_type = dynamics_type
        if self.dynamics_type == 'gnn':
            self.dynamics_learner = DynamicsLearner(state_dim)
        # elif self.dynamics_type == 'transformer':
        #     self.dynamics_learner = TransformerDynamicsLearner(state_dim)
        else:
            raise ValueError(f"Unknown dynamics_type: {self.dynamics_type}")
            
        # (关联和更新模块的参数，当前模型中暂未完全整合)
        self.association_mlp = nn.Sequential(
            nn.Linear(256 * 2, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.sinkhorn_iters = 20
        self.sinkhorn_epsilon = 0.1

    def forward(self, point_clouds, prev_tracks):
        #定义模型的前向传播逻辑
        
        # ===== 1. 传感器学习器: 处理点云，提取特征 =====
        batch = torch.zeros(point_clouds.shape[0], dtype=torch.long, device=point_clouds.device)
        data_obj = Data(pos=point_clouds[:, :3], x=point_clouds, batch=batch)
        point_features, _, _ = self.sensor_learner_backbone(data_obj)
        # `point_features` 是从PointNet++骨干网络输出的1024维特征

        # ===== 2. 动力学学习器: 预测已有轨迹的下一状态 =====
        if prev_tracks is not None and prev_tracks.numel() > 0:
            # (这里的简化实现需要完善，例如prev_tracks应该是一个包含状态和ID的结构体)
            # num_tracks = len(prev_tracks)
            # edge_index = ...
            # propagated_tracks = self.dynamics_learner(prev_tracks, edge_index)
            # 为了简化，我们暂时假设动力学部分返回空
            propagated_tracks = torch.tensor([], device=point_clouds.device)
        else:
            propagated_tracks = torch.tensor([], device=point_clouds.device)
            
        # ===== 3. 返回结果 =====
        # `forward` 方法的核心输出是：
        #   - propagated_tracks: 由动力学模型预测的上一时刻目标的位置
        #   - point_features: 从当前点云中提取的、用于新生目标检测的特征
        return propagated_tracks, point_features

    # ======================================================================
    # 新增的、完整的 extract_states 方法
    # ======================================================================
    def extract_states(self, track_features):
       
        #从原始特征中提取最终的目标状态集合。这个方法是可微分的，并且是PMBM理念的体现。
    
        if track_features.numel() == 0:
            # 如果没有输入特征，返回一个空的、形状正确的状态张量
            return torch.empty((0, self.state_dim), device=track_features.device, dtype=track_features.dtype)
        
        # 1. 通过检测头得到状态(均值、协方差)和存在概率
        # raw_output 的形状是 [N, output_dim]
        raw_output = self.detection_head(track_features)
        
        # 2. 解码存在概率
        # 假设存在概率的logit值是最后一个元素
        existence_logits = raw_output[:, self.state_dim]
        existence_probs = torch.sigmoid(existence_logits)
        
        # 3. 根据阈值筛选出存在的目标
        keep_indices = existence_probs > self.existence_threshold
        
        if not torch.any(keep_indices):
            return torch.empty((0, self.state_dim), device=track_features.device, dtype=track_features.dtype)

        # 4. 只返回被筛选出的目标的状态部分
        final_states = raw_output[keep_indices, :self.state_dim]
        
        # (可选) 如果需要返回协方差，在这里解码
        # if self.predict_covariance:
        #    cov_elements = raw_output[keep_indices, self.state_dim+1:]
        #    ... (解码逻辑) ...
        #    return final_states, final_covariances

        return final_states