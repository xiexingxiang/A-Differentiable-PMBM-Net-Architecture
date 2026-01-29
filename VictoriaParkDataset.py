import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VictoriaParkDataset(Dataset):
    """
    维多利亚公园数据集加载器
    处理激光雷达、GPS和DRS数据，生成用于多传感器融合任务的序列数据
    """
    collate_fn = None

    def __init__(self, data_root, sequence_length=10, split='train', verbose=True):
        """
        初始化维多利亚公园数据集

        参数:
            data_root (str): 数据根目录
            sequence_length (int): 序列长度，默认为10帧
            split (str): 数据集划分，'train'、'val'或'test'
            verbose (bool): 是否显示详细信息
        """
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.split = split
        self.verbose = verbose

        # 检查数据文件是否存在
        laser_file = os.path.join(data_root, 'LASER.txt')
        gps_file = os.path.join(data_root, 'GPS.txt')
        drs_file = os.path.join(data_root, 'DRS.txt')

        for file_path in [laser_file, gps_file, drs_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"数据文件不存在: {file_path}")

        # 加载数据
        self.laser_data = self._load_laser_data(laser_file)
        self.gps_data = self._load_gps_data(gps_file)
        self.drs_data = self._load_drs_data(drs_file)

        # 确保三个传感器数据长度一致
        # 修改这里：原来是对字典进行切片，现在分别对字典中的各个列表进行切片
        min_length = min(len(self.laser_data['pointclouds']),
                         len(self.gps_data['positions']),
                         len(self.drs_data['velocities']))

        # 修改这里：对激光数据中的每个列表进行切片
        self.laser_data['timestamps'] = self.laser_data['timestamps'][:min_length]
        self.laser_data['pointclouds'] = self.laser_data['pointclouds'][:min_length]

        # 修改这里：对GPS数据中的每个列表进行切片
        self.gps_data['timestamps'] = self.gps_data['timestamps'][:min_length]
        self.gps_data['positions'] = self.gps_data['positions'][:min_length]

        # 修改这里：对DRS数据中的每个列表进行切片
        self.drs_data['timestamps'] = self.drs_data['timestamps'][:min_length]
        self.drs_data['velocities'] = self.drs_data['velocities'][:min_length]
        self.drs_data['steering_angles'] = self.drs_data['steering_angles'][:min_length]

        if verbose:
            print(f"数据集加载完成:")
            print(f"  - 激光数据: {len(self.laser_data['pointclouds'])} 帧")
            print(f"  - GPS数据: {len(self.gps_data['positions'])} 帧")
            print(f"  - DRS数据: {len(self.drs_data['velocities'])} 帧")
            print(f"  - 序列长度: {sequence_length}")
            print(f"  - 可生成序列数: {max(0, len(self.laser_data['pointclouds']) - sequence_length + 1)}")

        # 生成序列索引
        self.sequence_indices = self._generate_sequence_indices()

        # 统计数据标准化参数
        self._compute_normalization_params()

    def _load_laser_data(self, file_path):
        """加载激光雷达数据"""
        with open(file_path, 'r') as f:
            # 读取单行数据，按制表符分割
            data_str = f.read().strip()
            values = np.array(data_str.split('\t'), dtype=float)

        # 每帧激光数据有362个值：1个时间戳 + 361个距离测量
        n_frames = len(values) // 362
        values = values[:n_frames * 362]  # 丢弃不完整帧
        laser_data = values.reshape(n_frames, 362)

        # 提取时间戳和距离测量值
        timestamps = laser_data[:, 0]
        ranges = laser_data[:, 1:]

        # 转换激光数据为点云坐标
        pointclouds = []
        for i in range(n_frames):
            pointcloud = self._convert_laser_to_pointcloud(ranges[i])
            pointclouds.append(pointcloud)

        return {
            'timestamps': timestamps,
            'pointclouds': pointclouds
        }

    def _convert_laser_to_pointcloud(self, ranges):
        """将激光测距数据转换为3D点云坐标"""
        # 激光扫描角度范围：-90度到+90度（-π/2到π/2）
        angles = np.linspace(-np.pi / 2, np.pi / 2, 361)

        # 过滤无效测量值（0或过大的值）
        valid_mask = (ranges > 0) & (ranges < 100)  # 假设有效距离在0-100米之间

        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]

        # 转换为笛卡尔坐标
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        z = np.zeros_like(x)  # 假设地面平面

        # 组合成点云 (N, 3)
        pointcloud = np.column_stack([x, y, z])
        return pointcloud

    def _load_gps_data(self, file_path):
        """加载GPS数据"""
        gps_data = np.loadtxt(file_path)

        # GPS数据格式: [时间, 纬度偏移, 经度偏移]
        timestamps = gps_data[:, 0]
        positions = gps_data[:, 1:]

        return {
            'timestamps': timestamps,
            'positions': positions
        }

    def _load_drs_data(self, file_path):
        """加载DRS（Dead Reckoning System）数据"""
        drs_data = np.loadtxt(file_path)

        # DRS数据格式: [时间, 速度, 转向角]
        timestamps = drs_data[:, 0]
        velocities = drs_data[:, 1]
        steering_angles = drs_data[:, 2]

        return {
            'timestamps': timestamps,
            'velocities': velocities,
            'steering_angles': steering_angles
        }

    def _generate_sequence_indices(self):
        """生成序列索引列表"""
        # 修改这里：使用点云列表的长度而不是字典
        n_frames = len(self.laser_data['pointclouds'])
        if n_frames < self.sequence_length:
            return []

        return list(range(n_frames - self.sequence_length + 1))

    def _compute_normalization_params(self):
        """计算数据标准化参数"""
        # 收集所有点云的点用于统计
        all_points = []
        for pointcloud in self.laser_data['pointclouds']:
            if len(pointcloud) > 0:
                all_points.append(pointcloud)

        if all_points:
            all_points = np.vstack(all_points)
            self.pointcloud_mean = np.mean(all_points, axis=0)
            self.pointcloud_std = np.std(all_points, axis=0)
        else:
            self.pointcloud_mean = np.zeros(3)
            self.pointcloud_std = np.ones(3)

        # GPS数据标准化参数
        gps_positions = self.gps_data['positions']
        self.gps_mean = np.mean(gps_positions, axis=0)
        self.gps_std = np.std(gps_positions, axis=0)

        # DRS数据标准化参数
        drs_velocities = self.drs_data['velocities']
        drs_steering = self.drs_data['steering_angles']

        self.drs_velocity_mean = np.mean(drs_velocities)
        self.drs_velocity_std = np.std(drs_velocities)
        self.drs_steering_mean = np.mean(drs_steering)
        self.drs_steering_std = np.std(drs_steering)

    def normalize_pointcloud(self, pointcloud):
        """标准化点云数据"""
        if len(pointcloud) == 0:
            return pointcloud
        return (pointcloud - self.pointcloud_mean) / (self.pointcloud_std + 1e-8)

    def __len__(self):
        """返回数据集中序列的数量"""
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        """
        获取一个序列样本

        返回包含以下内容的字典：
            - laser: 激光点云序列 (sequence_length, N_i, 3)
            - gps: GPS位置序列 (sequence_length, 2)
            - drs: DRS数据序列 (sequence_length, 2)
            - target: 监督信号 (下一帧的位姿变化)
            - metadata: 元数据
        """
        start_idx = self.sequence_indices[idx]
        end_idx = start_idx + self.sequence_length

        # 提取激光点云序列
        laser_sequence = []
        for i in range(start_idx, end_idx):
            pointcloud = self.laser_data['pointclouds'][i]
            # 标准化点云
            normalized_pointcloud = self.normalize_pointcloud(pointcloud)
            laser_sequence.append(normalized_pointcloud)

        # 提取GPS序列 (跳过时间戳，只取位置)
        gps_sequence = self.gps_data['positions'][start_idx:end_idx]
        # 标准化GPS数据
        gps_sequence = (gps_sequence - self.gps_mean) / (self.gps_std + 1e-8)

        # 提取DRS序列 (速度和转向角)
        drs_sequence = np.column_stack([
            self.drs_data['velocities'][start_idx:end_idx],
            self.drs_data['steering_angles'][start_idx:end_idx]
        ])
        # 标准化DRS数据
        drs_sequence[:, 0] = (drs_sequence[:, 0] - self.drs_velocity_mean) / (self.drs_velocity_std + 1e-8)
        drs_sequence[:, 1] = (drs_sequence[:, 1] - self.drs_steering_mean) / (self.drs_steering_std + 1e-8)

        # 生成监督信号：下一帧相对于当前帧的位姿变化
        # 使用GPS位置变化作为监督信号
        if end_idx < len(self.gps_data['positions']):
            # 当前帧的平均位置
            current_pos = np.mean(gps_sequence[-3:], axis=0) if len(gps_sequence) >= 3 else gps_sequence[-1]
            # 下一帧的GPS位置
            next_pos = self.gps_data['positions'][end_idx]
            # 位姿变化 (Δx, Δy)
            pose_change = next_pos - current_pos
        else:
            # 如果是最后一帧，使用零向量
            pose_change = np.zeros(2)

        # 转换为PyTorch张量
        # 注意：点云序列中每个点云的大小可能不同，所以保持为列表
        laser_tensors = [torch.from_numpy(pc).float() for pc in laser_sequence]
        gps_tensor = torch.from_numpy(gps_sequence).float()
        drs_tensor = torch.from_numpy(drs_sequence).float()
        target_tensor = torch.from_numpy(pose_change).float()

        return {
            "laser": laser_tensors,  # 激光点云序列列表
            "gps": gps_tensor,  # GPS位置序列 (sequence_length, 2)
            "drs": drs_tensor,  # DRS数据序列 (sequence_length, 2)
            "target": target_tensor,  # 监督信号：下一帧的位姿变化 (2,)
            "metadata": {
                "sequence_id": idx,
                "start_idx": start_idx,
                "end_idx": end_idx - 1,
                "timestamps": self.laser_data['timestamps'][start_idx:end_idx]
            }
        }

    def get_stats(self):
        """获取数据统计信息"""
        return {
            "pointcloud_mean": self.pointcloud_mean.tolist(),
            "pointcloud_std": self.pointcloud_std.tolist(),
            "gps_mean": self.gps_mean.tolist(),
            "gps_std": self.gps_std.tolist(),
            "drs_velocity_mean": float(self.drs_velocity_mean),
            "drs_velocity_std": float(self.drs_velocity_std),
            "drs_steering_mean": float(self.drs_steering_mean),
            "drs_steering_std": float(self.drs_steering_std),
            "total_sequences": len(self),
            "sequence_length": self.sequence_length
        }


def custom_collate_fn(batch):
    """
    自定义数据加载的collate函数
    处理变长点云序列
    """
    batch_laser = [item["laser"] for item in batch]  # 列表的列表
    batch_gps = torch.stack([item["gps"] for item in batch])
    batch_drs = torch.stack([item["drs"] for item in batch])
    batch_target = torch.stack([item["target"] for item in batch])
    batch_metadata = [item["metadata"] for item in batch]

    return {
        "laser": batch_laser,
        "gps": batch_gps,
        "drs": batch_drs,
        "target": batch_target,
        "metadata": batch_metadata
    }


# 设置自定义collate函数
VictoriaParkDataset.collate_fn = custom_collate_fn

if __name__ == "__main__":
    # 测试数据集
    import sys

    # 假设数据文件在当前目录
    data_root = "."

    try:
        # 创建数据集实例
        dataset = VictoriaParkDataset(
            data_root=data_root,
            sequence_length=5,  # 使用较短的序列进行测试
            split='train',
            verbose=True
        )

        print(f"\n数据集大小: {len(dataset)} 个序列")

        if len(dataset) > 0:
            # 获取第一个样本
            sample = dataset[0]

            print(f"\n样本包含的键: {list(sample.keys())}")
            print(f"激光点云序列长度: {len(sample['laser'])}")
            print(f"  第一帧点云形状: {sample['laser'][0].shape}")
            print(f"GPS序列形状: {sample['gps'].shape}")
            print(f"DRS序列形状: {sample['drs'].shape}")
            print(f"目标形状: {sample['target'].shape}")
            print(f"元数据: {sample['metadata']}")

            # 测试collate函数
            from torch.utils.data import DataLoader

            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=VictoriaParkDataset.collate_fn
            )

            batch = next(iter(dataloader))
            print(f"\n批次数据:")
            print(f"  激光点云: {len(batch['laser'])} 个序列")
            print(f"  GPS数据形状: {batch['gps'].shape}")
            print(f"  DRS数据形状: {batch['drs'].shape}")
            print(f"  目标数据形状: {batch['target'].shape}")

            # 显示统计信息
            stats = dataset.get_stats()
            print(f"\n数据统计信息:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保LASER.txt、GPS.txt和DRS.txt文件在当前目录中")
        sys.exit(1)
