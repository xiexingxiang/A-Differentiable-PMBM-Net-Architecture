import torch
import numpy as np
import os
import json
from torch.utils.data import Dataset

def load_point_cloud_from_bin(path):
    """
    从.bin文件中加载点云的示例函数。
    您需要根据您的点云文件格式进行修改。
    """
    # 假设是 KITTI 格式的 .bin 文件 (x, y, z, intensity)
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points # 返回 [N, 4] 的张量

class CustomDataset(Dataset):
    def __init__(self, dataroot, seq_len='full', split='test'): # split 参数用于保持接口一致性
        super().__init__()
        self.dataroot = dataroot
        self.seq_len = seq_len
        # 获取所有场景文件夹的路径
        self.scenes = sorted([os.path.join(dataroot, s) for s in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, s))])
        print(f"Found {len(self.scenes)} scenes in custom dataset at {dataroot}")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_path = self.scenes[idx]
        lidar_dir = os.path.join(scene_path, 'lidar')
        gt_path = os.path.join(scene_path, 'gt.json')

        # 加载点云文件路径并排序，确保时序正确
        lidar_files = sorted(os.listdir(lidar_dir))
        
        # 加载真值标注文件
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)

        point_clouds_seq = []
        gt_seq = []

        num_frames_to_load = len(lidar_files)
        if isinstance(self.seq_len, int) and self.seq_len < num_frames_to_load:
            num_frames_to_load = self.seq_len

        for i in range(num_frames_to_load):
            frame_filename = lidar_files[i]
            frame_basename = os.path.splitext(frame_filename)[0] # e.g., "00000"

            # 1. 加载点云
            pc_path = os.path.join(lidar_dir, frame_filename)
            pc = load_point_cloud_from_bin(pc_path)
            point_clouds_seq.append(torch.from_numpy(pc).float())

            # 2. 加载对应的真值
            if frame_basename in gt_data:
                # 假设gt是[N, 9]的列表 [x,y,z,w,l,h,yaw,id,class]
                # 我们只需要前8列用于跟踪评估
                gt_boxes = np.array(gt_data[frame_basename])[:, :8]
                gt_seq.append(torch.tensor(gt_boxes, dtype=torch.float))
            else:
                # 如果该帧没有物体，则添加一个空的真值张量
                gt_seq.append(torch.empty((0, 8)))

        return point_clouds_seq, gt_seq

    @staticmethod
    def collate_fn(batch):
        # 这个函数保持不变，用于组织一个批次的数据
        point_clouds_list = [item[0] for item in batch]
        ground_truths_list = [item[1] for item in batch]
        return point_clouds_list, ground_truths_list