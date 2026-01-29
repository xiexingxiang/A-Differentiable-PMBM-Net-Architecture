import torch
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from nuscenes.utils import splits


class VictoriaParkDataset(Dataset):  # It's good practice to inherit from Dataset
    # 在参数列表中添加 seq_len
    def __init__(self, version='victoria_park_datasets', dataroot='/home/huangjinrong/d_pmbm_net_test/data', verbose=True,
                 seq_len=10, split='train'):
        super().__init__()  # Good practice to call parent __init__ if you inherit from torch.utils.data.Dataset
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
        self.seq_len = seq_len
        self.split = split

        if version == 'victoria_park_datasets':
            # For the mini dataset, the splits are pre-defined
            if split == 'train':
                scene_names = splits.mini_train
            elif split == 'val':
                scene_names = splits.mini_val
            else:
                raise ValueError(f"Unknown split '{split}' for v1.0-mini")
        else:
            # Logic for the full dataset...
            scene_names = splits.train if split == 'train' else splits.val

        # This line now correctly uses 'self' because it's inside the method
        self.scenes = [s for s in self.nusc.scene if s['name'] in scene_names]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        # ... your existing __getitem__ logic ...
        pass

    def get_samples_for_scene(self, scene_name):
        scene = self.nusc.get('scene', self.nusc.field2token('scene', 'name', scene_name)[0])
        samples = []
        sample_token = scene['first_sample_token']
        while sample_token:
            samples.append(self.nusc.get('sample', sample_token))
            sample_token = self.nusc.get('sample', sample_token)['next']
        return samples

    def get_lidar_data(self, sample_token, nsweeps=1):
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_filepath = self.nusc.get_sample_data_path(lidar_token)

        pc = LidarPointCloud.from_file(lidar_filepath)

        # 可选：聚合多次扫描
        if nsweeps > 1:
            for _ in range(nsweeps - 1):
                lidar_token = self.nusc.get('sample_data', lidar_token)['prev']
                if not lidar_token:
                    break
                prev_pc = LidarPointCloud.from_file(self.nusc.get_sample_data_path(lidar_token))

                # 将点云转换到当前激光雷达坐标系
                prev_lidar_data = self.nusc.get('sample_data', lidar_token)
                current_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
                prev_pose = self.nusc.get('ego_pose', prev_lidar_data['ego_pose_token'])

                # 从传感器坐标到自车坐标
                current_cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
                prev_cs_record = self.nusc.get('calibrated_sensor', prev_lidar_data['calibrated_sensor_token'])

                prev_pc.rotate(Quaternion(prev_cs_record['rotation']).rotation_matrix)
                prev_pc.translate(np.array(prev_cs_record['translation']))

                # 从自车坐标到全局坐标
                prev_pc.rotate(Quaternion(prev_pose['rotation']).rotation_matrix)
                prev_pc.translate(np.array(prev_pose['translation']))

                # 从全局坐标到当前自车坐标
                prev_pc.translate(-np.array(current_pose['translation']))
                prev_pc.rotate(Quaternion(current_pose['rotation']).rotation_matrix.T)

                # 从当前自车坐标到当前传感器坐标
                prev_pc.translate(-np.array(current_cs_record['translation']))
                prev_pc.rotate(Quaternion(current_cs_record['rotation']).rotation_matrix.T)

                pc.points = np.hstack((pc.points, prev_pc.points))

        return pc.points.T

    def get_annotations(self, sample_token, coord_system='lidar'):
        sample = self.nusc.get('sample', sample_token)
        annotations = []
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']), name=ann['category_name'])

            if coord_system == 'lidar':
                # 将标注从全局坐标系转换到激光雷达坐标系
                lidar_token = sample['data']['LIDAR_TOP']
                lidar_data = self.nusc.get('sample_data', lidar_token)
                ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
                cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

                box.translate(-np.array(ego_pose['translation']))
                box.rotate(Quaternion(ego_pose['rotation']).inverse)

                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

            annotations.append(box)
        return annotations

    @staticmethod
    def collate_fn(batch):
        # 'batch' is a list of whatever your __getitem__ returns.
        # Let's assume __getitem__ returns a tuple (point_cloud, ground_truth)

        # This function will group all point clouds into one list
        # and all ground truths into another.
        point_clouds = [item[0] for item in batch]
        ground_truths = [item[1] for item in batch]

        # The collate_fn should return the organized batch
        return point_clouds, ground_truths

    def __len__(self):
        # This method should return the total number of items in the dataset.
        # Returning the number of scenes is a common approach.
        return len(self.scenes)

    # Add this method inside the NuScenesDataset class in nuscenes_dataset.py
    # Make sure to import torch at the top of the file: import torch

    def __getitem__(self, idx):
        # 1. Get the scene corresponding to the index
        scene = self.scenes[idx]

        # 2. Get all sample records for this scene
        samples = self.get_samples_for_scene(scene['name'])

        # 3. Initialize lists to hold the sequence data
        point_clouds_seq = []
        gt_seq = []

        # 4. Loop through the samples in the scene to create a sequence
        #    Use self.seq_len, which should be stored from __init__
        for sample in samples[:self.seq_len]:
            # Get point cloud and annotations for the current sample
            pc = self.get_lidar_data(sample['token'])
            annotations = self.get_annotations(sample['token'])

            # Convert data to PyTorch tensors
            point_clouds_seq.append(torch.from_numpy(pc).float())

            # You need a way to convert annotation boxes to a tensor for the ground truth
            # This is a placeholder for that logic
            gt_tensor = self.annotations_to_tensor(annotations)
            gt_seq.append(gt_tensor)

        # 5. Return the full sequence for one scene
        return point_clouds_seq, gt_seq

    # You also need to add this helper method to your class to handle annotations
    def annotations_to_tensor(self, annotations):
        # This function converts the list of Box objects into a single tensor.
        # The format depends on what your GOSPA loss function expects.
        # Example: Create a tensor of shape [num_boxes, 7] (x, y, z, w, l, h, yaw)
        if not annotations:
            # ADD INDENTATION HERE (e.g., 4 spaces)
            return torch.empty((0, 7))  # Return empty tensor if no objects

        box_data = []
        for box in annotations:
            # Get yaw from the quaternion
            yaw = box.orientation.yaw_pitch_roll[0]
            # Append [x, y, z, w, l, h, yaw]
            box_data.append(list(box.center) + list(box.wlh) + [yaw])

        return torch.tensor(box_data, dtype=torch.float)
