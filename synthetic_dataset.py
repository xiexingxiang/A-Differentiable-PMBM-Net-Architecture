import numpy as np

class SyntheticDataset:
    def __init__(self, num_targets=5, num_steps=100, x_dim=4, z_dim=2,
                 p_birth=0.03, p_death=0.01, clutter_rate=10, 
                 sensor_std=1.0, process_noise_std=0.1):
        self.num_targets = num_targets
        self.num_steps = num_steps
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.p_birth = p_birth
        self.p_death = p_death
        self.clutter_rate = clutter_rate
        self.sensor_std = sensor_std
        self.process_noise_std = process_noise_std
        
        # 状态转移矩阵 (匀速模型)
        dt = 1.0
        self.F = np.array([[1, dt, 0, 0],
                           [0, 1,  0, 0],
                           [0, 0,  1, dt],
                           [0, 0,  0, 1]])
        
        # 测量矩阵
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])
                           
        # 过程噪声协方差
        q = self.process_noise_std
        self.Q = q * np.array([[dt**3/3, dt**2/2, 0, 0],
                               [dt**2/2, dt,      0, 0],
                               [0, 0,      dt**3/3, dt**2/2],
                               [0, 0,      dt**2/2, dt]])

        # 测量噪声协方差
        self.R = self.sensor_std**2 * np.eye(self.z_dim)

    def generate_data(self):
        ground_truth = []
        measurements = []
        
        live_targets = {} # id -> state

        for t in range(self.num_steps):
            # 死亡
            dead_ids = [tid for tid, state in live_targets.items() if np.random.rand() < self.p_death]
            for tid in dead_ids:
                del live_targets[tid]
                
            # 状态传播
            for tid in live_targets:
                live_targets[tid] = self.F @ live_targets[tid] + \
                                    np.random.multivariate_normal(np.zeros(self.x_dim), self.Q)

            # 新生
            if np.random.rand() < self.p_birth:
                new_id = len(live_targets) + 1
                new_state = np.array([np.random.uniform(-50, 50), 
                                      np.random.uniform(-1, 1), 
                                      np.random.uniform(-50, 50),
                                      np.random.uniform(-1, 1)])
                live_targets[new_id] = new_state
                
            # 生成测量
            current_measurements = []
            for tid, state in live_targets.items():
                if np.random.rand() < 0.95: # 检测概率
                    z = self.H @ state + np.random.multivariate_normal(np.zeros(self.z_dim), self.R)
                    current_measurements.append(z)
            
            # 生成杂波
            num_clutter = np.random.poisson(self.clutter_rate)
            for _ in range(num_clutter):
                clutter = np.array([np.random.uniform(-100, 100), np.random.uniform(-100, 100)])
                current_measurements.append(clutter)
                
            ground_truth.append(list(live_targets.values()))
            measurements.append(current_measurements)
            
        return ground_truth, measurements