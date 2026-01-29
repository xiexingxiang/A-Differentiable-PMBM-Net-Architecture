import numpy as np
from scipy.stats import multivariate_normal

class PMBM_CV_Tracker:
    def __init__(self, config):
        self.config = config
        self.poisson_point_process = [] # (weight, mean, covariance)
        self.bernoulli_components = [] # (r, mean, covariance)

    def predict(self):
        # 预测泊松点过程 (新生和存活)
        # 预测伯努利分量
        pass # 具体的预测步骤依赖于详细的运动和新生模型

    def update(self, measurements):
        # 更新泊松点过程 (处理杂波和漏检)
        # 更新伯努リ分量 (为每个轨迹创建假设)
        # 生成新的伯努利分量 (对每个测量值)
        # 进行假设管理和修剪
        pass # 具体的更新步骤非常复杂，涉及到Murty's algorithm或类似的数据关联方法

    def estimate(self):
        # 从伯努利分量中提取状态估计
        estimates = []
        for r, mean, _ in self.bernoulli_components:
            if r > self.config['existence_threshold']:
                estimates.append(mean)
        return estimates