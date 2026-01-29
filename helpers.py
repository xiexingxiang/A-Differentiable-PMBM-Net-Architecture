import yaml
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def state_to_box(state, size=(1.8, 4.5, 1.6)):
    """Converts a state vector to a bounding box."""
    # 假设状态为 [x, y, vx, vy]
    return np.array([state[0], state[1], 0, size[0], size[1], size[2], 0])

def boxes_to_states(boxes):
    """Converts bounding boxes to state vectors."""
    states = []
    for box in boxes:
        # 假设我们只关心位置
        states.append(np.array([box[0], box[1], 0, 0])) 
    return states

def gate_measurements(measurements, predicted_state, predicted_covariance, gating_threshold=9.21):
    """
    Performs measurement gating using Mahalanobis distance.
    `gating_threshold` corresponds to chi2.ppf(0.99, df=2)
    """
    gated_measurements = []
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]]) # 示例测量矩阵
    R = np.eye(2) # 示例测量噪声
    
    for z in measurements:
        innovation = z - H @ predicted_state
        S = H @ predicted_covariance @ H.T + R
        mahalanobis_dist_sq = innovation.T @ np.linalg.inv(S) @ innovation
        
        if mahalanobis_dist_sq < gating_threshold:
            gated_measurements.append(z)
            
    return gated_measurements