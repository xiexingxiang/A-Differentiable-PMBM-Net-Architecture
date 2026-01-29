import numpy as np
from scipy.stats import chi2

def nees_test(x_true, x_est, P_est):
    """
    Calculates the NEES for a single time step.
    """
    error = x_true - x_est
    nees = error.T @ np.linalg.inv(P_est) @ error
    return nees

def nis_test(z_meas, z_pred, S):
    """
    Calculates the NIS for a single time step.
    """
    innovation = z_meas - z_pred
    nis = innovation.T @ np.linalg.inv(S) @ innovation
    return nis

def run_consistency_checks(true_states, estimated_states, estimated_covariances, measurements, predicted_measurements, innovation_covariances):
    num_steps = len(true_states)
    x_dim = true_states[0].shape[0]
    z_dim = measurements[0].shape[0]
    
    nees_values = np.zeros(num_steps)
    nis_values = np.zeros(num_steps)
    
    for t in range(num_steps):
        nees_values[t] = nees_test(true_states[t], estimated_states[t], estimated_covariances[t])
        nis_values[t] = nis_test(measurements[t], predicted_measurements[t], innovation_covariances[t])
        
    # 计算NEES和NIS的平均值
    avg_nees = np.mean(nees_values)
    avg_nis = np.mean(nis_values)
    
    # 计算置信区间
    alpha = 0.05
    nees_lower = chi2.ppf(alpha / 2, df=num_steps * x_dim) / num_steps
    nees_upper = chi2.ppf(1 - alpha / 2, df=num_steps * x_dim) / num_steps
    
    nis_lower = chi2.ppf(alpha / 2, df=num_steps * z_dim) / num_steps
    nis_upper = chi2.ppf(1 - alpha / 2, df=num_steps * z_dim) / num_steps

    results = {
        'avg_nees': avg_nees,
        'nees_bounds': (nees_lower, nees_upper),
        'avg_nis': avg_nis,
        'nis_bounds': (nis_lower, nis_upper),
        'nees_inside_bounds': nees_lower <= avg_nees <= nees_upper,
        'nis_inside_bounds': nis_lower <= avg_nis <= nis_upper
    }
    
    return results