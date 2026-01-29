import numpy as np
from scipy.optimize import linear_sum_assignment

def gospa(X, Y, p, c, alpha=2):
    """
    Computes the GOSPA metric between two sets of tracks.
    
    Args:
        X (list of arrays): First set of tracks (ground truth).
        Y (list of arrays): Second set of tracks (estimates).
        p (int): Order of the metric.
        c (float): Cut-off distance.
        alpha (int): Alpha parameter for the penalty of cardinality difference.
        
    Returns:
        float: GOSPA metric value.
        float: GOSPA localization component.
        float: GOSPA missed component.
        float: GOSPA false component.
    """
    
    n = len(X)
    m = len(Y)

    if n == 0 and m == 0:
        return 0, 0, 0, 0
    
    if n == 0:
        gospa_val = (c**p / alpha * m)**(1/p)
        return gospa_val, 0, 0, gospa_val
        
    if m == 0:
        gospa_val = (c**p / alpha * n)**(1/p)
        return gospa_val, 0, gospa_val, 0

    # 计算距离矩阵
    dist_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist = np.linalg.norm(X[i] - Y[j])
            dist_matrix[i, j] = min(dist, c)**p
            
    # 匈牙利算法进行分配
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    # 计算GOSPA
    gospa_dist = dist_matrix[row_ind, col_ind].sum()
    gospa_loc = (gospa_dist)**(1/p)
    
    # 惩罚项
    gospa_miss = (c**p / alpha * (n - len(row_ind)))**(1/p)
    gospa_false = (c**p / alpha * (m - len(row_ind)))**(1/p)

    gospa_val = (gospa_dist + (c**p / alpha) * (n + m - 2 * len(row_ind)))**(1/p)
    
    return gospa_val, gospa_loc, gospa_miss, gospa_false