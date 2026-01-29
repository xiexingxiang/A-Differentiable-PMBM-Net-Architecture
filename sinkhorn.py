import torch

def sinkhorn(cost_matrix, epsilon=0.1, n_iters=100):
    """
    Sinkhorn algorithm to approximate optimal transport.
    This version includes a stability improvement.
    """
    
    B, N, M = cost_matrix.shape
    
    # A small constant to prevent division by zero
    stability_eps = 1e-9
    
    # Initialize u and v
    # In the log-space, this would be torch.zeros, but we stick to the original here.
    u = torch.ones(B, N, device=cost_matrix.device) / N
    v = torch.ones(B, M, device=cost_matrix.device) / M
    
    # Sinkhorn-Knopp iteration
    K = torch.exp(-cost_matrix / epsilon)
    
    for _ in range(n_iters):
        # --- STABILITY FIX IS HERE ---
        # We add stability_eps to the denominator to avoid 1/0
        u = 1.0 / (torch.matmul(K, v.unsqueeze(-1)).squeeze(-1) + stability_eps)
        v = 1.0 / (torch.matmul(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + stability_eps)
        # --- END OF FIX ---
        
    transport_plan = u.unsqueeze(-1) * K * v.unsqueeze(-2)
    return transport_plan