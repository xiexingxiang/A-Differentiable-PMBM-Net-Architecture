# losses/gospa_loss.py
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from utils.sinkhorn import sinkhorn


class GOSPALoss(nn.Module):
    def __init__(self, c=10.0, p=1, alpha=2.0):
        super(GOSPALoss, self).__init__()
        self.c = c  # Cut-off distance
        self.p = p  # p-norm for distance
        self.alpha = alpha
        #self.sinkhorn = sinkhorn(n_iters=20, epsilon=0.1)
        
        # After
        self.sinkhorn_iters = 20
        self.sinkhorn_epsilon = 0.1

    def forward(self, predicted_set, target_set):
        # predicted_set, target_set: list of tensors (state vectors)
        n = len(predicted_set)
        m = len(target_set)

        if n == 0 and m == 0:
            return torch.tensor(0.0)
        
        # Create cost matrix
        # For differentiability, we use Sinkhorn instead of Hungarian algorithm
        # This compares only the first 3 columns (x,y,z) of both tensors
        cost_matrix = torch.cdist(predicted_set[:, :3], target_set[:, :3], p=self.p)
        
        cost_matrix = torch.min(cost_matrix, torch.tensor(self.c))**self.p

        # Use Sinkhorn for differentiable assignment
        # Note: Sinkhorn needs a square matrix, padding may be required
        # For simplicity, let's assume a conceptual cost matrix C
        # In a real implementation, this would be a core part of the work.
        
        # Simplified non-differentiable version for clarity
        # In the actual paper, this part IS differentiable via Sinkhorn
        with torch.no_grad():
             row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
        
        dist = cost_matrix[row_ind, col_ind].sum()
        
        # Penalty for cardinality mismatch
        card_error = self.c**self.p * abs(n - m)
        
        gospa_dist = (dist + card_error) / max(n, m)
        gospa_dist = gospa_dist**(1/self.p)
        
        # The actual loss will need to be carefully crafted with the soft assignment matrix
        # from Sinkhorn to maintain differentiability.
        # This simplified version shows the core logic.
        
        # Placeholder for a proper differentiable loss
        # The key is to use the soft assignment matrix from Sinkhorn to weight the distances

        # After
        # Call the imported sinkhorn function directly
        # After
        # Add a batch dimension with unsqueeze(0) before the call
        # Shape changes from [N, M] to [1, N, M]
        soft_assign = sinkhorn(
            cost_matrix.unsqueeze(0),
            n_iters=self.sinkhorn_iters,
            epsilon=self.sinkhorn_epsilon
        )
        # It's also good practice to remove the batch dimension from the output
        soft_assign = soft_assign.squeeze(0)

        gospa_loss = torch.sum(soft_assign * cost_matrix)
        
        # Add cardinality penalty
        # This also needs a differentiable formulation
        
        return gospa_loss