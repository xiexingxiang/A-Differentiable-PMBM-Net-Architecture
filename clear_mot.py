import numpy as np
from scipy.optimize import linear_sum_assignment

class MOTAccumulator:
    def __init__(self):
        self.acc = {}

    def update(self, gt_ids, hy_ids, dists):
        # Implementation of CLEAR MOT metrics update
        # This is a simplified version. For a full implementation,
        # refer to py-motmetrics library.
        
        if dists.shape[0] == 0:
            for h in hy_ids:
                if 'fp' not in self.acc: self.acc['fp'] = 0
                self.acc['fp'] += 1
            for g in gt_ids:
                if 'miss' not in self.acc: self.acc['miss'] = 0
                self.acc['miss'] += 1
            return
            
        row_ind, col_ind = linear_sum_assignment(dists)
        
        # Matches, FP, Misses
        matches = []
        for r, c in zip(row_ind, col_ind):
            if dists[r, c] < 1.0: # Distance threshold
                matches.append((gt_ids[r], hy_ids[c]))
        
        match_gt_ids = [m[0] for m in matches]
        match_hy_ids = [m[1] for m in matches]
        
        if 'matches' not in self.acc: self.acc['matches'] = 0
        self.acc['matches'] += len(matches)
        
        unmatched_gt = set(gt_ids) - set(match_gt_ids)
        if 'miss' not in self.acc: self.acc['miss'] = 0
        self.acc['miss'] += len(unmatched_gt)
        
        unmatched_hy = set(hy_ids) - set(match_hy_ids)
        if 'fp' not in self.acc: self.acc['fp'] = 0
        self.acc['fp'] += len(unmatched_hy)
        
        if 'gt' not in self.acc: self.acc['gt'] = 0
        self.acc['gt'] += len(gt_ids)

    def compute_metrics(self):
        metrics = {}
        gt = self.acc.get('gt', 0)
        if gt == 0:
            return metrics
            
        metrics['MOTA'] = 1.0 - (self.acc.get('fp', 0) + self.acc.get('miss', 0) + self.acc.get('id_switches', 0)) / gt
        metrics['MOTP'] = self.acc.get('total_dist', 0) / self.acc.get('matches', 1)
        metrics['FP'] = self.acc.get('fp', 0)
        metrics['Misses'] = self.acc.get('miss', 0)
        metrics['Matches'] = self.acc.get('matches', 0)
        
        return metrics

def calculate_clear_mot(gt_data, tracker_data):
    acc = MOTAccumulator()
    for t in range(len(gt_data)):
        gt_objects = gt_data[t]
        tracker_objects = tracker_data[t]
        
        gt_ids = [o['id'] for o in gt_objects]
        hy_ids = [o['id'] for o in tracker_objects]
        
        if not gt_objects or not tracker_objects:
            acc.update(gt_ids, hy_ids, np.array([[]]))
            continue
            
        dist_matrix = np.full((len(gt_objects), len(tracker_objects)), np.inf)
        for i, gt_obj in enumerate(gt_objects):
            for j, tr_obj in enumerate(tracker_objects):
                dist = np.linalg.norm(np.array(gt_obj['pos']) - np.array(tr_obj['pos']))
                dist_matrix[i,j] = dist
                
        acc.update(gt_ids, hy_ids, dist_matrix)
        
    return acc.compute_metrics()