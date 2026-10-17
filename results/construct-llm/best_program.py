# HeuriForge champion for cvrp_construct (cvrplib_x)
# program 28: train gap 23.08%, validation gap 20.35%
# idea: Replace the flawed distance penalty calculation with a simple distance-based heuristic that prefers customers closer to the depot when the truck is not nearly full, fixing the axis error and improving route compactness.
import math

import numpy as np

# EVOLVE-BLOCK-START
import numpy as np

def score_candidates(current, candidates, rest_capacity, capacity, demand, dist, coords, depot):
    scores = np.zeros(len(candidates))
    d = dist.astype(float)
    dem = demand.astype(float)
    mask = candidates != depot
    
    if np.any(mask):
        cand_indices = candidates[mask]
        dist_to_next = d[current, cand_indices]
        dist_to_depot = d[cand_indices, depot]
        savings = (dist_to_depot + dist_to_depot) - dist_to_next
        # Normalize savings by distance to prioritize near customers with high savings
        norm_savings = savings / (dist_to_next + 1e-9)
        # Adjust for capacity utilization: prefer filling the truck more if possible
        util = rest_capacity / capacity
        # Add a penalty for customers far from depot when truck is not full to avoid long routes
        dist_to_depot = d[cand_indices, depot]
        if util < 0.8:
            # Prefer customers closer to depot when truck is not full
            dist_factor = dist_to_depot / (np.mean(dist_to_depot) + 1e-9)
            scores[mask] = norm_savings * (1 + 0.5 * util) * (1 - 0.2 * dist_factor)
        else:
            scores[mask] = norm_savings * (1 + 0.5 * util)
    
    scores[candidates == depot] = -np.inf
    return scores
# EVOLVE-BLOCK-END
