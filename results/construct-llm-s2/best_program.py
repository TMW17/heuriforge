# HeuriForge champion for cvrp_construct (cvrplib_x)
# program 510: train gap 20.41%, validation gap 18.98%
# idea: Introduce a load factor penalty that explicitly penalizes candidates when the remaining capacity is low relative to the candidate's demand, encouraging more balanced route loads instead of overloading a single truck.
import math

import numpy as np

# EVOLVE-BLOCK-START
import numpy as np

def score_candidates(current, candidates, rest_capacity, capacity, demand, dist, coords, depot):
    d_curr = dist[current, candidates].astype(float)
    d_depot = dist[depot, candidates].astype(float)
    d_return = dist[current, depot].astype(float)
    savings = d_return + d_depot - d_curr
    d_curr_cand = d_curr
    d_depot_cand = d_depot
    load_factor = demand[candidates] / rest_capacity
    load_penalty = load_factor * d_curr
    load_penalty = load_factor * (capacity / rest_capacity) * d_curr
    # Penalize candidates far from depot to encourage compact routes
    depot_dist = dist[depot, candidates]
    distance_penalty = 0.1 * (depot_dist / d_return) * d_curr
    savings = savings - load_penalty * 0.5 - distance_penalty
    if candidates[0] == depot:
        scores = -np.inf
    else:
        scores = savings
    scores = scores.astype(float)
    return scores
# EVOLVE-BLOCK-END
