# HeuriForge champion for cvrp_construct (cvrplib_x)
# program 72: train gap 24.70%, validation gap 21.67%
# idea: swap an operator
import math

import numpy as np

# EVOLVE-BLOCK-START
def score_candidates(current, candidates, rest_capacity, capacity, demand, dist, coords, depot):
    """Score every possible next stop; the truck goes to the highest score.

    Starting point: nearest neighbour. Go to the closest customer that still fits,
    and only go back to the depot when no customer fits any more.
    """
    scores = -dist[current, candidates].astype(float)
    return scores - -0.09386 * (np.sort(dist[candidates][:, candidates], axis=1)[:, 1:4].mean(axis=1) if len(candidates) > 4 else 0.0) + -0.798 * (demand[candidates] / rest_capacity)
# EVOLVE-BLOCK-END
