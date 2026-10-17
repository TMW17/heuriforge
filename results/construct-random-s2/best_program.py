# HeuriForge champion for cvrp_construct (cvrplib_x)
# program 226: train gap 22.97%, validation gap 22.09%
# idea: change the constant 0.0238 to 0.04112
import math

import numpy as np

# EVOLVE-BLOCK-START
def score_candidates(current, candidates, rest_capacity, capacity, demand, dist, coords, depot):
    """Score every possible next stop; the truck goes to the highest score.

    Starting point: nearest neighbour. Go to the closest customer that still fits,
    and only go back to the depot when no customer fits any more.
    """
    scores = -dist[current, candidates].astype(float)
    return scores + 0.05822 * demand[candidates] + 0.1509 * (rest_capacity - demand[candidates]) + 0.04112 * demand[candidates] + 0.9394 * demand[candidates] + -0.028 * np.hypot(coords[candidates, 0] - coords[depot, 0], coords[candidates, 1] - coords[depot, 1]) + -0.173 * (demand[candidates] / rest_capacity) + -0.0145 * dist[candidates, depot]
# EVOLVE-BLOCK-END
