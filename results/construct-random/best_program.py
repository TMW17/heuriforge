# HeuriForge champion for cvrp_construct (cvrplib_x)
# program 112: train gap 22.20%, validation gap 22.17%
# idea: change the constant 0.254 to 0.1602
import math

import numpy as np

# EVOLVE-BLOCK-START
def score_candidates(current, candidates, rest_capacity, capacity, demand, dist, coords, depot):
    """Score every possible next stop; the truck goes to the highest score.

    Starting point: nearest neighbour. Go to the closest customer that still fits,
    and only go back to the depot when no customer fits any more.
    """
    scores = -dist[current, candidates].astype(float)
    return scores - 0.1602 * (demand[candidates] / rest_capacity) + -1.05 * (dist[current, depot] - dist[candidates, depot]) + -0.337 * demand[candidates]
# EVOLVE-BLOCK-END
