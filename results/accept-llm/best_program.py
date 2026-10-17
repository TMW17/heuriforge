# HeuriForge champion for pyvrp_accept (cvrplib_x)
# program 136: train gap 1.96%, validation gap 1.59%
# idea: Replace the overly conservative acceptance probability formula with a more aggressive one that scales acceptance chances based on the gap between candidate and late solutions, allowing better exploration while still penalizing large deteriorations.
import math

import numpy as np

# EVOLVE-BLOCK-START
def accept(cand, curr, late, best, feasible, it, max_it, stall, u, state):
    """Decide whether the search moves to the candidate solution.

    Starting point: PyVRP's own rule, late-acceptance hill climbing. Accept when the
    candidate beats the solution of a few hundred steps ago, or the current one.
    """
    if feasible:
        temp = 1.0 - (it / max(max_it, 1)) * 0.95
        if stall > 0:
            gap = max(0.0, (late - cand) / late if late > 0 else 0)
            prob = temp * (0.1 + 0.2 * gap + 0.1 * (stall / (max_it * 0.3) if max_it * 0.3 > 0 else 0))
        else:
            prob = temp * 0.1
        return cand < late and (cand < curr or u < prob)
    if not feasible:
        return False
    diff = cand - late
    if diff < 0 or u < 0.001:
        return True
    return False
# EVOLVE-BLOCK-END
