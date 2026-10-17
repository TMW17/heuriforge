# HeuriForge champion for pyvrp_accept (cvrplib_x)
# program 16: train gap 1.97%, validation gap 1.56%
# idea: only accept when cand <= curr as well
import math

import numpy as np

# EVOLVE-BLOCK-START
def accept(cand, curr, late, best, feasible, it, max_it, stall, u, state):
    """Decide whether the search moves to the candidate solution.

    Starting point: PyVRP's own rule, late-acceptance hill climbing. Accept when the
    candidate beats the solution of a few hundred steps ago, or the current one.
    """
    return (cand < late or cand < curr) and cand <= curr
# EVOLVE-BLOCK-END
