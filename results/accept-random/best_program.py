# HeuriForge champion for pyvrp_accept (cvrplib_x)
# program 298: train gap 1.69%, validation gap 1.61%
# idea: change the constant 0.1951 to 0.2194
import math

import numpy as np

# EVOLVE-BLOCK-START
def accept(cand, curr, late, best, feasible, it, max_it, stall, u, state):
    """Decide whether the search moves to the candidate solution.

    Starting point: PyVRP's own rule, late-acceptance hill climbing. Accept when the
    candidate beats the solution of a few hundred steps ago, or the current one.
    """
    return ((((cand < late or cand < curr) and cand < late * 1.005) and cand < best / (1 + 0.02 * (1 - it / max_it)) or u < math.exp(-max(0.2194, cand - curr) / max(1e-09, 0.002393 * curr * (1 - it / max_it)))) and cand < curr * 1.01) and cand < best * (1 + 0.02 * (1 - it / max_it))
# EVOLVE-BLOCK-END
