# HeuriForge champion for pyvrp_accept (cvrplib_x)
# program 78: train gap 1.83%, validation gap 1.43%
# idea: Use a simulated annealing schedule where the temperature decays based on the current stall count rather than the absolute iteration number, and introduce a "cooling rate" that adapts to the current search stagnation to allow more exploratory moves when the best solution hasn't improved in a while.
import math

import numpy as np

# EVOLVE-BLOCK-START
import math

def accept(cand, curr, late, best, feasible, it, max_it, stall, u, state):
    if not feasible:
        return False
    delta = cand - curr
    if delta < 0:
        return True
    if cand < best:
        return True
    T = state.get('temp', 1.0)
    if T is None:
        T = 1.0
    # Decay temperature based on stall count to encourage exploration when stuck
    decay = 0.995 ** stall
    T = T * decay
    state['temp'] = T
    if u < T:
        return True
    return False
# EVOLVE-BLOCK-END
