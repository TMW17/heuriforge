# HeuriForge champion for pyvrp_accept (cvrplib_x)
# program 471: train gap 1.52%, validation gap 1.47%
# idea: Strengthen the acceptance condition for improving solutions by using a stricter threshold (0.3) on the relative gap compared to the previous version (0.5), which reduces unnecessary rejections of marginal improvements while maintaining the adaptive temperature mechanism.
import math

import numpy as np

# EVOLVE-BLOCK-START
import math

def accept(cand, curr, late, best, feasible, it, max_it, stall, u, state):
    if not feasible:
        return False
    # Adaptive temperature decays with iteration and stagnation
    state['t'] = max(0.001, state.get('t', 1.0) * (0.99 ** (it / max_it)) * (0.98 ** stall))
    T = state['t']
    if u < T:
        return True
    delta = cand - curr
    if delta < 0 or abs(delta) < max(T * abs(curr - late) * 0.3, abs(curr - late) * 0.05):
        return True
    return False
# EVOLVE-BLOCK-END
