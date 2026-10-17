# HeuriForge champion for cvrp_construct (cvrplib_x)
# program 498: train gap 17.09%, validation gap 16.79%
# idea: Replace the undefined 'proximity' variable with a calculated proximity bonus based on the inverse of the distance to the current node, ensuring the scoring formula is mathematically consistent and avoids runtime errors.
import math

import numpy as np

# EVOLVE-BLOCK-START
import numpy as np

def score_candidates(current, candidates, rest_capacity, capacity, demand, dist, coords, depot):
    d = dist.astype(float)
    n = len(candidates)
    scores = np.full(n, -np.inf)
    fits = rest_capacity >= demand[candidates]
    
    if np.any(fits):
        valid_mask = fits
        cands = candidates[valid_mask]
        d_curr = d[current, cands]
        d_depot = d[cands, depot]
        d_curr_depot = d[current, depot]
        
        # Calculate savings for extending the route
        savings = d_curr_depot + d_depot - d_curr
        
        # Calculate centroid of the current route to score candidates relative to the whole route
        # We assume 'current' is the last stop, so we need the sum of distances from current to all
        # previously visited nodes in this route. However, we don't have the full route history here.
        # Instead, we use a proxy: the distance from the current node to the centroid of the route
        # is approximated by the average distance from current to all nodes in the route.
        # Since we can't access the route history, we use a heuristic based on the distance to the
        # "middle" of the route. A simpler proxy is the distance to the average of all nodes in the route.
        # To make this work without history, we assume the 'current' node is close to the route's center
        # if the route is balanced. We will use the distance to the depot as a baseline for the route's
        # "center" if the route is short, but a better approach is to use the distance from the current
        # node to the average of all nodes in the route.
        # Since we don't have the route history, we will use a fallback: if the route is long, 
        # we prefer nodes close to the current node; if short, we might prefer nodes closer to the depot.
        # However, the problem statement implies we are building one route at a time.
        # Let's use a simple spatial clustering score: score = savings - 0.5 * distance_to_centroid.
        # Since we don't have the centroid, we will use the distance from the current node to the 
        # average of all nodes in the route. But we don't have the route.
        # Let's use a different approach: score = savings * (1 - (distance_from_current_to_depot / d_curr_depot)).
        # This encourages nodes that are closer to the current node relative to the depot.
        
        # Revised: Use the distance from the current node to the depot as a measure of route length.
        # If the route is long, d_curr_depot is large, so we prefer nodes close to current.
        # If the route is short, d_curr_depot is small, so we are less restrictive.
        # We also add a capacity penalty to avoid overloading.
        
        load_ratio = demand[cands] / rest_capacity
        proximity = -1.0 / (d_curr + 1e-9)

        # Proximity bonus: prefer nodes close to current
        # Use -1/d^2 to give stronger relative weight to closer nodes
        proximity_bonus = -1.0 / (d_curr**2 + 1e-9)
        
        # Combine: savings + proximity * load_ratio
        # Using load_ratio as a weight for proximity ensures we prioritize customers
        # that fit well in the remaining capacity while being close to the current node.
        scores[valid_mask] = (savings * 0.7) + (proximity_bonus * 0.3 * load_ratio)
        
        # If the truck is nearly full, penalize nodes further from the current node to avoid
        # creating long detours that might fill the truck too quickly.
        urgency = (capacity - rest_capacity) / capacity
        if urgency > 0.3:
            scores[valid_mask] -= urgency * d_curr
            
    else:
        depot_mask = candidates == depot
        if np.any(depot_mask):
            scores[depot_mask] = 0.0
            
    return scores
# EVOLVE-BLOCK-END
