"""Task 1: evolve a constructive heuristic for the capacitated vehicle routing problem.

This is the classic "evolution of heuristics" set-up (EoH, ReEvo): a fixed loop builds
the routes one stop at a time, and the evolved function decides which stop comes next.
It is fast to evaluate (well under a second per instance), so it gives a strong signal
quickly, and it is the right task to start with.
"""

from __future__ import annotations

import numpy as np

from heuriforge.instances import Instance, load_instance
from heuriforge.tasks.base import InvalidOutput, ModifiedInput, Task, fingerprint, readonly

INITIAL = '''\
def score_candidates(current, candidates, rest_capacity, capacity, demand, dist, coords, depot):
    """Score every possible next stop; the truck goes to the highest score.

    Starting point: nearest neighbour. Go to the closest customer that still fits,
    and only go back to the depot when no customer fits any more.
    """
    scores = -dist[current, candidates].astype(float)
    scores[candidates == depot] = -np.inf
    return scores
'''

DESCRIPTION = """\
A depot (location 0) and n customers are spread over a plane. Each customer has a demand.
Identical trucks with a fixed capacity start and end their route at the depot. Every
customer must be visited exactly once, and the demand served on one route must not exceed
the capacity. The goal is the smallest total travel distance.

The solution is built one stop at a time by a fixed loop. A truck leaves the depot; at each
step, your function scores the possible next stops and the truck goes to the stop with the
highest score. When the truck goes back to the depot (or no remaining customer fits), the
next route starts. The loop always produces a valid solution: your function only decides
the order in which customers are visited and when routes end."""

API_DOC = """\
def score_candidates(current, candidates, rest_capacity, capacity, demand, dist, coords, depot):
    current        int: the location of the truck now (0 when it is at the depot)
    candidates     1-D int array: the possible next stops, in increasing order. These are the
                   unvisited customers whose demand fits in the truck, followed by the depot (0)
                   when the current route already has at least one customer. Giving the depot
                   the highest score ends the current route early
    rest_capacity  int: capacity left in the truck (at least 1)
    capacity       int: the full capacity of a truck
    demand         1-D int array: demand[i] of every location (demand[0] == 0)
    dist           2-D int array: dist[i, j] is the travel distance between i and j (symmetric)
    coords         2-D float array of shape (n, 2): the x, y position of every location
    depot          int: always 0
    returns        1-D array with one score per candidate (higher is better). NaN is not
                   allowed; -np.inf is allowed. Ties go to the first candidate.

All input arrays are read-only: make a copy (for example with .astype(float)) before changing
values. Instances have up to about 330 customers and the function is called about once per
customer, so keep each call fast: vectorised numpy, no Python loops over all pairs of
locations."""

SCORING_DOC = """\
Score = the average gap, in %, between the total distance of the built routes and the
best-known solution, over the training instances. Lower is better; 0% would match the
best-known solutions. Each instance must finish within the time limit. The final champion is
chosen on separate validation instances and reported once on sealed test instances, so tricks
that only fit the training instances do not pay off."""


class CvrpConstruct(Task):
    name = "cvrp_construct"
    entry = "score_candidates"
    params = [
        "current",
        "candidates",
        "rest_capacity",
        "capacity",
        "demand",
        "dist",
        "coords",
        "depot",
    ]
    initial_block = INITIAL
    title = "Constructive heuristic for the capacitated vehicle routing problem (CVRP)"
    description = DESCRIPTION
    api_doc = API_DOC
    scoring_doc = SCORING_DOC
    complexity_bins = [60, 120, 240]
    params_bins = [4, 5, 6]
    combine = "add"
    random_terms = [
        "dist[current, candidates]",
        "dist[candidates, depot]",
        "dist[current, depot] - dist[candidates, depot]",
        "demand[candidates]",
        "demand[candidates] / rest_capacity",
        "rest_capacity - demand[candidates]",
        "(rest_capacity - demand[candidates]) / capacity",
        "np.hypot(coords[candidates, 0] - coords[depot, 0], coords[candidates, 1] - coords[depot, 1])",
        "np.arctan2(coords[candidates, 1] - coords[depot, 1], coords[candidates, 0] - coords[depot, 0])",
        "np.sort(dist[candidates][:, candidates], axis=1)[:, 1:4].mean(axis=1) if len(candidates) > 4 else 0.0",
    ]

    def job_options(self, split: str, options: dict) -> dict:
        return {}

    def cpu_limit(self, split: str, options: dict) -> float:
        return float(options.get("per_instance_cpu_s", 10))

    # ----------------------------------------------------------------- worker side
    def prepare(self, path: str, seed: int, options: dict) -> dict:
        inst = load_instance(path)
        return {
            "n": inst.n,
            "capacity": inst.capacity,
            "demand": readonly(inst.demand),
            "dist": readonly(inst.dist),
            "coords": readonly(inst.coords),
        }

    def run(self, fn, prepared: dict, seed: int, options: dict) -> dict:
        routes, steps = construct(fn, prepared)
        return {"routes": routes, "steps": steps}


def construct(fn, p: dict) -> tuple[list[list[int]], int]:
    """The fixed construction loop. Always returns a valid solution."""
    n, capacity = p["n"], p["capacity"]
    demand, dist, coords = p["demand"], p["dist"], p["coords"]
    before = fingerprint([demand, dist, coords])
    unvisited = np.ones(n, dtype=bool)
    unvisited[0] = False
    remaining = n - 1
    routes: list[list[int]] = []
    route: list[int] = []
    load = 0
    current = 0
    steps = 0
    while remaining:
        rest = capacity - load
        feasible = np.flatnonzero(unvisited & (demand <= rest))
        if feasible.size == 0:
            if not route:
                raise RuntimeError("a customer's demand exceeds the truck capacity")
            routes.append(route)
            route, load, current = [], 0, 0
            continue
        candidates = np.append(feasible, 0) if route else feasible
        candidates.flags.writeable = False
        raw = fn(current, candidates, int(rest), int(capacity), demand, dist, coords, 0)
        steps += 1
        scores = as_scores(raw, len(candidates))
        nxt = int(candidates[int(np.argmax(scores))])
        if nxt == 0:
            routes.append(route)
            route, load, current = [], 0, 0
        else:
            route.append(nxt)
            load += int(demand[nxt])
            unvisited[nxt] = False
            remaining -= 1
            current = nxt
    if route:
        routes.append(route)
    if fingerprint([demand, dist, coords]) != before:
        raise ModifiedInput("the function changed its input data")
    return routes, steps


def as_scores(raw, m: int) -> np.ndarray:
    try:
        scores = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidOutput(f"return an array of numbers, not {type(raw).__name__}") from exc
    if scores.shape != (m,):
        raise InvalidOutput(
            f"return one score per candidate: expected shape ({m},), got {scores.shape}"
        )
    if np.isnan(scores).any():
        raise InvalidOutput("the scores contain NaN")
    return scores


# --------------------------------------------------------------------------- baselines
def savings_routes(inst: Instance) -> list[list[int]]:
    """Clarke and Wright's parallel savings algorithm (1964): the classic baseline."""
    n, d, demand, cap = inst.n, inst.dist, inst.demand, inst.capacity
    i, j = np.triu_indices(n, k=1)
    keep = i > 0
    i, j = i[keep], j[keep]
    saving = d[0, i] + d[0, j] - d[i, j]
    order = np.lexsort((j, i, -saving))  # largest saving first, ties by index
    routes = {c: [c] for c in range(1, n)}  # route id -> customers (id = first customer)
    owner = {c: c for c in range(1, n)}  # customer -> route id
    load = {c: int(demand[c]) for c in range(1, n)}
    for k in order:
        if saving[k] <= 0:
            break
        a, b = int(i[k]), int(j[k])
        ia, ib = owner[a], owner[b]
        if ia == ib or load[ia] + load[ib] > cap:
            continue
        ra, rb = routes[ia], routes[ib]
        if ra[-1] == a and rb[0] == b:
            merged = ra + rb
        elif ra[0] == a and rb[-1] == b:
            merged = rb + ra
        elif ra[-1] == a and rb[-1] == b:
            merged = ra + rb[::-1]
        elif ra[0] == a and rb[0] == b:
            merged = ra[::-1] + rb
        else:
            continue  # a or b sits inside its route: the routes cannot be joined there
        routes[ia] = merged
        load[ia] += load.pop(ib)
        del routes[ib]
        for c in rb:
            owner[c] = ia
    return sorted(routes.values())
