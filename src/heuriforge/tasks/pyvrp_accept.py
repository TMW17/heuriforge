"""Task 2: evolve the acceptance rule inside PyVRP, a state-of-the-art VRP solver.

PyVRP (0.14) solves vehicle routing problems with iterated local search: perturb the
current solution, improve it with local search, then decide whether to *accept* the
result as the new current solution. PyVRP decides with late-acceptance hill climbing.
Here that single decision is the evolvable function; everything else is PyVRP's own code,
replicated step for step in ``run_ils`` (a test checks that the replica with the initial
rule gives exactly the same result as ``pyvrp.solve``).

Improving a strong solver is much harder than improving a constructive heuristic, so this
task is the stretch goal: a small, honest improvement here is worth more than a big one on
Task 1.
"""

from __future__ import annotations

import math

import numpy as np

from heuriforge.tasks.base import InvalidOutput, Task

INITIAL = '''\
def accept(cand, curr, late, best, feasible, it, max_it, stall, u, state):
    """Decide whether the search moves to the candidate solution.

    Starting point: PyVRP's own rule, late-acceptance hill climbing. Accept when the
    candidate beats the solution of a few hundred steps ago, or the current one.
    """
    return cand < late or cand < curr
'''

DESCRIPTION = """\
A depot and up to about 330 customers with demands must be served by trucks of a fixed
capacity; the goal is the smallest total travel distance (the capacitated vehicle routing
problem). The solver is PyVRP, a state-of-the-art iterated local search. Every iteration it
perturbs the current solution and improves it with local search, which gives a candidate
solution. Your function then decides whether the candidate becomes the new current solution.
The best feasible solution ever seen is kept separately, so accepting a worse candidate is
safe: it only changes where the search goes next.

Costs are "penalised costs": the travel distance plus a penalty when a solution overloads a
truck. The penalty weights adapt during the search, so the same solution can have a different
cost in different iterations. The run has a fixed number of iterations."""

API_DOC = """\
def accept(cand, curr, late, best, feasible, it, max_it, stall, u, state):
    cand      float: penalised cost of the candidate solution
    curr      float: penalised cost of the current solution
    late      float: penalised cost of the solution in the late-acceptance history (the
              current solution of a few hundred steps ago), or of the initial solution
    best      float: cost of the best feasible solution so far (math.inf if none yet)
    feasible  bool: whether the candidate is feasible (no overloaded truck)
    it        int: iteration number, from 0 to max_it - 1
    max_it    int: total number of iterations of the run
    stall     int: iterations since the best solution last improved
    u         float: a random number in [0, 1), fresh every iteration (use it for random
              decisions; it keeps runs reproducible)
    state     dict: an empty dict at the start of each run that you may use to keep values
              between iterations (for example a temperature)
    returns   True to move to the candidate, False to stay on the current solution

The function is called once per iteration, so it must be cheap."""

SCORING_DOC = """\
Score = the average gap, in %, between the best solution found and the best-known solution,
over the training instances, after a fixed number of iterations. Lower is better. PyVRP's
default rule (the starting point) is already strong, so improvements are small: a few tenths
of a percent is a real gain. The champion is chosen on validation instances and seeds, and
reported once on sealed test instances with new seeds."""


class PyvrpAccept(Task):
    name = "pyvrp_accept"
    entry = "accept"
    params = ["cand", "curr", "late", "best", "feasible", "it", "max_it", "stall", "u", "state"]
    initial_block = INITIAL
    title = "Acceptance rule for PyVRP's iterated local search (CVRP)"
    description = DESCRIPTION
    api_doc = API_DOC
    scoring_doc = SCORING_DOC
    complexity_bins = [30, 60, 120]
    params_bins = [3, 4, 6]
    combine = "bool"
    option_keys = Task.option_keys | {
        "iterations",
        "smoke_iterations",
        "train_seeds",
        "val_seeds",
        "test_seeds",
    }
    random_terms = [
        "cand <= curr",
        "cand < best",
        "u < 0.01",
        "cand < curr * 1.01",
        "cand < late * 1.005",
        "stall > 200 and cand < curr * 1.02",
        "cand < best * (1 + 0.02 * (1 - it / max_it))",
        "u < math.exp(-max(0.0, cand - curr) / max(1e-9, 0.001 * curr * (1 - it / max_it)))",
        "feasible and cand < late",
    ]

    def seeds(self, split: str, options: dict) -> list[int]:
        if split == "smoke":
            return list(options.get("train_seeds", [1]))[:1]
        return list(options.get(f"{split}_seeds", [1]))

    def job_options(self, split: str, options: dict) -> dict:
        iterations = int(options.get("iterations", 1000))
        if split == "smoke":
            iterations = int(options.get("smoke_iterations", 200))
        return {"iterations": iterations}

    def cpu_limit(self, split: str, options: dict) -> float:
        return float(options.get("per_instance_cpu_s", 60))

    # ----------------------------------------------------------------- worker side
    def prepare(self, path: str, seed: int, options: dict):
        from pyvrp import read

        return read(path, round_func="round")

    def run(self, fn, data, seed: int, options: dict) -> dict:
        from heuriforge.instances import solution_routes

        best, accepted = run_ils(data, fn, seed=seed, iterations=int(options["iterations"]))
        if not best.is_feasible():
            raise InvalidOutput("the search never found a feasible solution")
        return {"routes": solution_routes(best, data), "accepted": accepted}


def lahc(cand, curr, late, best, feasible, it, max_it, stall, u, state):
    """PyVRP's rule, as a plain function (used by the replica test and the baselines)."""
    return cand < late or cand < curr


def run_ils(data, accept, seed: int, iterations: int):
    """PyVRP 0.14's ``solve`` + ``IteratedLocalSearch.run`` with a pluggable acceptance rule.

    Returns the best solution and how many candidates were accepted.
    """
    from pyvrp import PenaltyManager, RandomNumberGenerator, Solution, SolveParams
    from pyvrp.RingBuffer import RingBuffer
    from pyvrp.search import LocalSearch, PerturbationManager, compute_neighbours

    params = SolveParams()
    rng = RandomNumberGenerator(seed=seed)
    neighbours = compute_neighbours(data, params.neighbourhood)
    ls = LocalSearch(data, rng, neighbours, PerturbationManager(params.perturbation))
    for op in params.operators:
        if op.supports(data):
            ls.add_operator(op(data))
    pm = PenaltyManager(params.penalty.midpoint_penalties(data), params.penalty)
    init = ls(Solution.make_random(data, rng), pm.max_cost_evaluator(), exhaustive=True)

    ils = params.ils
    history = RingBuffer(ils.history_length)
    best = curr = init
    stall = 0
    accepted = 0
    state: dict = {}
    u_stream = np.random.default_rng(seed)  # separate from PyVRP's own random numbers
    for it in range(iterations):
        if stall == ils.num_iters_no_improvement:  # restart, as PyVRP does
            history.clear()
            curr = best
            stall = 0

        cost_eval = pm.cost_evaluator()
        cand = ls(curr, cost_eval, exhaustive=False)
        pm.register(cand)

        stall += 1
        if cost_eval.cost(cand) < cost_eval.cost(best):
            best = cand
            stall = 0
            if ils.exhaustive_on_best:
                cand = ls(cand, cost_eval, exhaustive=True)
                if cand.is_feasible():
                    best = cand

        cand_cost = cost_eval.penalised_cost(cand)
        curr_cost = cost_eval.penalised_cost(curr)
        late_cost = cost_eval.penalised_cost(init)
        if (late := history.peek()) is not None:
            late_cost = cost_eval.penalised_cost(late)
        best_cost = cost_eval.cost(best) if best.is_feasible() else math.inf
        u = float(u_stream.random())

        decision = accept(
            float(cand_cost), float(curr_cost), float(late_cost), float(best_cost),
            bool(cand.is_feasible()), it, iterations, stall, u, state,
        )  # fmt: skip
        if not isinstance(decision, (bool, np.bool_)):
            raise InvalidOutput(
                f"accept() must return True or False, not {type(decision).__name__}"
            )
        if decision:
            curr = cand
            curr_cost = cand_cost
            accepted += 1

        # The history update is part of the fixed algorithm, as in PyVRP.
        if curr_cost < late_cost or late is None:
            history.append(curr)
        else:
            history.skip()
    return best, accepted
