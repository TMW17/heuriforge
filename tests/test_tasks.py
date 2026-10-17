import numpy as np
import pytest
from pyvrp import read, solve
from pyvrp.stop import MaxIterations

from heuriforge.instances import check_routes, load_instance, route_cost, solution_routes
from heuriforge.sandbox.safe_exec import load_program
from heuriforge.tasks import TASKS, get_task
from heuriforge.tasks.base import InvalidOutput
from heuriforge.tasks.cvrp_construct import as_scores, construct
from heuriforge.tasks.pyvrp_accept import lahc, run_ils

from .conftest import DATA


def test_registry():
    assert set(TASKS) == {"cvrp_construct", "pyvrp_accept"}
    with pytest.raises(ValueError):
        get_task("tsp")


@pytest.mark.parametrize(
    ("name", "iterations", "seed"), [("S-n21-s101", 150, 1), ("S-n26-s103", 300, 7)]
)
def test_ils_replica_matches_pyvrp_exactly(settings, name, iterations, seed):
    """With PyVRP's own acceptance rule, the replica must reproduce pyvrp.solve exactly."""
    path = settings.paths.data / "tiny" / f"{name}.vrp"
    data = read(path, round_func="round")
    expected = solve(data, stop=MaxIterations(iterations), seed=seed, collect_stats=False)
    best, _ = run_ils(data, lahc, seed=seed, iterations=iterations)
    assert best.distance() == expected.best.distance()
    assert solution_routes(best, data) == solution_routes(expected.best, data)


@pytest.mark.slow
def test_ils_replica_matches_pyvrp_on_a_real_instance():
    data = read(DATA / "X-n101-k25.vrp", round_func="round")
    expected = solve(data, stop=MaxIterations(300), seed=3, collect_stats=False)
    best, _ = run_ils(data, lahc, seed=3, iterations=300)
    assert best.distance() == expected.best.distance()


def test_acceptance_rule_matters(settings):
    data = read(settings.paths.data / "tiny" / "S-n41-s3.vrp", round_func="round")
    never, accepted = run_ils(data, lambda *a: False, seed=1, iterations=100)
    assert accepted == 0 and never.is_feasible()
    with pytest.raises(InvalidOutput):
        run_ils(data, lambda *a: "yes", seed=1, iterations=5)


def test_constructive_skeleton_always_builds_valid_routes():
    inst = load_instance(DATA / "X-n101-k25.vrp")
    task = get_task("cvrp_construct")
    prepared = task.prepare(str(inst.path), 0, {})
    nearest = load_program("import numpy as np\n" + task.initial_block, task.entry)
    routes, steps = construct(nearest, prepared)
    assert check_routes(inst, routes) == [] and steps >= inst.n - 1

    def constant(*args):  # all ties: the skeleton must still make progress
        return np.zeros(len(args[1]))

    routes, _ = construct(constant, prepared)
    assert check_routes(inst, routes) == []

    def back_home(current, candidates, *rest):  # always prefers the depot when offered
        return (candidates == 0).astype(float)

    routes, _ = construct(back_home, prepared)
    assert check_routes(inst, routes) == [] and len(routes) == inst.n - 1
    assert route_cost(inst.dist, routes) > inst.bks


def test_scores_validation():
    assert as_scores([1, 2.5, -np.inf], 3).shape == (3,)
    for bad in ([1, 2], [np.nan, 1, 2], "abc", [[1, 2, 3]]):
        with pytest.raises(InvalidOutput):
            as_scores(bad, 3)
