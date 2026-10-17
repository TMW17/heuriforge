import numpy as np
import pytest
import vrplib

from heuriforge.config import load_settings, load_splits
from heuriforge.instances import (
    check_routes,
    gap_percent,
    generate_synthetic,
    load_instance,
    route_cost,
    synthetic_spec,
)
from heuriforge.tasks.cvrp_construct import savings_routes

from .conftest import DATA


@pytest.fixture(scope="module")
def x101():
    return load_instance(DATA / "X-n101-k25.vrp")


def test_best_known_cost_is_recomputed_exactly(x101):
    sol = vrplib.read_solution(DATA / "X-n101-k25.sol")
    routes = [list(map(int, r)) for r in sol["routes"]]
    assert x101.n == 101 and x101.capacity == 206 and x101.bks == 27591
    assert check_routes(x101, routes) == []
    assert route_cost(x101.dist, routes) == 27591  # rounded Euclidean distances, as CVRPLIB


def test_check_routes_catches_every_kind_of_invalid_solution(x101):
    sol = vrplib.read_solution(DATA / "X-n101-k25.sol")
    routes = [list(map(int, r)) for r in sol["routes"]]
    assert "never visited" in check_routes(x101, routes[:-1])[0]
    assert "visited twice" in check_routes(x101, routes + [[routes[0][0]]])[-1]
    assert "capacity" in check_routes(x101, [routes[0] + routes[1]] + routes[2:])[0]
    assert "not a customer" in check_routes(x101, [[0]] + routes)[0]
    assert "not a customer" in check_routes(x101, [[101]] + routes)[0]
    assert "not a location number" in check_routes(x101, [[1.5]] + routes)[0]
    assert "not a location number" in check_routes(x101, [[True]] + routes)[0]
    assert check_routes(x101, "routes")
    assert check_routes(x101, [routes[0], "x"])


def test_savings_baseline_is_valid_and_reasonable(x101):
    routes = savings_routes(x101)
    assert check_routes(x101, routes) == []
    assert 0 < gap_percent(route_cost(x101.dist, routes), x101.bks) < 15


def test_synthetic_instances_are_reproducible(tmp_path):
    a = generate_synthetic("S-n21-s7", tmp_path / "a", ref_iterations=200)
    b = generate_synthetic("S-n21-s7", tmp_path / "b", ref_iterations=200)
    assert a.read_text() == b.read_text()
    assert a.with_suffix(".sol").read_text() == b.with_suffix(".sol").read_text()
    inst = load_instance(a)
    assert inst.n == 21 and inst.bks_kind == "reference" and inst.demand[0] == 0
    assert inst.demand[1:].min() >= 1 and inst.capacity >= inst.demand.max()


def test_synthetic_name_parsing():
    assert synthetic_spec("S-n41-s3") == (41, 3)
    with pytest.raises(ValueError):
        synthetic_spec("X-n101-k25")


def test_repository_splits_are_disjoint_and_complete():
    for dataset, sizes in {"cvrplib_x": (3, 20, 10, 20), "synthetic": (3, 12, 6, 10)}.items():
        splits = load_splits(dataset)
        assert tuple(len(splits[k]) for k in ("smoke", "train", "val", "test")) == sizes
        names = splits["train"] + splits["val"] + splits["test"]
        assert len(names) == len(set(names))


def test_x_split_covers_short_and_long_routes():
    splits = load_splits("cvrplib_x")
    for split in ("train", "val", "test"):
        lengths = [(int(n.split("-")[1][1:]) - 1) / int(n.split("-")[2][1:]) for n in splits[split]]
        assert min(lengths) < 5 and max(lengths) > 15


def test_overlapping_splits_are_refused(tmp_path):
    (tmp_path / "config" / "splits").mkdir(parents=True)
    (tmp_path / "config" / "splits" / "bad.yaml").write_text("train: [A]\nval: [B]\ntest: [A]\n")
    with pytest.raises(ValueError, match="appears in both"):
        load_splits("bad", tmp_path)


def test_settings_load():
    s = load_settings()
    assert s.llm.provider == "server"
    assert s.llm.thinking is False
    assert s.llm.models[0].base_url.startswith("http://127.0.0.1")
    assert set(s.tasks) == {"cvrp_construct", "pyvrp_accept"}
    assert np.isclose(s.llm.temperature, 0.7)


def test_settings_typos_are_errors():
    from pydantic import ValidationError

    from heuriforge.config import Settings
    from heuriforge.tasks import get_task

    with pytest.raises(ValidationError):
        Settings(run={"max_candidate": 3})
    with pytest.raises(ValueError, match="unknown setting"):
        get_task("pyvrp_accept").check_options({"iteration": 2000})
    s = load_settings()
    for name in ("cvrp_construct", "pyvrp_accept"):
        get_task(name).check_options(s.task_options(name))
