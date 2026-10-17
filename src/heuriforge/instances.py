"""Benchmark instances: download, generation, loading, and the trusted cost checker.

Every instance is a CVRP problem in VRPLIB format (``<name>.vrp``) with a best-known or
reference solution (``<name>.sol``). Locations are numbered 0..n-1 and location 0 is the
depot. Distances are Euclidean, rounded to the nearest integer: the CVRPLIB convention that
the published best-known solutions (BKS) use.

The functions ``check_routes`` and ``route_cost`` are the evaluator's ground truth. They run
in the trusted parent process and never use anything a candidate program returned except
the list of routes, which they check from scratch.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import vrplib

log = logging.getLogger(__name__)

X_URL = "https://raw.githubusercontent.com/PyVRP/Instances/main/CVRP/{name}.{ext}"
SYNTHETIC = re.compile(r"^S-n(\d+)-s(\d+)$")


@dataclass
class Instance:
    name: str
    path: Path
    coords: np.ndarray  # (n, 2) float
    demand: np.ndarray  # (n,) int, demand[0] = 0 (depot)
    capacity: int
    dist: np.ndarray  # (n, n) int64, rounded Euclidean
    bks: float | None  # best-known (or reference) cost
    bks_kind: str  # "bks" for published values, "reference" for our own long runs

    @property
    def n(self) -> int:
        return len(self.demand)

    @property
    def num_customers(self) -> int:
        return self.n - 1


def load_instance(path: str | Path) -> Instance:
    path = Path(path)
    raw = vrplib.read_instance(path)
    depot = int(np.asarray(raw["depot"]).ravel()[0])
    if depot != 0:
        raise ValueError(f"{path.name}: the depot must be the first location")
    coords = np.asarray(raw["node_coord"], dtype=float)
    dist = np.round(np.asarray(raw["edge_weight"], dtype=float)).astype(np.int64)
    demand = np.asarray(raw["demand"], dtype=np.int64)
    sol = path.with_suffix(".sol")
    bks = read_cost(sol) if sol.exists() else None
    kind = "reference" if SYNTHETIC.match(path.stem) else "bks"
    return Instance(path.stem, path, coords, demand, int(raw["capacity"]), dist, bks, kind)


def read_cost(sol_path: Path) -> float | None:
    for line in sol_path.read_text().splitlines()[::-1]:
        if line.strip().lower().startswith("cost"):
            return float(line.split()[-1])
    return None


# --------------------------------------------------------------------------- checking
def check_routes(inst: Instance, routes: object) -> list[str]:
    """Problems with a proposed solution; an empty list means it is valid.

    A valid solution visits every customer exactly once, never visits the depot in the
    middle of a route, and never loads a vehicle beyond its capacity.
    """
    if not isinstance(routes, list) or not all(isinstance(r, list) for r in routes):
        return ["routes must be a list of lists of location numbers"]
    seen = np.zeros(inst.n, dtype=np.int64)
    problems: list[str] = []
    for k, route in enumerate(routes):
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in route):
            return [f"route {k} contains something that is not a location number"]
        for v in route:
            if not 1 <= v < inst.n:
                return [f"route {k} visits location {v}, which is not a customer"]
            seen[v] += 1
        load = int(inst.demand[route].sum()) if route else 0
        if load > inst.capacity:
            problems.append(f"route {k} carries {load} > capacity {inst.capacity}")
    missing = np.flatnonzero(seen[1:] == 0) + 1
    repeated = np.flatnonzero(seen[1:] > 1) + 1
    if missing.size:
        problems.append(f"{missing.size} customers are never visited (e.g. {missing[:5].tolist()})")
    if repeated.size:
        problems.append(
            f"{repeated.size} customers are visited twice (e.g. {repeated[:5].tolist()})"
        )
    return problems


def route_cost(dist: np.ndarray, routes: list[list[int]]) -> int:
    total = 0
    for route in routes:
        if route:
            path = [0, *route, 0]
            total += int(dist[path[:-1], path[1:]].sum())
    return total


def gap_percent(cost: float, bks: float) -> float:
    return 100.0 * (cost - bks) / bks


# --------------------------------------------------------------------------- writing
def write_vrp(
    path: Path, name: str, coords: np.ndarray, demand: np.ndarray, capacity: int, comment: str
) -> None:
    lines = [
        f"NAME : {name}",
        f"COMMENT : {comment}",
        "TYPE : CVRP",
        f"DIMENSION : {len(demand)}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        f"CAPACITY : {capacity}",
        "NODE_COORD_SECTION",
        *(f"{i + 1}\t{int(x)}\t{int(y)}" for i, (x, y) in enumerate(coords)),
        "DEMAND_SECTION",
        *(f"{i + 1}\t{int(d)}" for i, d in enumerate(demand)),
        "DEPOT_SECTION",
        "1",
        "-1",
        "EOF",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_sol(path: Path, routes: list[list[int]], cost: int) -> None:
    lines = [f"Route #{k + 1}: " + " ".join(map(str, r)) for k, r in enumerate(routes) if r]
    path.write_text("\n".join([*lines, f"Cost {cost}"]) + "\n")


# --------------------------------------------------------------------------- synthetic
def synthetic_spec(name: str) -> tuple[int, int]:
    m = SYNTHETIC.match(name)
    if not m:
        raise ValueError(f"{name!r} is not a synthetic instance name (expected S-n<N>-s<seed>)")
    return int(m.group(1)), int(m.group(2))


def generate_synthetic(name: str, out_dir: Path, ref_iterations: int = 3000) -> Path:
    """A random CVRP instance plus a reference solution from a long PyVRP run.

    Customers are uniform on a 1000 x 1000 grid, demands are 1-10, and the capacity
    allows about one route per eight customers.
    """
    from pyvrp import read as pyvrp_read
    from pyvrp import solve
    from pyvrp.stop import MaxIterations

    n, seed = synthetic_spec(name)
    if n < 3:
        raise ValueError("a synthetic instance needs at least two customers")
    rng = np.random.default_rng(seed)
    coords = rng.integers(0, 1001, size=(n, 2))
    demand = rng.integers(1, 11, size=n)
    demand[0] = 0
    routes_target = max(2, round((n - 1) / 8))
    capacity = max(int(demand.max()), math.ceil(demand.sum() / routes_target))
    out_dir.mkdir(parents=True, exist_ok=True)
    vrp = out_dir / f"{name}.vrp"
    write_vrp(vrp, name, coords, demand, capacity, f"synthetic HeuriForge instance, seed {seed}")

    data = pyvrp_read(vrp, round_func="round")
    result = solve(data, stop=MaxIterations(ref_iterations), seed=0, collect_stats=False)
    routes = solution_routes(result.best, data)
    inst = load_instance(vrp)
    problems = check_routes(inst, routes)
    if problems:
        raise RuntimeError(f"reference solution for {name} is invalid: {problems}")
    write_sol(vrp.with_suffix(".sol"), routes, route_cost(inst.dist, routes))
    return vrp


def solution_routes(solution, data) -> list[list[int]]:
    """Routes of a PyVRP solution as location numbers (depot = 0)."""
    offset = data.num_depots
    return [[a.idx + offset for a in route if a.is_client()] for route in solution.routes()]


# --------------------------------------------------------------------------- downloads
def download_x(name: str, out_dir: Path, client: httpx.Client | None = None) -> Path:
    """Download a CVRPLIB X instance and its best-known solution from PyVRP/Instances."""
    out_dir.mkdir(parents=True, exist_ok=True)
    own = client is None
    client = client or httpx.Client(timeout=60, follow_redirects=True)
    try:
        for ext in ("vrp", "sol"):
            target = out_dir / f"{name}.{ext}"
            if target.exists() and target.stat().st_size > 0:
                continue
            for attempt in range(4):
                try:
                    resp = client.get(X_URL.format(name=name, ext=ext))
                    resp.raise_for_status()
                    target.write_bytes(resp.content)
                    break
                except httpx.HTTPError as exc:
                    if attempt == 3:
                        raise RuntimeError(f"could not download {name}.{ext}: {exc}") from exc
                    time.sleep(2 * (attempt + 1))
    finally:
        if own:
            client.close()
    inst = load_instance(out_dir / f"{name}.vrp")
    sol = vrplib.read_solution(out_dir / f"{name}.sol")
    cost = route_cost(inst.dist, [list(map(int, r)) for r in sol["routes"]])
    if inst.bks is not None and abs(cost - inst.bks) > 0.5:
        log.warning("%s: recomputed BKS %s differs from the file's %s", name, cost, inst.bks)
    return out_dir / f"{name}.vrp"


def prepare_dataset(
    data_dir: Path, dataset: str, splits: dict[str, list[str]], ref_iterations: int = 3000
) -> Path:
    """Make sure every instance named in the splits exists; write an index file."""
    out = data_dir / dataset
    out.mkdir(parents=True, exist_ok=True)
    names = [n for split in ("smoke", "train", "val", "test") for n in splits[split]]
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for name in dict.fromkeys(names):
            vrp = out / f"{name}.vrp"
            if vrp.exists() and vrp.with_suffix(".sol").exists():
                continue
            if SYNTHETIC.match(name):
                log.info("generating %s", name)
                generate_synthetic(name, out, ref_iterations)
            else:
                log.info("downloading %s", name)
                download_x(name, out, client)
    index = []
    for split in ("smoke", "train", "val", "test"):
        for name in splits[split]:
            inst = load_instance(out / f"{name}.vrp")
            index.append(
                {"name": name, "split": split, "n": inst.n, "bks": inst.bks, "kind": inst.bks_kind}
            )
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return out


def load_split(data_dir: Path, dataset: str, names: list[str]) -> list[Instance]:
    out = data_dir / dataset
    missing = [
        n for n in names if not (out / f"{n}.vrp").exists() or not (out / f"{n}.sol").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} instances of dataset {dataset!r} are missing (e.g. {missing[:3]}). "
            f"Run: heuriforge data --dataset {dataset}"
        )
    return [load_instance(out / f"{n}.vrp") for n in names]
