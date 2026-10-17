"""Repeated runs: seeds, the sealed test of every finished run, results pooled over runs."""

import json
import shutil

import pytest

from heuriforge.cli import main
from heuriforge.database import Database
from heuriforge.results import SealedRun, format_spread, markdown_table, pool


@pytest.fixture
def fresh(project, tmp_path, monkeypatch):
    """A project of its own (the shared one keeps the runs of other tests)."""
    root = tmp_path / "project"
    shutil.copytree(project / "config", root / "config")
    shutil.copytree(project / "data", root / "data")
    monkeypatch.setenv("HEURIFORGE_ROOT", str(root))
    return root


def evolve(name, *extra):
    main(["evolve", "--task", "cvrp_construct", "--dataset", "tiny", "--name", name, *extra])


def seed(root, name):
    db = Database(root / "runs" / name / "programs.sqlite", readonly=True)
    try:
        return db.get_meta("settings")["run"]["seed"]
    finally:
        db.close()


def test_a_run_keeps_its_seed_when_resumed(fresh, capsys):
    evolve("r-s3", "--provider", "random", "--max-candidates", "2", "--seed", "3")
    assert seed(fresh, "r-s3") == 3
    evolve("r-s3", "--provider", "random", "--max-candidates", "4", "--resume")  # no --seed
    assert (
        "resuming with this run's own seed (3)" in (fresh / "runs" / "r-s3" / "log.txt").read_text()
    )
    main(["status", "--name", "r-s3"])
    out = capsys.readouterr().out
    assert "seed 3" in out and "Candidates: 4" in out
    # Resuming a finished run (what `make repeats-*` does after an interruption) adds nothing.
    evolve("r-s3", "--provider", "random", "--max-candidates", "4", "--resume")
    main(["status", "--name", "r-s3"])
    assert "Candidates: 4" in capsys.readouterr().out


def test_sealed_tests_of_every_finished_run_then_the_pooled_table(fresh, capsys):
    evolve("construct-llm", "--provider", "mock", "--max-candidates", "3")
    evolve("construct-random", "--provider", "random", "--max-candidates", "3")
    evolve("construct-random-s1", "--provider", "random", "--max-candidates", "3", "--seed", "1")
    evolve("unfinished", "--provider", "random", "--max-candidates", "3", "--max-hours", "0")
    capsys.readouterr()
    main(["test", "--all"])
    out = capsys.readouterr().out
    assert "unfinished: not finished" in out
    assert out.count("Sealed test of run") == 3
    main(["test", "--all"])
    assert "construct-llm: sealed test already done, skipped" in capsys.readouterr().out

    main(["table"])
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("| 1 · Constructive"))
    assert "(1 run) |" in row and "(2 runs) |" in row and " pts [" in row
    assert "| `construct-random-s1` | 1 |" in out
    assert "Clarke-Wright savings" in out and "| 2 · PyVRP acceptance rule | – |" in out

    main(["export", "--all", "--out", str(fresh / "results")])
    assert (fresh / "results" / "construct-random-s1" / "sealed_test.json").exists()
    capsys.readouterr()
    main(["table", "--from", str(fresh / "results")])  # the exported runs give the same table
    assert out.split("| Run |")[0] == capsys.readouterr().out.split("| Run |")[0]


def sealed_run(name, kind, gaps, seed=0, errors=None):
    report = {
        "mean_gap": {"champion": sum(gaps.values()) / len(gaps), "initial": 30.0} if gaps else {},
        "per_item": {"champion": gaps} if gaps else {},
        "errors": errors or {},
    }
    return SealedRun(name, "cvrp_construct", kind, seed, report)


def test_pooling_averages_each_item_over_the_runs():
    runs = [
        sealed_run("llm", "llm", {"a": 10.0, "b": 12.0}),
        sealed_run("llm-s1", "llm", {"a": 12.0, "b": 14.0}, seed=1),
        sealed_run("random", "random", {"a": 11.0, "b": 13.0}),
        sealed_run("llm-s2", "llm", {}, seed=2, errors={"champion": "timeout: too slow"}),
    ]
    p = pool(runs, "cvrp_construct")
    assert [r.name for r in p.failed] == ["llm-s2"]
    assert p.items == 2 and p.comparison["mean_diff"] == pytest.approx(0.0)
    assert format_spread(p.llm) == "12.00% ± 1.41 (2 runs)"
    assert format_spread(p.base) == "12.00% (1 run)"
    table = markdown_table(runs)
    assert (
        "Run `llm-s2`: the champion failed on the test split (champion: timeout: too slow)."
        in table
    )
    assert "| `llm-s2` | 2 | failed |" in table
    json.dumps(p.comparison)  # plain numbers, ready for the report
