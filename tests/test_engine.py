"""End to end: a small evolution with the offline mock model, resumed, tested and reported."""

import asyncio
import json

import pytest

from heuriforge.database import Database
from heuriforge.engine import Engine, champion
from heuriforge.report import build_site
from heuriforge.sealed_test import AlreadyTested, run_sealed_test
from heuriforge.tasks import get_task

pytestmark = pytest.mark.slow


def evolve(settings, path, provider, n, resume=False, task="cvrp_construct"):
    settings.evolution.migration_interval = 4
    settings.evolution.reflection_interval = 5
    engine = Engine(
        settings, get_task(task), "tiny", path, provider=provider, resume=resume, max_candidates=n
    )
    return asyncio.run(engine.run())


def test_evolution_resume_sealed_test_and_report(settings, tmp_path):
    run = tmp_path / "runs" / "mock"
    summary = evolve(settings, run, "mock", 10)
    assert summary["candidates"] == 10 and summary["stop_reason"].startswith(
        "reached max_candidates"
    )
    assert (run / "best_program.py").exists() and (run / "run.yaml").exists()

    with pytest.raises(FileExistsError):  # a finished run is never overwritten by accident
        evolve(settings, run, "mock", 12)
    summary = evolve(settings, run, "mock", 14, resume=True)
    assert summary["candidates"] == 14

    db = Database(run / "programs.sqlite", readonly=True)
    programs = db.programs()
    attempts = sorted(p.attempt for p in programs if p.origin == "llm")
    assert attempts == list(range(1, 15))  # every attempt is recorded exactly once
    assert {p.origin for p in programs} >= {"initial", "llm"}
    for p in programs:
        if p.status == "evaluated":
            assert p.train_score is not None and p.code_hash and p.cell is not None
        else:
            assert p.error  # every failure says why
    best = champion(db)
    assert best is not None and best.val_score is not None
    assert len(db.cells()) >= 2  # both islands hold the initial program at least
    assert db.lessons()  # reflection ran with the mock model
    db.close()

    report = asyncio.run(run_sealed_test(settings, run))
    assert set(report["mean_gap"]) == {"champion", "initial", "savings"}
    assert report["comparisons"]["initial"]["p_holm"] <= 1.0
    with pytest.raises(AlreadyTested):
        asyncio.run(run_sealed_test(settings, run))
    again = asyncio.run(run_sealed_test(settings, run, force=True))
    assert again["forced"] is True and again["previous_runs"] == 1

    baseline = tmp_path / "runs" / "random"
    evolve(settings, baseline, "random", 6)
    page = build_site([run, baseline], tmp_path / "site")
    text = page.read_text()
    assert (
        "Offline demo" in text
        and "Run <code>mock</code>" in text
        and "Run <code>random</code>" in text
    )
    assert "cdn.plot.ly" in text and (tmp_path / "site" / ".nojekyll").exists()
    assert json.loads((run / "summary.json").read_text())["candidates"] == 14


def test_task2_runs_end_to_end(settings, tmp_path):
    summary = evolve(settings, tmp_path / "accept", "random", 4, task="pyvrp_accept")
    assert summary["candidates"] == 4 and summary["initial_train_gap"] is not None


def test_stop_file_ends_a_run(settings, tmp_path):
    run = tmp_path / "stopped"
    run.mkdir()
    (run / "STOP").write_text("")
    summary = evolve(settings, run, "random", 50)
    assert summary["candidates"] == 0 and "STOP" in summary["stop_reason"]


def test_missing_server_stops_cleanly(settings, tmp_path):
    from heuriforge.llm import LLMError

    settings.llm.models[0].base_url = "http://127.0.0.1:9/v1"
    with pytest.raises(LLMError, match="make server"):
        evolve(settings, tmp_path / "noserver", "server", 5)
