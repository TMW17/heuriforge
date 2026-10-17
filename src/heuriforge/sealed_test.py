"""The sealed test: score the champion once, on instances evolution never saw.

The champion is the program with the best *validation* score. It is compared with the
initial program and with a classic baseline on the *test* split, instance by instance
(and seed by seed for Task 2), with paired statistics. The result is written to
``runs/<name>/sealed_test.json`` and logged in the run's database, and a second run is
refused unless ``--force`` is given (the forced re-run is recorded as such).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from heuriforge.codeblock import build_program
from heuriforge.config import Settings, load_splits
from heuriforge.database import Database
from heuriforge.engine import champion
from heuriforge.evaluator import Evaluator
from heuriforge.instances import gap_percent, route_cost
from heuriforge.sandbox.runner import Sandbox
from heuriforge.stats import compare, holm
from heuriforge.tasks import get_task

log = logging.getLogger("heuriforge")


class AlreadyTested(RuntimeError):
    pass


def split_fingerprint(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]


async def run_sealed_test(settings: Settings, run_dir: Path, force: bool = False) -> dict:
    db = Database(run_dir / "programs.sqlite")
    task = get_task(db.get_meta("task"))
    dataset = db.get_meta("dataset")
    previous = db.events("sealed_test")
    if previous and not force:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(previous[-1]["created_at"]))
        raise AlreadyTested(
            f"the sealed test of this run already ran ({when}); see {run_dir / 'sealed_test.json'}. "
            "Running it again would turn the test set into a second validation set. If you really "
            "must (for example after fixing a bug), use --force: the re-run is recorded."
        )
    best = champion(db)
    if best is None:
        raise RuntimeError("this run has no validated program yet")
    initial = db.programs("origin = 'initial'")[0]

    splits = load_splits(dataset, settings.root)
    options = task.check_options(settings.task_options(task.name))
    sandbox = Sandbox(settings.eval.workers, settings.eval.memory_mb)
    evaluator = Evaluator(task, options, settings.paths.data, dataset, splits, sandbox)

    contenders = {"champion": best.code, "initial": initial.code}
    gaps: dict[str, dict[str, float]] = {}
    errors: dict[str, str] = {}
    for name, code in contenders.items():
        log.info(
            "sealed test: running the %s program on %d test items",
            name,
            len(evaluator.items("test")),
        )
        result = await evaluator.stage(build_program(task.header, code), "test")
        if result.ok:
            gaps[name] = result.per_item
        else:
            errors[name] = f"{result.error_type}: {result.error}"
    for name, per_item in baselines(task.name, evaluator).items():
        gaps[name] = per_item

    report = {
        "task": task.name,
        "dataset": dataset,
        "run": run_dir.name,
        "provider": db.get_meta("provider"),
        "created_at": time.time(),
        "forced": bool(previous),
        "previous_runs": len(previous),
        "test_split_fingerprint": split_fingerprint(splits["test"]),
        "items": len(evaluator.items("test")),
        "champion": {
            "id": best.id,
            "train_gap": best.train_score,
            "val_gap": best.val_score,
            "idea": best.idea,
            "code": best.code,
        },
        "mean_gap": {name: sum(v.values()) / len(v) for name, v in gaps.items()},
        "per_item": gaps,
        "errors": errors,
        "comparisons": {},
    }
    if "champion" in gaps:
        others = [name for name in gaps if name != "champion"]
        keys = list(gaps["champion"])
        rows = {
            name: compare([gaps["champion"][k] for k in keys], [gaps[name][k] for k in keys])
            for name in others
        }
        adjusted = holm([rows[name]["p_value"] for name in others])
        for name, p in zip(others, adjusted, strict=True):
            rows[name]["p_holm"] = p
        report["comparisons"] = rows

    (run_dir / "sealed_test.json").write_text(json.dumps(report, indent=2))
    db.log_event(
        "sealed_test",
        {
            "champion": best.id,
            "mean_gap": report["mean_gap"],
            "forced": report["forced"],
            "fingerprint": report["test_split_fingerprint"],
        },
    )
    db.close()
    return report


def baselines(task_name: str, evaluator: Evaluator) -> dict[str, dict[str, float]]:
    """Trusted reference methods, run in this process (no sandbox needed)."""
    if task_name != "cvrp_construct":
        return {}  # Task 2's baseline *is* the initial program: PyVRP's own rule
    from heuriforge.tasks.cvrp_construct import savings_routes

    out = {}
    for item in evaluator.items("test"):
        routes = savings_routes(item.instance)
        out[item.key] = gap_percent(route_cost(item.instance.dist, routes), item.instance.bks)
    return {"savings": out}
