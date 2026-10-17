"""Evaluate a program on one split: run it in the sandbox, then check and score the routes.

The sandbox only returns routes. This trusted code verifies them (every customer once,
no overloaded truck) and recomputes their cost from the instance file, so a candidate
cannot report a better score than it earned.
"""

from __future__ import annotations

import asyncio
import time
from statistics import fmean

from heuriforge.instances import Instance, check_routes, gap_percent, load_split, route_cost
from heuriforge.sandbox.runner import Sandbox
from heuriforge.tasks.base import Item, StageResult, Task

# Error types that are the evaluator's problem, not the candidate's.
SETUP_ERRORS = {"setup_error"}


class SetupError(RuntimeError):
    """The evaluation could not run for reasons unrelated to the candidate (e.g. missing data)."""


class Evaluator:
    def __init__(
        self,
        task: Task,
        options: dict,
        data_dir,
        dataset: str,
        splits: dict[str, list[str]],
        sandbox: Sandbox,
    ):
        self.task = task
        self.options = options
        self.data_dir = data_dir
        self.dataset = dataset
        self.splits = splits
        self.sandbox = sandbox
        self._instances: dict[str, list[Instance]] = {}

    def instances(self, split: str) -> list[Instance]:
        if split not in self._instances:
            self._instances[split] = load_split(self.data_dir, self.dataset, self.splits[split])
        return self._instances[split]

    def items(self, split: str) -> list[Item]:
        return self.task.items(self.instances(split), split, self.options)

    async def stage(self, program: str, split: str, repeat: int = 1) -> StageResult:
        """Run the program on every item of a split. Stops early at the first failure."""
        t0 = time.perf_counter()
        items = self.items(split)
        if not items:
            return StageResult(
                split, False, error_type="setup_error", error=f"the {split} split is empty"
            )
        cpu_limit = self.task.cpu_limit(split, self.options)
        n_chunks = min(self.sandbox.workers, len(items))
        # Interleave so that every chunk gets a mix of small and large instances.
        chunks = [items[k::n_chunks] for k in range(n_chunks)]
        job_options = self.task.job_options(split, self.options)
        jobs = [
            {
                "task": self.task.name,
                "program": program,
                "items": [
                    {"key": it.key, "instance": str(it.instance.path), "seed": it.seed}
                    for it in chunk
                ],
                "options": job_options,
                "cpu_s_per_item": cpu_limit,
                "repeat": repeat,
            }
            for chunk in chunks
        ]
        walls = [cpu_limit * (len(chunk) * repeat + 1) * 1.5 + 30 for chunk in chunks]
        pending = {
            asyncio.create_task(self.sandbox.run(job, wall))
            for job, wall in zip(jobs, walls, strict=True)
        }
        outputs: list[dict] = []
        failure: dict | None = None
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for fut in done:
                    out = fut.result()
                    outputs.append(out)
                    if not out.get("ok") and failure is None:
                        failure = out
                if failure is not None:
                    break  # no need to wait for the other chunks
        finally:
            for fut in pending:
                fut.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        seconds = time.perf_counter() - t0
        if failure is not None:
            error_type, error = first_error(failure)
            if error_type in SETUP_ERRORS:
                raise SetupError(error)
            return StageResult(split, False, seconds=seconds, error_type=error_type, error=error)

        by_key = {it.key: it for it in items}
        result = StageResult(split, True, seconds=seconds)
        for out in outputs:
            for row in out.get("items", []):
                item = by_key.get(row.get("key"))
                if item is None:
                    return StageResult(
                        split, False, seconds=seconds, error_type="crash", error="unexpected result"
                    )
                problems = check_routes(item.instance, row.get("routes"))
                if problems:
                    result.ok = False
                    result.error_type = "invalid_output"
                    result.error = f"{item.instance.name}: {problems[0]}"
                    return result
                cost = route_cost(item.instance.dist, row["routes"])
                result.costs[item.key] = cost
                result.per_item[item.key] = gap_percent(cost, item.instance.bks)
                result.cpu_s += float(row.get("cpu_s", 0.0))
        missing = set(by_key) - set(result.per_item)
        if missing:
            result.ok = False
            result.error_type = "crash"
            result.error = f"no result for {len(missing)} items"
            return result
        result.per_item = {k: result.per_item[k] for k in by_key}  # stable order
        result.costs = {k: result.costs[k] for k in by_key}
        result.score = fmean(result.per_item.values())
        return result


def first_error(out: dict) -> tuple[str, str]:
    """The most informative error in a failed job."""
    for row in out.get("items", []):
        if not row.get("ok") and row.get("error_type") not in (None, "skipped"):
            return row["error_type"], row.get("error", "")
    return out.get("error_type") or "crash", out.get("error") or "unknown error"
