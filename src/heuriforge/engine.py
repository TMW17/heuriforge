"""The evolution loop: propose, check, evaluate, keep the good ones, repeat.

One *attempt* (= one candidate):

1. pick an island (round robin) and a parent from its archive;
2. pick an arm (model x mutation type) with the bandit;
3. ask the model for a change (or let the random mutator make one);
4. reject unsafe code (static checks) and repeats (novelty checks) without running them;
5. run the smoke test (tiny instances, twice, to catch crashes and non-determinism);
6. score it on the training instances;
7. offer it to the archive; if it is the island's new best, also score it on the
   validation instances (the champion is chosen on validation, never on training);
8. tell the bandit whether the child beat its parent, and save everything.

Every ``migration_interval`` attempts the islands pass their best program to the next
island; every ``reflection_interval`` attempts the model writes a one-line lesson that
later prompts include. ``parallel`` attempts run at the same time, so the model server
and the sandbox stay busy together.
"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from heuriforge import __version__
from heuriforge.archive import Archive, better
from heuriforge.bandit import Bandit, arm_key
from heuriforge.codeblock import EditError, build_program, complexity, params_used
from heuriforge.config import Settings, load_splits
from heuriforge.database import Database, Program
from heuriforge.evaluator import Evaluator
from heuriforge.llm import LLMClient, LLMError, mock_transport
from heuriforge.mutators import RandomMutator, child_from_answer
from heuriforge.novelty import code_hash, structure_hash
from heuriforge.prompts import SCHEMAS, lesson_messages, mutation_messages
from heuriforge.sandbox.runner import Sandbox
from heuriforge.sandbox.static import check_code
from heuriforge.tasks.base import Task

log = logging.getLogger("heuriforge")


@dataclass
class RunPaths:
    root: Path

    @property
    def db(self) -> Path:
        return self.root / "programs.sqlite"

    @property
    def stop(self) -> Path:
        return self.root / "STOP"

    @property
    def best(self) -> Path:
        return self.root / "best_program.py"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"


def champion(db: Database) -> Program | None:
    """The program with the best validation score (ties: better training score)."""
    validated = [p for p in db.evaluated() if p.val_score is not None]
    if not validated:
        return None
    return min(validated, key=lambda p: (p.val_score, p.train_score, p.complexity, p.id))


class Engine:
    def __init__(
        self,
        settings: Settings,
        task: Task,
        dataset: str,
        run_dir: Path,
        provider: str | None = None,
        transport=None,
        resume: bool = False,
        max_candidates: int | None = None,
        max_hours: float | None = None,
        tracker=None,
    ):
        self.settings = settings
        self.task = task
        self.dataset = dataset
        self.paths = RunPaths(Path(run_dir))
        self.provider = provider or settings.llm.provider
        self.transport = transport
        self.resume = resume
        self.max_candidates = max_candidates or settings.run.max_candidates
        self.max_hours = max_hours if max_hours is not None else settings.run.max_hours
        self.tracker = tracker
        self.options = task.check_options(settings.task_options(task.name))
        evo = settings.evolution
        self.archive = Archive(
            evo.islands,
            self.options.get("complexity_bins", task.complexity_bins),
            self.options.get("params_bins", task.params_bins),
        )
        self.rng = np.random.default_rng(settings.run.seed)
        self.stop_reason: str | None = None
        self.last_error: dict[int, str] = {}
        self.completed = 0

    # ----------------------------------------------------------------- set-up
    def _open(self) -> None:
        exists = self.paths.db.exists()
        if exists and not self.resume:
            raise FileExistsError(
                f"{self.paths.root} already has a run. Continue it with --resume, or choose another --name."
            )
        if self.resume and not exists:
            raise FileNotFoundError(f"there is no run to resume in {self.paths.root}")
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.paths.db)
        if exists:
            task, dataset = self.db.get_meta("task"), self.db.get_meta("dataset")
            if (task, dataset) != (self.task.name, self.dataset):
                raise ValueError(
                    f"this run is for task {task} on {dataset}, not {self.task.name} on {self.dataset}"
                )
            provider = self.db.get_meta("provider")
            if provider and provider != self.provider:
                raise ValueError(
                    f"this run used provider {provider!r}; resume it with the same provider"
                )
            # A resumed run keeps its own search seed, whatever the command line says.
            seed = ((self.db.get_meta("settings") or {}).get("run") or {}).get("seed")
            if seed is not None and seed != self.settings.run.seed:
                log.info("resuming with this run's own seed (%s)", seed)
                self.settings.run.seed = seed
        else:
            self.db.set_meta("task", self.task.name)
            self.db.set_meta("dataset", self.dataset)
            self.db.set_meta("provider", self.provider)
            self.db.set_meta("started_at", time.time())
            self.db.set_meta("settings", self.settings.model_dump(mode="json"))
            self._write_run_file()
        self.db.log_event(
            "session_start", {"resume": self.resume, "max_candidates": self.max_candidates}
        )

        self.splits = load_splits(self.dataset, self.settings.root)
        self.sandbox = Sandbox(self.settings.eval.workers, self.settings.eval.memory_mb)
        self.evaluator = Evaluator(
            self.task,
            self.options,
            self.settings.paths.data,
            self.dataset,
            self.splits,
            self.sandbox,
        )
        if self.provider == "random":
            arms = {arm_key("random", "random"): 1.0}
        elif self.provider == "mock":
            arms = {arm_key("mock", mut): 1.0 for mut in self.settings.llm.mutations}
        else:
            arms = {
                arm_key(m.name, mut): m.cost
                for m in self.settings.llm.models
                for mut in self.settings.llm.mutations
            }
        self.bandit = Bandit(arms, self.db)
        # A resumed run continues with a fresh random stream instead of replaying the first one.
        done = self.db.attempts()
        self.rng = np.random.default_rng([self.settings.run.seed, done])
        self.mutator = RandomMutator(self.task, self.rng)
        if self.provider == "mock" and self.transport is None:
            self.transport = mock_transport(self.task, seed=self.settings.run.seed + done)
        self.llm = (
            LLMClient(self.settings.llm, self.transport) if self.provider != "random" else None
        )

        self.seen_hashes, self.structure_counts = self.db.hash_counts()
        for island, cell, program_id in self.db.cells():
            p = self.db.get(program_id)
            if p is not None:
                self.archive.put(island, cell, p)
        self.next_attempt = self.db.attempts()

    def _write_run_file(self) -> None:
        info = {
            "task": self.task.name,
            "dataset": self.dataset,
            "provider": self.provider,
            "heuriforge": __version__,
            "python": platform.python_version(),
            "machine": f"{platform.system()} {platform.machine()}",
            "settings": self.settings.model_dump(mode="json", exclude={"root", "paths"}),
        }
        (self.paths.root / "run.yaml").write_text(yaml.safe_dump(info, sort_keys=False))

    async def _check_server(self) -> None:
        if self.llm is None:
            return
        for endpoint in self.settings.llm.models:
            if not await self.llm.health(endpoint):
                raise LLMError(
                    f"no model server answers at {endpoint.base_url}. Start it first (make server), "
                    "or run the no-LLM baseline with --provider random.",
                    fatal=True,
                )

    async def _seed_archive(self) -> None:
        """Evaluate the initial program and place it on every island."""
        if self.db.evaluated():
            return
        log.info("evaluating the initial program (smoke, train, validation)")
        block = self.task.initial_block.strip("\n")
        program = build_program(self.task.header, block)
        smoke = await self.evaluator.stage(program, "smoke", repeat=2)
        train = await self.evaluator.stage(program, "train") if smoke.ok else smoke
        val = await self.evaluator.stage(program, "val") if train.ok else train
        if not val.ok:
            raise RuntimeError(
                f"the initial program failed ({val.stage}): {val.error_type}: {val.error}"
            )
        for island in range(self.archive.islands):
            p = Program(
                island=island,
                code=block,
                origin="initial",
                status="evaluated",
                code_hash=code_hash(block),
                structure_hash=structure_hash(block),
                smoke_score=smoke.score,
                train_score=train.score,
                val_score=val.score,
                train_details=train.per_item,
                val_details=val.per_item,
                eval_seconds=smoke.seconds + train.seconds + val.seconds,
                complexity=complexity(block),
                params_used=params_used(block, self.task.entry, self.task.params),
            )
            self.db.insert(p)
            self.archive.add(p)
            self.db.save_cell(p.island, p.cell, p.id)
            self.db.update(p)
        self.seen_hashes.add(code_hash(block))
        log.info("initial program: train gap %.2f%%, validation gap %.2f%%", train.score, val.score)

    # ----------------------------------------------------------------- main loop
    async def run(self) -> dict:
        self._open()
        self.session_start = time.time()
        try:
            await self._check_server()
            await self._seed_archive()
            workers = [
                asyncio.create_task(self._worker())
                for _ in range(max(1, self.settings.run.parallel))
            ]
            try:
                await asyncio.gather(*workers)
            finally:
                for w in workers:
                    w.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
        finally:
            summary = self.finish()
            if self.llm is not None:
                await self.llm.close()
            self.db.close()
        return summary

    def _should_stop(self) -> str | None:
        if self.stop_reason:
            return self.stop_reason
        if self.next_attempt >= self.max_candidates:
            return f"reached max_candidates ({self.max_candidates})"
        if (time.time() - self.session_start) / 3600 >= self.max_hours:
            return f"reached max_hours ({self.max_hours})"
        if self.paths.stop.exists():
            return "found the STOP file"
        return None

    async def _worker(self) -> None:
        while True:
            reason = self._should_stop()
            if reason:
                self.stop_reason = self.stop_reason or reason
                return
            self.next_attempt += 1
            await self.attempt(self.next_attempt)
            self.completed += 1
            await self._periodic()

    async def _periodic(self) -> None:
        evo = self.settings.evolution
        done = self.db.attempts()
        if (
            evo.islands > 1
            and evo.migration_interval > 0
            and self.completed % evo.migration_interval == 0
        ):
            self.migrate()
        if (
            self.llm is not None
            and evo.reflection_interval > 0
            and self.completed % evo.reflection_interval == 0
        ):
            await self.reflect(island=done % self.archive.islands)

    # ----------------------------------------------------------------- one attempt
    async def attempt(self, attempt: int) -> Program:
        evo = self.settings.evolution
        island = (attempt - 1) % self.archive.islands
        parent = self.archive.sample_parent(
            island, self.rng, evo.exploit_prob, evo.selection_lambda
        )
        self.db.add_child(parent.id)
        parent.children += 1
        arm = self.bandit.choose(self.rng)
        model, mutation = arm.split("|", 1)
        child = Program(
            island=island,
            code=None,
            parent_id=parent.id,
            generation=parent.generation + 1,
            origin="random" if self.provider == "random" else "llm",
            attempt=attempt,
            model=model,
            mutation=mutation,
        )
        t0 = time.perf_counter()
        try:
            await self._propose(child, parent, mutation)
            if child.status == "new":
                await self._check_and_evaluate(child, parent)
        except LLMError as exc:
            child.status, child.error_type, child.error = "llm_error", "llm_error", str(exc)
            if exc.fatal:
                self.stop_reason = "the model server cannot be reached"
        success = child.status == "evaluated" and child.train_score < parent.train_score - 1e-9
        self.bandit.update(arm, success)
        if child.status == "evaluated":
            self.last_error.pop(parent.id, None)
        elif child.status != "llm_error":  # tell the model next time what went wrong
            self.last_error[parent.id] = f"{child.error_type}: {child.error}"[:500]
        new_best = self._save(child)
        if new_best:
            await self._validate(child)
        self._log(child, parent, time.perf_counter() - t0, new_best)
        if self.tracker is not None:
            self.tracker.attempt(child, self.archive.best())
        return child

    async def _propose(self, child: Program, parent: Program, mutation: str) -> None:
        if self.llm is None:
            child.code, child.idea = self.mutator.mutate(parent.code)
            return
        if self.provider == "mock":
            endpoint = self.settings.llm.models[0]
        else:
            endpoint = self.llm.endpoint(child.model)
        inspirations = self.archive.inspirations(
            child.island, parent, self.settings.evolution.inspirations, self.rng
        )
        lessons = self.db.lessons(self.settings.evolution.max_lessons_in_prompt)
        messages = mutation_messages(
            self.task, parent, mutation, inspirations, lessons, self.last_error.get(parent.id)
        )
        seed = self.settings.run.seed * 1_000_003 + child.attempt
        reply = await self.llm.complete(endpoint, messages, mutation, SCHEMAS[mutation], seed=seed)
        child.prompt_tokens = reply.prompt_tokens
        child.completion_tokens = reply.completion_tokens
        child.cached_tokens = reply.cached_tokens
        child.llm_seconds = reply.seconds
        try:
            child.code, child.idea = child_from_answer(parent.code, mutation, reply.data)
        except EditError as exc:
            child.code = None
            child.idea = " ".join(str(reply.data.get("idea", "")).split())[:300]
            child.status, child.error_type, child.error = "edit_failed", "edit_failed", str(exc)

    async def _check_and_evaluate(self, child: Program, parent: Program) -> None:
        task = self.task
        report = check_code(child.code, task.entry, task.params)
        if not report.ok:
            child.status, child.error_type = "rejected_static", report.category
            line = f" (line {report.line})" if report.line else ""
            child.error = f"{report.message}{line}"
            return
        child.code_hash = code_hash(child.code)
        child.structure_hash = structure_hash(child.code)
        child.complexity = complexity(child.code)
        child.params_used = params_used(child.code, task.entry, task.params)
        if child.code_hash in self.seen_hashes:
            child.status, child.error_type = "rejected_duplicate", "duplicate"
            child.error = (
                "the same program as an earlier one (only names, comments or formatting differ)"
            )
            return
        if (
            self.structure_counts.get(child.structure_hash, 0)
            >= self.settings.evolution.max_param_variants
        ):
            child.status, child.error_type = "rejected_duplicate", "too_many_variants"
            child.error = (
                "only constants changed, and this structure has been tuned enough; try a new idea"
            )
            return
        self.seen_hashes.add(child.code_hash)
        self.structure_counts[child.structure_hash] = (
            self.structure_counts.get(child.structure_hash, 0) + 1
        )

        program = build_program(task.header, child.code)
        smoke = await self.evaluator.stage(program, "smoke", repeat=2)
        child.eval_seconds = smoke.seconds
        if not smoke.ok:
            child.status, child.error_type, child.error = (
                "failed_smoke",
                smoke.error_type,
                smoke.error,
            )
            return
        child.smoke_score = smoke.score
        train = await self.evaluator.stage(program, "train")
        child.eval_seconds += train.seconds
        if not train.ok:
            child.status, child.error_type, child.error = (
                "failed_train",
                train.error_type,
                train.error,
            )
            return
        child.train_score, child.train_details = train.score, train.per_item
        child.status = "evaluated"

    def _save(self, child: Program) -> bool:
        """Store the candidate; True if it is its island's new best."""
        self.db.insert(child)
        if child.status != "evaluated":
            return False
        previous_best = self.archive.best(child.island)
        if self.archive.add(child):
            self.db.save_cell(child.island, child.cell, child.id)
        self.db.update(child)
        return previous_best is not None and better(child, previous_best)

    async def _validate(self, p: Program) -> None:
        """Score a new island best on the validation split (it is a champion candidate)."""
        twin = self.db.programs("code_hash = ? AND val_score IS NOT NULL", (p.code_hash,))
        if twin:
            p.val_score, p.val_details = twin[0].val_score, twin[0].val_details
        else:
            val = await self.evaluator.stage(build_program(self.task.header, p.code), "val")
            p.eval_seconds += val.seconds
            if val.ok:
                p.val_score, p.val_details = val.score, val.per_item
            else:  # works on training but breaks on new instances: never a champion
                p.val_score = float("inf")
                p.val_details = {"error": f"{val.error_type}: {val.error}"}
                log.warning("program %s failed on validation: %s", p.id, val.error)
        self.db.update(p)

    def _log(self, child: Program, parent: Program, seconds: float, new_best: bool) -> None:
        if child.status == "evaluated":
            result = f"train {child.train_score:6.2f}%  (parent {parent.train_score:.2f}%)"
            if new_best:
                result += f"  * island best, validation {child.val_score:.2f}%"
        else:
            result = f"{child.error_type}: {(child.error or '')[:70]}"
        log.info(
            "[%4d/%d] island %d  %-24s %-18s %s  %.1fs",
            child.attempt, self.max_candidates, child.island, f"{child.model}/{child.mutation}",
            child.status, result, seconds,
        )  # fmt: skip

    # ----------------------------------------------------------------- periodic steps
    def migrate(self) -> None:
        """Ring migration: each island's best goes to the next island."""
        n = self.archive.islands
        bests = [self.archive.best(i) for i in range(n)]
        for i, best in enumerate(bests):
            if best is None:
                continue
            j = (i + 1) % n
            if any(p.code_hash == best.code_hash for p in self.archive.elites(j)):
                continue
            copy = Program(
                island=j,
                code=best.code,
                parent_id=best.id,
                generation=best.generation,
                origin="migrated",
                status="evaluated",
                code_hash=best.code_hash,
                structure_hash=best.structure_hash,
                model=best.model,
                mutation=best.mutation,
                idea=best.idea,
                smoke_score=best.smoke_score,
                train_score=best.train_score,
                val_score=best.val_score,
                train_details=best.train_details,
                val_details=best.val_details,
                complexity=best.complexity,
                params_used=best.params_used,
            )
            cell = self.archive.cell_of(copy)
            current = self.archive.grid[j].get(cell)
            if current is not None and not better(copy, current):
                continue
            self.db.insert(copy)
            self.archive.add(copy)
            self.db.save_cell(j, copy.cell, copy.id)
            self.db.update(copy)
            self.db.log_event(
                "migration", {"from": i, "to": j, "program": best.id, "copy": copy.id}
            )

    async def reflect(self, island: int) -> None:
        """Ask the model for a one-line lesson from a better and a worse program."""
        best = self.archive.best(island)
        others = [
            p
            for p in self.db.programs(
                "status = 'evaluated' AND island = ? AND origin != 'migrated'", (island,)
            )
            if best is not None and p.train_score > best.train_score + 1e-9
        ]
        if best is None or not others:
            return
        worse = others[int(self.rng.integers(len(others)))]
        endpoint = self.settings.llm.models[0]
        try:
            reply = await self.llm.complete(
                endpoint, lesson_messages(self.task, best, worse), "lesson", SCHEMAS["lesson"]
            )
        except LLMError as exc:
            log.warning("reflection failed: %s", exc)
            return
        text = " ".join(reply.data["lesson"].split())[:400]
        if text and text not in self.db.lessons():
            self.db.add_lesson(text, island, [best.id, worse.id])
            log.info("lesson: %s", text)

    # ----------------------------------------------------------------- the end
    def finish(self) -> dict:
        db = self.db
        summary = self.summary()
        self.paths.summary.write_text(json.dumps(summary, indent=2))
        best = champion(db)
        if best is not None:
            header = (
                f"# HeuriForge champion for {self.task.name} ({self.dataset})\n"
                f"# program {best.id}: train gap {best.train_score:.2f}%, validation gap {best.val_score:.2f}%\n"
                f"# idea: {best.idea or 'initial program'}\n"
            )
            self.paths.best.write_text(header + build_program(self.task.header, best.code))
        db.log_event("session_end", {"reason": self.stop_reason, "completed": self.completed})
        if self.tracker is not None:
            self.tracker.finish(summary, self.paths)
        return summary

    def summary(self) -> dict:
        return run_summary(self.db, self.stop_reason)


def run_summary(db: Database, stop_reason: str | None = None) -> dict:
    programs = db.programs("origin IN ('llm', 'random')")
    evaluated = db.evaluated()
    initial = db.programs("origin = 'initial'")
    best_train = min(evaluated, key=lambda p: (p.train_score, p.complexity)) if evaluated else None
    best = champion(db)
    tokens = sum((p.prompt_tokens or 0) + (p.completion_tokens or 0) for p in programs)
    cached = sum(p.cached_tokens or 0 for p in programs)
    prompt = sum(p.prompt_tokens or 0 for p in programs)
    return {
        "task": db.get_meta("task"),
        "dataset": db.get_meta("dataset"),
        "provider": db.get_meta("provider"),
        "candidates": len(programs),
        "statuses": db.status_counts(),
        "errors": db.error_counts(),
        "initial_train_gap": initial[0].train_score if initial else None,
        "initial_val_gap": initial[0].val_score if initial else None,
        "best_train_gap": best_train.train_score if best_train else None,
        "champion_id": best.id if best else None,
        "champion_train_gap": best.train_score if best else None,
        "champion_val_gap": best.val_score if best else None,
        "llm_seconds": round(sum(p.llm_seconds or 0 for p in programs), 1),
        "eval_seconds": round(sum(p.eval_seconds or 0 for p in programs), 1),
        "tokens": tokens,
        "prompt_cache_hit_rate": round(cached / prompt, 3) if prompt else None,
        "lessons": len(db.lessons()),
        "stop_reason": stop_reason,
    }
