"""Results pooled over repeated runs: the README table and the report's comparison.

Each configuration (a task, with or without the language model) is evolved several times
with different search seeds. The sealed test of a run scores its champion on every test
item. Two summaries come out of that:

- run level: the mean test gap of each run's champion, reported as mean ± standard
  deviation over the runs (how much the outcome depends on the luck of one run);
- item level: for every test item, the champions' gaps averaged over the runs of each
  kind, then the two kinds compared with paired statistics (bootstrap confidence
  interval, Wilcoxon signed-rank test, Vargha-Delaney A12).

A run whose champion fails on the test split is not dropped silently: it is listed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, stdev

from heuriforge.stats import compare

TASKS = (
    ("cvrp_construct", "1 · Constructive heuristic"),
    ("pyvrp_accept", "2 · PyVRP acceptance rule"),
)


@dataclass
class SealedRun:
    name: str
    task: str
    kind: str  # "llm" (a model proposed the changes) or "random" (the no-model baseline)
    seed: int
    report: dict

    @property
    def gaps(self) -> dict[str, float] | None:
        """The champion's gap on every test item, or None if it failed on the test split."""
        return (self.report.get("per_item") or {}).get("champion")

    @property
    def mean_gap(self) -> float | None:
        return (self.report.get("mean_gap") or {}).get("champion")


@dataclass
class Pooled:
    task: str
    llm: list[SealedRun] = field(default_factory=list)
    base: list[SealedRun] = field(default_factory=list)
    failed: list[SealedRun] = field(default_factory=list)
    initial: float | None = None
    savings: float | None = None
    comparison: dict | None = None
    items: int = 0


def kind_of(provider: str | None, name: str) -> str:
    if provider:
        return "random" if provider == "random" else "llm"
    return "random" if "random" in name else "llm"


def task_of(report: dict, name: str) -> str:
    if report.get("task"):
        return report["task"]
    if name.startswith("construct"):
        return "cvrp_construct"
    if name.startswith("accept"):
        return "pyvrp_accept"
    return ""


def seed_of(run_dir: Path) -> int:
    """The run's search seed: from its database or run.yaml, else a -sN name suffix, else 0."""
    seed = None
    if (run_dir / "programs.sqlite").exists():
        from heuriforge.database import Database

        db = Database(run_dir / "programs.sqlite", readonly=True)
        try:
            seed = ((db.get_meta("settings") or {}).get("run") or {}).get("seed")
        finally:
            db.close()
    elif (run_dir / "run.yaml").exists():  # an exported run (results/<run>/)
        import yaml

        info = yaml.safe_load((run_dir / "run.yaml").read_text()) or {}
        seed = ((info.get("settings") or {}).get("run") or {}).get("seed")
    if seed is not None:
        return int(seed)
    m = re.search(r"-s(\d+)$", run_dir.name)
    return int(m.group(1)) if m else 0


def load_tested_runs(dirs) -> list[SealedRun]:
    runs = []
    for d in sorted(Path(x) for x in dirs):
        path = d / "sealed_test.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        runs.append(
            SealedRun(
                name=d.name,
                task=task_of(report, d.name),
                kind=kind_of(report.get("provider"), d.name),
                seed=seed_of(d),
                report=report,
            )
        )
    return sorted(runs, key=lambda r: (r.task, r.kind, r.seed, r.name))


def pool(runs: list[SealedRun], task: str) -> Pooled | None:
    mine = [r for r in runs if r.task == task]
    if not mine:
        return None
    first = mine[0].report.get("mean_gap") or {}
    out = Pooled(task=task, initial=first.get("initial"), savings=first.get("savings"))
    for r in mine:
        if not r.gaps:
            out.failed.append(r)
        elif r.kind == "llm":
            out.llm.append(r)
        else:
            out.base.append(r)
    if out.llm and out.base:
        keys = sorted(set.intersection(*(set(r.gaps) for r in out.llm + out.base)))
        if keys:
            a = [fmean(r.gaps[k] for r in out.llm) for k in keys]
            b = [fmean(r.gaps[k] for r in out.base) for k in keys]
            out.comparison = compare(a, b)
            out.items = len(keys)
    return out


def spread(runs: list[SealedRun]) -> tuple[float, float | None, int]:
    """Mean of the runs' mean test gaps, their standard deviation (None for one run), count."""
    values = [r.mean_gap for r in runs if r.mean_gap is not None]
    return fmean(values), (stdev(values) if len(values) > 1 else None), len(values)


def format_spread(runs: list[SealedRun]) -> str:
    mean, sd, n = spread(runs)
    if sd is None:
        return f"{mean:.2f}% (1 run)"
    return f"{mean:.2f}% ± {sd:.2f} ({n} runs)"


def format_comparison(c: dict) -> str:
    return (
        f"{c['mean_diff']:+.2f} pts [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}], "
        f"p = {c['p_value']:.2g}, A12 = {c['a12']:.2f}"
    )


def markdown_table(runs: list[SealedRun]) -> str:
    """The README results table (pooled over runs), its caption, and one row per run."""
    lines = [
        "| Task | Starting point | No-LLM baseline | LLM | LLM vs no-LLM (per test item) |",
        "|---|---|---|---|---|",
    ]
    notes = []
    for task, title in TASKS:
        p = pool(runs, task)
        if p is None or not p.llm or not p.base:
            have = "none" if p is None else f"{len(p.llm)} with the model, {len(p.base)} without"
            lines.append(f"| {title} | – | – | – | (sealed tests so far: {have}) |")
            continue
        verdict = format_comparison(p.comparison) if p.comparison else "–"
        initial = f"{p.initial:.2f}%" if p.initial is not None else "–"
        lines.append(
            f"| {title} | {initial} | {format_spread(p.base)} | {format_spread(p.llm)} | {verdict} |"
        )
        if p.savings is not None:
            notes.append(
                f"Task {title[0]} reference on the same test instances: Clarke-Wright savings "
                f"(1964) {p.savings:.2f}%."
            )
        for r in p.failed:
            error = "; ".join(f"{k}: {v}" for k, v in (r.report.get("errors") or {}).items())
            notes.append(f"Run `{r.name}`: the champion failed on the test split ({error}).")
    lines.append("")
    lines.append(
        "*Mean gap (%) to the best-known solutions on the sealed test items (Task 1: 20 "
        "instances; Task 2: 20 instances × 5 seeds), as mean ± standard deviation over "
        "independent runs. Last column: the champions' gaps are averaged over the runs of "
        "each kind, item by item, then compared: mean difference with a 95% bootstrap "
        "confidence interval, two-sided Wilcoxon signed-rank p-value, and Vargha-Delaney "
        "A12 (the probability that the LLM side does better on an item; 0.5 = no "
        "difference). Negative differences favour the LLM.*"
    )
    if notes:
        lines.append("")
        lines.extend(notes)
    lines.append("")
    lines.append("| Run | Seed | Champion's test gap | Champion vs starting point |")
    lines.append("|---|---|---|---|")
    for r in runs:
        gap = f"{r.mean_gap:.2f}%" if r.mean_gap is not None else "failed"
        c = (r.report.get("comparisons") or {}).get("initial")
        versus = (
            f"{c['mean_diff']:+.2f} pts, p = {c.get('p_holm', c['p_value']):.2g} (Holm)"
            if c
            else "–"
        )
        lines.append(f"| `{r.name}` | {r.seed} | {gap} | {versus} |")
    return "\n".join(lines)
