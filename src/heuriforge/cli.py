"""Command line: ``heuriforge <command>``. Run ``heuriforge <command> --help`` for details."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from statistics import fmean

from heuriforge import __version__
from heuriforge.config import Settings, load_settings, load_splits

log = logging.getLogger("heuriforge")
DATASETS = ["cvrplib_x", "synthetic"]


def setup_logging(verbose: bool, logfile: Path | None = None) -> None:
    root = logging.getLogger("heuriforge")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(logfile)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S"))
        root.addHandler(handler)
    root.propagate = False
    logging.getLogger("httpx").setLevel(logging.WARNING)


def run_dir(settings: Settings, name: str) -> Path:
    path = Path(name)
    return (
        path if (path / "programs.sqlite").exists() or "/" in name else settings.paths.runs / name
    )


# --------------------------------------------------------------------------- commands
def cmd_data(args, settings: Settings) -> None:
    from heuriforge.instances import prepare_dataset

    names = DATASETS if args.dataset == "all" else [args.dataset]
    for name in names:
        t0 = time.time()
        log.info("preparing dataset %s", name)
        out = prepare_dataset(settings.paths.data, name, load_splits(name, settings.root))
        n = len(list(out.glob("*.vrp")))
        log.info("  %d instances in %s (%.0f s)", n, out, time.time() - t0)


def cmd_eval(args, settings: Settings) -> None:
    from heuriforge.codeblock import build_program, extract_block
    from heuriforge.evaluator import Evaluator
    from heuriforge.instances import gap_percent, route_cost
    from heuriforge.sandbox.runner import Sandbox
    from heuriforge.sandbox.static import check_code
    from heuriforge.tasks import get_task

    if args.split == "test":
        sys.exit("The test split is sealed. Use `heuriforge test --name <run>` once, at the end.")
    task = get_task(args.task)
    if args.program:
        text = Path(args.program).read_text()
        block = extract_block(text) if "EVOLVE-BLOCK-START" in text else text
    else:
        block = task.initial_block
    report = check_code(block, task.entry, task.params)
    if not report.ok:
        sys.exit(f"static check failed: {report.category}: {report.message} (line {report.line})")
    splits = load_splits(args.dataset, settings.root)
    options = task.check_options(settings.task_options(task.name))
    evaluator = Evaluator(
        task,
        options,
        settings.paths.data,
        args.dataset,
        splits,
        Sandbox(settings.eval.workers, settings.eval.memory_mb),
    )
    t0 = time.time()
    result = asyncio.run(
        evaluator.stage(
            build_program(task.header, block), args.split, repeat=2 if args.split == "smoke" else 1
        )
    )
    if not result.ok:
        sys.exit(f"failed: {result.error_type}: {result.error}")
    savings = {}
    if task.name == "cvrp_construct":
        from heuriforge.tasks.cvrp_construct import savings_routes

        for item in evaluator.items(args.split):
            savings[item.key] = gap_percent(
                route_cost(item.instance.dist, savings_routes(item.instance)), item.instance.bks
            )
    print(f"{'item':<18} {'gap %':>8}" + (f" {'savings %':>10}" if savings else ""))
    for key, gap in result.per_item.items():
        print(f"{key:<18} {gap:>8.2f}" + (f" {savings[key]:>10.2f}" if savings else ""))
    print(
        f"{'mean':<18} {result.score:>8.2f}"
        + (f" {fmean(savings.values()):>10.2f}" if savings else "")
    )
    print(
        f"\n{len(result.per_item)} items in {time.time() - t0:.1f} s ({result.cpu_s:.1f} s of CPU in the sandbox)"
    )


def cmd_evolve(args, settings: Settings) -> None:
    from heuriforge.engine import Engine
    from heuriforge.evaluator import SetupError
    from heuriforge.llm import LLMError
    from heuriforge.tasks import get_task

    task = get_task(args.task)
    path = settings.paths.runs / args.name
    setup_logging(args.verbose, path / "log.txt")
    if args.islands:
        settings.evolution.islands = args.islands
    if args.seed is not None:
        settings.run.seed = args.seed
    tracker = None
    if settings.tracking.mlflow:
        from heuriforge.tracking import MlflowTracker

        tracker = MlflowTracker(
            args.name,
            {
                "task": task.name,
                "dataset": args.dataset,
                **settings.model_dump(mode="json", exclude={"root", "paths"}),
            },
            settings.root,
        )
    provider = args.provider or settings.llm.provider
    engine = Engine(
        settings, task, args.dataset, path, provider=provider, resume=args.resume,
        max_candidates=args.max_candidates, max_hours=args.max_hours, tracker=tracker,
    )  # fmt: skip
    log.info(
        "HeuriForge %s: %s on %s, provider %s, run folder %s",
        __version__,
        task.name,
        args.dataset,
        provider,
        path,
    )
    log.info(
        "Stop safely at any time with Ctrl+C or by creating the file %s; continue later with --resume.",
        path / "STOP",
    )
    try:
        summary = asyncio.run(engine.run())
    except KeyboardInterrupt:
        log.info(
            "\nStopped. Everything up to the last finished candidate is saved. Continue with --resume."
        )
        return
    except (LLMError, SetupError, FileExistsError, FileNotFoundError, ValueError) as exc:
        sys.exit(f"\n{exc}")
    print_summary(summary, path)


def print_summary(s: dict, path: Path) -> None:
    log.info("")
    log.info("Run %s finished (%s)", path.name, s.get("stop_reason"))
    log.info(
        "  candidates: %d   %s",
        s["candidates"],
        ", ".join(f"{k} {v}" for k, v in sorted(s["statuses"].items())),
    )
    log.info(
        "  initial program:  train %.2f%%  validation %.2f%%",
        s["initial_train_gap"],
        s["initial_val_gap"],
    )
    if s.get("champion_id") is not None:
        log.info(
            "  champion (#%s):   train %.2f%%  validation %.2f%%",
            s["champion_id"], s["champion_train_gap"], s["champion_val_gap"],
        )  # fmt: skip
    log.info("  best program: %s", path / "best_program.py")


def cmd_status(args, settings: Settings) -> None:
    from heuriforge.database import Database
    from heuriforge.engine import run_summary

    path = run_dir(settings, args.name)
    db = Database(path / "programs.sqlite", readonly=True)
    s = run_summary(db)
    started = db.get_meta("started_at")
    programs = db.programs("origin IN ('llm', 'random')")
    seed = ((db.get_meta("settings") or {}).get("run") or {}).get("seed", 0)
    print(f"Run {path}: {s['task']} on {s['dataset']}, provider {s['provider']}, seed {seed}")
    if started:
        print(f"Started {time.strftime('%Y-%m-%d %H:%M', time.localtime(started))}", end="")
        if programs:
            ago = (time.time() - programs[-1].created_at) / 60
            print(f", last candidate {ago:.0f} min ago", end="")
        print()
    print(f"Candidates: {s['candidates']}")
    for status, count in sorted(s["statuses"].items(), key=lambda kv: -kv[1]):
        print(f"  {status:<20} {count}")
    print(
        f"Initial program:   train {s['initial_train_gap']:.2f}%   validation {s['initial_val_gap']:.2f}%"
    )
    if s["best_train_gap"] is not None:
        print(f"Best on training:  {s['best_train_gap']:.2f}%")
    if s["champion_id"] is not None:
        print(
            f"Champion (#{s['champion_id']}): validation {s['champion_val_gap']:.2f}%  (train {s['champion_train_gap']:.2f}%)"
        )
    calls = [p for p in programs if p.prompt_tokens]
    if calls:
        print(
            f"Model: {len(calls)} calls, {fmean(p.llm_seconds for p in calls):.1f} s and "
            f"{fmean(p.completion_tokens for p in calls):.0f} answer tokens per call, prompt cache reuse "
            f"{100 * (s['prompt_cache_hit_rate'] or 0):.0f}%"
        )
    arms = db.arms()
    if arms:
        print(
            "Bandit (beat parent / tried):",
            ", ".join(f"{k} {v['successes']}/{v['pulls']}" for k, v in arms.items()),
        )
    if programs:
        print("Last candidates:")
        for p in programs[-8:]:
            result = f"train {p.train_score:.2f}%" if p.status == "evaluated" else f"{p.error_type}"
            print(
                f"  #{p.attempt:<5} {p.model}/{p.mutation:<8} {p.status:<18} {result}  {(p.idea or '')[:60]}"
            )
    db.close()


def cmd_test(args, settings: Settings) -> None:
    from heuriforge.sealed_test import AlreadyTested, run_sealed_test

    if not args.all:
        path = run_dir(settings, args.name)
        try:
            report = asyncio.run(run_sealed_test(settings, path, force=args.force))
        except AlreadyTested as exc:
            sys.exit(str(exc))
        print_sealed(report)
        return
    # Every finished run that has not had its sealed test yet; the others are left alone.
    for path in finished_dirs(settings):
        summary = path / "summary.json"
        reason = json.loads(summary.read_text()).get("stop_reason", "") if summary.exists() else ""
        if not reason.startswith("reached max_candidates"):
            print(f"\n{path.name}: not finished ({reason or 'still running?'}), skipped")
            continue
        try:
            report = asyncio.run(run_sealed_test(settings, path))
        except AlreadyTested:
            print(f"\n{path.name}: sealed test already done, skipped")
            continue
        print_sealed(report)


def print_sealed(report: dict) -> None:
    print(f"\nSealed test of run {report['run']}: {report['items']} test items")
    for name, gap in sorted(report["mean_gap"].items(), key=lambda kv: kv[1]):
        print(f"  {name:<10} mean gap {gap:6.2f}%")
    for name, c in report["comparisons"].items():
        print(
            f"  champion vs {name}: {c['mean_diff']:+.2f} points [95% CI {c['ci_low']:+.2f}, {c['ci_high']:+.2f}], "
            f"p = {c['p_holm']:.3g} (Holm), A12 = {c['a12']:.2f} ({c['effect']}), "
            f"wins/losses/ties {c['wins']}/{c['losses']}/{c['ties']}"
        )
    for name, error in report.get("errors", {}).items():
        print(f"  {name} failed: {error}")


def cmd_report(args, settings: Settings) -> None:
    from heuriforge.report import build_site

    dirs = [run_dir(settings, n) for n in args.name] if args.name else finished_dirs(settings)
    out = Path(args.out) if args.out else settings.paths.site
    target = build_site(dirs, out, title=args.title, offline=args.offline)
    print(f"Report written to {target}")


def cmd_export(args, settings: Settings) -> None:
    """Copy a run's small result files into results/<run>/, ready to commit."""
    paths = finished_dirs(settings) if args.all else [run_dir(settings, args.name)]
    for path in paths:
        out = Path(args.out) / path.name
        out.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in ("best_program.py", "summary.json", "sealed_test.json", "run.yaml"):
            if (path / name).exists():
                shutil.copy2(path / name, out / name)
                copied.append(name)
        print(f"Copied {', '.join(copied)} to {out}")


def cmd_table(args, settings: Settings) -> None:
    """Print the README results table from the sealed tests, pooled over repeated runs."""
    from heuriforge.results import load_tested_runs, markdown_table

    folder = Path(args.source) if args.source else settings.paths.runs
    dirs = [
        p
        for p in folder.glob("*")
        if (p / "sealed_test.json").exists() and not p.name.startswith("demo-")
    ]
    print(markdown_table(load_tested_runs(dirs)))


def finished_dirs(settings: Settings) -> list[Path]:
    """Every run folder except the offline demo's."""
    return sorted(
        p
        for p in settings.paths.runs.glob("*")
        if (p / "programs.sqlite").exists() and not p.name.startswith("demo-")
    )


def cmd_bench(args, settings: Settings) -> None:
    from heuriforge.bench import print_bench, run_bench, save_bench
    from heuriforge.llm import LLMError

    concurrency = tuple(sorted({1, max(1, settings.run.parallel)}))
    print(
        f"Benchmarking {settings.llm.models[0].base_url} with {args.requests} requests per setting ..."
    )
    try:
        result = asyncio.run(run_bench(settings, args.task, args.requests, concurrency))
    except LLMError as exc:
        sys.exit(str(exc))
    print_bench(result)
    print(f"\nSaved to {save_bench(result, settings.paths.bench)}")


def cmd_demo(args, settings: Settings) -> None:
    from heuriforge.engine import Engine
    from heuriforge.instances import prepare_dataset
    from heuriforge.report import build_site
    from heuriforge.sealed_test import run_sealed_test
    from heuriforge.tasks import get_task

    task = get_task("cvrp_construct")
    log.info(
        "Offline demo: synthetic instances + a mock model (random mutations). No server, no download."
    )
    prepare_dataset(settings.paths.data, "synthetic", load_splits("synthetic", settings.root))
    settings.run.max_candidates = args.candidates
    settings.evolution.migration_interval = 10
    settings.evolution.reflection_interval = 10
    dirs = []
    for provider in ("mock", "random"):
        path = settings.paths.runs / f"demo-{provider}"
        if path.exists():
            shutil.rmtree(path)
        log.info("\n== run demo-%s ==", provider)
        summary = asyncio.run(Engine(settings, task, "synthetic", path, provider=provider).run())
        print_summary(summary, path)
        print_sealed(asyncio.run(run_sealed_test(settings, path)))
        dirs.append(path)
    target = build_site(dirs, Path(args.out), title="HeuriForge offline demo", offline=True)
    log.info("\nDemo report: %s  (open it in a browser)", target)


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="heuriforge", description="LLM-guided evolution of vehicle-routing heuristics."
    )
    p.add_argument("--settings", help="settings file (default: config/settings.yaml)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"heuriforge {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("data", help="download / generate the benchmark instances")
    s.add_argument("--dataset", default="all", help="cvrplib_x, synthetic, or all (default)")
    s.set_defaults(func=cmd_data)

    s = sub.add_parser("eval", help="score one program (the initial one by default) on a split")
    s.add_argument("--task", required=True, choices=["cvrp_construct", "pyvrp_accept"])
    s.add_argument("--dataset", default="cvrplib_x", help="cvrplib_x (default) or synthetic")
    s.add_argument("--split", default="train", choices=["smoke", "train", "val", "test"])
    s.add_argument("--program", help="a Python file with the program (or just its function)")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("evolve", help="run (or resume) an evolution")
    s.add_argument("--task", required=True, choices=["cvrp_construct", "pyvrp_accept"])
    s.add_argument("--dataset", default="cvrplib_x", help="cvrplib_x (default) or synthetic")
    s.add_argument("--name", required=True, help="run name (folder under runs/)")
    s.add_argument(
        "--provider",
        choices=["server", "random", "mock"],
        help="server: local llama-server (default); random: no-LLM baseline; mock: offline fake model",
    )
    s.add_argument("--max-candidates", type=int)
    s.add_argument("--max-hours", type=float)
    s.add_argument("--islands", type=int)
    s.add_argument(
        "--seed",
        type=int,
        help="search seed (default: run.seed in the settings); repeated runs use different seeds",
    )
    s.add_argument("--resume", action="store_true", help="continue an existing run")
    s.set_defaults(func=cmd_evolve)

    s = sub.add_parser("status", help="progress of a run (works while it is running)")
    s.add_argument("--name", required=True)
    s.set_defaults(func=cmd_status)

    s = sub.add_parser(
        "test", help="the sealed test: score a run's champion on the test split, once"
    )
    which = s.add_mutually_exclusive_group(required=True)
    which.add_argument("--name")
    which.add_argument(
        "--all", action="store_true", help="every finished run without a sealed test yet"
    )
    s.add_argument(
        "--force", action="store_true", help="run again although it already ran (recorded)"
    )
    s.set_defaults(func=cmd_test)

    s = sub.add_parser("report", help="build the static results website")
    s.add_argument("--name", nargs="*", help="runs to include (default: every run except demo-*)")
    s.add_argument("--out", help="output folder (default: site/)")
    s.add_argument("--title", default="HeuriForge results")
    s.add_argument(
        "--offline", action="store_true", help="embed plotly.js so the page works without internet"
    )
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("export", help="copy a run's result files to results/<run>/ (to commit)")
    which = s.add_mutually_exclusive_group(required=True)
    which.add_argument("--name")
    which.add_argument("--all", action="store_true", help="every run except the demo's")
    s.add_argument("--out", default="results")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("table", help="print the README results table from the sealed tests")
    s.add_argument(
        "--from",
        dest="source",
        help="folder of runs (default: runs/; results/ reads the exported runs)",
    )
    s.set_defaults(func=cmd_table)

    s = sub.add_parser("bench", help="Phase 0: measure the local model server")
    s.add_argument("--task", default="cvrp_construct", choices=["cvrp_construct", "pyvrp_accept"])
    s.add_argument("--requests", type=int, default=8)
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("demo", help="offline end-to-end demo (no model server needed)")
    s.add_argument("--candidates", type=int, default=40)
    s.add_argument("--out", default="site-demo")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    settings = load_settings(args.settings)
    args.func(args, settings)


if __name__ == "__main__":
    main()
