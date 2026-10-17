"""Phase 0: measure the local model server before spending hours on a run.

Sends real HeuriForge prompts to llama-server, one at a time and then several at once,
and reports what decides whether evolution is practical on this computer: generation
speed, seconds per candidate, prompt-cache reuse, and how many answers are usable
(valid JSON whose edits apply and whose code passes the static checks).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from statistics import fmean, median

from heuriforge.codeblock import EditError
from heuriforge.config import Settings
from heuriforge.database import Program
from heuriforge.llm import LLMClient, LLMError
from heuriforge.mutators import child_from_answer
from heuriforge.prompts import SCHEMAS, mutation_messages
from heuriforge.sandbox.static import check_code
from heuriforge.tasks import get_task


async def _one(
    client: LLMClient, endpoint, task, parent: Program, mutation: str, seed: int
) -> dict:
    messages = mutation_messages(task, parent, mutation, [], [])
    t0 = time.perf_counter()
    row = {"mutation": mutation, "ok": False, "usable": False}
    try:
        reply = await client.complete(endpoint, messages, mutation, SCHEMAS[mutation], seed=seed)
    except LLMError as exc:
        row.update(
            error=str(exc)[:200],
            seconds=time.perf_counter() - t0,
            answer_tail=exc.text[-240:],
            answer=exc.text,
        )
        if exc.fatal:
            raise
        return row
    t = reply.timings
    row.update(
        ok=True,
        seconds=reply.seconds,
        prompt_tokens=reply.prompt_tokens,
        cached_tokens=reply.cached_tokens,
        completion_tokens=reply.completion_tokens,
        prompt_ms=t.get("prompt_ms"),
        prompt_per_second=t.get("prompt_per_second"),
        predicted_per_second=t.get("predicted_per_second"),
        idea=reply.data.get("idea", "")[:160],
        answer=reply.text,  # kept in the saved file, to look at failures
    )
    try:
        block, _ = child_from_answer(parent.code, mutation, reply.data)
        report = check_code(block, task.entry, task.params)
        row["usable"] = report.ok
        if not report.ok:
            row["error"] = f"{report.category}: {report.message}"
    except EditError as exc:
        row["error"] = f"edit_failed: {exc}"
    return row


async def run_bench(
    settings: Settings,
    task_name: str = "cvrp_construct",
    requests: int = 6,
    concurrency=(1, 2),
    transport=None,
) -> dict:
    task = get_task(task_name)
    client = LLMClient(settings.llm, transport)
    endpoint = settings.llm.models[0]
    try:
        if not await client.health(endpoint):
            raise LLMError(
                f"no model server answers at {endpoint.base_url}. Start it with: make server",
                fatal=True,
            )
        props = {}
        try:
            root = endpoint.base_url.rstrip("/").removesuffix("/v1")
            r = await client.client.get(f"{root}/props", timeout=10.0)
            if r.status_code == 200:
                data = r.json()
                props = {
                    "model_path": data.get("model_path"),
                    "total_slots": data.get("total_slots"),
                    "n_ctx": (data.get("default_generation_settings") or {}).get("n_ctx"),
                    "build": data.get("build_info"),
                }
        except Exception:  # /props is informative only
            pass
        parent = Program(island=0, code=task.initial_block.strip("\n"), train_score=30.0, id=0)
        rounds = []
        seed = 1
        for conc in concurrency:
            rows: list[dict] = []
            t0 = time.perf_counter()
            for start in range(0, requests, conc):
                batch = []
                for k in range(start, min(start + conc, requests)):
                    mutation = settings.llm.mutations[k % len(settings.llm.mutations)]
                    batch.append(_one(client, endpoint, task, parent, mutation, seed))
                    seed += 1
                rows += await asyncio.gather(*batch)
            wall = time.perf_counter() - t0
            ok = [r for r in rows if r["ok"]]
            rounds.append(
                {
                    "concurrency": conc,
                    "requests": len(rows),
                    "wall_seconds": wall,
                    "candidates_per_hour": 3600 * len(rows) / wall if wall else 0,
                    "median_seconds": median(r["seconds"] for r in rows),
                    "generation_tokens_per_s": fmean(
                        r["predicted_per_second"] for r in ok if r.get("predicted_per_second")
                    )
                    if any(r.get("predicted_per_second") for r in ok)
                    else None,
                    "prompt_tokens_per_s": fmean(
                        r["prompt_per_second"] for r in ok if r.get("prompt_per_second")
                    )
                    if any(r.get("prompt_per_second") for r in ok)
                    else None,
                    "mean_prompt_tokens": fmean(r["prompt_tokens"] for r in ok) if ok else None,
                    "mean_completion_tokens": fmean(r["completion_tokens"] for r in ok)
                    if ok
                    else None,
                    "cache_reuse": (
                        sum(r["cached_tokens"] for r in ok)
                        / max(1, sum(r["prompt_tokens"] for r in ok))
                    )
                    if ok
                    else None,
                    "valid_json": len(ok) / len(rows),
                    "usable": sum(r["usable"] for r in rows) / len(rows),
                    "rows": rows,
                }
            )
        return {
            "task": task_name,
            "endpoint": endpoint.base_url,
            "model": endpoint.model,
            "server": props,
            "rounds": rounds,
        }
    finally:
        await client.close()


def print_bench(result: dict) -> None:
    print(
        f"\nModel server {result['endpoint']}  ({result['server'].get('model_path') or result['model']})"
    )
    if result["server"]:
        print(
            f"slots: {result['server'].get('total_slots')}   context: {result['server'].get('n_ctx')}"
        )
    print()
    print(
        f"{'in flight':>9} {'cand/hour':>9} {'s/cand':>7} {'gen tok/s':>9} {'prompt tok/s':>12} {'cache':>6} {'parsed':>7} {'usable':>6}"
    )
    for r in result["rounds"]:
        gen = f"{r['generation_tokens_per_s']:.1f}" if r["generation_tokens_per_s"] else "–"
        pp = f"{r['prompt_tokens_per_s']:.0f}" if r["prompt_tokens_per_s"] else "–"
        cache = f"{100 * r['cache_reuse']:.0f}%" if r["cache_reuse"] is not None else "–"
        print(
            f"{r['concurrency']:>9} {r['candidates_per_hour']:>9.0f} {r['median_seconds']:>7.1f} {gen:>9} {pp:>12} "
            f"{cache:>6} {100 * r['valid_json']:>6.0f}% {100 * r['usable']:>5.0f}%"
        )
    best = max(result["rounds"], key=lambda r: r["candidates_per_hour"])
    print(
        f"\nBest setting: {best['concurrency']} in flight, about {best['candidates_per_hour']:.0f} candidates per hour "
        f"(model time only; evaluation runs alongside)."
    )
    shown_tail = shown_answer = False
    for r in result["rounds"]:
        for row in r["rows"]:
            if not row.get("usable"):
                print(f"  not usable ({row['mutation']}): {row.get('error', '?')[:150]}")
                if row.get("answer_tail") and not shown_tail:
                    print(f"    end of that answer: {row['answer_tail']!r}")
                    shown_tail = True
                elif row.get("ok") and row.get("answer") and not shown_answer:
                    print(f"    that answer: {row['answer'][:400]!r}")
                    shown_answer = True
    print("  (every answer is in the saved file below)")


def save_bench(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2))
    return path
