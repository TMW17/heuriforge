"""The sandboxed child process: run one candidate program on a list of items.

Started by ``runner.py`` as ``python -P -s -B -m heuriforge.sandbox.worker`` in an empty
temporary folder. It reads one job (JSON) from stdin and writes one result (JSON) to
stdout. Order of events:

1. trusted code loads the instances (before any limit applies);
2. limits: CPU time, no file writes, memory (address space on Linux, a peak-memory
   watchdog everywhere), and a CPU timer per item;
3. the candidate is loaded with restricted builtins and run inside the task's skeleton;
4. errors are classified so the next prompt can say exactly what went wrong.
"""

from __future__ import annotations

import json
import math
import os
import resource
import signal
import sys
import threading
import time
import traceback
from contextlib import contextmanager

from heuriforge.codeblock import START
from heuriforge.sandbox.safe_exec import load_program
from heuriforge.sandbox.static import check_code
from heuriforge.tasks import get_task
from heuriforge.tasks.base import InvalidOutput, ModifiedInput


class ItemTimeout(BaseException):
    """Raised by the CPU timer. Derives from BaseException so `except Exception` cannot catch it."""


_out_lock = threading.Lock()
_out_file = None
_emitted = False


def emit(result: dict) -> None:
    """Write the one and only result (the memory watchdog may race the main thread)."""
    global _emitted
    with _out_lock:
        if _emitted:
            return
        _emitted = True
        _out_file.write(json.dumps(result).encode())
        _out_file.flush()


def _on_timer(signum, frame):
    raise ItemTimeout()


@contextmanager
def cpu_timer(seconds: float):
    # Fires after `seconds` of CPU time, then every half second in case the first one is
    # swallowed by a `finally:` block in the candidate.
    signal.setitimer(signal.ITIMER_PROF, max(seconds, 0.01), 0.5)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0, 0)


def set_limits(cpu_seconds: float, memory_mb: int) -> None:
    cpu = math.ceil(cpu_seconds)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # no file writes (Python ignores SIGXFSZ)
    if sys.platform.startswith("linux"):
        # Address space, not resident memory: leave room for the interpreter and libraries.
        limit = (memory_mb + 1536) * 2**20
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def watch_memory(memory_mb: int) -> None:
    """Peak-memory watchdog (macOS has no working address-space limit)."""
    unit = 1 if sys.platform == "darwin" else 1024  # ru_maxrss is bytes on macOS, KiB on Linux

    def loop():
        while True:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
            if peak > memory_mb * 2**20:
                emit(
                    {
                        "ok": False,
                        "error_type": "memory_limit",
                        "error": f"used more than {memory_mb} MB of memory",
                        "items": [],
                    }
                )
                os._exit(0)
            time.sleep(0.05)

    threading.Thread(target=loop, daemon=True).start()


def block_line_offset(program: str) -> int:
    for i, line in enumerate(program.splitlines(), start=1):
        if line.strip() == START:
            return i
    return 0


def describe(exc: BaseException, program: str) -> tuple[str, str]:
    """(error type, message pointing at the candidate's line)."""
    message = f"{type(exc).__name__}: {exc}".strip()
    if isinstance(exc, SyntaxError):
        offset = block_line_offset(program)
        line = (exc.lineno or 0) - offset if offset and (exc.lineno or 0) > offset else exc.lineno
        return "syntax_error", f"syntax error: {exc.msg} (line {line} of the code block)"
    if isinstance(exc, ValueError) and "read-only" in str(exc):
        return "modified_input", f"the function tried to change a read-only input ({exc})"
    if isinstance(exc, MemoryError):
        return "memory_limit", "ran out of memory"
    if isinstance(exc, RecursionError):
        return "runtime_error", "RecursionError: maximum recursion depth exceeded"
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == "<candidate>"]
    if frames:
        lineno = frames[-1].lineno or 0
        lines = program.splitlines()
        text = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""
        offset = block_line_offset(program)
        where = lineno - offset if offset and lineno > offset else lineno
        message += f" (line {where} of the code block: {text})"
    return "runtime_error", message[:600]


def run_job(job: dict) -> dict:
    task = get_task(job["task"])
    options = job.get("options", {})
    items = job["items"]
    repeat = int(job.get("repeat", 1))
    limit = float(job["cpu_s_per_item"])
    program = job["program"]

    # The parent already ran the static checks; running them again here means a bug or a
    # bypass in the parent cannot get unchecked code executed.
    report = check_code(program, task.entry, task.params)
    if not report.ok:
        return {
            "ok": False,
            "error_type": "rejected_static",
            "error": f"{report.category}: {report.message}",
            "items": [],
        }

    try:
        prepared = [task.prepare(it["instance"], it["seed"], options) for it in items]
    except Exception as exc:  # trusted code failed: not the candidate's fault
        return {"ok": False, "error_type": "setup_error", "error": repr(exc), "items": []}

    set_limits(limit * (len(items) * repeat + 1) + 5, int(job.get("memory_mb", 2048)))
    watch_memory(int(job.get("memory_mb", 2048)))
    signal.signal(signal.SIGPROF, _on_timer)

    try:
        with cpu_timer(limit):
            fn = load_program(program, task.entry)
    except ItemTimeout:
        return {
            "ok": False,
            "error_type": "timeout",
            "error": "the program took too long to load",
            "items": [],
        }
    except BaseException as exc:
        error_type, message = describe(exc, program)
        return {"ok": False, "error_type": error_type, "error": message, "items": []}

    results = []
    failed = False
    for it, prep in zip(items, prepared, strict=True):
        if failed:
            results.append({"key": it["key"], "ok": False, "error_type": "skipped", "error": ""})
            continue
        outs, res = [], None
        for _ in range(repeat):
            t0 = time.process_time()
            try:
                with cpu_timer(limit):
                    out = task.run(fn, prep, it["seed"], options)
                outs.append(out)
            except ItemTimeout:
                res = {
                    "error_type": "timeout",
                    "error": f"took more than {limit:g} s of CPU time on {it['key']}",
                }
            except InvalidOutput as exc:
                res = {"error_type": "invalid_output", "error": str(exc)}
            except ModifiedInput as exc:
                res = {"error_type": "modified_input", "error": str(exc)}
            except BaseException as exc:
                error_type, message = describe(exc, program)
                res = {"error_type": error_type, "error": message}
            cpu = time.process_time() - t0
            if res is not None:
                break
        if res is None and repeat > 1 and any(o["routes"] != outs[0]["routes"] for o in outs[1:]):
            res = {
                "error_type": "nondeterministic",
                "error": "two runs on the same input gave different results (keep no state between calls)",
            }
        if res is not None:
            failed = True
            results.append({"key": it["key"], "ok": False, **res})
            continue
        out = outs[0]
        results.append({"key": it["key"], "ok": True, "cpu_s": round(cpu, 4), **out})
    return {"ok": not failed, "items": results}


def main() -> None:
    global _out_file
    job = json.loads(sys.stdin.buffer.read())
    # Keep the real stdout for the result; everything else printed goes nowhere.
    _out_file = os.fdopen(os.dup(1), "wb")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    result = run_job(job)
    signal.setitimer(signal.ITIMER_PROF, 0, 0)
    emit(result)


if __name__ == "__main__":
    main()
