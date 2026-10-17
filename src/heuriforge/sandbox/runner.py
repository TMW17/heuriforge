"""Start sandbox processes and collect their results (parent side)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

import heuriforge

# The worker must import the same heuriforge package as the parent, installed or not.
PACKAGE_PARENT = str(Path(heuriforge.__file__).resolve().parents[1])
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def worker_env(home: str) -> dict[str, str]:
    """A minimal environment: no inherited secrets, one thread per numeric library."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": home,
        "TMPDIR": home,
        "PYTHONPATH": PACKAGE_PARENT,
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.update({name: "1" for name in THREAD_VARS})
    return env


def _kill(proc: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)


def parse_output(out: bytes, err: bytes, returncode: int | None) -> dict:
    text = out.decode("utf-8", errors="replace").strip()
    if text:
        try:
            result, _ = json.JSONDecoder().raw_decode(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    if returncode in (-signal.SIGXCPU, -signal.SIGKILL):
        return {
            "ok": False,
            "error_type": "timeout",
            "error": "the CPU time limit was reached",
            "items": [],
        }
    tail = err.decode("utf-8", errors="replace").strip().splitlines()[-3:]
    return {
        "ok": False,
        "error_type": "crash",
        "error": f"the sandbox process stopped (exit code {returncode}): {' | '.join(tail)}"[:600],
        "items": [],
    }


class Sandbox:
    """Runs jobs in separate processes, at most ``workers`` at a time."""

    def __init__(self, workers: int = 4, memory_mb: int = 2048):
        self.workers = max(1, int(workers))
        self.memory_mb = int(memory_mb)
        self._sem = asyncio.Semaphore(self.workers)

    async def run(self, job: dict, wall_timeout: float) -> dict:
        job = {**job, "memory_mb": self.memory_mb}
        payload = json.dumps(job).encode()
        async with self._sem:
            with tempfile.TemporaryDirectory(prefix="heuriforge-") as tmp:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-P", "-s", "-B", "-m", "heuriforge.sandbox.worker",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=tmp,
                    env=worker_env(tmp),
                    start_new_session=True,  # its own process group, so a kill gets everything
                )  # fmt: skip
                try:
                    out, err = await asyncio.wait_for(
                        proc.communicate(payload), timeout=wall_timeout
                    )
                except TimeoutError:
                    _kill(proc)
                    await proc.wait()
                    return {
                        "ok": False,
                        "error_type": "timeout",
                        "error": f"no result within {wall_timeout:.0f} s",
                        "items": [],
                    }
                except asyncio.CancelledError:
                    # Another chunk of the same stage failed: stop this one too, and reap it
                    # so no orphan process or open pipe is left behind.
                    _kill(proc)
                    with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(proc.wait(), timeout=10)
                    raise
        return parse_output(out, err, proc.returncode)
