"""The second wall: what happens when misbehaving code actually runs.

Every test here plays a candidate that tries something it should not: loop forever, eat
memory, change its inputs, keep state between calls, return garbage, or escape the
sandbox. The evaluator must catch each one and say what happened.
"""

import asyncio

import pytest

from heuriforge.codeblock import build_program
from heuriforge.config import load_splits
from heuriforge.evaluator import Evaluator
from heuriforge.sandbox.runner import Sandbox
from heuriforge.tasks import get_task

TASK = get_task("cvrp_construct")
SIG = "def score_candidates(current, candidates, rest_capacity, capacity, demand, dist, coords, depot):\n"
OK = "    return -dist[current, candidates].astype(float)\n"


def evaluate(
    settings, block: str, split: str = "smoke", repeat: int = 2, task=TASK, memory_mb: int = 1024
):
    options = dict(settings.task_options(task.name), per_instance_cpu_s=2)
    evaluator = Evaluator(
        task,
        options,
        settings.paths.data,
        "tiny",
        load_splits("tiny", settings.root),
        Sandbox(2, memory_mb),
    )
    return asyncio.run(evaluator.stage(build_program(task.header, block), split, repeat=repeat))


def test_initial_programs_pass(settings):
    for name in ("cvrp_construct", "pyvrp_accept"):
        task = get_task(name)
        result = evaluate(settings, task.initial_block, split="train", repeat=1, task=task)
        assert result.ok, result.error
        assert result.score is not None and result.score > -1e-9
        assert len(result.per_item) == 3 * len(task.seeds("train", settings.task_options(name)))


@pytest.mark.parametrize(
    ("body", "error_type"),
    [
        ("    while True:\n        pass\n", "timeout"),
        # a `finally:` that loops again after the first timer signal
        (
            "    try:\n        while True:\n            pass\n    finally:\n        while True:\n            pass\n",
            "timeout",
        ),
        ("    x = np.ones((25000, 12000))\n" + OK, "memory_limit"),
        ("    x = [0.5] * (3 * 10**8)\n" + OK, "memory_limit"),
        ("    dist[0, 0] = 5\n" + OK, "modified_input"),
        ("    demand += 1\n" + OK, "modified_input"),
        ("    return np.full(len(candidates), np.nan)\n", "invalid_output"),
        ("    return np.zeros(len(candidates) + 1)\n", "invalid_output"),
        ("    return 'scores'\n", "invalid_output"),
        ("    return demand[10**6]\n", "runtime_error"),
        (
            "    return score_candidates(current, candidates, rest_capacity, capacity, demand, dist, coords, depot)\n",
            "runtime_error",
        ),
    ],
)
def test_misbehaviour_is_caught(settings, body, error_type):
    result = evaluate(settings, SIG + body)
    assert not result.ok
    assert result.error_type == error_type, result.error


def test_runtime_errors_point_at_the_line(settings):
    result = evaluate(settings, SIG + "    x = 1\n    y = demand[10**6]\n" + OK)
    assert result.error_type == "runtime_error"
    assert "IndexError" in result.error and "line 3 of the code block" in result.error


def test_state_kept_between_calls_is_caught(settings):
    block = (
        "HISTORY = []\n"
        + SIG
        + "    HISTORY.append(1)\n    return -dist[current, candidates] + len(HISTORY) * (candidates % 3)\n"
    )
    result = evaluate(settings, block)
    assert result.error_type == "nondeterministic"


@pytest.mark.parametrize(
    "line",
    [
        "    k = ().__class__.__base__.__subclasses__()\n",  # the classic escape
        "    np.lib.format.open_memmap('x.npy', mode='w+', shape=(3,))\n",
        "    dist.flags.writeable = True\n    dist[0, 0] = 0\n",
        "    __import__('os').system('echo hacked')\n",
    ],
)
def test_escapes_are_blocked_inside_the_worker_too(settings, line):
    # The parent normally rejects these before evaluation. Here they are sent straight to
    # the worker, which must refuse them as well (defence in depth).
    result = evaluate(settings, SIG + line + OK)
    assert not result.ok
    assert result.error_type == "rejected_static"


def test_restricted_builtins_stop_runtime_tricks(tmp_path):
    from heuriforge.sandbox.safe_exec import load_program

    with pytest.raises(NameError):
        load_program("open('/etc/passwd')\ndef f():\n    return 1\n", "f")
    with pytest.raises(ImportError):
        load_program("import os\ndef f():\n    return 1\n", "f")
    with pytest.raises(ImportError):
        load_program("from numpy import load\ndef f():\n    return 1\n", "f")
    fn = load_program("import numpy as np\ndef f():\n    return float(np.arange(4).mean())\n", "f")
    assert fn() == 1.5  # numpy's own lazy imports still work


def test_routes_are_checked_by_the_parent(settings, monkeypatch):
    # Even if the worker returned bad routes, the trusted checker in the parent rejects them.
    from heuriforge.sandbox import runner

    async def fake_run(self, job, wall_timeout):
        return {
            "ok": True,
            "items": [
                {"key": it["key"], "ok": True, "routes": [[1, 1]], "cpu_s": 0.0}
                for it in job["items"]
            ],
        }

    monkeypatch.setattr(runner.Sandbox, "run", fake_run)
    result = evaluate(settings, TASK.initial_block)
    assert not result.ok and result.error_type == "invalid_output"
