"""The first wall: code that breaks the rules is rejected before it runs."""

import pytest

from heuriforge.sandbox.static import check_code
from heuriforge.tasks import get_task

TASK = get_task("cvrp_construct")
SIG = "def score_candidates(current, candidates, rest_capacity, capacity, demand, dist, coords, depot):\n"
OK_BODY = "    return -dist[current, candidates]\n"


def check(code: str):
    return check_code(code, TASK.entry, TASK.params)


@pytest.mark.parametrize(
    ("code", "category"),
    [
        ("import os\n" + SIG + OK_BODY, "forbidden_import"),
        ("import subprocess\n" + SIG + OK_BODY, "forbidden_import"),
        ("import numpy.random\n" + SIG + OK_BODY, "forbidden_import"),
        ("from numpy import load\n" + SIG + OK_BODY, "forbidden_import"),
        ("from numpy import *\n" + SIG + OK_BODY, "forbidden_import"),
        ("from . import x\n" + SIG + OK_BODY, "forbidden_import"),
        (SIG + "    open('/etc/passwd')\n" + OK_BODY, "forbidden_call"),
        (SIG + "    eval('1')\n" + OK_BODY, "forbidden_call"),
        (SIG + "    g = getattr\n" + OK_BODY, "forbidden_call"),
        (SIG + "    t = type(dist)\n" + OK_BODY, "forbidden_call"),
        (SIG + "    __import__('os')\n" + OK_BODY, "dunder_name"),
        (SIG + "    k = ().__class__.__base__\n" + OK_BODY, "forbidden_attribute"),
        (SIG + "    x = np.load('data.npy')\n" + OK_BODY, "forbidden_attribute"),
        (SIG + "    x = np.random.rand(3)\n" + OK_BODY, "forbidden_attribute"),
        (SIG + "    dist.flags.writeable = True\n" + OK_BODY, "forbidden_attribute"),
        (SIG + "    d = dist.base\n" + OK_BODY, "forbidden_attribute"),
        (SIG + "    np.lib.format.open_memmap('x')\n" + OK_BODY, "forbidden_attribute"),
        (SIG + "    global X\n" + OK_BODY, "forbidden_statement"),
        ("class A:\n    pass\n" + SIG + OK_BODY, "forbidden_statement"),
        (SIG + "    with open('x') as f:\n        pass\n" + OK_BODY, "forbidden_statement"),
        (
            SIG + "    try:\n        pass\n    except:\n        pass\n" + OK_BODY,
            "forbidden_statement",
        ),
        (
            SIG + "    try:\n        pass\n    except BaseException:\n        pass\n" + OK_BODY,
            "forbidden_name",
        ),
        ("while True:\n    pass\n" + SIG + OK_BODY, "forbidden_statement"),
        ("print(1)\n" + SIG + OK_BODY, "forbidden_statement"),
        ("def score_candidates(current, candidates):\n" + OK_BODY, "signature_changed"),
        (SIG.replace("depot)", "depot=0)") + OK_BODY, "signature_changed"),
        ("def other(x):\n    return x\n", "signature_changed"),
        (SIG + OK_BODY + SIG + OK_BODY, "signature_changed"),
        (SIG + "    return (\n", "syntax_error"),
        (SIG + "    x = 1\n" * 2000 + OK_BODY, "too_large"),
    ],
)
def test_rule_breaking_code_is_rejected(code, category):
    report = check(code)
    assert not report.ok
    assert report.category == category, report


def test_good_code_passes():
    code = """
import heapq
from collections import Counter
from functools import lru_cache

WEIGHT = 0.5

def helper(x):
    return float(np.linalg.norm(x))

def score_candidates(current, candidates, rest_capacity, capacity, demand, dist, coords, depot):
    \"\"\"Docstring.\"\"\"
    d = dist[current, candidates].astype(float)
    try:
        ratio = demand[candidates] / rest_capacity
    except ZeroDivisionError:
        ratio = 0.0
    scores = -d + WEIGHT * ratio + sorted([1, 2])[0] * 0 + helper(coords[0]) * 0
    print("debug output is allowed and goes nowhere")
    scores[candidates == depot] = -np.inf
    return scores
"""
    assert check(code).ok
    for task in ("cvrp_construct", "pyvrp_accept"):
        t = get_task(task)
        assert check_code(t.initial_block, t.entry, t.params).ok
