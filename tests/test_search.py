"""Archive, parent selection, bandit, prompts, statistics."""

import numpy as np
import pytest

from heuriforge.archive import Archive
from heuriforge.bandit import Bandit, arm_key
from heuriforge.database import Database, Program
from heuriforge.prompts import DIFF_SCHEMA, lesson_messages, mutation_messages, task_prefix
from heuriforge.stats import a12_lower_is_better, bootstrap_ci, compare, holm, wilcoxon_p
from heuriforge.tasks import get_task


def prog(pid, score, complexity=50, params=4, island=0, children=0, code=None):
    return Program(
        island=island, code=code or f"# {pid}", id=pid, train_score=score,
        complexity=complexity, params_used=params, children=children, status="evaluated",
    )  # fmt: skip


def test_archive_keeps_the_best_per_cell():
    a = Archive(2, [60, 120], [4, 6])
    assert a.add(prog(1, 30.0, 50, 4))
    assert not a.add(prog(2, 31.0, 55, 3))  # same cell, worse
    assert a.add(prog(3, 29.0, 58, 4))  # same cell, better
    assert a.add(prog(4, 40.0, 200, 8))  # new cell
    assert {p.id for p in a.elites(0)} == {3, 4}
    assert a.best(0).id == 3 and a.best(1) is None
    assert a.cell_of(prog(5, 1.0, 61, 5)) == "c1-p1"


def test_parent_selection_prefers_good_and_unexplored_programs():
    a = Archive(1, [60, 120, 240], [4, 6])
    a.add(prog(1, 10.0, 50))
    a.add(prog(2, 20.0, 100))
    a.add(prog(3, 30.0, 200))
    rng = np.random.default_rng(0)
    picks = [a.sample_parent(0, rng, exploit_prob=0.0, lam=10.0).id for _ in range(2000)]
    counts = np.bincount(picks, minlength=4)
    assert counts[1] > counts[2] > counts[3]
    a.grid[0]["c0-p0"].children = 50  # the best program has had many children already
    picks = [a.sample_parent(0, rng, exploit_prob=0.0, lam=10.0).id for _ in range(2000)]
    assert np.bincount(picks, minlength=4)[1] < counts[1]


def test_inspirations_exclude_the_parent():
    a = Archive(1, [60, 120, 240], [4, 6])
    for pid, score, cx in [(1, 10.0, 50), (2, 20.0, 100), (3, 30.0, 200), (4, 40.0, 300)]:
        a.add(prog(pid, score, cx))
    rng = np.random.default_rng(0)
    parent = a.best(0)
    ins = a.inspirations(0, parent, 2, rng)
    assert len(ins) == 2 and parent.id not in {p.id for p in ins} and ins[0].id == 2


def test_bandit_learns_and_persists(tmp_path):
    db = Database(tmp_path / "p.sqlite")
    good, bad = arm_key("m", "diff"), arm_key("m", "rewrite")
    bandit = Bandit({good: 1.0, bad: 1.0}, db)
    rng = np.random.default_rng(1)
    for _ in range(300):
        arm = bandit.choose(rng)
        bandit.update(
            arm,
            success=(arm == good and rng.random() < 0.5) or (arm == bad and rng.random() < 0.05),
        )
    assert bandit.stats[good]["pulls"] > 3 * bandit.stats[bad]["pulls"]
    again = Bandit({good: 1.0, bad: 1.0}, Database(tmp_path / "p.sqlite"))
    assert again.stats[good] == bandit.stats[good]


def test_bandit_accounts_for_cost():
    cheap, dear = arm_key("small", "diff"), arm_key("large", "diff")
    bandit = Bandit({cheap: 0.25, dear: 1.0})
    for key in (cheap, dear):  # same success rate
        bandit.stats[key].update(pulls=100, successes=20, failures=80)
    rng = np.random.default_rng(0)
    picks = [bandit.choose(rng) for _ in range(500)]
    assert picks.count(cheap) > 450


def test_fixed_text_is_in_the_system_message():
    """Everything fixed sits before the last user message, where llama.cpp checkpoints."""
    task = get_task("cvrp_construct")
    a = mutation_messages(task, prog(1, 20.0, code="def a(): pass"), "diff", [], [])
    b = mutation_messages(
        task,
        prog(2, 25.0, code="def b(): pass"),
        "rewrite",
        [prog(3, 19.0, code="def c(): pass")],
        ["lesson"],
    )
    reflection = lesson_messages(task, prog(4, 18.0, code="def d(): pass"), prog(5, 30.0))
    assert a[0] == b[0] == reflection[0]  # one identical system message for every request
    assert task_prefix(task) in a[0]["content"]
    assert [m["role"] for m in a] == ["system", "user"]
    assert a[1]["content"].startswith("## Current program")
    assert "def a(): pass" in a[1]["content"] and '"edits"' in a[1]["content"]
    assert (
        "def score_candidates(current, candidates" in b[1]["content"]
        and "lesson" in b[1]["content"]
    )
    assert DIFF_SCHEMA["properties"]["edits"]["maxItems"] == 3


def test_statistics():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert a12_lower_is_better([1, 2, 3], [4, 5, 6]) == 1.0
    assert a12_lower_is_better([1, 2], [1, 2]) == 0.5
    assert wilcoxon_p([1, 2, 3], [1, 2, 3]) == 1.0
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    b = [x + 1.0 for x in a]
    mean, lo, hi = bootstrap_ci(a, b)
    assert mean == pytest.approx(-1.0) and lo == pytest.approx(-1.0) and hi == pytest.approx(-1.0)
    c = compare(a, b)
    assert (
        c["wins"] == 8
        and c["p_value"] < 0.01
        and c["effect"] in {"small", "medium", "large", "negligible"}
    )
