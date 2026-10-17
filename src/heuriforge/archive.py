"""MAP-Elites archive with islands, and the parent-selection rule.

Each island keeps a small grid. A program's cell is given by two descriptors: how big its
code is, and how many of the function's inputs it uses. A cell holds the best program with
those descriptors, so the archive keeps *different kinds* of good programs instead of many
copies of the single best one. Islands evolve separately and exchange their best program
every ``migration_interval`` candidates (ring migration), which keeps diversity without
losing good ideas.
"""

from __future__ import annotations

from bisect import bisect_left

import numpy as np

from heuriforge.database import Program


def better(a: Program, b: Program) -> bool:
    """Lower training gap wins; on a tie, the smaller program wins."""
    if a.train_score is None:
        return False
    if b.train_score is None:
        return True
    if abs(a.train_score - b.train_score) > 1e-9:
        return a.train_score < b.train_score
    return a.complexity < b.complexity


class Archive:
    def __init__(self, islands: int, complexity_bins: list[int], params_bins: list[int]):
        self.islands = islands
        self.complexity_bins = sorted(complexity_bins)
        self.params_bins = sorted(params_bins)
        self.grid: list[dict[str, Program]] = [{} for _ in range(islands)]

    def cell_of(self, p: Program) -> str:
        c = bisect_left(self.complexity_bins, p.complexity)
        k = bisect_left(self.params_bins, p.params_used)
        return f"c{c}-p{k}"

    def add(self, p: Program) -> bool:
        """Offer an evaluated program to its island. True if it became its cell's elite."""
        p.cell = self.cell_of(p)
        cell = self.grid[p.island]
        current = cell.get(p.cell)
        if current is None or better(p, current):
            cell[p.cell] = p
            return True
        return False

    def put(self, island: int, cell: str, p: Program) -> None:
        """Restore a saved cell (when resuming a run)."""
        self.grid[island][cell] = p

    def elites(self, island: int | None = None) -> list[Program]:
        if island is None:
            return [p for g in self.grid for p in g.values()]
        return list(self.grid[island].values())

    def best(self, island: int | None = None) -> Program | None:
        best = None
        for p in self.elites(island):
            if best is None or better(p, best):
                best = p
        return best

    def sample_parent(
        self, island: int, rng: np.random.Generator, exploit_prob: float, lam: float
    ) -> Program:
        """Pick the program to mutate.

        With probability ``exploit_prob``: the island's best. Otherwise a weighted draw
        (as in ShinkaEvolve): weight = sigmoid(lam * normalised fitness) / (1 + children).
        Good programs are likely, but a program that already has many children becomes
        less likely, so the search does not keep hammering on the same parent.
        """
        elites = self.elites(island) or self.elites()
        if not elites:
            raise RuntimeError("the archive is empty")
        if len(elites) == 1:
            return elites[0]
        if rng.random() < exploit_prob:
            return self.best(island) or elites[0]
        fitness = np.array([-p.train_score for p in elites])
        spread = float(fitness.max() - fitness.min())
        z = (fitness - np.median(fitness)) / spread if spread > 0 else np.zeros_like(fitness)
        weights = 1.0 / (1.0 + np.exp(-lam * z)) / (1.0 + np.array([p.children for p in elites]))
        return elites[int(rng.choice(len(elites), p=weights / weights.sum()))]

    def inspirations(
        self, island: int, parent: Program, k: int, rng: np.random.Generator
    ) -> list[Program]:
        """Other programs to show in the prompt: the best one, then random other elites."""
        others = [p for p in self.elites(island) if p.id != parent.id and p.code != parent.code]
        if len(others) < k:
            seen = {p.id for p in others} | {parent.id}
            others += [p for p in self.elites() if p.id not in seen and p.code != parent.code]
        if not others or k <= 0:
            return []
        others = list({p.code: p for p in reversed(others)}.values())  # one per distinct code
        others.sort(key=lambda p: (p.train_score, p.complexity, p.id))
        chosen = [others[0]]
        rest = others[1:]
        while rest and len(chosen) < k:
            chosen.append(rest.pop(int(rng.integers(len(rest)))))
        return chosen
