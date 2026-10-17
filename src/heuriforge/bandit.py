"""Cost-aware Thompson sampling over (model, mutation type) arms.

Each arm keeps a Beta(1 + successes, 1 + failures) belief about how often it produces a
child that beats its parent. Every candidate, each arm draws a sample from its belief,
divides it by the arm's relative cost, and the highest value wins. Arms that work get
used more; a cheap model that works almost as well as an expensive one gets used more
too. The counts live in the database, so a resumed run keeps what it learned.
"""

from __future__ import annotations

import numpy as np

from heuriforge.database import Database


def arm_key(model: str, mutation: str) -> str:
    return f"{model}|{mutation}"


class Bandit:
    def __init__(self, arms: dict[str, float], db: Database | None = None):
        """``arms`` maps an arm key to its relative cost."""
        if not arms:
            raise ValueError("the bandit needs at least one arm")
        saved = db.arms() if db is not None else {}
        self.db = db
        self.stats = {
            key: {
                "pulls": saved.get(key, {}).get("pulls", 0),
                "successes": saved.get(key, {}).get("successes", 0),
                "failures": saved.get(key, {}).get("failures", 0),
                "cost": float(cost),
            }
            for key, cost in arms.items()
        }

    def choose(self, rng: np.random.Generator) -> str:
        best_key, best_value = None, -1.0
        for key, s in self.stats.items():
            value = rng.beta(1 + s["successes"], 1 + s["failures"]) / max(s["cost"], 1e-9)
            if value > best_value:
                best_key, best_value = key, value
        return best_key

    def update(self, key: str, success: bool) -> None:
        s = self.stats[key]
        s["pulls"] += 1
        s["successes" if success else "failures"] += 1
        if self.db is not None:
            self.db.save_arm(key, s["pulls"], s["successes"], s["failures"], s["cost"])

    def table(self) -> list[dict]:
        rows = []
        for key, s in self.stats.items():
            model, mutation = key.split("|", 1)
            rate = s["successes"] / s["pulls"] if s["pulls"] else 0.0
            rows.append({"model": model, "mutation": mutation, **s, "success_rate": rate})
        return rows
