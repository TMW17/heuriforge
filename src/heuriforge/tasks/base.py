"""What every task provides: the evolvable function, how to run it, and how to score it.

A task has two halves. The *worker half* (``prepare`` and ``run``) runs inside the
sandbox process: it loads an instance and calls the candidate function inside a fixed,
trusted skeleton. The *parent half* (``seeds`` and ``job_options``) runs in the main
process, which then checks the returned routes and recomputes their cost itself
(``heuriforge.evaluator``). The candidate never reports its own score.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from heuriforge.instances import Instance


class InvalidOutput(Exception):
    """The candidate returned something the task cannot use."""


class ModifiedInput(Exception):
    """The candidate changed data it was only allowed to read."""


@dataclass
class Item:
    """One unit of evaluation: an instance, plus a seed for randomised tasks."""

    instance: Instance
    seed: int = 0

    @property
    def key(self) -> str:
        return f"{self.instance.name}#{self.seed}"


@dataclass
class StageResult:
    stage: str
    ok: bool
    score: float | None = None  # mean gap to the best-known solution, in %; lower is better
    per_item: dict[str, float] = field(default_factory=dict)  # item key -> gap %
    costs: dict[str, int] = field(default_factory=dict)  # item key -> verified route cost
    seconds: float = 0.0  # wall-clock time of the stage
    cpu_s: float = 0.0  # CPU time used by the candidate's runs
    error_type: str | None = None
    error: str | None = None


def readonly(array: np.ndarray) -> np.ndarray:
    """A copy the candidate cannot modify (writing to it raises an error)."""
    out = np.array(array, copy=True)
    out.flags.writeable = False
    return out


def fingerprint(arrays: list[np.ndarray]) -> bytes:
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.digest()


class Task:
    """Base class. Subclasses fill in the attributes and the worker-side methods."""

    name: str = ""
    entry: str = ""  # the function the language model evolves
    params: list[str] = []  # its exact parameter list
    header: str = "import math\n\nimport numpy as np\n"  # fixed code above the evolve block
    initial_block: str = ""

    # Text for the prompt.
    title: str = ""
    description: str = ""
    api_doc: str = ""
    scoring_doc: str = ""

    # MAP-Elites axes: code size (syntax nodes) and number of inputs used.
    complexity_bins: list[int] = [60, 120, 240]
    params_bins: list[int] = [3, 4, 6]

    # Ingredients for the random-mutation baseline (no language model).
    random_terms: list[str] = []
    combine: str = "add"  # "add": scores + c * term;  "bool": rule or/and term

    # Settings this task understands (tasks.<name> in config/settings.yaml).
    option_keys: frozenset[str] = frozenset(
        {"per_instance_cpu_s", "complexity_bins", "params_bins"}
    )

    def check_options(self, options: dict) -> dict:
        unknown = set(options) - self.option_keys
        if unknown:
            raise ValueError(
                f"unknown setting(s) for task {self.name}: {', '.join(sorted(unknown))} "
                f"(known: {', '.join(sorted(self.option_keys))})"
            )
        return options

    # ----------------------------------------------------------------- parent side
    def seeds(self, split: str, options: dict) -> list[int]:
        return [0]

    def items(self, instances: list[Instance], split: str, options: dict) -> list[Item]:
        return [Item(inst, seed) for inst in instances for seed in self.seeds(split, options)]

    def job_options(self, split: str, options: dict) -> dict:
        """The options the worker needs for this split."""
        return {}

    def cpu_limit(self, split: str, options: dict) -> float:
        """CPU seconds allowed per item."""
        return float(options.get("per_instance_cpu_s", 30))

    # ----------------------------------------------------------------- worker side
    def prepare(self, path: str, seed: int, options: dict) -> object:
        """Load what ``run`` needs. Trusted code; runs before the resource limits."""
        raise NotImplementedError

    def run(self, fn, prepared: object, seed: int, options: dict) -> dict:
        """Run the candidate function inside the trusted skeleton.

        Returns ``{"routes": [[...], ...]}`` plus any extra JSON-safe details. Raises
        ``InvalidOutput`` or ``ModifiedInput`` when the candidate misbehaves.
        """
        raise NotImplementedError
