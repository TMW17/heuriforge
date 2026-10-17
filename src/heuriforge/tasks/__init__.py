"""The two evolution tasks. Add a new task by subclassing ``Task`` and registering it here."""

from __future__ import annotations

from heuriforge.tasks.base import Task
from heuriforge.tasks.cvrp_construct import CvrpConstruct
from heuriforge.tasks.pyvrp_accept import PyvrpAccept

TASKS: dict[str, Task] = {t.name: t for t in (CvrpConstruct(), PyvrpAccept())}


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError:
        raise ValueError(f"unknown task {name!r}; choose from {', '.join(TASKS)}") from None
