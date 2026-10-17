"""Load a candidate program with restricted builtins and a guarded import.

This is the runtime half of the rules in ``static.py``: even if some trick got past
the static checks, the candidate's global namespace has no ``open``, no ``getattr``,
no ``__import__`` beyond the allowed modules, and nothing that reaches the file system.
"""

from __future__ import annotations

import builtins
import importlib

from heuriforge.sandbox.static import ALLOWED_MODULES, BANNED_ATTRIBUTES

_SAFE_NAMES = [
    "abs", "all", "any", "bool", "callable", "chr", "complex", "dict", "divmod",
    "enumerate", "filter", "float", "frozenset", "int", "isinstance", "iter", "len",
    "list", "map", "max", "min", "next", "ord", "pow", "range", "reversed", "round",
    "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    # exceptions a heuristic may reasonably raise or catch
    "ArithmeticError", "AssertionError", "Exception", "FloatingPointError", "IndexError",
    "KeyError", "LookupError", "NotImplementedError", "OverflowError", "RuntimeError",
    "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
]  # fmt: skip


def _no_print(*args, **kwargs) -> None:
    """Printing is allowed but goes nowhere (output would only slow the search down)."""


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.startswith("numpy."):
        # numpy's C code imports its own helper modules lazily (for example
        # numpy._core._methods for ndarray.mean). Those imports run with the builtins of
        # the frame that called numpy, i.e. the candidate's, so they arrive here. The
        # candidate itself cannot write such an import: the static checks only accept the
        # exact module names in ALLOWED_MODULES and reject every name with an underscore.
        leaf = importlib.import_module(name)
        return leaf if fromlist else importlib.import_module("numpy")
    if level != 0 or name not in ALLOWED_MODULES:
        raise ImportError(f"import of {name!r} is not allowed")
    module = importlib.import_module(name)
    for attr in fromlist or ():
        if attr == "*" or attr.startswith("_") or attr in BANNED_ATTRIBUTES:
            raise ImportError(f"from {name} import {attr} is not allowed")
    return module


def safe_builtins() -> dict:
    table = {name: getattr(builtins, name) for name in _SAFE_NAMES}
    table["print"] = _no_print
    table["__import__"] = _guarded_import
    table["__build_class__"] = None  # no classes
    return table


def load_program(program: str, entry: str):
    """Execute the program in a fresh restricted namespace and return its entry function."""
    namespace = {"__builtins__": safe_builtins(), "__name__": "candidate"}
    code = compile(program, "<candidate>", "exec")
    exec(code, namespace)  # noqa: S102 - this is the sandboxed child process
    fn = namespace.get(entry)
    if not callable(fn):
        raise TypeError(f"the program does not define a function called {entry}")
    return fn
