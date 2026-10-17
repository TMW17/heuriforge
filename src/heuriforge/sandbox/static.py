"""Static checks: reject unsafe or rule-breaking code before it ever runs.

These checks are the first of two walls. The second is the sandbox process
(``runner.py`` / ``worker.py``): restricted builtins, a guarded import, CPU, file and
memory limits, and a fresh process per evaluation. A candidate has to get past both to
do anything outside its job, so the rules here can stay simple and strict.

Every rejection has a category. The categories are counted per run (the reward-hacking
audit), because a model that keeps reaching for files or introspection is trying to
win in a way the task does not allow.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

ALLOWED_MODULES = {
    "math",
    "numpy",
    "heapq",
    "itertools",
    "functools",
    "collections",
    "bisect",
    "statistics",
}

BANNED_CALLS = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "hasattr", "input", "breakpoint", "help", "exit",
    "quit", "memoryview", "super", "dir", "id", "hash", "type", "object", "classmethod",
    "staticmethod", "property",
}  # fmt: skip

BANNED_NAMES = {"BaseException", "KeyboardInterrupt", "SystemExit", "GeneratorExit"}

BANNED_ATTRIBUTES = {
    # frames, code objects and class internals (classic sandbox escapes)
    "f_globals", "f_locals", "f_back", "f_builtins", "f_code", "tb_frame", "tb_next",
    "gi_frame", "gi_code", "ag_frame", "cr_frame", "co_code", "mro", "func_globals",
    # numpy file input/output, randomness and low-level access
    "load", "loads", "save", "savez", "savez_compressed", "loadtxt", "savetxt",
    "genfromtxt", "fromfile", "tofile", "memmap", "ctypeslib", "ctypes", "lib",
    "distutils", "f2py", "testing", "random", "DataSource", "fromregex", "dump", "dumps",
    "core", "rec", "info", "show_config", "show_runtime", "get_include", "os", "sys",
    "system", "popen", "subprocess", "builtins", "compat", "matlib",
    # making read-only inputs writable again
    "flags", "setflags", "base",
}  # fmt: skip

MAX_CHARS = 12_000
MAX_NODES = 3_000


@dataclass
class StaticReport:
    ok: bool
    category: str | None = None
    message: str = ""
    line: int | None = None

    @classmethod
    def fail(cls, category: str, message: str, node: ast.AST | None = None) -> StaticReport:
        return cls(False, category, message, getattr(node, "lineno", None))


def check_code(code: str, entry: str, params: list[str]) -> StaticReport:
    """Check the code of an evolve block (or a whole program)."""
    if len(code) > MAX_CHARS:
        return StaticReport.fail("too_large", f"the code is longer than {MAX_CHARS} characters")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return StaticReport(False, "syntax_error", f"syntax error: {exc.msg}", exc.lineno)
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        return StaticReport.fail("too_large", f"the code has more than {MAX_NODES} syntax nodes")

    for node in nodes:
        report = _check_node(node)
        if report is not None:
            return report

    entries = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry]
    if len(entries) != 1:
        return StaticReport.fail("signature_changed", f"define the function {entry}() exactly once")
    fn = entries[0]
    a = fn.args
    got = [arg.arg for arg in a.posonlyargs + a.args]
    if got != params or a.vararg or a.kwarg or a.kwonlyargs or a.defaults or a.kw_defaults:
        return StaticReport.fail(
            "signature_changed",
            f"keep the signature exactly: def {entry}({', '.join(params)})",
            fn,
        )
    if fn.decorator_list:
        return StaticReport.fail("signature_changed", f"{entry}() must not have decorators", fn)
    for stmt in tree.body:
        if not isinstance(
            stmt, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.Expr)
        ):
            return StaticReport.fail(
                "forbidden_statement",
                "only imports, constants and function definitions are allowed at the top level",
                stmt,
            )
        if isinstance(stmt, ast.Expr) and not (
            isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)
        ):
            return StaticReport.fail(
                "forbidden_statement", "top-level expressions are not allowed", stmt
            )
    return StaticReport(True)


def _check_node(node: ast.AST) -> StaticReport | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name not in ALLOWED_MODULES:
                return StaticReport.fail(
                    "forbidden_import",
                    f"import {alias.name} is not allowed (allowed: {', '.join(sorted(ALLOWED_MODULES))})",
                    node,
                )
    elif isinstance(node, ast.ImportFrom):
        if node.level or node.module not in ALLOWED_MODULES:
            return StaticReport.fail(
                "forbidden_import", f"from {node.module} import ... is not allowed", node
            )
        for alias in node.names:
            if alias.name == "*" or alias.name.startswith("_") or alias.name in BANNED_ATTRIBUTES:
                return StaticReport.fail(
                    "forbidden_import",
                    f"from {node.module} import {alias.name} is not allowed",
                    node,
                )
    elif isinstance(node, ast.Name):
        if node.id.startswith("__"):
            return StaticReport.fail("dunder_name", f"the name {node.id} is not allowed", node)
        if node.id in BANNED_NAMES:
            return StaticReport.fail("forbidden_name", f"the name {node.id} is not allowed", node)
        if node.id in BANNED_CALLS:
            return StaticReport.fail("forbidden_call", f"{node.id} is not allowed", node)
    elif isinstance(node, ast.Attribute):
        if node.attr.startswith("_"):
            return StaticReport.fail(
                "forbidden_attribute", f"private attributes like .{node.attr} are not allowed", node
            )
        if node.attr in BANNED_ATTRIBUTES:
            return StaticReport.fail("forbidden_attribute", f".{node.attr} is not allowed", node)
    elif isinstance(node, (ast.Global, ast.Nonlocal)):
        return StaticReport.fail("forbidden_statement", "global and nonlocal are not allowed", node)
    elif isinstance(node, ast.ClassDef):
        return StaticReport.fail("forbidden_statement", "classes are not allowed", node)
    elif isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await)):
        return StaticReport.fail("forbidden_statement", "async code is not allowed", node)
    elif isinstance(node, ast.With):
        return StaticReport.fail("forbidden_statement", "with blocks are not allowed", node)
    elif isinstance(node, ast.ExceptHandler):
        if node.type is None:
            return StaticReport.fail(
                "forbidden_statement", "a bare except: is not allowed; catch specific errors", node
            )
    elif isinstance(node, ast.arg):
        if node.arg.startswith("__"):
            return StaticReport.fail("dunder_name", f"the name {node.arg} is not allowed", node)
    elif isinstance(node, ast.FunctionDef) and node.name.startswith("__"):
        return StaticReport.fail("dunder_name", f"the name {node.name} is not allowed", node)
    return None
