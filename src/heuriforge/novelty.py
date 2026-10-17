"""Cheap novelty checks that run before any evaluation.

Two programs are *duplicates* when they differ only in comments, docstrings, formatting
or local variable names. They share a *structure* when, in addition, only their numeric
constants differ. Duplicates are rejected outright. Constant-only variants are allowed
up to a cap, so that the search keeps exploring new ideas instead of tuning one.
"""

from __future__ import annotations

import ast
import builtins
import hashlib

_BUILTIN_NAMES = set(dir(builtins))


class _Canonicalize(ast.NodeTransformer):
    def __init__(self, keep: set[str], abstract_constants: bool):
        self.keep = keep
        self.abstract = abstract_constants
        self.names: dict[str, str] = {}

    def _rename(self, name: str) -> str:
        if name in self.keep or name in _BUILTIN_NAMES:
            return name
        if name not in self.names:
            self.names[name] = f"v{len(self.names)}"
        return self.names[name]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id=self._rename(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return ast.copy_location(ast.arg(arg=self._rename(node.arg)), node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        value = node.value
        if (
            self.abstract
            and isinstance(value, (int, float, complex))
            and not isinstance(value, bool)
        ):
            return ast.copy_location(ast.Constant(value="#num"), node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = _drop_docstring(node.body)
        node.returns = None
        self.generic_visit(node)
        return node


def _drop_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    first = body[0] if body else None
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        body = body[1:] or [ast.Pass()]
    return body


def _canonical(code: str, keep: set[str], abstract_constants: bool) -> str:
    tree = ast.parse(code)
    tree.body = _drop_docstring(tree.body)
    for node in ast.walk(tree):  # strip type annotations: they do not change behaviour
        if isinstance(node, ast.arg):
            node.annotation = None
    imported = {
        (alias.asname or alias.name).split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    tree = _Canonicalize(keep | imported | functions, abstract_constants).visit(tree)
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


def code_hash(code: str, keep: set[str] = frozenset()) -> str:
    """Identical for programs that differ only in formatting, comments or local names."""
    return hashlib.sha1(_canonical(code, set(keep), False).encode()).hexdigest()[:16]


def structure_hash(code: str, keep: set[str] = frozenset()) -> str:
    """Identical for programs that differ only in their numeric constants (as well)."""
    return hashlib.sha1(_canonical(code, set(keep), True).encode()).hexdigest()[:16]
