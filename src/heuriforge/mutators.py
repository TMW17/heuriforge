"""Turning a model's answer into a child program, and the no-LLM random mutator.

``RandomMutator`` changes code without any language model: it nudges a constant, adds a
task-specific term, swaps an operator or drops a line. Run with ``--provider random`` it
is the baseline that shows how much the language model really adds; the evolution engine,
archive and evaluator are exactly the same. It also drives the offline mock model used by
the tests and the demo.
"""

from __future__ import annotations

import ast
import copy
import math

import numpy as np

from heuriforge.codeblock import END, START, EditError, apply_edits, strip_code_fences
from heuriforge.tasks.base import Task


def child_from_answer(parent_block: str, mutation: str, answer: dict) -> tuple[str, str]:
    """(new code block, idea) from a validated JSON answer."""
    if mutation == "diff":
        block = apply_edits(parent_block, answer["edits"])
    elif mutation == "rewrite":
        block = strip_code_fences(answer["code"])
    else:
        raise ValueError(f"unknown mutation {mutation!r}")
    lines = [line for line in block.split("\n") if line.strip() not in (START, END)]
    block = "\n".join(lines).strip("\n")
    if not block.strip():
        raise EditError("the new code is empty")
    return block, " ".join(str(answer.get("idea", "")).split())[:300]


class RandomMutator:
    def __init__(self, task: Task, rng: np.random.Generator):
        self.task = task
        self.rng = rng
        self.terms = [ast.parse(t, mode="eval").body for t in task.random_terms]

    def mutate(self, block: str) -> tuple[str, str]:
        tree = ast.parse(block)
        original = ast.unparse(tree)
        ops = [self._tweak_constant, self._add_term, self._swap_operator, self._drop_statement]
        weights = np.array([0.4, 0.35, 0.15, 0.1])
        for _ in range(20):
            op = ops[int(self.rng.choice(len(ops), p=weights))]
            result = op(copy.deepcopy(tree))
            if result is None:
                continue
            new_tree, idea = result
            code = ast.unparse(ast.fix_missing_locations(new_tree))
            if code != original:
                return code, idea
        return original, "no change"

    # ----------------------------------------------------------------- operators
    def _entry(self, tree: ast.Module) -> ast.FunctionDef | None:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == self.task.entry:
                return node
        return None

    def _tweak_constant(self, tree: ast.Module):
        nodes = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool)
        ]
        if not nodes:
            return None
        node = nodes[int(self.rng.integers(len(nodes)))]
        old = node.value
        if isinstance(old, int):
            node.value = old + int(self.rng.choice([-2, -1, 1, 2]))
        else:
            new = (
                old * math.exp(self.rng.normal(0.0, 0.5))
                if old != 0
                else float(self.rng.normal(0, 0.1))
            )
            node.value = float(f"{new:.4g}")
        return tree, f"change the constant {old} to {node.value}"

    def _add_term(self, tree: ast.Module):
        fn = self._entry(tree)
        if fn is None or not self.terms:
            return None
        returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
        if not returns:
            return None
        ret = returns[-1]
        k = int(self.rng.integers(len(self.terms)))
        term = copy.deepcopy(self.terms[k])
        text = self.task.random_terms[k]
        if self.task.combine == "bool":
            if self.rng.random() < 0.6:
                ret.value = ast.BoolOp(op=ast.Or(), values=[ret.value, term])
                idea = f"also accept when {text}"
            else:
                ret.value = ast.BoolOp(op=ast.And(), values=[ret.value, term])
                idea = f"only accept when {text} as well"
        else:
            coef = float(f"{self.rng.choice([-1, 1]) * 10 ** self.rng.uniform(-2, 0.5):.3g}")
            ret.value = ast.BinOp(
                left=ret.value,
                op=ast.Add(),
                right=ast.BinOp(left=ast.Constant(coef), op=ast.Mult(), right=term),
            )
            idea = f"add {coef} * ({text}) to the scores"
        return tree, idea

    def _swap_operator(self, tree: ast.Module):
        swaps = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt}
        binops = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}
        sites = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Compare):
                sites += [(n, i) for i, op in enumerate(n.ops) if type(op) in swaps]
            elif isinstance(n, ast.BinOp) and type(n.op) in binops:
                sites.append((n, None))
        if not sites:
            return None
        node, i = sites[int(self.rng.integers(len(sites)))]
        if i is None:
            node.op = binops[type(node.op)]()
        else:
            node.ops[i] = swaps[type(node.ops[i])]()
        return tree, "swap an operator"

    def _drop_statement(self, tree: ast.Module):
        fn = self._entry(tree)
        if fn is None:
            return None
        body = fn.body
        start = (
            1
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            else 0
        )
        sites = [i for i in range(start, len(body)) if not isinstance(body[i], ast.Return)]
        if not sites or len(body) - start < 2:
            return None
        del body[sites[int(self.rng.integers(len(sites)))]]
        return tree, "remove a line"
