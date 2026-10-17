"""The evolvable part of a program, and the edits a language model proposes for it.

A program is a small Python module. Only the code between the two marker comments
changes during evolution; everything outside them (the imports) stays fixed:

    import numpy as np

    # EVOLVE-BLOCK-START
    def score_candidates(...):
        ...
    # EVOLVE-BLOCK-END
"""

from __future__ import annotations

import ast
import itertools
import re
import textwrap

START = "# EVOLVE-BLOCK-START"
END = "# EVOLVE-BLOCK-END"


class EditError(ValueError):
    """A proposed edit could not be applied. The message is shown to the model."""


def build_program(header: str, block: str) -> str:
    return f"{header.rstrip()}\n\n{START}\n{block.strip(chr(10))}\n{END}\n"


def split_program(program: str) -> tuple[str, str, str]:
    """Split a program into (text before the block, the block, text after the block)."""
    lines = program.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.strip() == START]
    ends = [i for i, line in enumerate(lines) if line.strip() == END]
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise ValueError("a program needs exactly one EVOLVE-BLOCK-START ... EVOLVE-BLOCK-END pair")
    s, e = starts[0], ends[0]
    return "".join(lines[: s + 1]), "".join(lines[s + 1 : e]), "".join(lines[e:])


def extract_block(program: str) -> str:
    return split_program(program)[1].strip("\n")


def replace_block(program: str, block: str) -> str:
    before, _, after = split_program(program)
    return f"{before}{block.strip(chr(10))}\n{after}"


# --------------------------------------------------------------------------- edits
_TOP_LEVEL_OK = (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)


def _plausible(block: str) -> bool:
    """The block compiles and has only definitions at the top level."""
    try:
        tree = ast.parse(block)
        compile(tree, "<block>", "exec")  # also catches `return` outside a function
    except (SyntaxError, ValueError):
        return False
    return all(
        isinstance(stmt, _TOP_LEVEL_OK)
        or (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
        for stmt in tree.body
    )


def apply_edits(block: str, edits: list[dict]) -> str:
    """Apply SEARCH/REPLACE edits one after the other.

    Each SEARCH text must appear exactly once in the current block. When it does not
    appear verbatim, a second attempt ignores differences in indentation and trailing
    spaces (the most common mistake small models make), and re-indents the replacement.
    Edits whose SEARCH and REPLACE are identical are skipped.

    When the result does not compile, or leaves statements at the top level, the
    REPLACE texts were probably indented wrongly (seen often with Qwen3.5-4B: lines
    without the function's indentation, or a first line without indentation followed by
    lines with it). The edits are then applied again with the replacements re-indented
    to the matched lines, and the first version that compiles is kept.
    """
    if not edits:
        raise EditError("no edits were proposed")
    result = _apply_all(block, edits, None)
    if _plausible(result) or not _plausible(block):
        return result
    for modes in itertools.product(_MODES, repeat=len(edits)):
        try:
            repaired = _apply_all(block, edits, modes)
        except EditError:
            continue
        if _plausible(repaired):
            return repaired
    return result


_MODES = ("as_is", "dedent", "first_line")


def _reindent(replace: str, base: str, mode: str) -> str:
    """Re-indent REPLACE so that it starts at indentation ``base``."""
    lines = replace.split("\n")
    if mode == "dedent":  # keep the relative indentation, move the whole text to base
        lines = [
            base + line if line.strip() else "" for line in textwrap.dedent(replace).split("\n")
        ]
    elif mode == "first_line":  # only the first line lost its indentation
        lines = [base + lines[0].lstrip()] + lines[1:]
    return "\n".join(lines)


def _apply_all(block: str, edits: list[dict], modes: tuple[str, ...] | None) -> str:
    changed = False
    for k, edit in enumerate(edits, start=1):
        search = edit.get("search", "")
        replace = edit.get("replace", "")
        if not search.strip():
            raise EditError(f"edit {k}: SEARCH is empty; copy the exact lines you want to change")
        if search == replace:
            continue  # a no-op edit next to real ones: ignore it
        count = block.count(search)
        if count > 1:
            raise EditError(
                f"edit {k}: SEARCH appears {count} times; include more surrounding lines so it is unique"
            )
        if count == 1:
            start = block.index(search)
            line_start = block.rfind("\n", 0, start) + 1
            prefix = block[line_start:start]
            mode = modes[k - 1] if modes else "as_is"
            if prefix.strip():  # SEARCH starts in the middle of a line: splice as it is
                block = block[:start] + replace + block[start + len(search) :]
            elif mode != "as_is":
                line_end = block.find("\n", line_start)
                first = block[line_start : line_end if line_end >= 0 else len(block)]
                new = _reindent(replace, _indent(first), mode)
                block = block[:line_start] + new + block[start + len(search) :]
            elif prefix and replace[:1] in (" ", "\t"):
                # SEARCH skipped the line's indentation but REPLACE brings its own:
                # replace from the start of the line, or the indentation doubles.
                block = block[:line_start] + replace + block[start + len(search) :]
            else:
                block = block[:start] + replace + block[start + len(search) :]
        else:
            block = _apply_loose(block, search, replace, k)
        changed = True
    if not changed:
        raise EditError("every SEARCH is identical to its REPLACE, so nothing changes")
    return block


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _apply_loose(block: str, search: str, replace: str, k: int) -> str:
    block_lines = block.split("\n")
    search_lines = [line for line in search.strip("\n").split("\n")]
    while search_lines and not search_lines[-1].strip():
        search_lines.pop()
    m = len(search_lines)
    matches = [
        i
        for i in range(len(block_lines) - m + 1)
        if all(block_lines[i + j].strip() == search_lines[j].strip() for j in range(m))
    ]
    if not matches:
        raise EditError(
            f"edit {k}: SEARCH does not match the current code; copy the lines exactly as they are"
        )
    if len(matches) > 1:
        raise EditError(f"edit {k}: SEARCH matches {len(matches)} places; include more lines")
    i = matches[0]
    have, want = _indent(block_lines[i]), _indent(search_lines[0])
    new_lines = []
    for line in replace.strip("\n").split("\n") if replace.strip() else []:
        if not line.strip():
            new_lines.append("")
        elif len(have) >= len(want):
            new_lines.append(have[: len(have) - len(want)] + line)
        else:
            cut = len(want) - len(have)
            new_lines.append(line[cut:] if line[:cut].strip() == "" else line.lstrip())
    return "\n".join(block_lines[:i] + new_lines + block_lines[i + m :])


# --------------------------------------------------------------------------- measures
def complexity(code: str) -> int:
    """Size of the code: the number of nodes in its syntax tree."""
    try:
        return sum(1 for _ in ast.walk(ast.parse(code)))
    except SyntaxError:
        return 0


def params_used(code: str, entry: str, params: list[str]) -> int:
    """How many of the entry function's inputs its body actually reads."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry:
            used = {
                n.id
                for n in ast.walk(node)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in params
            }
            return len(used)
    return 0


def strip_code_fences(text: str) -> str:
    """Remove ```python fences a model may wrap around code in a JSON string."""
    m = re.match(r"^\s*```[a-zA-Z0-9_-]*\n(.*?)\n?```\s*$", text, flags=re.S)
    return m.group(1) if m else text
