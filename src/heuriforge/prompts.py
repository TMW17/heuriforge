"""Prompts and JSON answer formats for the language model.

Prompt layout matters for speed on a laptop. Everything that never changes during a run
(the rules and the task description) goes in the *system* message, which is identical in
every request; only the *user* message changes from one candidate to the next.

This split is not cosmetic. Qwen3.5 is a hybrid model (most of its layers are linear-
attention layers with a running state), and llama.cpp cannot cut such a state back to an
arbitrary shared prefix. Instead it saves "checkpoints", and it always saves one at the
start of the last user message. With all the fixed text before that point, the next
request restores the checkpoint and only processes the new user message.
"""

from __future__ import annotations

from heuriforge.database import Program
from heuriforge.sandbox.static import ALLOWED_MODULES
from heuriforge.tasks.base import Task

RULES = f"""\
You improve heuristics for combinatorial optimisation by editing Python code.
Rules for the code you write:
- Keep the function name and its parameters exactly as they are.
- Allowed imports: {", ".join(sorted(ALLOWED_MODULES))}. Nothing else: no files, no os, no \
sys, no random module, no printing.
- No attributes that start with an underscore (such as x._data or x.__class__), no names \
that start with two underscores, no classes, no global or nonlocal.
- No global variables that change between calls (when the function has a `state` input, \
use that instead). The code must be deterministic and fast.
Reply only in the format asked for at the end of the user message."""

DIFF_SCHEMA = {
    "type": "object",
    "properties": {
        "idea": {"type": "string"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"search": {"type": "string"}, "replace": {"type": "string"}},
                "required": ["search", "replace"],
                "additionalProperties": False,
            },
            "minItems": 1,
            "maxItems": 3,
        },
    },
    "required": ["idea", "edits"],
    "additionalProperties": False,
}

LESSON_SCHEMA = {
    "type": "object",
    "properties": {"lesson": {"type": "string"}},
    "required": ["lesson"],
    "additionalProperties": False,
}

# Full rewrites come back as a short plan and a plain ```python block, not inside JSON: small
# models write much cleaner code when it does not have to be escaped into a JSON string. The
# plan comes first because, without one, Qwen3.5-4B (thinking off) reasons in long code
# comments instead and runs into the token limit.
SCHEMAS = {"diff": DIFF_SCHEMA, "rewrite": None, "lesson": LESSON_SCHEMA}

DIFF_INSTRUCTION = """\
Make one focused improvement to the current program.
Reply with JSON: {"idea": "<one sentence: what you change and why>", "edits": [{"search": \
"<lines copied exactly from the current program>", "replace": "<the new lines>"}]}
Use 1 to 3 edits. Every "search" must be copied character for character from the current \
program, including indentation, and must appear in it only once. Indent the "replace" lines \
exactly as they must appear in the program."""

REWRITE_INSTRUCTION = """\
Write a new version of the code that tries a different idea; it may be very different from \
the current program.
Reply in exactly this format and nothing else:
Plan: <two or three short sentences: the idea, and how the code does it>
```python
<the complete new code: imports and helper functions if you need them, and \
def {entry}({params}):>
```
Think in the plan, not in the code: write plain code with no comments and no docstrings, \
at most about 40 lines."""

LESSON_INSTRUCTION = """\
In one or two sentences, state a general lesson for writing better versions of this \
heuristic, based on the difference between the two versions. Do not repeat the code.
Reply with JSON: {"lesson": "<the lesson>"}"""


def task_prefix(task: Task) -> str:
    """The task description: fixed for the whole run."""
    return (
        f"# Task: {task.title}\n\n{task.description}\n\n"
        f"## The function you are improving\n{task.api_doc}\n\n"
        f"## How programs are scored\n{task.scoring_doc}"
    )


def system_message(task: Task) -> dict:
    """Rules + task description: byte-for-byte identical in every request of a run."""
    return {"role": "system", "content": f"{RULES}\n\n{task_prefix(task)}"}


def code(text: str) -> str:
    return f"```python\n{text.strip(chr(10))}\n```"


def mutation_messages(
    task: Task,
    parent: Program,
    mutation: str,
    inspirations: list[Program],
    lessons: list[str],
    last_error: str | None = None,
) -> list[dict]:
    parts = [
        f"## Current program\nTraining gap: {parent.train_score:.2f}% (lower is better).\n\n{code(parent.code)}\n",
    ]
    if last_error:
        parts.append(
            f"## A recent change to this program failed\n{last_error.strip()}\nAvoid this mistake.\n"
        )
    if inspirations:
        parts.append("## Other programs for inspiration")
        for p in inspirations:
            parts.append(f"### Training gap {p.train_score:.2f}%\n{code(p.code)}")
        parts.append("")
    if lessons:
        parts.append(
            "## Lessons from earlier attempts\n" + "\n".join(f"- {t}" for t in lessons) + "\n"
        )
    if mutation == "diff":
        instruction = DIFF_INSTRUCTION
    else:
        instruction = REWRITE_INSTRUCTION.format(entry=task.entry, params=", ".join(task.params))
    parts.append(f"## Your answer\n{instruction}")
    return [system_message(task), {"role": "user", "content": "\n".join(parts)}]


def lesson_messages(task: Task, better: Program, worse: Program) -> list[dict]:
    body = (
        "## Two versions of the function, with their training gap (lower is better)\n\n"
        f"Better ({better.train_score:.2f}%):\n{code(better.code)}\n\n"
        f"Worse ({worse.train_score:.2f}%):\n{code(worse.code)}\n\n"
        f"## Your answer\n{LESSON_INSTRUCTION}"
    )
    return [system_message(task), {"role": "user", "content": body}]
