import pytest

from heuriforge.codeblock import (
    EditError,
    apply_edits,
    build_program,
    complexity,
    extract_block,
    params_used,
    replace_block,
    split_program,
    strip_code_fences,
)
from heuriforge.novelty import code_hash, structure_hash

BLOCK = """def f(a, b, c):
    x = a + 1
    if x > 2:
        x = x * 2
    return x + b"""


def test_program_round_trip():
    program = build_program("import numpy as np\n", BLOCK)
    assert extract_block(program) == BLOCK
    before, block, after = split_program(program)
    assert before.startswith("import numpy") and after.strip() == "# EVOLVE-BLOCK-END"
    assert (
        extract_block(replace_block(program, "def f(a, b, c):\n    return 0"))
        == "def f(a, b, c):\n    return 0"
    )
    with pytest.raises(ValueError):
        split_program("def f(): pass")


def test_exact_edit():
    out = apply_edits(BLOCK, [{"search": "    x = a + 1", "replace": "    x = a + 2"}])
    assert "x = a + 2" in out and "x = a + 1" not in out


def test_several_edits_apply_in_order():
    out = apply_edits(
        BLOCK,
        [
            {"search": "x = a + 1", "replace": "x = a + 5"},
            {"search": "x = a + 5", "replace": "x = a + 6"},
        ],
    )
    assert "x = a + 6" in out


def test_edit_with_wrong_indentation_is_repaired():
    # The model forgot the indentation of the block: matched loosely, then re-indented.
    edit = {"search": "if x > 2:\n    x = x * 2", "replace": "if x > 3:\n    x = x * 3"}
    out = apply_edits(BLOCK, [edit])
    assert "    if x > 3:\n        x = x * 3" in out
    compile(out, "<test>", "exec")


def test_edit_errors_are_explained():
    with pytest.raises(EditError, match="does not match"):
        apply_edits(BLOCK, [{"search": "y = 3", "replace": "y = 4"}])
    with pytest.raises(EditError, match="empty"):
        apply_edits(BLOCK, [{"search": "  ", "replace": "y = 4"}])
    with pytest.raises(EditError, match="appears 2 times"):
        apply_edits(BLOCK, [{"search": "x = ", "replace": "y = "}])
    with pytest.raises(EditError, match="identical"):
        apply_edits(BLOCK, [{"search": "x = a + 1", "replace": "x = a + 1"}])
    with pytest.raises(EditError):
        apply_edits(BLOCK, [])


def test_measures():
    assert complexity(BLOCK) > 10
    assert complexity("def (") == 0
    assert params_used(BLOCK, "f", ["a", "b", "c"]) == 2


def test_strip_code_fences():
    assert strip_code_fences("```python\nx = 1\n```") == "x = 1"
    assert strip_code_fences("x = 1") == "x = 1"


def test_formatting_comments_and_names_do_not_make_a_new_program():
    renamed = '''def f(a, b, c):
    """A docstring."""
    # a comment
    value = a + 1
    if value > 2:
        value = value * 2
    return value + b'''
    assert code_hash(BLOCK) == code_hash(renamed)
    assert structure_hash(BLOCK) == structure_hash(renamed)


def test_constants_change_the_program_but_not_its_structure():
    tuned = BLOCK.replace("a + 1", "a + 7")
    assert code_hash(BLOCK) != code_hash(tuned)
    assert structure_hash(BLOCK) == structure_hash(tuned)


def test_new_logic_is_a_new_structure():
    other = BLOCK.replace("x + b", "x - b")
    assert structure_hash(BLOCK) != structure_hash(other)
    called = BLOCK.replace("x * 2", "max(x, 2)")
    assert structure_hash(BLOCK) != structure_hash(called)


def test_search_without_indentation_and_replace_with_it():
    # Seen with Qwen3.5-4B: SEARCH skips the indentation, REPLACE includes it.
    out = apply_edits(BLOCK, [{"search": "x = a + 1", "replace": "    x = a + 3"}])
    assert "\n    x = a + 3\n" in out
    compile(out, "<test>", "exec")


def test_no_op_edits_are_skipped():
    out = apply_edits(
        BLOCK,
        [
            {"search": "x = a + 1", "replace": "x = a + 4"},
            {"search": "return x + b", "replace": "return x + b"},
        ],
    )
    assert "x = a + 4" in out


def test_replacements_that_lost_their_indentation_are_reindented():
    # Seen with Qwen3.5-4B: REPLACE lines at column 0, which would put `return` at the
    # top level, and a REPLACE that indents everything one level too deep.
    tail = {"search": "    return x + b", "replace": "y = x + b\nreturn y"}
    out = apply_edits(BLOCK, [tail])
    assert out.endswith("    y = x + b\n    return y")
    deep = {"search": "    x = a + 1", "replace": "        x = a + 2\n        x = x - 1"}
    out = apply_edits(BLOCK, [deep])
    assert "\n    x = a + 2\n    x = x - 1\n    if x > 2:" in out
    compile(out, "<test>", "exec")


def test_correct_indentation_is_left_alone():
    nested = {"search": "        x = x * 2", "replace": "        x = x * 2\n        x += 1"}
    out = apply_edits(BLOCK, [nested])
    assert "        x = x * 2\n        x += 1\n    return" in out
