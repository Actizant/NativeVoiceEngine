"""Suite hygiene.

A duplicated `def` inside a class does not fail: the second definition silently
replaces the first, the test count drops, and everything still reports green. That
is how a test suite can lose coverage without anyone noticing, so it is checked
here rather than trusted.
"""

from __future__ import annotations

import ast
import collections
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def _scope_names(tree: ast.AST) -> list[str]:
    """Function names qualified by the *full* scope they are defined in.

    Two same-named methods in two same-named classes collide. Two local helper
    classes with the same name inside different test functions do not - they are
    separate scopes, and flagging those teaches people to stop writing helpers.
    """

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found.append(f"{prefix}{child.name}")
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            else:
                walk(child, prefix)

    found: list[str] = []
    walk(tree, "")
    return [name for name in found if name.split(".")[-1].startswith("test_")]


def test_no_file_defines_the_same_test_twice():
    offenders = []
    for path in sorted(TESTS.glob("test_*.py")):
        counts = collections.Counter(_scope_names(ast.parse(path.read_text(encoding="utf-8"))))
        offenders.extend(f"{path.name}:{name} x{count}" for name, count in counts.items() if count > 1)
    assert offenders == []


def test_every_test_file_contributes_at_least_one_test():
    empty = []
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tests = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
        ]
        if not tests:
            empty.append(path.name)
    assert empty == []


def test_no_test_is_unconditionally_skipped():
    """A bare `@pytest.mark.skip` is how a suite quietly stops measuring anything.

    Conditional skips are fine and are used deliberately by the real-inference
    tests, which must not run without an approved asset.
    """
    offenders = []
    for path in sorted(TESTS.glob("test_*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                dotted = ast.unparse(target)
                if dotted.endswith("mark.skip") and isinstance(decorator, ast.Name):
                    offenders.append(f"{path.name}:{node.name}:{node.lineno}")
                elif dotted.endswith("mark.skip") and isinstance(decorator, ast.Call) and not decorator.args:
                    offenders.append(f"{path.name}:{node.name}:{node.lineno}")
    assert offenders == []
