"""Telling an absent optional backend apart from one that is present and broken.

Both are an import failure, and only one of them is an installation gap. A package
that is not installed can be installed; a package that is installed and then raises
while loading means the environment is corrupt, and calling that "not installed"
invites the host to reinstall the thing that is already fine and to keep looking for
a missing file. The distinction is made from the exception the import machinery
raises, never from a message match.

A package whose own dependency is missing raises `ModuleNotFoundError` naming *that*
dependency, and a failing native extension raises `ImportError` or `OSError` - neither
is absence of the package the caller asked for.

`sys.modules[name] = None` classifies as absent, which is what a test that poisons the
module table is usually trying to simulate: measured on CPython 3.12.10 and 3.14.7 both
raise `ModuleNotFoundError` with `name` set to the poisoned module. The classifier itself
never consults the module table, so a future interpreter that reports it differently moves
that simulation, not the rule.
"""

from __future__ import annotations

#: The named top-level module could not be found at all.
ABSENT = "package-absent"
#: The import failed for some other reason, which includes any failure that cannot
#: be attributed to the requested name - the loud answer, never a quiet "missing".
BROKEN = "package-broken"


def import_state(module_name: str, error: BaseException) -> str:
    """Classify `error` from a failed `import <module_name>`.

    Only a `ModuleNotFoundError` whose `name` is the requested module itself counts as
    absence. A failure carrying no `name` at all is answered `BROKEN`, because guessing
    absence from an unattributable error would turn a corrupt environment into a skip.
    """
    if isinstance(error, ModuleNotFoundError) and error.name == module_name:
        return ABSENT
    return BROKEN
