"""Redaction of account-bearing paths at the serialization boundary of committed evidence.

WHY THE BOUNDARY AND NOT THE SOURCE. A tool has to see the real path to do its job: an
asset root is digested, a working copy is opened, a persistence check lists a directory.
Rewriting those values where they are computed would silently defeat the check they
belong to. So redaction happens once, on the way out - a record keeps its real paths in
memory and the artefact that lands in the repository carries ``[USER_HOME]`` in place of
the home directory and nothing else.

The substitution removes the account without removing the reader's ability to check what it
records. ``[USER_HOME]`` is a stable marker for this machine's home directory, so a digest
written beside a redacted path stays re-verifiable by re-joining the marker with a home the
reader resolves for itself. Nothing here performs that join: no consumer needs a filesystem
path back out of published evidence, and the marker is not a variable that expands on read.
What the artefact loses is the account that produced it - the one thing
``tests/test_no_personal_paths.py`` exists to keep out of the published tree.

Console output is deliberately not covered here. It is presentation for the operator, who
already knows their own home directory, and a tool that could not print a real path would
be harder to use than a redacted artefact is to read.
"""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import Any

#: The placeholder the repository publishes home directories as. The spelling is pinned
#: by the positive controls in ``tests/test_no_personal_paths.py``; do not vary it.
PLACEHOLDER = "[USER_HOME]"


def home_prefixes() -> tuple[str, ...]:
    """Every way this machine spells its own home directory, longest form first.

    ``str(Path.home())`` covers the native form and :meth:`Path.as_posix` the
    forward-slash form a tool reaches for when it writes a URI-style path. The Git-Bash
    spelling is included because an environment override set from this shell -
    ``ML_EVAL_WORK``, ``NATIVEVOICE_HOME`` - arrives in exactly that shape and still
    resolves to the same account.
    """
    home = Path.home()
    prefixes = [str(home), home.as_posix()]
    windows = PureWindowsPath(str(home))
    if windows.drive:
        prefixes.append(f"/{windows.drive[0].lower()}{home.as_posix()[2:]}")
    return tuple(sorted(dict.fromkeys(prefixes), key=len, reverse=True))


def redact(value: str) -> str:
    """Replace every occurrence of this machine's home directory with ``PLACEHOLDER``.

    The remainder is kept verbatim, separators and all, because the tail of a path is the
    part evidence is read for.
    """
    if not isinstance(value, str):
        raise TypeError(f"redact takes a string, got {type(value).__name__}")
    for prefix in home_prefixes():
        value = value.replace(prefix, PLACEHOLDER)
    return value


def scrub(node: Any) -> Any:
    """Redact every string in a nested record, returning a copy.

    A copy is the whole point: the caller's dict keeps its real paths, so a record can be
    written redacted and still be used for verification afterwards.
    """
    if isinstance(node, str):
        return redact(node)
    if isinstance(node, dict):
        return {key: scrub(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [scrub(item) for item in node]
    return node


def dumps(obj: Any, **kwargs: Any) -> str:
    """Serialize ``obj`` with account-bearing paths redacted.

    ``indent`` and ``ensure_ascii`` default to the shape the committed artefacts already
    use, so a non-ASCII sample name stays readable rather than becoming escape sequences.
    """
    kwargs.setdefault("indent", 2)
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(scrub(obj), **kwargs)


def write_json(path: Path | str, obj: Any, **kwargs: Any) -> Path:
    """Write ``obj`` as JSON with account-bearing paths redacted."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps(obj, **kwargs), encoding="utf-8")
    return target


def redact_copy(source: Path | str, target: Path | str) -> Path:
    """Re-serialize a JSON file into a committed copy, with its paths redacted.

    A byte copy is wrong here even though it is what a backup usually wants: the live file
    carries the asset root the integrity check digests, and committing it verbatim would
    publish the account that provisioned it. So the copy is a serialization of the same
    record rather than a mirror of the same bytes - which is also why the live file is
    never touched by this module.

    One byte-level convention the copy does mirror, deliberately: whether the source ends
    in a final newline. Reserialization neither adds nor removes one, so a backup of a live
    manifest (`models/manifest_io.save_manifest` writes a trailing newline) keeps the
    terminator the committed copies already carry instead of silently dropping it. The
    mirror does not control the inter-line separator: text written here is translated by
    the platform, so on Windows a fresh copy is CRLF in the working tree and matches its
    committed LF blob only once ``.gitattributes`` normalizes it on staging.
    """
    text = Path(source).read_text(encoding="utf-8")
    serialised = dumps(json.loads(text))
    if text.endswith("\n"):
        serialised += "\n"
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialised, encoding="utf-8")
    return destination
