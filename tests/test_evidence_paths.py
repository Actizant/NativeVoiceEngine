"""The `[USER_HOME]` boundary: what it rewrites, what it must leave alone, and what it copies.

No model, no asset, no inference. The unit under test is a string substitution that stands
between a tool's real working paths and the artefact a machine commits, so the cases are the
spellings a home directory takes, the paths that are not a home directory, and the guarantee
that the record in memory still names the file the tool has to open afterwards.

The convention these tests pin is the one `tests/test_no_personal_paths.py` already enforces
on published evidence: `[USER_HOME]` for the home directory, everything after it verbatim.
Fixtures are assembled from fragments at runtime because a test that hardcodes a real home
path would itself be the leak it is trying to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nativevoice.diagnostics.evidence_paths import (
    PLACEHOLDER,
    dumps,
    home_prefixes,
    redact,
    redact_copy,
    scrub,
    write_json,
)

HOME = str(Path.home())
TAIL = "Downloads/say-this-one.wav"


def home_forms() -> tuple[str, ...]:
    """Every spelling of this home directory that serialized evidence could carry.

    Each prefix from the boundary's own list, plus its JSON form: `json.dumps` doubles every
    backslash, so a leaked Windows path appears in an artefact as the escaped shape and a
    check against the raw spelling alone passes on a file that leaks.
    """
    forms = []
    for prefix in home_prefixes():
        forms.append(prefix)
        forms.append(json.dumps(prefix)[1:-1])
    return tuple(dict.fromkeys(forms))


def assert_no_home_path(text: str, evidence: str = "") -> None:
    """Fail if any form of this machine's home directory survives in serialized text."""
    leaked = [form for form in home_forms() if form in text]
    assert not leaked, f"home path survived in {evidence or 'serialized text'}: {leaked}\n{text[:200]}"


def test_home_is_named_in_every_spelling_a_writer_can_produce() -> None:
    """The native form and the forward-slash form are both matched, or the guard is a joke."""
    prefixes = home_prefixes()
    assert HOME in prefixes
    assert Path(HOME).as_posix() in prefixes
    assert all(not item.endswith(("\\", "/")) for item in prefixes), prefixes


def test_redact_replaces_the_account_directory_wherever_it_sits() -> None:
    """A path that merely *contains* the home directory is still the account's own."""
    assert redact(f"{HOME}\\{TAIL}") == f"{PLACEHOLDER}\\{TAIL}"
    assert redact(f"{HOME}/{TAIL}") == f"{PLACEHOLDER}/{TAIL}"
    assert redact(f"models_root={HOME}/AppData/Local/ultron-voice") == (
        f"models_root={PLACEHOLDER}/AppData/Local/ultron-voice"
    )


def test_redact_leaves_no_trace_of_the_account_name() -> None:
    """The account name is the secret; the placeholder is only the visible half of the test."""
    account = Path(HOME).name
    for value in (f"{HOME}\\x.wav", f"{HOME}/x.wav", f"note {HOME} note"):
        assert account not in redact(value), value


@pytest.mark.parametrize("prefix", home_prefixes())
def test_every_spelling_of_home_redacts_to_the_same_placeholder(prefix: str) -> None:
    """A drive-letter, POSIX or Git-Bash spelling of one home directory is one home directory.

    `ML_EVAL_WORK` and `NATIVEVOICE_HOME` set from a Git-Bash shell arrive in the third form,
    and a path taken from either is written straight into the inventory.
    """
    redacted = redact(f"{prefix}/sample/a.wav")
    assert redacted == f"{PLACEHOLDER}/sample/a.wav"


def test_the_tail_of_the_path_is_preserved_verbatim() -> None:
    """Redaction removes the account, not the evidence: separators, spaces and non-ASCII stay.

    A sample named in another script is the reason `ensure_ascii=False` is the default here - an
    artefact that turned a Japanese filename into escape sequences would be unreadable, and a
    digest recorded beside a redacted path has to stay re-computable.
    """
    value = f"{HOME}\\Downloads\\元気な 女性-2026-10-13 (1).wav"
    assert redact(value) == f"{PLACEHOLDER}\\Downloads\\元気な 女性-2026-10-13 (1).wav"


@pytest.mark.parametrize(
    "value",
    [
        "artifacts/device/real-speech-read-aloud.json",
        "./models/manifests/faster-whisper-base.json",
        "C:\\Users\\tester\\AppData\\Local\\ultron-voice",
        "/home/account/models/kokoro",
        "the Users group cannot be granted on this share",
        "",
    ],
)
def test_a_path_that_is_not_this_home_is_not_touched(value: str) -> None:
    """Placeholder names, another account and repo-relative forms pass through unchanged.

    Rewriting a path that carries no account would cost information for nothing, and a
    placeholder in someone else's home directory would read as if it were this machine's.
    """
    assert redact(value) == value


def test_scrub_returns_a_copy_and_leaves_the_record_real() -> None:
    """The boundary is on the way out: the caller's dict still names the file it must open.

    `tools/multilingual_eval.py` reads back the working copy through the same dict it writes
    into the inventory, and `tools/platform_demo.py` compares two host data roots for
    distinctness from the report it is about to publish. Redacting in place would have made
    either the pass or the check impossible to run.
    """
    record = {
        "paths": [f"{HOME}\\Downloads\\a.mp3"],
        "nested": {"asset_root": f"{HOME}/AppData/Local/ultron-voice", "frames": 480, "warm": None},
        "pair": (f"{HOME}/x", "unchanged"),
    }
    scrubbed = scrub(record)
    assert record["paths"][0].startswith(HOME)
    assert record["nested"]["asset_root"].startswith(HOME)
    assert scrubbed["paths"] == [f"{PLACEHOLDER}\\Downloads\\a.mp3"]
    assert scrubbed["nested"]["asset_root"] == f"{PLACEHOLDER}/AppData/Local/ultron-voice"
    assert scrubbed["pair"] == [f"{PLACEHOLDER}/x", "unchanged"]
    assert scrubbed["nested"] == {"asset_root": scrubbed["nested"]["asset_root"], "frames": 480, "warm": None}


def test_scrub_is_idempotent() -> None:
    """A record read back from committed evidence can be re-serialized without drift."""
    once = scrub({"p": f"{HOME}\\a.wav"})
    assert scrub(once) == once


def test_redact_refuses_a_path_object() -> None:
    """A caller that hands over a `Path` would silently win an un-redacted `PurePath` repr."""
    with pytest.raises(TypeError):
        redact(Path(HOME) / "a.wav")  # type: ignore[arg-type]


def test_dumps_keeps_the_shape_the_committed_artefacts_already_use() -> None:
    text = dumps({"samples": [f"{HOME}\\Downloads\\示例名.mp3"], "bytes": 161400})
    parsed = json.loads(text)
    assert parsed["samples"] == [f"{PLACEHOLDER}\\Downloads\\示例名.mp3"]
    assert parsed["bytes"] == 161400
    assert "\\u" not in text, "non-ASCII must stay readable in published evidence"
    assert_no_home_path(text, "dumps output")


def test_write_json_creates_the_directory_and_writes_utf8(tmp_path: Path) -> None:
    target = write_json(tmp_path / "deep" / "run.json", {"source": f"{HOME}/Downloads/a.mp3"})
    written = target.read_text(encoding="utf-8")
    assert json.loads(written) == {"source": f"{PLACEHOLDER}/Downloads/a.mp3"}
    assert written.count(PLACEHOLDER) == 1, "the placeholder replaces the prefix, it does not duplicate it"
    assert_no_home_path(written, "write_json output")
    assert not written.endswith("\n"), (
        "write_json emits no final newline; redact_copy is the function that mirrors one. "
        "The split is deliberate and is what the next audit run's bytes will show."
    )


def test_write_json_can_be_pointed_at_a_directory_outside_the_home(tmp_path: Path) -> None:
    """A redacted value is still the same path on this machine, so nothing may be lost around it."""
    outside = tmp_path / "evidence.json"
    write_json(outside, {"kept": "/srv/models/base", "warm": 0.612})
    assert json.loads(outside.read_text(encoding="utf-8")) == {"kept": "/srv/models/base", "warm": 0.612}


def test_redact_copy_redacts_the_committed_copy_and_not_the_live_file(tmp_path: Path) -> None:
    """A byte copy of a manifest would publish the asset root the integrity check digests.

    The live file keeps the real path because that is what `models/integrity.py` opens and
    hashes; the committed copy is a serialization of the same record with the account removed,
    so the same `sha256` values and the same relative layout remain readable in it.
    """
    live = tmp_path / "manifests" / "faster-whisper-base.json"
    live.parent.mkdir(parents=True)
    live.write_text(
        json.dumps({"model_id": "faster-whisper-base", "asset_root": f"{HOME}\\.cache\\huggingface\\hub"}, indent=2),
        encoding="utf-8",
    )
    copy = redact_copy(live, tmp_path / "snapshots" / "faster-whisper-base.json.pre-audit")
    assert json.loads(live.read_text(encoding="utf-8"))["asset_root"].startswith(HOME)
    written = json.loads(copy.read_text(encoding="utf-8"))
    assert written["asset_root"] == f"{PLACEHOLDER}\\.cache\\huggingface\\hub"
    assert written["model_id"] == "faster-whisper-base"


def test_an_account_repeated_outside_the_home_prefix_is_not_covered() -> None:
    """A recorded limit, asserted so nobody mistakes the boundary for a name scrubber.

    A scratch root like `pytest-of-<account>` carries the account twice: once inside the home
    prefix, which this module replaces, and once again as a segment of its own, which it does
    not. Evidence published from such a directory therefore still names the account, and
    `tests/test_no_personal_paths.py` rejects it - fail-closed, with the operator pointing
    `--out` back at the repository. Widening the substitution into a blanket find-and-replace
    would also rewrite transcript text that happens to contain the same word, which is a
    measurement change dressed up as redaction.
    """
    account = Path(HOME).name
    text = redact(f"{HOME}\\AppData\\Local\\Temp\\pytest-of-{account}\\evidence.json")
    assert text == f"{PLACEHOLDER}\\AppData\\Local\\Temp\\pytest-of-{account}\\evidence.json"
    assert text.startswith(PLACEHOLDER), "the prefix is replaced at the head, not merely absent somewhere"
    assert text.count(account) == 1, "the residue this limit documents is exactly one scratch slug"
    assert_no_home_path(json.dumps({"path": text}), "the serialized form of the redacted path")


@pytest.mark.parametrize("trailing", [True, False], ids=["source-ends-in-newline", "source-does-not"])
def test_redact_copy_mirrors_the_source_final_newline(tmp_path: Path, trailing: bool) -> None:
    """A backup that dropped the newline would change bytes while changing no content.

    The live manifests this copies come from `models/manifest_io.save_manifest`, which writes a
    trailing newline, and the committed `.pre-audit` copies already read that way. The copy's
    job is to remove the account and nothing else, so the final newline is the one byte-level
    convention it follows rather than sets - and `write_json`, which has no source to follow,
    stays exactly as its own tests pin it.

    The text-mode assertions cannot see the separator: `read_text` translates, so a copy written
    with CRLF reads back as LF. The trailing byte-level assertions check the bytes that actually
    reached the disk, which is where a future `newline=""` change would show up first.
    """
    live = tmp_path / "manifests" / "faster-whisper-base.json"
    live.parent.mkdir(parents=True)
    body = json.dumps(
        {"model_id": "faster-whisper-base", "asset_root": f"{HOME}\\.cache\\huggingface\\hub"}, indent=2
    )
    live.write_text(body + ("\n" if trailing else ""), encoding="utf-8")

    copy = redact_copy(live, tmp_path / "snapshots" / "faster-whisper-base.json.pre-audit")
    written = copy.read_text(encoding="utf-8")
    assert written == json.dumps(
        {"model_id": "faster-whisper-base", "asset_root": f"{PLACEHOLDER}\\.cache\\huggingface\\hub"}, indent=2
    ) + ("\n" if trailing else "")
    assert written.endswith("\n") is trailing
    assert_no_home_path(written, "a pre-audit copy")
    assert json.loads(live.read_text(encoding="utf-8"))["asset_root"].startswith(HOME), (
        "the live manifest is never touched by the boundary"
    )

    raw = copy.read_bytes()
    source_raw = live.read_bytes()
    assert raw.endswith(b"\n") is trailing
    assert raw.count(b"\r\n") == source_raw.count(b"\r\n"), (
        "the copy mirrors the source's separator style rather than setting a new one; read_text "
        "translates on the way in, so the byte count is the only assertion that can see this"
    )
    assert raw.count(b"\n") == source_raw.count(b"\n"), "the copy adds and drops no line terminators"
    assert Path(HOME).name.encode("utf-8") not in raw, "the account name is absent from the published bytes"
