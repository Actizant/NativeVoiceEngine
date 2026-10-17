"""Committed evidence must not carry a real account's home-directory identity.

Machine-generated artefacts recorded absolute paths containing the operator's Windows account
name, and those artefacts were published. The repository convention is now `[USER_HOME]` for
home-rooted paths, repository-relative paths for files inside the repository, and the literal
`ACCOUNT` for an account component embedded inside a folder slug.

Personal-path fixtures in this file are assembled from fragments so this source does not itself
contain a matchable path. No line anywhere is exempted from the scan.
"""

from __future__ import annotations

import getpass
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Build, environment and trial residue: never tracked, so not part of the repository scan.
SCAN_EXCLUSIONS = {
    ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".trials", ".venv",
    "Python", "__pycache__", "build", "dist", "node_modules", "venv",
}

# Ignored roots that are still inspected, and reported instead of silently skipped.
DECLARED_LOCAL_ONLY_ROOTS = [".trials"]

# A path segment may name an account only through these tokens.
PLACEHOLDER_SEGMENTS = {"account", "tester", "[user_home]"}

LOCAL_ONLY_EXCEPTIONS = {".trials/kitten/models/manifests/kitten-nano-en-v0_8-int8.json"}

ACCOUNT = "privateuser"

WINDOWS_DOUBLE = "C:\\\\Users\\\\"
WINDOWS_SINGLE = "C:\\Users\\"
GIT_BASH = "/c/Users/"
POSIX_HOME = "/home/"
SLUG = "C--Users-"

SEGMENT = r"(?P<segment>[A-Za-z0-9._-]{2,64})"
# A slug uses the dash as its separator, so the segment must stop at the first dash or the token
# comparison never sees just the account component.
SLUG_SEGMENT = r"(?P<segment>[A-Za-z0-9._]{2,64})"
DETECTORS = (
    re.compile(r"Users\\\\" + SEGMENT),        # JSON-escaped Windows path
    re.compile(r"Users\\" + SEGMENT),          # single-backslash Windows path
    re.compile(r"Users/" + SEGMENT),           # forward-slash and Git-Bash paths
    re.compile(r"Users-{1,2}" + SLUG_SEGMENT),  # account id embedded in a folder slug
    re.compile(r"/home/" + SEGMENT),           # POSIX home directory
)

EVIDENCE_MUST_BE_SCANNED = (
    "artifacts/evaluation/multilingual-inventory.json",
    "artifacts/evaluation/multilingual-large-v3-int8.json",
    "artifacts/audit/manifest-snapshots/faster-whisper-base.json.pre-audit",
    "artifacts/audit/asset-status-corrections.json",
    "artifacts/audit/mechanical-findings.json",
    "artifacts/device/real-speech-read-aloud.json",
    "artifacts/platform/platform-run.json",
    "artifacts/platform/voices/voice.platform.default.json",
    "docs/relocation.md",
    "tests/test_config.py",
    # The rest of the tracked evidence that carries the committed placeholder. Each of these
    # was covered by the whole-tree scan but undeclared here, so a future narrowing of that
    # scan would have dropped it from the guarantee silently - see
    # `test_every_placeholder_carrying_artefact_is_named`.
    "artifacts/audit/manifest-snapshots/faster-whisper-small.json.pre-audit",
    "artifacts/audit/manifest-snapshots/kokoro-int8-multi-lang-v1_1.json.pre-audit",
    "artifacts/device/mic-baseline-post-go.json",
    "artifacts/device/real-speech-read-aloud.20261013-145157.human-run.original.json",
    "artifacts/device/real-speech-read-aloud.20261013-145157.previous.json",
    "artifacts/device/real-speech-read-aloud.20261013-151857.previous.json",
    "artifacts/device/real-speech-read-aloud.20261013-153133.previous.json",
    "artifacts/device/real-speech-read-aloud.20261013-171915.previous.json",
    "artifacts/device/real-speech-read-aloud.20261013-180908.previous.json",
    "artifacts/evaluation/multilingual-base-default.json",
    "artifacts/evaluation/multilingual-base-int8.json",
    "artifacts/provenance/faster-whisper-large-v3.json",
    "artifacts/platform/voices/voice.host.fieldmind.speaker7.json",
)

#: The published stand-in for a home directory, as it appears inside committed evidence.
#: `artifacts/` is the tree the serialization boundary writes to, so it is what the
#: inventory below has to cover; prose that merely names the token is not evidence.
PLACEHOLDER = "[USER_HOME]"
EVIDENCE_TREE = "artifacts/"


def personal_path_findings(text: str) -> list[str]:
    """Return every home-directory segment in text that is not an approved placeholder."""
    found = []
    for detector in DETECTORS:
        for match in detector.finditer(text):
            if match.group("segment").lower() not in PLACEHOLDER_SEGMENTS:
                found.append(f"{detector.pattern}:{match.group('segment')}")
    return found


def scanned_text_files() -> list[Path]:
    files = []
    for path in sorted(REPO.rglob("*")):
        if not path.is_file():
            continue
        if set(path.relative_to(REPO).parts[:-1]) & SCAN_EXCLUSIONS:
            continue
        try:
            path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        files.append(path)
    return files


@pytest.fixture(scope="module")
def files_and_text() -> list[tuple[Path, str]]:
    return [(path, path.read_text(encoding="utf-8")) for path in scanned_text_files()]


@pytest.mark.parametrize(
    "fixture",
    [
        WINDOWS_DOUBLE + ACCOUNT,
        WINDOWS_SINGLE + ACCOUNT,
        GIT_BASH + ACCOUNT,
        POSIX_HOME + ACCOUNT,
        SLUG + ACCOUNT + "-Documents-Qoder",
        WINDOWS_DOUBLE + ACCOUNT + "-Documents-Qoder-nativevoice-artifacts-device-",
    ],
    ids=["windows-json", "windows-single", "git-bash", "posix-home", "folder-slug", "repo-internal-json"],
)
def test_every_supported_personal_path_format_is_detected(fixture: str) -> None:
    """Negative control: a format this guard cannot see is a format it cannot protect."""
    assert personal_path_findings(fixture + "Downloads"), f"undetected: {fixture!r}"


@pytest.mark.parametrize(
    "scrubbed",
    [
        "[USER_HOME]\\Downloads\\a.mp3",
        "[USER_HOME]\\\\.cache\\\\huggingface\\\\hub",
        "[USER_HOME]/AppData/Local/ultron-voice",
        "artifacts/device/real-speech-read-aloud.json",
        SLUG + "ACCOUNT-Documents-Qoder-2026-10-12-76969c5c",
        WINDOWS_DOUBLE + "tester" + "\\\\AppData\\\\Local",
        "the Users group cannot be granted on this share",
    ],
    ids=["token-single", "token-json", "token-forward", "repo-relative", "account-token", "fixture-name", "prose"],
)
def test_scrubbed_forms_are_not_flagged(scrubbed: str) -> None:
    """Positive control: approved forms must pass, so the guard is not a rubber stamp."""
    assert not personal_path_findings(scrubbed), f"false positive on approved form: {scrubbed!r}"


def test_no_home_path_names_a_real_account(files_and_text: list[tuple[Path, str]]) -> None:
    """Any home-directory segment in the scanned tree must be an approved placeholder."""
    bad = [
        f"{path.relative_to(REPO)}:{number}: {hit}"
        for path, text in files_and_text
        for number, line in enumerate(text.splitlines(), start=1)
        for hit in personal_path_findings(line)
    ]
    assert not bad, f"{len(bad)} real-account path(s):\n" + "\n".join(bad[:20])


def test_operator_account_name_is_not_committed(files_and_text: list[tuple[Path, str]]) -> None:
    """Derived at run time, so this file never stores the name it guards against."""
    names = {name for name in (getpass.getuser(), Path.home().name) if len(name) >= 4}
    assert names, "derived account name is too short to be a meaningful guard"
    for name in sorted(names):
        hits = [
            f"{path.relative_to(REPO)}:{number}"
            for path, text in files_and_text
            for number, line in enumerate(text.splitlines(), start=1)
            if name in line
        ]
        assert not hits, f"account name appears in {len(hits)} line(s):\n" + "\n".join(hits[:20])


def _json_errors(text: str) -> list[str]:
    try:
        json.loads(text)
    except json.JSONDecodeError as error:
        return [str(error)]
    return []


def test_evidence_files_still_parse(files_and_text: list[tuple[Path, str]]) -> None:
    """Artefacts and pre-audit manifest copies are JSON: a path edit must not corrupt evidence."""
    broken = [
        f"{path.relative_to(REPO)}: {error}"
        for path, text in files_and_text
        if path.suffix in {".json", ".pre-audit"}
        for error in _json_errors(text)
    ]
    assert not broken, "invalid evidence file(s):\n" + "\n".join(broken)


def test_local_only_personal_paths_are_declared() -> None:
    """Ignored files are reported, not hidden.

    These files are never published, but they still hold the real account name on this machine, so the
    exception list is explicit: it fails if another one appears, or if a declared one is cleaned and
    the list was not updated.

    The two halves of this check have different scopes, deliberately. The ignore-rule assertion runs for
    every declared root whether or not the root exists, because a root that stopped being ignored is a
    finding on any machine. The inventory comparison is scoped to the roots that exist: an absent trial
    directory contains no personal path, so asserting against it would report "nothing is here" with the
    same message as "the declared exception was cleaned and the list was not updated". Machines without a
    trial root therefore exercise detection of nothing - see A53 in `docs/audit-register.md`.
    """
    gitignore = (REPO / ".gitignore").read_text(encoding="utf-8")
    offenders: set[str] = set()
    expected: set[str] = set()
    for root_name in DECLARED_LOCAL_ONLY_ROOTS:
        assert f"{root_name}/" in gitignore, f"{root_name} is no longer git-ignored"
        root = REPO / root_name
        if not root.is_dir():
            continue
        expected |= {entry for entry in LOCAL_ONLY_EXCEPTIONS if entry.split("/", 1)[0] == root_name}
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if personal_path_findings(text) or Path.home().name in text:
                offenders.add(str(path.relative_to(REPO)).replace("\\", "/"))
    assert offenders == expected, (
        f"local-only personal paths changed: added={sorted(offenders - expected)} "
        f"gone={sorted(expected - offenders)}"
    )


def test_scan_coverage_is_reported(files_and_text: list[tuple[Path, str]]) -> None:
    """A guard that scanned almost nothing is indistinguishable from a clean repository.

    The floor is derived from `git ls-files` rather than a pinned count, so it cannot silently rot as
    files are added. Where a tracked file is genuinely binary there is nothing to scan, and that is
    counted rather than assumed.
    """
    names = {str(path.relative_to(REPO)).replace("\\", "/") for path, _ in files_and_text}
    git = shutil.which("git")
    if git is None:
        pytest.skip("NOT RUN: git is unresolvable, so the tracked-file coverage floor cannot be derived")
    listed = subprocess.run([git, "ls-files"], cwd=REPO, capture_output=True)
    tracked = {line for line in listed.stdout.decode("utf-8", "replace").splitlines() if line}
    assert tracked, "NOT RUN: git ls-files produced no paths, so there is no coverage floor"

    skipped_binary = []
    outside_scan = []
    for name in sorted(tracked):
        path = REPO / name
        if not path.is_file():
            continue
        try:
            path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            skipped_binary.append(name)
            continue
        if name not in names:
            outside_scan.append(name)
    assert not outside_scan, f"tracked text files outside the scan: {outside_scan[:10]}"

    missing = [name for name in EVIDENCE_MUST_BE_SCANNED if name not in names]
    assert not missing, f"evidence files outside the scan: {missing}"


def test_every_placeholder_carrying_artefact_is_named() -> None:
    """Deleting a path from the inventory must fail, not quietly shrink the guarantee.

    `test_scan_coverage_is_reported` proves every named entry is scanned; it cannot see an
    entry that was removed, which is how a list of ten could sit over eighteen artefacts that
    carry the placeholder. The whole-tree scan is what protects them today, so a shortened
    list is invisible until that scan is narrowed - and by then the evidence has already been
    published. This test makes the removal itself the failure.

    Derived from `git ls-files` rather than pinned, so a new redacted artefact has to be
    declared before it can be tracked. Scoped to `artifacts/` and to tracked paths: prose
    that merely names the token (this repository's own guard, the audit register) is not
    output of the serialization boundary, and an untracked file is not yet evidence.
    """
    git = shutil.which("git")
    if git is None:
        pytest.skip("NOT RUN: git is unresolvable, so the tracked-artefact set cannot be derived")
    listed = subprocess.run([git, "ls-files", "-z"], cwd=REPO, capture_output=True)
    tracked = sorted(name for name in listed.stdout.decode("utf-8", "replace").split("\0") if name.startswith(EVIDENCE_TREE))
    assert tracked, "NOT RUN: git ls-files produced no paths under the evidence tree"

    carrying = []
    for name in tracked:
        path = REPO / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if PLACEHOLDER in text:
            carrying.append(name)
    assert carrying, "NOT RUN: no tracked artefact carries the placeholder, so nothing was proved"

    undeclared = sorted(set(carrying) - set(EVIDENCE_MUST_BE_SCANNED))
    assert not undeclared, (
        f"{len(carrying)} tracked artefact(s) under {EVIDENCE_TREE} carry {PLACEHOLDER!r} and "
        f"{len(undeclared)} of them are not in EVIDENCE_MUST_BE_SCANNED: {undeclared}"
    )
