"""Tests for `tools/audit.py` itself: the taxonomy exemption and the write contract.

Both features exist to stop the audit tool from damaging the evidence it produces, so
neither is tested by reading the code. Every test here drives the shipped `main()` or the
shipped check function, and the negative controls point the tool at synthetic source trees
so a guard that stops working fails loudly instead of quietly narrowing the guarantee.

The full 12-check audit run is exercised here as a *read-only* claim only: one test runs the
shipped check set exactly as `python tools/audit.py` does and inspects the repository before and
after. It is in the suite because that run measures about 1.5s in process, which is cheap enough to
prove the operator's own command creates nothing rather than leaving that claim to manual checking.
The other tests use real check functions from the fast static set, because the write-contract
assertions do not need the runtime checks.

One rule for anyone extending this file: a test that passes `--publish-new … --yes` must call
`_repository_areas` first. Creation is confined to a dated name directly inside
`artifacts/audit/`, and that convention is what makes a conforming target writable, so an
unrelocated call would write a real file into the repository's evidence area — which is what
the guard exists to make deliberate.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import tools.audit as audit  # noqa: E402

ARTEFACT = ROOT / "artifacts" / "audit" / "mechanical-findings.json"
GUARD = "diagnostics/evidence_paths.py::redact"


#: A stand-in for the real exempt guard, so a synthetic tree that says nothing about
#: `redact()` does not also report the exemption as stale and muddy the assertion.
STAND_IN_GUARD = "def redact(value):\n    if not isinstance(value, str):\n        raise TypeError('takes a string')\n    return value\n"


def _synthetic(monkeypatch, tmp_path: Path, sources: dict[str, str], guard: bool = True) -> Path:
    """Point the check at a source tree we control, leaving the repository untouched.

    `guard=False` builds a tree without the exempt site, which is how the stale-exception
    control is tested.
    """
    tree = tmp_path / "nativevoice"
    bodies = dict(sources)
    if guard and "diagnostics/evidence_paths.py" not in bodies:
        bodies["diagnostics/evidence_paths.py"] = STAND_IN_GUARD
    for relative, body in bodies.items():
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    monkeypatch.setattr(audit, "SRC", tree)
    return tree


# --- the taxonomy exemption ---------------------------------------------------------


def test_the_shipped_tree_complies_and_shows_its_one_exception():
    finding = audit.check_error_taxonomy()
    assert finding.status == "PASS", finding.detail
    assert "0 raise site(s)" in finding.detail
    assert GUARD in finding.detail, "the admitted guard must be visible in the finding, not silent"
    assert ":58" in finding.detail, f"the exemption names its line: {finding.detail}"


def test_a_type_error_at_an_unreviewed_site_is_still_an_offender(monkeypatch, tmp_path):
    """The exemption is per site. This is the control that would catch a blanket allowlist."""
    _synthetic(monkeypatch, tmp_path, {"leaks.py": "def parse(value):\n    raise TypeError('nope')\n"})
    finding = audit.check_error_taxonomy()
    assert finding.status == "FAIL", finding.detail
    assert "leaks.py:2:TypeError" in finding.detail
    granted = finding.detail.split("admitted argument-type guard(s):")[-1]
    assert "leaks" not in granted, f"the unreviewed site must not appear among the granted ones: {granted}"


def test_any_other_builtin_category_is_an_offender_even_at_the_exempt_site(monkeypatch, tmp_path):
    """`ValueError` in `redact()` is a different claim, and the grant does not cover it."""
    _synthetic(
        monkeypatch,
        tmp_path,
        {"diagnostics/evidence_paths.py": "def redact(value):\n    raise ValueError('wrong kind')\n"},
    )
    finding = audit.check_error_taxonomy()
    assert finding.status == "FAIL"
    assert "evidence_paths.py:2:ValueError" in finding.detail


def test_an_exception_whose_guard_has_gone_fails_the_check(monkeypatch, tmp_path):
    """A grant cannot outlive the guard that earned it.

    Without this, deleting `redact()` would leave the key behind as a standing permission
    for that path and function name, and the suite would report a green taxonomy.
    """
    _synthetic(monkeypatch, tmp_path, {"diagnostics/other.py": "def helper():\n    return 1\n"}, guard=False)
    finding = audit.check_error_taxonomy()
    assert finding.status == "FAIL"
    assert "stale-exception:" in finding.detail
    assert GUARD in finding.detail


def test_a_type_error_outside_any_function_is_not_exempted(monkeypatch, tmp_path):
    """A module-level raise has no guarding function, so there is nothing to grant."""
    _synthetic(monkeypatch, tmp_path, {"diagnostics/evidence_paths.py": "raise TypeError('module level')\n"})
    finding = audit.check_error_taxonomy()
    assert finding.status == "FAIL", finding.detail
    assert "evidence_paths.py:1:TypeError" in finding.detail


def test_domain_errors_are_unaffected(monkeypatch, tmp_path):
    """A subsystem raising its own derived type stays compliant through the rewrite."""
    _synthetic(
        monkeypatch,
        tmp_path,
        {"core/errors.py": "class VoiceError(Exception):\n    pass\n\nclass Boom(VoiceError):\n    pass\n",
         "stt/thing.py": "class Boom(VoiceError):\n    pass\n\n\ndef go():\n    raise Boom('fine')\n"},
    )
    finding = audit.check_error_taxonomy()
    assert finding.status == "PASS", finding.detail


def test_the_exempted_guard_is_the_redaction_control_it_claims():
    """The behaviour the exception was granted for must still be the behaviour.

    This is the drift detector: if `redact()` stops refusing a `Path`, or refuses it with a
    different category, or the pinned test goes away, the exemption is describing a guard
    that no longer exists and the published audit would assert something untrue.
    """
    from nativevoice.diagnostics.evidence_paths import redact

    with pytest.raises(TypeError):
        redact(Path.home() / "a.wav")  # type: ignore[arg-type]

    site = ROOT / "src" / "nativevoice" / "diagnostics" / "evidence_paths.py"
    tree = audit._tree(site)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "redact")
    raises = [node for node in ast.walk(function) if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)]
    assert len(raises) == 1, f"redact() is expected to raise exactly once, found {len(raises)}"
    raised = ast.unparse(raises[0].exc.func)
    assert raised == "TypeError", f"the exemption covers a TypeError guard, and the site now raises {raised}"
    guards = [node for node in ast.walk(function) if isinstance(node, ast.If)]
    tested_isinstance = any(
        "isinstance" in ast.unparse(test) for node in guards for test in ast.walk(node.test)
    )
    assert tested_isinstance, "the refusal is expected to sit behind an isinstance() argument check"

    pinned = (ROOT / "tests" / "test_evidence_paths.py").read_text(encoding="utf-8")
    assert "pytest.raises(TypeError)" in pinned and "redact(Path" in pinned, (
        "the pinned redaction test is the reason this site is exempt; it must stay"
    )


# --- the write contract -----------------------------------------------------------


class _Stub:
    """A check-shaped object so `main()` can be exercised without the runtime checks."""

    def __init__(self, name: str, status: str = "PASS", detail: str = "stubbed") -> None:
        self.__name__ = name
        self._finding = audit.Finding(name, status, detail)

    def __call__(self) -> audit.Finding:
        return self._finding


FAST = (_Stub("error taxonomy"), _Stub("offline isolation (static)"))
FAILING = (_Stub("error taxonomy", "FAIL", "1 offender"),)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_state() -> tuple[set[str], dict[str, str]]:
    """Every file name in the repository, plus the digest of everything in the evidence areas.

    Two observations rather than one, because they fail for different reasons: the name set
    catches a file that appeared or vanished anywhere, and the digest map catches an in-place
    edit of evidence that never changed name or size. `.git` and bytecode caches are excluded
    because the test runner itself touches those.
    """
    names = {str(x.relative_to(ROOT)).replace("\\", "/")
             for x in ROOT.rglob("*")
             if x.is_file() and ".git" not in x.parts and "__pycache__" not in x.parts}
    digests = {
        str(x.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(x.read_bytes()).hexdigest()
        for area in ("artifacts", "evidence")
        for x in (ROOT / area).rglob("*")
        if x.is_file()
    }
    return names, digests


def test_an_ordinary_run_leaves_the_committed_artefact_byte_identical(tmp_path, monkeypatch, capsys):
    """The core guarantee: verifying is not writing.

    Both write mechanisms are watched. Spying only `write_findings` was the blind spot a
    refutation probe found: the new-record path goes through `_create_exclusive`, so an
    ordinary run that minted a dated artefact would have left this test green.
    """
    before = _sha(ARTEFACT)
    called: list[str] = []
    monkeypatch.setattr(audit, "write_findings",
                        lambda findings, out: called.append(f"write_findings {out}") or out)
    monkeypatch.setattr(audit, "_create_exclusive",
                        lambda target, payload: called.append(f"_create_exclusive {target}") or None)
    assert audit.main([], checks=FAST) == 0
    assert called == [], f"an ordinary run reached a writer: {called}"
    assert _sha(ARTEFACT) == before
    printed = capsys.readouterr().out
    assert "0 failed" in printed and "wrote" not in printed


def test_the_shipped_twelve_check_audit_gains_no_file_anywhere(monkeypatch, capsys):
    """The operator's own command, run for real inside the suite: it must change no file.

    This test is about the filesystem, so the filesystem decides it. Everything is collected
    immediately after the run and before any assertion — the two writer spies, the file names
    in the tree and the digests of both evidence areas — so no failure below can mask a leak.
    The write invariants are then asserted without reference to what the audit concluded, and
    the audit's verdict is asserted afterwards as its own claim: a host that cannot execute one
    runtime check has produced a finding about that host, not a file the tool wrote.
    """
    attempted: list[str] = []
    monkeypatch.setattr(audit, "write_findings",
                        lambda findings, out: attempted.append(f"write_findings {out}") or out)
    monkeypatch.setattr(audit, "_create_exclusive",
                        lambda target, payload: attempted.append(f"_create_exclusive {target}") or None)
    names, digests = _tree_state()

    exit_code = audit.main([], checks=audit.CHECKS)
    printed = capsys.readouterr().out
    after_names, after_digests = _tree_state()

    # The no-write guarantee: reached by neither the pass count nor the host's capabilities.
    assert attempted == [], f"a run with no destination reached a writer: {attempted}"
    assert after_names == names, f"the tree gained or lost files: {sorted(after_names ^ names)}"
    assert after_digests == digests, "a file in the evidence areas changed bytes"

    # The verdict, kept as a separate claim so a host limitation reads as one.
    summaries = [line for line in printed.splitlines() if line.strip().endswith("not run")]
    assert summaries, f"the shipped audit printed no result summary (exit {exit_code}):\n{printed}"
    counts = [int(word) for word in summaries[-1].split() if word.isdigit()]
    assert len(counts) == 3, f"unrecognised summary line, expected 'N passed, M failed, K not run': {summaries[-1]}"
    passed, failed, not_run = counts
    assert passed + failed + not_run == len(audit.CHECKS), (
        f"the summary counted {passed}+{failed}+{not_run}, not the {len(audit.CHECKS)} shipped checks: {summaries[-1]}"
    )
    assert failed == 0 and not_run == 0, (
        f"the audit's verdict on this host was {passed} PASS, {failed} FAIL, {not_run} NOT RUN. The "
        "write invariants above still hold; this is the audit's own finding and is reported as one."
    )
    assert exit_code == 0, f"a verdict with no FAIL and no NOT RUN must exit 0, got {exit_code}"


def test_a_failing_check_still_exits_non_zero_and_still_writes_nothing(monkeypatch):
    """A FAIL is a reason to report, not a licence to write: both writers are watched."""
    written: list[str] = []
    monkeypatch.setattr(audit, "write_findings",
                        lambda findings, out: written.append(f"write_findings {out}") or out)
    monkeypatch.setattr(audit, "_create_exclusive",
                        lambda target, payload: written.append(f"_create_exclusive {target}") or None)
    assert audit.main([], checks=FAILING) == 1
    assert written == []


def test_out_writes_only_where_the_operator_pointed(tmp_path):
    target = tmp_path / "findings.json"
    assert audit.main(["--out", str(target)], checks=FAST) == 0
    rows = json.loads(target.read_text(encoding="utf-8"))
    assert [row["check"] for row in rows] == ["error taxonomy", "offline isolation (static)"]


def test_out_refuses_a_destination_whose_directory_does_not_exist(tmp_path):
    assert audit.main(["--out", str(tmp_path / "absent" / "f.json")], checks=FAST) == 1
    assert not (tmp_path / "absent").exists(), "the tool must not create directories for you"


class _Spy:
    """A check that records that it ran, so "refused before executing" can be proven.

    The guarantee is about ordering, not outcome: an invalid destination must be rejected
    without spending a full audit run, and a refusal that ran the runtime checks first has
    already opened threads and driven whole conversations for nothing.
    """

    def __init__(self, name: str, calls: list[str]) -> None:
        self.__name__ = name
        self._calls = calls

    def __call__(self) -> audit.Finding:
        self._calls.append(self.__name__)
        return audit.Finding(self.__name__, "PASS", "stubbed")


def _spies(calls: list[str]) -> tuple:
    return (_Spy("error taxonomy", calls), _Spy("offline isolation (static)", calls))


def _repository_areas(tmp_path: Path, monkeypatch) -> Path:
    """A stand-in repository root with its own evidence areas.

    The guards compare against constants frozen at import, so a test that only moved `ROOT`
    would be checking the real repository's areas while claiming to check a temporary one.
    All three are relocated together, and a separate test below pins the real values.
    """
    artifacts = tmp_path / "artifacts"
    evidence = tmp_path / "evidence"
    (artifacts / "audit").mkdir(parents=True)
    (artifacts / "benchmarks").mkdir(parents=True)
    (artifacts / "corpus").mkdir(parents=True)
    evidence.mkdir(parents=True)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "PUBLISHED_AREA", (artifacts / "audit").resolve())
    monkeypatch.setattr(audit, "PROTECTED_AREAS", (artifacts.resolve(), evidence.resolve()))
    return tmp_path


def test_out_refuses_every_protected_area_including_a_new_file_name(tmp_path, monkeypatch, capsys):
    """The guard covers the whole evidence tree, not one sub-directory of it.

    `artifacts/` also holds corpus, benchmark, device, evaluation, platform and provenance
    records, and `evidence/` is the attribution-correction set. An ordinary `--out` into any
    of them would create or replace committed evidence under a name nobody reviewed.
    """
    root = _repository_areas(tmp_path, monkeypatch)
    targets = [
        root / "artifacts" / "audit" / "new-revision.json",
        root / "artifacts" / "new-root-file.json",
        root / "artifacts" / "benchmarks" / "new-benchmark.json",
        root / "artifacts" / "corpus" / "new-corpus-file.json",
        root / "evidence" / "new-evidence.json",
    ]
    for target in targets:
        calls: list[str] = []
        assert audit.main(["--out", str(target)], checks=_spies(calls)) == 1, target
        assert not target.exists(), f"{target} was created by a refused run"
        assert calls == [], f"{target} was refused only after {calls} ran"
    printed = capsys.readouterr().out
    assert printed.count("REFUSED --out") == len(targets)
    for area in ("artifacts/", "evidence/"):
        assert area in printed, f"the refusal must name the area it protects, missing {area}"


def test_out_refuses_a_relative_spelling_of_a_protected_area(tmp_path, monkeypatch, capsys):
    """`../` and a relative path are judged on what they resolve to, not how they were typed."""
    root = _repository_areas(tmp_path, monkeypatch)
    monkeypatch.chdir(root)
    calls: list[str] = []
    assert audit.main(["--out", "artifacts/audit/../../evidence/sneaky.json"], checks=_spies(calls)) == 1
    assert calls == []
    assert not (root / "evidence" / "sneaky.json").exists()
    assert "REFUSED --out" in capsys.readouterr().out


def test_the_protected_areas_are_the_real_ones():
    """A guard is only as good as its constants; the relocation above must not hide a mistake."""
    assert audit.PROTECTED_AREAS == ((ROOT / "artifacts").resolve(), (ROOT / "evidence").resolve())
    assert audit.PUBLISHED_AREA == (ROOT / "artifacts" / "audit").resolve()
    assert audit.PROTECTED_AREAS[0] in audit.PUBLISHED_AREA.parents, (
        "the publish area must sit inside a protected area, so --publish is the only way in"
    )
    assert (audit.PUBLISHED_AREA / "mechanical-findings.json") == ARTEFACT.resolve()


def test_out_still_writes_outside_the_protected_areas(tmp_path):
    target = tmp_path / "somewhere" / "findings.json"
    target.parent.mkdir()
    assert audit.main(["--out", str(target)], checks=FAST) == 0
    assert target.is_file()


def test_out_replaces_an_existing_destination_and_revalidates_nothing(tmp_path, capsys):
    """`--out` is not a publisher: it names no baseline, so it re-checks nothing before writing.

    This pins behaviour that already exists rather than changing it. Both publish operations
    re-read something immediately before writing — an overwrite re-reads its `HEAD` baseline, a
    create claims its name with `O_EXCL` — and `--out` does neither, because outside the
    evidence areas there is no committed record to compare against and the operator's own file
    is the thing they asked to replace. The check below edits the destination after validation;
    the run still overwrites it, and the result is the findings record, not the edit.
    """
    target = tmp_path / "somewhere" / "findings.json"
    target.parent.mkdir()
    target.write_text("PRE-EXISTING OPERATOR FILE", encoding="utf-8")
    calls: list[str] = []

    class _Editor(_Spy):
        def __call__(self) -> audit.Finding:
            finding = super().__call__()
            if self.__name__ == "offline isolation (static)":
                target.write_text("changed during the run", encoding="utf-8")
            return finding

    checks = (_Editor("error taxonomy", calls), _Editor("offline isolation (static)", calls))
    assert audit.main(["--out", str(target)], checks=checks) == 0
    assert len(calls) == 2
    rows, problem = audit._reference_rows(target)
    assert problem is None, f"--out must leave a readable record at its destination: {problem}"
    assert [row["check"] for row in rows] == ["error taxonomy", "offline isolation (static)"]
    assert "wrote" in capsys.readouterr().out


@pytest.mark.parametrize(
    "content, expected",
    [
        ("{}", "the top level must be a list"),
        ("null", "the top level must be a list"),
        ('"a string"', "the top level must be a list"),
        ('["a"]', "row 0 must be an object, found str"),
        ('[["check"]]', "row 0 must be an object, found list"),
        ('[{"check":"x"}]', "row 0 is missing ['status', 'detail']"),
        ('[{"check":"x","status":"PASS"}]', "row 0 is missing ['detail']"),
        ('[{"check":"x","status":"PASS","detail":"y","extra":1}]', "unexpected keys ['extra']"),
        ('[{"check":7,"status":"PASS","detail":"y"}]', "must have string `check` and `detail`"),
        ('[{"check":"x","status":"PASS","detail":7}]', "must have string `check` and `detail`"),
        ('[{"check":"x","status":"MAYBE","detail":"y"}]', "has status 'MAYBE'"),
        ('[{"check":"x","status":"PASS","detail":"a"},{"check":"x","status":"FAIL","detail":"b"}]',
         "repeats check 'x'"),
        ('[{"check":"a","status":"PASS","detail":"x"},{"check":"b","status":"PASS","detail":"y"},'
         '{"check":"a","status":"FAIL","detail":"z"}]', "repeats check 'a'"),
        ('[{"check":"x","status":null,"detail":"y"}]', "has status None"),
        ("{not json", "it is not readable JSON"),
    ],
)
def test_check_artefact_refuses_a_malformed_reference_cleanly(tmp_path, capsys, content, expected):
    """A malformed reference is a refusal, not a traceback and not a plausible drift report.

    `json.loads("{}")` yields a dict, and iterating a dict yields its keys, so an unvalidated
    empty object would have been reported as "every check is missing from the artefact" — the
    kind of confident nonsense this tool exists to prevent.
    """
    reference = tmp_path / "reference.json"
    reference.write_text(content, encoding="utf-8")
    before = _sha(reference)
    calls: list[str] = []
    assert audit.main(["--check-artefact", str(reference)], checks=_spies(calls)) == 1
    assert calls == [], f"checks ran against a malformed reference: {calls}"
    assert _sha(reference) == before, "--check-artefact must never modify its reference"
    printed = capsys.readouterr().out
    assert "REFUSED --check-artefact" in printed and expected in printed, printed


def test_check_artefact_accepts_a_well_formed_reference(tmp_path, capsys):
    rows = [_Stub("error taxonomy")._finding._asdict(), _Stub("offline isolation (static)")._finding._asdict()]
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(rows), encoding="utf-8")
    calls: list[str] = []
    assert audit.main(["--check-artefact", str(reference)], checks=_spies(calls)) == 0
    assert calls == [spy.__name__ for spy in _spies([])], "a valid reference must be compared, not refused"
    assert "no drift" in capsys.readouterr().out


def test_the_committed_artefact_is_structurally_valid_and_only_content_has_drifted(tmp_path, capsys):
    """The 2026-10-13 artefact parses and is well-shaped, so drift is content, not format."""
    calls: list[str] = []
    assert audit.main(["--check-artefact", str(ARTEFACT)], checks=_spies(calls)) == 1
    assert len(calls) == 2, "a well-formed committed artefact must reach the comparison"
    assert "REFUSED" not in capsys.readouterr().out


def test_publish_refuses_to_create_a_new_file_even_with_yes(tmp_path, monkeypatch, capsys):
    """Publication overwrites the committed record; it never mints one.

    A mistyped name with `--yes` previously created a brand-new file inside the protected
    area — a typo turned into evidence. Creating a new dated revision is a separate,
    explicitly reviewed step, so an absent target is refused before anything runs.
    """
    root = _repository_areas(tmp_path, monkeypatch)
    target = root / "artifacts" / "audit" / "mechanical-findings-2099.json"
    calls: list[str] = []
    assert audit.main(["--publish", str(target), "--yes"], checks=_spies(calls)) == 1
    assert calls == [], f"a create attempt ran {calls}"
    assert not target.exists(), "the refused publish still created a file"
    printed = capsys.readouterr().out
    assert "not an existing committed findings record" in printed


def test_publish_new_creates_a_record_and_reports_what_it_will_write(tmp_path, monkeypatch, capsys):
    """Deliberate creation, in a relocated repository: printed first, then written on `--yes`."""
    root = _repository_areas(tmp_path, monkeypatch)
    target = root / "artifacts" / "audit" / "mechanical-findings-2099-01-01.json"
    calls: list[str] = []

    assert audit.main(["--publish-new", str(target)], checks=_spies(calls)) == 0
    assert not target.exists(), "a dry run must not create the record"
    printed = capsys.readouterr().out
    assert "create target: artifacts/audit/mechanical-findings-2099-01-01.json" in printed
    assert str(target) in printed, "the resolved destination must be shown, not just the relative name"
    assert "findings: 2 (2 PASS, 0 FAIL, 0 NOT_RUN)" in printed
    digest = [line for line in printed.splitlines() if "proposed content sha256" in line][0].rsplit(" ", 1)[-1]
    assert len(digest) == 16

    assert audit.main(["--publish-new", str(target), "--yes"], checks=_spies(calls)) == 0
    assert target.is_file(), "--yes must actually create the record"
    written = target.read_bytes()
    assert written.replace(b"\r\n", b"\n").startswith(b"[")
    assert hashlib.sha256(written.replace(b"\r\n", b"\n")).hexdigest()[:16] == digest, (
        "the digest shown before writing must be the content that was written"
    )
    rows, problem = audit._reference_rows(target)
    assert problem is None and [row["check"] for row in rows] == [spy.__name__ for spy in _spies([])]


def test_publish_new_refuses_an_existing_target_without_running_anything(tmp_path, monkeypatch, capsys):
    """Creation cannot degrade into an overwrite, so an existing target is refused outright.

    The name is a legal one, so this is the existence guard being tested and not the naming
    rule firing first.
    """
    root = _repository_areas(tmp_path, monkeypatch)
    target = root / "artifacts" / "audit" / "mechanical-findings-2099-01-02.json"
    original = b"ORIGINAL COMMITTED CONTENT"
    target.write_bytes(original)
    calls: list[str] = []
    assert audit.main(["--publish-new", str(target), "--yes"], checks=_spies(calls)) == 1
    assert calls == [], f"an existing target ran {calls}"
    assert target.read_bytes() == original, "the refused create must not touch the existing file"
    assert "already exists" in capsys.readouterr().out


def test_publish_refuses_an_absent_target_without_running_anything(tmp_path, monkeypatch, capsys):
    """And overwrite cannot degrade into a creation: the two flags have disjoint domains."""
    root = _repository_areas(tmp_path, monkeypatch)
    target = root / "artifacts" / "audit" / "not-committed.json"
    calls: list[str] = []
    assert audit.main(["--publish", str(target), "--yes"], checks=_spies(calls)) == 1
    assert calls == [], f"an absent target ran {calls}"
    assert not target.exists()
    assert "not an existing committed findings record" in capsys.readouterr().out


@pytest.mark.parametrize(
    "relative",
    [
        "../mechanical-findings.json",
        "../benchmarks/x.json",
        "sub/x.json",
        "../../evidence/x.json",
        "../../x.json",
        # A name that satisfies the convention is still refused outside the area, so the
        # two guards cannot substitute for each other.
        "../benchmarks/mechanical-findings-2099-01-04.json",
        "sub/mechanical-findings-2099-01-05.json",
    ],
)
def test_publish_new_is_confined_to_a_direct_child_of_the_artefact_area(
    tmp_path, monkeypatch, capsys, relative
):
    """Every way out of the area, spelled relative so resolution is what is being tested."""
    root = _repository_areas(tmp_path, monkeypatch)
    area = root / "artifacts" / "audit"
    target = (area / relative).resolve()
    calls: list[str] = []
    assert audit.main(["--publish-new", str(target), "--yes"], checks=_spies(calls)) == 1
    assert calls == [], f"{relative} ran {calls} before refusing"
    assert not target.exists(), f"{relative} was created"
    assert "does not resolve directly inside" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("x.json", "is not a new-record file name"),
        ("final-v2.json", "is not a new-record file name"),
        ("mechanical-findings.json", "is not a new-record file name"),
        ("mechanical-findings-2099-1-1.json", "is not a new-record file name"),
        ("mechanical-findings-99-01-01.json", "is not a new-record file name"),
        ("mechanical-findings-2099-01-01.jsonl", "is not a new-record file name"),
        ("mechanical-findings-2099-01-01.JSON", "is not a new-record file name"),
        ("mechanical-findings-2099-13-01.json", "names a date that does not exist"),
        ("mechanical-findings-2099-02-30.json", "names a date that does not exist"),
        ("mechanical-findings-2099-00-10.json", "names a date that does not exist"),
    ],
)
def test_publish_new_refuses_a_name_outside_the_documented_convention(
    tmp_path, monkeypatch, capsys, name, expected
):
    """The artefact area is evidence, so an unexplained file name is not publishable.

    `mechanical-findings.json` is refused here because it is the committed record and
    `--publish` owns it; a dated name with an impossible month or day is refused by the
    calendar, not only the pattern. Every refusal happens before a check runs and writes
    nothing.
    """
    root = _repository_areas(tmp_path, monkeypatch)
    target = root / "artifacts" / "audit" / name
    calls: list[str] = []
    assert audit.main(["--publish-new", str(target), "--yes"], checks=_spies(calls)) == 1, name
    assert calls == [], f"{name} ran {calls} before refusing"
    assert not target.exists(), f"{name} was created despite the refusal"
    printed = capsys.readouterr().out
    assert expected in printed, printed
    assert "mechanical-findings-YYYY-MM-DD.json" in printed, "the refusal must state the convention"


def test_publish_new_accepts_the_documented_convention(tmp_path, monkeypatch, capsys):
    """The positive control: a conforming name in the conforming place is not refused."""
    root = _repository_areas(tmp_path, monkeypatch)
    target = root / "artifacts" / "audit" / "mechanical-findings-2099-02-28.json"
    calls: list[str] = []
    assert audit.main(["--publish-new", str(target), "--yes"], checks=_spies(calls)) == 0
    assert len(calls) == 2
    assert target.is_file()
    assert "REFUSED" not in capsys.readouterr().out


# --- the check-then-write window ----------------------------------------------------


def test_publish_new_refuses_a_target_that_appeared_during_the_run(tmp_path, monkeypatch, capsys):
    """Validation said the path was free; something disagreed before the write.

    The planter runs as a check, which is exactly where the real window sits: the destination
    is judged before the run, and a run takes minutes. `O_EXCL` is what closes it, so the
    answer must be a refusal that leaves the other writer's bytes untouched — not a silent
    overwrite of a file nobody reviewed.
    """
    root = _repository_areas(tmp_path, monkeypatch)
    target = root / "artifacts" / "audit" / "mechanical-findings-2099-06-01.json"
    racer_bytes = b"WRITTEN BY SOMETHING ELSE"
    calls: list[str] = []

    class _Planter(_Spy):
        def __call__(self) -> audit.Finding:
            finding = super().__call__()
            if self.__name__ == "offline isolation (static)":
                target.write_bytes(racer_bytes)
            return finding

    checks = (_Planter("error taxonomy", calls), _Planter("offline isolation (static)", calls))
    assert audit.main(["--publish-new", str(target), "--yes"], checks=checks) == 1
    assert len(calls) == 2, "the race has to happen inside the run, after validation"
    assert target.read_bytes() == racer_bytes, "the create followed the file that appeared"
    assert "created after validation" in capsys.readouterr().out


def test_publish_refuses_when_the_working_file_moves_during_the_run(tmp_path, monkeypatch, capsys):
    """The baseline was the committed bytes at validation; a local edit during the run voids it.

    Without the re-read the tool would publish over an edit made while it was busy, and the
    only trace would be a printed "published" line. This drives the real `main()` with the
    edit planted by a check, so the guard is proven on the shipped ordering, not on a
    hand-called helper.
    """
    root = _repository_areas(tmp_path, monkeypatch)
    target = root / "artifacts" / "audit" / "mechanical-findings.json"
    committed = json.dumps([audit.Finding("error taxonomy", "PASS", "stubbed")._asdict()])
    target.write_text(committed, encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(audit, "_committed_bytes", lambda resolved: committed.encode("utf-8"))

    class _Editor(_Spy):
        def __call__(self) -> audit.Finding:
            finding = super().__call__()
            if self.__name__ == "offline isolation (static)":
                target.write_text("someone edited this during the run", encoding="utf-8")
            return finding

    checks = (_Editor("error taxonomy", calls), _Editor("offline isolation (static)", calls))
    assert audit.main(["--publish", str(target), "--yes"], checks=checks) == 1
    assert len(calls) == 2
    assert target.read_text(encoding="utf-8") == "someone edited this during the run", (
        "the refused overwrite replaced the local edit anyway"
    )
    assert "no longer match HEAD" in capsys.readouterr().out


def test_publish_refuses_when_the_head_baseline_moves_during_the_run(tmp_path, monkeypatch, capsys):
    """The other half of the window: the file is untouched, but HEAD is no longer what was approved.

    `git` cannot be made to move from inside a test without running a real repository, so the
    baseline reader is given a two-stage answer — the bytes HEAD held at validation, then the
    bytes it holds at the write. That is a stand-in for `git checkout`, `git pull` or an
    amend, and it is the reader call that the guard is built on.
    """
    root = _repository_areas(tmp_path, monkeypatch)
    target = root / "artifacts" / "audit" / "mechanical-findings.json"
    committed = json.dumps([audit.Finding("error taxonomy", "PASS", "stubbed")._asdict()])
    target.write_text(committed, encoding="utf-8")
    before = target.read_bytes()
    reads: list[bytes] = []

    def baseline_moves_after_the_first_look(resolved: Path) -> bytes:
        reads.append(committed.encode("utf-8") if not reads else b"different HEAD bytes")
        return reads[-1]

    monkeypatch.setattr(audit, "_committed_bytes", baseline_moves_after_the_first_look)
    assert audit.main(["--publish", str(target), "--yes"], checks=FAST) == 1
    assert target.read_bytes() == before, "the refused write moved the file"
    assert len(reads) == 2, "the baseline is read once at validation and once immediately before writing"
    assert "HEAD baseline changed during the run" in capsys.readouterr().out


def test_publish_overwrites_while_the_baseline_still_matches(tmp_path, monkeypatch, capsys):
    """Positive control for the re-read: an unchanged baseline is not a refusal.

    Without this, both race tests would pass against a guard that simply always refused.
    """
    root = _repository_areas(tmp_path, monkeypatch)
    target = root / "artifacts" / "audit" / "mechanical-findings.json"
    committed = json.dumps([audit.Finding("error taxonomy", "PASS", "stubbed")._asdict()])
    target.write_text(committed, encoding="utf-8")
    monkeypatch.setattr(audit, "_committed_bytes", lambda resolved: committed.encode("utf-8"))
    assert audit.main(["--publish", str(target), "--yes"], checks=FAST) == 0
    assert "overwritten" in capsys.readouterr().out
    assert target.read_text(encoding="utf-8") != committed


def test_a_partially_written_new_record_is_not_left_behind(tmp_path, monkeypatch):
    """A create that opened the file then failed must not leave a half record as evidence."""
    target = tmp_path / "mechanical-findings-2099-06-02.json"

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fdopen", refuse)
    problem = audit._create_exclusive(target, b'[{"check": "x"}]')
    assert problem is not None and "could not be written" in problem, problem
    assert "removed" in problem and "LEFT IN PLACE" not in problem, problem
    assert not target.exists(), "a partially created record stayed in the artefact area"


def test_publish_new_refuses_a_symlink_destination(tmp_path, monkeypatch, capsys):
    """A link is refused even though its resolved parent is the allowed directory."""
    root = _repository_areas(tmp_path, monkeypatch)
    elsewhere = root / "somewhere-else.json"
    elsewhere.write_text("not evidence", encoding="utf-8")
    link = root / "artifacts" / "audit" / "linked.json"
    try:
        os.symlink(elsewhere, link)
    except (OSError, NotImplementedError):
        pytest.skip("NOT RUN: this platform/user does not permit creating symlinks")
    calls: list[str] = []
    assert audit.main(["--publish-new", str(link), "--yes"], checks=_spies(calls)) == 1
    assert calls == []
    assert elsewhere.read_text(encoding="utf-8") == "not evidence", "the write reached the link target"
    assert not link.exists(), "a dangling entry was created where the link was refused"
    assert "symlink" in capsys.readouterr().out


def test_publish_new_refuses_a_broken_symlink_destination(tmp_path, monkeypatch, capsys):
    """`lexists` catches a link that points at nothing, which `.exists()` reports as absent."""
    root = _repository_areas(tmp_path, monkeypatch)
    missing = root / "does-not-exist-yet.json"
    link = root / "artifacts" / "audit" / "broken.json"
    try:
        os.symlink(missing, link)
    except (OSError, NotImplementedError):
        pytest.skip("NOT RUN: this platform/user does not permit creating symlinks")
    calls: list[str] = []
    assert audit.main(["--publish-new", str(link), "--yes"], checks=_spies(calls)) == 1
    assert calls == []
    assert not missing.exists(), "the refused write followed the broken link and created its target"
    printed = capsys.readouterr().out
    assert "symlink" in printed, printed  # a broken link is still a link, and is refused as one


def test_publish_refuses_to_write_findings_that_are_not_a_valid_record(tmp_path, monkeypatch, capsys):
    """The record about to be published is held to the same rules as a committed one.

    Two checks sharing a name would otherwise be published as a record that collapses on
    re-read — the same silent-loss defect the validator refuses in a file. The checks have
    run by this point (the findings are their output), but nothing is written and the exit
    status reports the refusal.
    """
    root = _repository_areas(tmp_path, monkeypatch)
    target = root / "artifacts" / "audit" / "mechanical-findings-2099-01-03.json"
    calls: list[str] = []
    colliding = (_Spy("error taxonomy", calls), _Spy("error taxonomy", calls))
    assert audit.main(["--publish-new", str(target), "--yes"], checks=colliding) == 1
    assert not target.exists(), "an invalid record was written anyway"
    assert "repeats check 'error taxonomy'" in capsys.readouterr().out


def test_publish_refusals_run_no_checks(tmp_path, monkeypatch):
    """Refusal before execution, for all three publish rejection paths."""
    root = _repository_areas(tmp_path, monkeypatch)
    target = root / "artifacts" / "benchmarks" / "x.json"
    for argv in (["--publish", str(target), "--yes"],
                 ["--publish", str(root / "elsewhere.json"), "--yes"],
                 ["--publish", str(root / "artifacts" / "audit"), "--yes"]):
        calls: list[str] = []
        assert audit.main(argv, checks=_spies(calls)) == 1, argv
        assert calls == [], f"{argv} ran {calls} before refusing"


def test_publish_refuses_when_there_is_no_committed_baseline(tmp_path, monkeypatch, capsys):
    """An existing file with nothing committed behind it cannot be "restored" over."""
    root = _repository_areas(tmp_path, monkeypatch)
    target = root / "artifacts" / "audit" / "mechanical-findings.json"
    target.write_text("bytes", encoding="utf-8")
    monkeypatch.setattr(audit, "_committed_bytes", lambda resolved: None)
    calls: list[str] = []
    assert audit.main(["--publish", str(target), "--yes"], checks=_spies(calls)) == 1
    assert target.read_text(encoding="utf-8") == "bytes", "a refusal must leave the file alone"
    assert calls == [], f"the checks ran for a publish that was refused: {calls}"
    assert "no baseline" in capsys.readouterr().out


def test_publish_refuses_a_directory_sitting_in_the_artefact_area(tmp_path, monkeypatch, capsys):
    """A real branch on this machine, unlike the symlink case, which needs privileges."""
    root = _repository_areas(tmp_path, monkeypatch)
    target = root / "artifacts" / "audit"
    calls: list[str] = []
    assert audit.main(["--publish", str(target), "--yes"], checks=_spies(calls)) == 1
    assert target.is_dir() and list(target.iterdir()) == []
    assert calls == [], f"the checks ran for a publish that was refused: {calls}"
    assert "REFUSED --publish" in capsys.readouterr().out


def test_yes_without_publish_and_a_bad_out_are_rejected_before_any_check(capsys):
    """Argument validation happens before a single check executes."""
    calls: list[str] = []
    with pytest.raises(SystemExit):
        audit.main(["--yes"], checks=_spies(calls))
    assert calls == []
    assert "--yes only means something" in capsys.readouterr().err
    assert audit.main(["--out", "definitely-absent-dir/f.json"], checks=_spies(calls)) == 1
    assert calls == [], "a missing destination directory must not cost a full run"


@pytest.mark.parametrize(
    "baseline, expected",
    [
        (b"{}", "the top level must be a list"),
        (b"null", "the top level must be a list"),
        (b'["a"]', "row 0 must be an object, found str"),
        (b'[{"check":"x"}]', "row 0 is missing ['status', 'detail']"),
        (b"{not json", "not readable JSON"),
        (b'[{"check":"x","status":"MAYBE","detail":"y"}]', "has status 'MAYBE'"),
        (b'[{"check":"x","status":"PASS","detail":"a"},'
         b'{"check":"x","status":"FAIL","detail":"b"}]', "repeats check 'x'"),
    ],
)
def test_publish_refuses_a_malformed_committed_baseline_before_running_anything(
    tmp_path, monkeypatch, capsys, baseline, expected
):
    """The publisher's baseline gets the same structural rules as `--check-artefact`.

    Until this test existed, `--publish` validated everything except the bytes it compares
    against, and parsed them with a bare `json.loads` *after* a full run: a committed record
    of the wrong shape crashed at the last step, and `{}` was read as "no rows recorded" —
    a confident wrong answer on the one path that writes. Zero checks must run, and the
    target must be untouched. The repository is relocated to `tmp_path`, so the real
    historical artefact is never the thing under test.
    """
    root = _repository_areas(tmp_path, monkeypatch)
    target = root / "artifacts" / "audit" / "mechanical-findings.json"
    target.write_bytes(baseline)
    monkeypatch.setattr(audit, "_committed_bytes", lambda resolved: baseline)
    calls: list[str] = []
    assert audit.main(["--publish", str(target), "--yes"], checks=_spies(calls)) == 1
    assert calls == [], f"a malformed baseline ran {calls} before refusing"
    assert target.read_bytes() == baseline, "a refusal must not touch the target"
    printed = capsys.readouterr().out
    assert "REFUSED --publish" in printed and expected in printed, printed


def test_publish_accepts_a_well_formed_committed_baseline(tmp_path, monkeypatch, capsys):
    """The control for the test above: a valid baseline still reaches the comparison."""
    root = _repository_areas(tmp_path, monkeypatch)
    rows = [{"check": "error taxonomy", "status": "FAIL", "detail": "recorded"}]
    blob = json.dumps(rows).encode("utf-8")
    target = root / "artifacts" / "audit" / "mechanical-findings.json"
    target.write_bytes(blob)
    monkeypatch.setattr(audit, "_committed_bytes", lambda resolved: blob)
    calls: list[str] = []
    assert audit.main(["--publish", str(target)], checks=_spies(calls)) == 0
    assert len(calls) == 2, "a valid baseline must not block the run"
    assert target.read_bytes() == blob, "a dry run must not write"
    printed = capsys.readouterr().out
    assert "overwrite target: artifacts/audit/mechanical-findings.json" in printed
    assert "this was a dry run: nothing was written" in printed
    assert "error taxonomy: FAIL -> PASS" in printed


def test_reporting_no_findings_is_a_zero_run_not_a_crash(capsys):
    """`max()` over no findings used to raise, turning "nothing ran" into a traceback."""
    assert audit._report([]) == (0, 0, 0)
    assert "0 passed, 0 failed, 0 not run" in capsys.readouterr().out
    assert audit.main([], checks=()) == 0


def test_no_write_to_protected_paths_leaves_no_file_behind_anywhere_in_the_repository():
    """The real repository's evidence areas are unwritable by this tool, in every mode.

    Absolute paths, deliberately: a relative `artifacts/audit/x.json` would resolve against
    whatever directory the test ran from and could write *there*, while the assertion below
    still reported the repository as clean — a green test proving nothing.
    """
    targets = [ROOT / "artifacts" / "audit" / "x.json", ROOT / "evidence" / "x.json",
               ROOT / "artifacts" / "benchmarks" / "x.json", ROOT / "artifacts" / "x.json"]
    for target in targets:
        for flag, extra in (("--out", []), ("--publish", ["--yes"])):
            calls: list[str] = []
            assert audit.main([flag, str(target), *extra], checks=_spies(calls)) == 1, target
            assert calls == [], f"{flag} {target} ran {calls} before refusing"
            assert not target.exists(), f"{target} was created despite the refusal"


def test_publish_requires_yes_and_writes_nothing_without_it(tmp_path, monkeypatch):
    root = _repository_areas(tmp_path, monkeypatch)
    target = root / "artifacts" / "audit" / "mechanical-findings.json"
    committed = json.dumps([audit.Finding("error taxonomy", "PASS", "stubbed")._asdict()])
    target.write_text(committed, encoding="utf-8")
    monkeypatch.setattr(audit, "_committed_bytes", lambda resolved: committed.encode("utf-8"))
    before = _sha(target)
    assert audit.main(["--publish", str(target)], checks=FAST) == 0
    assert _sha(target) == before, "a dry run must not move the bytes"
    assert audit.main(["--publish", str(target), "--yes"], checks=FAST) == 0
    assert _sha(target) != before, "with --yes the publish really publishes"


def test_publish_refuses_a_target_whose_bytes_differ_from_head(tmp_path, monkeypatch, capsys):
    root = _repository_areas(tmp_path, monkeypatch)
    target = root / "artifacts" / "audit" / "mechanical-findings.json"
    target.write_text("locally edited", encoding="utf-8")
    monkeypatch.setattr(audit, "_committed_bytes", lambda resolved: b"different committed bytes")
    assert audit.main(["--publish", str(target), "--yes"], checks=FAST) == 1
    assert target.read_text(encoding="utf-8") == "locally edited"
    assert "differ from HEAD" in capsys.readouterr().out


def test_publish_refuses_a_path_that_is_not_directly_in_the_artefact_area(tmp_path, monkeypatch, capsys):
    root = _repository_areas(tmp_path, monkeypatch)
    nested = root / "artifacts" / "audit" / "sub"
    nested.mkdir(parents=True)
    assert audit.main(["--publish", str(nested / "f.json"), "--yes"], checks=FAST) == 1
    assert not (nested / "f.json").exists()
    assert "does not resolve directly inside" in capsys.readouterr().out


def test_publish_refuses_a_symlink_into_the_area(tmp_path, monkeypatch, capsys):
    """A link can name a file outside the area while looking like it belongs to it."""
    root = _repository_areas(tmp_path, monkeypatch)
    elsewhere = root / "somewhere-else.json"
    elsewhere.write_text("not evidence", encoding="utf-8")
    link = root / "artifacts" / "audit" / "sneaky.json"
    try:
        os.symlink(elsewhere, link)
    except (OSError, NotImplementedError):
        pytest.skip("NOT RUN: this platform/user does not permit creating symlinks")
    assert audit.main(["--publish", str(link), "--yes"], checks=FAST) == 1
    assert elsewhere.read_text(encoding="utf-8") == "not evidence", "the write must never reach the link target"
    assert "is a symlink" in capsys.readouterr().out


def test_publish_refuses_a_relative_escape_that_resolves_outside(tmp_path, monkeypatch):
    root = _repository_areas(tmp_path, monkeypatch)
    monkeypatch.chdir(root)
    assert audit.main(["--publish", "artifacts/audit/../../elsewhere.json", "--yes"], checks=FAST) == 1
    assert not (root / "elsewhere.json").exists()


DESTINATIONS = ("--out", "--publish", "--publish-new", "--check-artefact")


@pytest.mark.parametrize("pair", list(combinations(DESTINATIONS, 2)), ids=lambda p: "+".join(p))
def test_no_two_destinations_can_be_combined(pair, tmp_path):
    """One operation per run, tested on all six pairs rather than a chosen few.

    Overwrite and creation are separate flags precisely so neither can fall back into the
    other, and a mutually exclusive argument group is the mechanism that enforces that. The
    pairs are generated from the flag names, so a fifth destination cannot be added and go
    untested against the others.

    The destinations are named under `tmp_path` rather than as bare relative file names. A
    pair that argparse rejects never reaches the filesystem, but if the exclusivity guard were
    ever removed, the surviving `--out` would be judged against the real repository — its
    parent directory exists and it sits outside the protected areas, so it would be accepted
    and the run would write into whatever directory the suite happened to run from. A scratch
    destination cannot do that, so a broken guard breaks the test without dirtying the tree.
    """
    argv = [value for flag in pair for value in (flag, str(tmp_path / f"{flag.lstrip('-')}.json"))]
    with pytest.raises(SystemExit) as exit_info:
        audit.main(argv, checks=FAST)
    assert exit_info.value.code == 2, argv
    assert not any(Path(value).exists() for value in argv[1::2]), (
        f"{pair} was refused at the parse step, so no destination may exist afterwards: "
        f"{[value for value in argv[1::2] if Path(value).exists()]}"
    )


def test_check_artefact_reports_drift_without_touching_the_file(capsys):
    """The historical artefact is a snapshot of an older tree, and the tool says so.

    The stub set is deliberately smaller than the artefact's eleven rows, because that is
    exactly the shape of the real staleness: rows recorded by a check set that no longer
    runs, and details that have moved. The point of this test is that the drift is named and
    the reference file is not modified.
    """
    before = _sha(ARTEFACT)
    assert audit.main(["--check-artefact", str(ARTEFACT)], checks=FAST) == 1
    assert _sha(ARTEFACT) == before, "--check-artefact must never modify its reference"
    printed = capsys.readouterr().out
    assert "drift item(s)" in printed
    recorded = json.loads(ARTEFACT.read_text(encoding="utf-8"))
    removed = [line for line in printed.splitlines() if "no longer computed" in line]
    assert len(removed) == len(recorded) - len(FAST), (
        f"{len(recorded)} recorded rows minus the {len(FAST)} computed ones should be named: {removed}"
    )
    assert any("error taxonomy" in line for line in printed.splitlines())


def test_check_artefact_names_every_kind_of_drift(tmp_path, capsys):
    """Added, removed, status-changed and detail-changed rows are each reported as such."""
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            [
                audit.Finding("a check that went away", "PASS", "gone")._asdict(),
                audit.Finding("error taxonomy", "FAIL", "old detail")._asdict(),
                audit.Finding("offline isolation (static)", "PASS", "stubbed")._asdict(),
            ]
        ),
        encoding="utf-8",
    )
    assert audit.main(["--check-artefact", str(reference)], checks=FAST) == 1
    printed = capsys.readouterr().out
    drift_lines = [line.strip() for line in printed.splitlines() if line.startswith("  ")]
    assert any("a check that went away" in line and "no longer computed" in line for line in drift_lines)
    assert any(line.startswith("~ error taxonomy: FAIL -> PASS") for line in drift_lines)
    assert not any("offline isolation (static)" in line for line in drift_lines), (
        f"a row whose status and detail both match must not be reported as drift: {drift_lines}"
    )


def test_check_artefact_reports_no_drift_for_an_identical_record(tmp_path):
    rows = [audit.Finding("error taxonomy", "PASS", "stubbed")._asdict(),
            audit.Finding("offline isolation (static)", "PASS", "stubbed")._asdict()]
    target = tmp_path / "match.json"
    target.write_text(json.dumps(rows), encoding="utf-8")
    assert audit.main(["--check-artefact", str(target)], checks=FAST) == 0


def test_check_artefact_refuses_a_missing_or_malformed_reference(tmp_path, capsys):
    assert audit.main(["--check-artefact", str(tmp_path / "absent.json")], checks=FAST) == 1
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert audit.main(["--check-artefact", str(broken)], checks=FAST) == 1
    assert "REFUSED --check-artefact" in capsys.readouterr().out


def test_the_committed_artefact_equals_its_published_bytes():
    """The file this suite keeps touching is checked against what the repository publishes.

    Read straight from `HEAD`, not from a constant, so the assertion cannot be satisfied by
    editing the test. If any test in this file ever writes into `artifacts/`, this fails.
    """
    git = shutil.which("git")
    if git is None:
        pytest.skip("NOT RUN: git is unresolvable, so the published bytes cannot be read")
    published = subprocess.run([git, "show", "HEAD:artifacts/audit/mechanical-findings.json"], cwd=ROOT, capture_output=True)
    assert published.returncode == 0, published.stderr.decode("utf-8", "replace")
    assert ARTEFACT.read_bytes() == published.stdout, (
        "the working copy of the historical findings differs from HEAD; nothing in this suite may write it"
    )


def test_refused_runs_leave_no_new_file_behind_in_the_repository(tmp_path, monkeypatch):
    """Refusals must refuse, not partially write, checked against the real repository tree.

    The ordinary-run half of this claim is now
    `test_the_shipped_twelve_check_audit_gains_no_file_anywhere`; what is exercised here is
    every refusal path that carries a destination, including a `--publish-new` whose name is
    refused and one whose directory is refused. The repository is inspected by name and the
    evidence areas by digest before and after, so a refusal that leaked a file fails here no
    matter which guard was supposed to stop it.
    """
    names, digests = _tree_state()
    root = _repository_areas(tmp_path, monkeypatch)
    area = root / "artifacts" / "audit"
    (area / "sub").mkdir(parents=True)
    refusals = [
        ["--out", str(tmp_path / "absent" / "f.json")],
        ["--publish", str(area / "sub" / "f.json"), "--yes"],
        ["--publish-new", str(area / "sub" / "mechanical-findings-2099-07-07.json"), "--yes"],
        ["--publish-new", str(area / "x.json"), "--yes"],
        ["--check-artefact", str(tmp_path / "absent.json")],
    ]
    for argv in refusals:
        assert audit.main(argv, checks=FAST) == 1, argv
    after_names, after_digests = _tree_state()
    assert after_names == names, f"the repository gained or lost files: {sorted(after_names ^ names)}"
    assert after_digests == digests, "a refused run changed bytes in the evidence areas"
