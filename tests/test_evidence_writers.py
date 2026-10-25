"""Every writer of committed evidence serializes it through the `[USER_HOME]` boundary.

No model, no asset, no download, no device, no inference: each case hands a writer its own
real function, a record shaped like the one it really publishes, and a temporary directory,
then reads the bytes back. The unit under test is the difference between the record in memory
and the artefact on disk - the first has to keep working, the second may not name the account
that produced it.

The identity package is the case where the two destinations differ: the same `VoiceRegistry`
class writes runtime state under the data root and tracked evidence under `artifacts/`, and
only the second is redacted.

The static fence at the end is the part that keeps this true after today. A new tool that opens
an artefact with `json.dumps` directly, or a field that starts carrying a home path in one of the
files listed as carrying none, fails here rather than in the published tree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import tools.asset_status_audit as asset_status_audit  # noqa: E402
import tools.audit as audit  # noqa: E402
import tools.multilingual_eval as mle  # noqa: E402
import tools.platform_demo as platform_demo  # noqa: E402
import tools.real_speech_check as rsc  # noqa: E402
import tools.seed_voices as seed_voices  # noqa: E402
import tools.selftest_audio as selftest  # noqa: E402
from nativevoice.config import data_root  # noqa: E402
from nativevoice.core.language import CapabilityStatus  # noqa: E402
from nativevoice.diagnostics.evidence_paths import PLACEHOLDER  # noqa: E402
from nativevoice.voices.identity import (  # noqa: E402
    Compatibility,
    InstallRecord,
    LanguageSupport,
    Measurements,
    VoiceIdentity,
)
from nativevoice.voices.registry import (  # noqa: E402
    PLATFORM_DEFAULT,
    VoiceRegistry,
    identity_from_json,
    identity_to_json,
)

HOME = str(Path.home())
ACCOUNT = Path(HOME).name


def home_forms() -> tuple[str, ...]:
    """Every spelling of this home directory that a serialized artefact could carry.

    The boundary's own prefix list, each in its raw and its JSON form: `json.dumps` doubles
    every backslash, so a leaked Windows path appears in an artefact as the escaped shape and
    a check against the raw spelling alone passes on a file that leaks. A tool that wrote a
    forward-slash path - `as_posix`, a URI, a Git-Bash override - leaks in the raw shape
    instead, which is why both are needed.
    """
    from nativevoice.diagnostics.evidence_paths import home_prefixes

    forms = []
    for prefix in home_prefixes():
        forms.append(prefix)
        forms.append(json.dumps(prefix)[1:-1])
    return tuple(dict.fromkeys(forms))


def assert_redacted(path: Path, account_check: bool = True) -> str:
    """Read an artefact back and refuse it if it still names this machine's account.

    `account_check` is off only where a test's own scratch directory is recorded inside the
    artefact: the pytest temp root repeats the account in a segment of its own, which the
    boundary deliberately does not rewrite - see
    `test_an_account_repeated_outside_the_home_prefix_is_not_covered`.
    """
    text = path.read_text(encoding="utf-8")
    leaked = [form for form in home_forms() if form in text]
    assert not leaked, f"{path.name} still carries the home directory as {leaked}: {text[:200]}"
    assert PLACEHOLDER in text, text[:200]
    if account_check:
        assert ACCOUNT not in text, text[:200]
    return text


def packaged_identity() -> VoiceIdentity:
    """An identity with the three path fields a measured package really carries.

    `install.source` is the asset root, `languages[].evidence` the file that earned the
    capability, `measurements.source` the evaluation directory - all absolute, all under home,
    because that is how the measuring tool records them.
    """
    return VoiceIdentity(
        voice_id=PLATFORM_DEFAULT,
        display_name=PLATFORM_DEFAULT,
        named=False,
        compatibility=Compatibility(
            engine_id="sherpa-kokoro", model_id="kokoro-int8-multi-lang-v1_1", speaker_slot=0
        ),
        languages=(
            LanguageSupport(
                language="en",
                status=CapabilityStatus.MEASURED,
                metric_name="roundtrip_wer",
                metric_value=0.07,
                measured_on="2026-10-15",
                evidence=f"{HOME}\\Documents\\Qoder\\nativevoice\\artifacts\\platform\\platform-run.json",
                limitations=("round-trip on machine-generated speech",),
            ),
        ),
        measurements=Measurements(host="host-1", source=f"{HOME}\\Documents\\Qoder\\nativevoice\\artifacts\\evaluation"),
        install=InstallRecord(
            installed_on="2026-10-15T10:00:00",
            source=f"{HOME}\\AppData\\Local\\ultron-voice\\models\\kokoro-int8-multi-lang-v1_1",
            digests={"model.onnx": "a" * 64},
            removal="delete this identity JSON; the model asset is untouched",
        ),
    )


def test_eval_partial_record_is_redacted_and_the_engine_still_has_its_paths(tmp_path: Path, monkeypatch) -> None:
    """A pass record names the hub cache, the snapshot and the asset root, all three under home."""
    monkeypatch.setattr(mle, "OUT", tmp_path)
    record = {
        "pass": "large-v3-int8",
        "hub_cache_root": f"{HOME}\\.cache\\huggingface\\hub",
        "repository": f"{HOME}\\.cache\\huggingface\\hub\\models--Systran--faster-whisper-large-v3",
        "asset_root": f"{HOME}\\AppData\\Local\\ultron-voice-measurement\\models\\faster-whisper-large-v3",
        "results": [{"sample": "a.mp3", "working_copy": f"{HOME}\\work\\a.16k.wav"}],
        "engine_warm_seconds": 4.2,
    }
    mle._write_partial("large-v3-int8", record)
    written = assert_redacted(tmp_path / "multilingual-large-v3-int8.json")
    assert "ultron-voice-measurement" in written, "the measurement isolation stays visible"
    assert record["asset_root"].startswith(HOME), "the loaded manifest is still addressable in memory"


def test_eval_inventory_is_redacted_while_the_pass_reads_the_real_working_copy(
    tmp_path: Path, monkeypatch
) -> None:
    """The inventory is the artefact; the dict is the pass's own handle on the decoded audio."""
    monkeypatch.setattr(mle, "OUT", tmp_path)
    working = f"{HOME}\\Documents\\Qoder\\_multilingual_eval\\a.16k.wav"
    inventory = {"downloads": f"{HOME}\\Downloads", "workspace": f"{HOME}\\Documents\\Qoder\\_multilingual_eval",
                 "samples": [{"paths": [f"{HOME}\\Downloads\\a.mp3"], "working_copy": working}]}
    mle._write_inventory(inventory)
    written = assert_redacted(tmp_path / "multilingual-inventory.json")
    assert "Downloads" in written
    assert inventory["samples"][0]["working_copy"] == working


def test_real_speech_artefact_is_redacted_but_the_backup_path_stays_real(tmp_path: Path) -> None:
    """The persistence block proves nothing if it can no longer say which files appeared.

    `preserve_previous` returns the path it copied so the caller can verify the copy exists;
    only the published form carries the placeholder.
    """
    target = tmp_path / "artifacts" / "device" / "real-speech-read-aloud.json"
    earlier = {"generated": "2026-10-15T09:00:00", "persistence": {"data_root": f"{HOME}\\AppData\\Local\\ultron-voice"}}
    rsc.write_artefact(earlier, target)
    assert PLACEHOLDER in target.read_text(encoding="utf-8")

    report = {"generated": "2026-10-15T10:00:00",
              "recordings_written": [f"{HOME}\\AppData\\Local\\ultron-voice\\hosts\\selftest\\rec_00.wav"],
              "persistence": {"data_root": f"{HOME}\\AppData\\Local\\ultron-voice",
                              "created_during_run": [f"{HOME}\\AppData\\Local\\ultron-voice\\approvals.json"]}}
    kept = rsc.write_artefact(report, target)
    assert Path(kept).is_file(), "the previous generation is kept under its real path"
    # The backup is a byte copy of an artefact that was already redacted, so the whole
    # chain of runs stays free of the account without this tool rewriting history.
    backup_text = Path(kept).read_text(encoding="utf-8")
    leaked = [form for form in home_forms() if form in backup_text]
    assert not leaked, f"the kept backup carries the home directory as {leaked}: {backup_text[:200]}"
    assert PLACEHOLDER in backup_text, "the previous generation is kept redacted, not raw"
    written = assert_redacted(target, account_check=False)
    assert "approvals.json" in written
    assert report["persistence"]["data_root"].startswith(HOME)


def test_platform_run_is_redacted_after_the_host_roots_were_compared(tmp_path: Path) -> None:
    """The verdict compares two hosts' data roots; redaction must not make them look identical."""
    ultron = f"{HOME}\\Documents\\Qoder\\nativevoice\\artifacts\\platform\\hosts\\ultron"
    fieldmind = f"{HOME}\\Documents\\Qoder\\nativevoice\\artifacts\\platform\\hosts\\fieldmind"
    report = {"hosts": {"ultron": {"data_root": ultron, "store_isolated": f"{ultron}\\recordings",
                                   "audio_file": f"{ultron}\\reply.wav"},
                        "fieldmind": {"data_root": fieldmind}}}
    assert report["hosts"]["ultron"]["data_root"] != report["hosts"]["fieldmind"]["data_root"]
    target = tmp_path / "platform-run.json"
    platform_demo.write_run(report, target)
    written = assert_redacted(target)
    published = json.loads(written)
    assert published["hosts"]["ultron"]["data_root"].replace("\\", "/").endswith("hosts/ultron")
    assert report["hosts"]["ultron"]["data_root"] == ultron


def test_asset_corrections_are_redacted_without_touching_the_live_manifest(tmp_path: Path) -> None:
    """The audit rewrites the manifest it corrects; the copy it publishes must not be the same."""
    out = tmp_path / "artifacts" / "audit" / "asset-status-corrections.json"
    payload = {"generated": "2026-10-15T10:00:00", "corrections": [{
        "model_id": "faster-whisper-base",
        "evidence_inspected": {"checked_directory": f"{HOME}\\.cache\\huggingface\\hub\\models--Systran--faster-whisper-base"},
        "backup_of_previous_manifest": f"{HOME}\\Documents\\Qoder\\nativevoice\\artifacts\\audit\\manifest-snapshots\\faster-whisper-base.json.pre-audit",
    }]}
    asset_status_audit.write_corrections(payload, out)
    written = assert_redacted(out)
    assert "models--Systran--faster-whisper-base" in written

    live = tmp_path / "manifests" / "faster-whisper-base.json"
    live.parent.mkdir(parents=True)
    live.write_text(json.dumps({"asset_root": f"{HOME}\\.cache\\huggingface\\hub"}), encoding="utf-8")
    copy = asset_status_audit.redact_copy(live, tmp_path / "pre-audit.json")
    assert json.loads(live.read_text(encoding="utf-8"))["asset_root"].startswith(HOME)
    assert json.loads(copy.read_text(encoding="utf-8"))["asset_root"] == (
        f"{PLACEHOLDER}\\.cache\\huggingface\\hub"
    )


def test_one_corrupt_manifest_costs_only_its_own_backup(tmp_path: Path) -> None:
    """The audit snapshots every installed manifest, so one bad record cannot take the rest down.

    Before this the loop called `redact_copy` bare: a manifest that was not valid JSON raised
    out of `main()`, the pass stopped at that asset, and the operator lost the snapshots of the
    assets that came after it - the same records the next run would overwrite without a
    preserved copy of the previous claim. Reporting per asset is also what makes the failure
    inspectable: the reason belongs in the corrections record, not on a traceback.
    """
    manifests = tmp_path / "manifests"
    snapshots = tmp_path / "snapshots"
    manifests.mkdir()
    sources = {
        "good-a": json.dumps({"model_id": "good-a", "asset_root": f"{HOME}\\.cache\\a"}, indent=2) + "\n",
        "good-b": json.dumps({"model_id": "good-b", "asset_root": f"{HOME}\\.cache\\b"}, indent=2) + "\n",
        "broken": '{"model_id": "broken", "asset_root": ',
    }
    for name, text in sources.items():
        (manifests / f"{name}.json").write_text(text, encoding="utf-8")

    results = {
        name: asset_status_audit.backup_manifest(
            manifests / f"{name}.json", snapshots / f"{name}.json.pre-audit"
        )
        for name in sources
    }

    assert results["good-a"] is None and results["good-b"] is None, results
    assert results["broken"] is not None and "broken.json" in results["broken"], results["broken"]
    assert (snapshots / "good-a.json.pre-audit").is_file() and (snapshots / "good-b.json.pre-audit").is_file(), (
        "the valid manifests were still backed up"
    )
    assert not (snapshots / "broken.json.pre-audit").exists(), "a corrupt source leaves no truncated backup"

    written = (snapshots / "good-a.json.pre-audit").read_text(encoding="utf-8")
    assert json.loads(written)["asset_root"] == f"{PLACEHOLDER}\\.cache\\a"
    assert written.endswith("\n"), "the live manifest's final newline is mirrored, not dropped"


def test_audit_findings_are_redacted_only_in_the_published_form(tmp_path: Path) -> None:
    """A PASS detail names a root by its last component; a FAIL detail quotes the resolved path.

    The operator has to see both roots to fix an isolation break, so the console keeps the real
    strings and only the artefact loses the account.
    """
    findings = [
        audit.Finding("measurement assets cannot reach the product path", "PASS",
                      f"roots distinct as spelled and after resolution ({Path(HOME, 'appdata', 'ultron-voice').name})"),
        audit.Finding("assets isolated", "FAIL",
                      f"the measurement root is not a separate tree once resolved: "
                      f"{HOME}\\AppData\\Local\\ultron-voice-measurement vs {HOME}\\AppData\\Local\\ultron-voice"),
    ]
    out = tmp_path / "artifacts" / "audit" / "mechanical-findings.json"
    audit.write_findings(findings, out)
    written = assert_redacted(out)
    assert "ultron-voice-measurement" in written
    assert findings[1].detail.count(HOME) == 2, "the check still reports the paths it compared"


def test_selftest_artefact_is_redacted_while_the_recording_stays_reachable(tmp_path: Path) -> None:
    """A retained recording is named by the artefact and opened by the operator afterwards."""
    recording = f"{HOME}\\AppData\\Local\\ultron-voice\\hosts\\selftest\\recordings\\speech_00.wav"
    result = {"runs": [{"capture": {"recordings_written": [recording], "retained": True,
                                    "peak_input_level": 0.11}}],
              "verdict": "PASS"}
    out = tmp_path / "artifacts" / "device" / "selftest.json"
    selftest.write_result(result, out)
    written = assert_redacted(out)
    assert "speech_00.wav" in written
    assert json.loads(out.read_text(encoding="utf-8"))["runs"][0]["capture"]["retained"] is True
    assert result["runs"][0]["capture"]["recordings_written"] == [recording]


def test_a_committed_identity_copy_is_redacted_while_the_live_package_keeps_its_paths(
    tmp_path: Path,
) -> None:
    """One registry class, two destinations: the data root is runtime state, `artifacts/` is evidence."""
    identity = packaged_identity()
    live_path = VoiceRegistry(tmp_path / "data" / "voices").save(identity)
    published_path = VoiceRegistry(tmp_path / "repo" / "voices", committed=True).save(identity)

    live = live_path.read_text(encoding="utf-8")
    assert json.loads(live)["install"]["source"] == identity.install.source
    assert json.loads(live)["languages"][0]["evidence"] == identity.languages[0].evidence
    assert PLACEHOLDER not in live, "a runtime package must not be sanitised to satisfy the scan"
    assert live == identity_to_json(identity), "a live package keeps the exact bytes it always had"

    published = assert_redacted(published_path)
    data = json.loads(published)
    published_form = json.dumps(data)
    leaked = [form for form in home_forms() if form in published_form]
    assert not leaked, f"the published copy keeps the home prefix in some field as {leaked}"
    assert data["install"]["source"] == (
        f"{PLACEHOLDER}\\AppData\\Local\\ultron-voice\\models\\kokoro-int8-multi-lang-v1_1"
    )
    assert data["languages"][0]["evidence"] == (
        f"{PLACEHOLDER}\\Documents\\Qoder\\nativevoice\\artifacts\\platform\\platform-run.json"
    )
    assert data["measurements"]["source"].startswith(PLACEHOLDER)

    assert identity.install.source.startswith(HOME), "saving must not sanitise the object in memory"
    assert identity.languages[0].evidence.startswith(HOME)
    assert identity.measurements.source.startswith(HOME)

    assert data["compatibility"] == json.loads(live)["compatibility"], "the engine binding is load-bearing"
    assert data["install"]["digests"] == {"model.onnx": "a" * 64}, "verification evidence stays in the copy"
    restored = identity_from_json(published)
    assert restored.voice_id == PLATFORM_DEFAULT and restored.speaks("en") is True
    assert restored.compatibility.speaker_slot == 0


def test_the_tracked_platform_demo_route_sanitises_the_identities_it_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The demonstration binds the flag to where it is writing, so a repo root is never raw."""
    monkeypatch.setattr(platform_demo, "ROOT", (tmp_path / "repo").resolve())
    identity = packaged_identity()
    published = platform_demo.voice_registry(tmp_path / "repo" / "artifacts" / "platform" / "voices").save(identity)
    assert_redacted(published)
    live = platform_demo.voice_registry(tmp_path / "voices").save(identity)
    assert json.loads(live.read_text(encoding="utf-8"))["install"]["source"].startswith(HOME), (
        "a run pointed outside the repository keeps real paths"
    )


def test_a_default_platform_demo_run_publishes_identities_into_the_repository() -> None:
    """The route above is not hypothetical: this is where the tracked voice packages came from."""
    voices_root = platform_demo.DEFAULT_OUT / "voices"
    assert voices_root.is_relative_to(platform_demo.ROOT)
    assert platform_demo.voice_registry(voices_root).committed is True
    tracked = voices_root / f"{PLATFORM_DEFAULT}.json"
    assert tracked.is_file(), "the default run writes a file the repository already publishes"
    assert json.loads(tracked.read_text(encoding="utf-8"))["install"]["source"].startswith(f"{PLACEHOLDER}\\")


def test_seed_voices_redacts_a_repository_out_root_and_leaves_the_data_root_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--out` decides the treatment; with none, the identity is installed as runtime state."""
    repo = (tmp_path / "repo").resolve()
    monkeypatch.setattr(seed_voices, "ROOT", repo)
    identity = packaged_identity()

    default = seed_voices.voice_registry("")
    assert default.committed is False
    assert default.root == Path(data_root()) / "voices"

    tracked = seed_voices.voice_registry(str(repo / "artifacts" / "platform" / "voices"))
    assert tracked.committed is True
    assert_redacted(tracked.save(identity))

    live = seed_voices.voice_registry(str(tmp_path / "voices"))
    assert live.committed is False
    assert json.loads(live.save(identity).read_text(encoding="utf-8"))["install"]["source"].startswith(HOME)


def test_a_relative_out_root_inside_the_repository_is_still_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative `--out` resolves against the working directory, so it cannot slip the fence."""
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    monkeypatch.setattr(seed_voices, "ROOT", repo)
    monkeypatch.chdir(repo)
    assert seed_voices.voice_registry("artifacts/platform/voices").committed is True
    assert seed_voices.voice_registry("..\\elsewhere\\voices").committed is False, (
        "a predicate that redacts everything is not a predicate about the repository"
    )


def test_the_identity_writer_imports_the_boundary() -> None:
    """The package serializer is a committed-evidence writer, not a tool-internal detail."""
    source = (ROOT / "src" / "nativevoice" / "voices" / "registry.py").read_text(encoding="utf-8")
    assert "from ..diagnostics import evidence_paths" in source
    assert "identity_to_json(identity, committed=self.committed)" in source


@pytest.mark.parametrize(
    "tool",
    ["multilingual_eval.py", "real_speech_check.py", "platform_demo.py", "asset_status_audit.py",
     "audit.py", "offline_proof.py", "selftest_audio.py"],
)
def test_in_scope_writers_import_the_boundary(tool: str) -> None:
    """A writer that stops importing the helper has started publishing raw paths again."""
    source = (ROOT / "tools" / tool).read_text(encoding="utf-8")
    assert "from nativevoice.diagnostics.evidence_paths import" in source, tool


# Writers reviewed in the A53 inventory and found to publish no path at all: each payload is
# numbers, model ids, licence strings and clip names, never a filesystem path. They stay on this
# list until a field in one of them starts naming a file. `offline_proof.py` left it when its
# publication path was gated and routed through the boundary above: the scan reports findings
# relative to the directory scanned, but that is a construction this writer no longer has to
# rely on, so it is now classified with the writers the boundary is asserted over rather than
# declared exempt. The set equality below is unchanged and remains an equality.
PATH_FREE_WRITERS = {
    "ab_samples.py": "records `file` by name only",
    "benchmark_profiles.py": "records thread counts and timings",
    "corpus_forge.py": "records clip names under the corpus directory",
    "evaluate.py": "records transcripts and scores",
    "multilingual_compare.py": "copies configuration fields, never a path field",
    "tts_head_to_head.py": "records medians; the model directory stays in the engine",
    "tts_latency_lab.py": "records medians",
}


def test_no_tool_writes_committed_json_around_the_boundary() -> None:
    """Every publisher of evidence is either redacting or explicitly declared path-free."""
    raw = {}
    for path in sorted((ROOT / "tools").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if ".write_text(json.dumps(" in source or "json.dump(" in source:
            raw[path.name] = source
    assert set(raw) == set(PATH_FREE_WRITERS), {name: PATH_FREE_WRITERS.get(name, "unreviewed") for name in raw}
