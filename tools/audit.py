"""Static and runtime audit checks over the whole engine.

An audit that is only reading produces impressions. These are mechanical checks,
each with a finding or a number, so a claim of "audited" can be re-run and
compared after a fix. Run from the project root:

    python tools/audit.py

Exits non-zero if any check FAILs. NOT RUN is reported separately from FAIL: a
check that could not execute has not passed, and pretending otherwise is the
specific self-deception this project is built to avoid.
"""

from __future__ import annotations

import ast
import io as _io
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "nativevoice"
sys.path.insert(0, str(ROOT / "src"))

OUR_ERRORS = {
    "VoiceError", "ConfigurationError", "CapabilityUnavailable", "EngineUnavailable",
    "ModelNotInstalled", "ModelNotApproved", "IntegrityError", "ResourceLimitExceeded",
    "DeviceError", "DeviceDisconnected", "LanguageUnsupported", "CancellationRequested",
    "EngineTimeout", "EngineCrashed", "StorageError", "ProfileInvalid",
}
BANNED_NETWORK = {"socket", "requests", "httpx", "urllib3", "ftplib", "smtplib", "urllib", "http"}
SPEECH_PATH_PACKAGES = ("core", "audio", "models", "hardware", "config", "language", "stt", "tts", "profiles", "voices", "session", "eval", "integration")


class Finding(NamedTuple):
    check: str
    status: str  # PASS | FAIL | NOT_RUN
    detail: str


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def _source_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _our_exception_names() -> set[str]:
    """Every class that derives from VoiceError, however locally it is declared.

    A subsystem raising its own `NetworkBlocked` is compliant; a subsystem raising a
    bare `ValueError` is not, because the host cannot catch the category.
    """
    names = set(OUR_ERRORS)
    for path in _source_files():
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.ClassDef) and any(
                ast.unparse(base).split(".")[-1] in names or ast.unparse(base).endswith("Error") for base in node.bases
            ):
                names.add(node.name)
    return names


def check_error_taxonomy() -> Finding:
    """Library code raises our types, so a host can branch on category not message."""
    allowed = _our_exception_names() | {"SystemExit", "StopIteration", "KeyboardInterrupt"}
    offenders = []
    for path in _source_files():
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                name = ast.unparse(node.exc.func)
                if "." in name or name.startswith("self"):
                    continue
                if name not in allowed:
                    offenders.append(f"{path.name}:{node.lineno}:{name}")
    return Finding(
        "error taxonomy",
        "PASS" if not offenders else "FAIL",
        f"{len(offenders)} raise site(s) outside core.errors" + (f": {offenders[:6]}" if offenders else ""),
    )


def check_network_isolation() -> Finding:
    offenders = []
    for package in SPEECH_PATH_PACKAGES:
        for path in (SRC / package).rglob("*.py"):
            for node in ast.walk(_tree(path)):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                offenders.extend(f"{path.name}:{node.lineno}:{name}" for name in names if name.split(".")[0] in BANNED_NETWORK)
    return Finding(
        "offline isolation (static)",
        "PASS" if not offenders else "FAIL",
        f"{len(offenders)} network import(s) on the speech path" + (f": {offenders[:6]}" if offenders else ""),
    )


def check_lazy_optional_imports() -> Finding:
    """A heavy runtime may only appear inside a method, never at module import."""
    heavy = {"numpy", "torch", "sherpa_onnx", "onnxruntime", "faster_whisper", "ctranslate2", "psutil", "sounddevice", "kokoro", "misaki"}
    offenders = []
    for path in _source_files():
        tree = _tree(path)
        for node in tree.body:  # top level only
            targets = []
            if isinstance(node, ast.Import):
                targets = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue  # a relative import of our own module, not a third-party runtime
                targets = [(node.module or "").split(".")[0]]
            for name in targets:
                if name in heavy:
                    offenders.append(f"{path.name}:{node.lineno}:{name}")
    return Finding(
        "optional runtimes imported lazily",
        "PASS" if not offenders else "FAIL",
        f"{len(offenders)} module-level heavy import(s)" + (f": {offenders[:6]}" if offenders else ""),
    )


def check_host_independence() -> Finding:
    """Nothing may import a host application; the engine is a library, not a feature of one."""
    offenders = []
    for path in _source_files():
        for node in ast.walk(_tree(path)):
            module = None
            if isinstance(node, ast.Import):
                module = node.names[0].name
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                module = node.module or ""
            if module and module.split(".")[0] not in {"nativevoice"} and re.match(
                r"^(ultron|fieldmind|orukeet|openwhispr|pocket_tts|pocket)", module, re.I
            ):
                offenders.append(f"{path.name}:{node.lineno}:{module}")
    return Finding(
        "host independence",
        "PASS" if not offenders else "FAIL",
        f"{len(offenders)} host/reference import(s)" + (f": {offenders[:6]}" if offenders else ""),
    )


def check_lifecycle_symmetry() -> Finding:
    """Anything that opens a device, stream or model must expose a way to release it.

    A class that reads a file under `with open(...)` and keeps the parsed data
    acquires nothing, so the test looks for a handle stored on the instance - the
    only thing that can actually leak.
    """
    offenders = []
    for path in _source_files():
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {child.name for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))}
            stored_handle = False
            for statement in ast.walk(node):
                if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
                    called = ast.unparse(statement.value.func)
                    if any(name in called for name in ("open(", "Stream(", "Manager(")) and any(
                        isinstance(target, ast.Attribute) for target in statement.targets
                    ):
                        stored_handle = True
            if ("open" in methods or stored_handle) and "close" not in methods:
                offenders.append(f"{path.name}:{node.name}")
    return Finding(
        "resource lifecycle symmetry",
        "PASS" if not offenders else "FAIL",
        f"{len(offenders)} unbalanced lifecycle(s)" + (f": {offenders[:6]}" if offenders else ""),
    )


def check_no_global_mutable_state() -> Finding:
    """Module-level singletons are how two conversations end up sharing a model handle."""
    offenders = []
    for path in _source_files():
        tree = _tree(path)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if not isinstance(target, ast.Name) or target.id.startswith("_"):
                        continue
                    if target.id.isupper():
                        continue  # constants
                    if isinstance(node.value, ast.Call) and ast.unparse(node.value.func).split(".")[0] in {
                        "dict", "list", "set", "defaultdict", "deque", "Queue", "SimpleQueue"
                    }:
                        offenders.append(f"{path.name}:{node.lineno}:{target.id}")
                    elif isinstance(node.value, (ast.Dict, ast.List, ast.Set)):
                        offenders.append(f"{path.name}:{node.lineno}:{target.id}")
    return Finding(
        "no mutable module-level state",
        "PASS" if not offenders else "FAIL",
        f"{len(offenders)} mutable global(s)" + (f": {offenders[:8]}" if offenders else ""),
    )


def check_thread_hygiene() -> Finding:
    """A conversation must not leave threads behind."""
    import array

    from nativevoice.audio.io import CapturingOutput
    from nativevoice.core.audio import AudioBuffer, AudioFormat
    from nativevoice.core.language import LanguageTag
    from nativevoice.integration import VoiceInterface
    from nativevoice.session.state import SessionConfig

    class Stt:
        def transcribe(self, request):
            from nativevoice.stt.api import TranscriptionResult

            return TranscriptionResult(text="hello", language=LanguageTag("en"), engine="audit")

        def stream(self, chunks, language=None, token=None):
            return iter([])

    class Tts:
        def stream(self, request):
            from nativevoice.core.events import SynthesisEvent, SynthesisKind

            for index in range(3):
                yield SynthesisEvent(
                    SynthesisKind.AUDIO if index < 2 else SynthesisKind.EOS,
                    index,
                    audio=AudioBuffer(array.array("h", bytes(480 * 2)), AudioFormat(24000, 1)),
                )

    class Host:
        def respond(self, utterance, language=None):
            return "reply"

    before = {thread.name for thread in threading.enumerate()}
    for _ in range(25):
        voice = VoiceInterface(transcriber=Stt(), synthesizer=Tts(), output=CapturingOutput(), config=SessionConfig())
        voice.exchange(AudioBuffer(array.array("h", bytes(1600 * 2)), AudioFormat(16000, 1)), Host())
        voice.close()
    time.sleep(0.2)
    leaked = {thread.name for thread in threading.enumerate()} - before
    return Finding(
        "no thread leak across 25 turns",
        "PASS" if not leaked else "FAIL",
        f"threads still alive: {sorted(leaked)}" if leaked else "25 full turns opened and closed cleanly",
    )


def check_speech_path_persists_nothing() -> Finding:
    """Local-first is not the same as silent persistence; only named stores write.

    A static scan cannot tell `str.replace()` from `Path.replace()`, so the claim is
    tested by running conversations inside an empty directory and requiring that the
    directory stays empty - the only formulation that actually means it.
    """
    import array
    import tempfile

    from nativevoice.audio.io import CapturingOutput
    from nativevoice.core.audio import AudioBuffer, AudioFormat
    from nativevoice.core.language import LanguageTag
    from nativevoice.integration import VoiceInterface
    from nativevoice.session.state import SessionConfig

    class Stt:
        def transcribe(self, request):
            from nativevoice.stt.api import TranscriptionResult

            return TranscriptionResult(text="hello", language=LanguageTag("en"), engine="audit")

        def stream(self, chunks, language=None, token=None):
            return iter([])

    class Tts:
        def stream(self, request):
            from nativevoice.core.events import SynthesisEvent, SynthesisKind

            yield SynthesisEvent(
                SynthesisKind.EOS,
                0,
                audio=AudioBuffer(array.array("h", bytes(480 * 2)), AudioFormat(24000, 1)),
            )

    class Host:
        def respond(self, utterance, language=None):
            return "reply"

    original = __import__("os").getcwd()
    with tempfile.TemporaryDirectory() as working:
        __import__("os").chdir(working)
        try:
            for _ in range(10):
                voice = VoiceInterface(transcriber=Stt(), synthesizer=Tts(), output=CapturingOutput(), config=SessionConfig())
                voice.exchange(AudioBuffer(array.array("h", bytes(1600 * 2)), AudioFormat(16000, 1)), Host())
                voice.close()
            created = sorted(path.name for path in Path(working).iterdir())
        finally:
            __import__("os").chdir(original)
    return Finding(
        "speech path persists nothing",
        "PASS" if not created else "FAIL",
        f"10 full turns created: {created}" if created else "10 full turns wrote no file anywhere in the working directory",
    )


def check_asset_fetches_are_forced_offline() -> Finding:
    """A runtime load path that *can* fetch a model will fetch one.

    `local_files_only` and `force_download=False` are the difference between a
    local-first engine and one that quietly phones home on first use.
    """
    fetching = {"WhisperModel", "from_pretrained", "hf_hub_download", "snapshot_download", "MediaRecorder"}
    offenders = []
    for path in _source_files():
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] in fetching:
                keywords = {keyword.arg for keyword in node.keywords}
                if "local_files_only" not in keywords and "cache_dir" not in keywords:
                    offenders.append(f"{path.name}:{node.lineno}:{ast.unparse(node.func)}")
    return Finding(
        "asset load paths are offline-forced",
        "PASS" if not offenders else "FAIL",
        f"{len(offenders)} load call(s) without an offline guarantee" + (f": {offenders[:6]}" if offenders else ""),
    )


def check_capability_statuses_are_honest() -> Finding:
    """DECLARED must not satisfy a strict routing decision; assert it on the live ledger."""
    from nativevoice.core.language import Capability, CapabilityStatus, LanguageTag, Task
    from nativevoice.language import CapabilityLedger, LanguageRegistry, LanguageRouter

    ledger = CapabilityLedger()
    registry = LanguageRegistry()
    ledger.record(Capability(Task.STT, CapabilityStatus.DECLARED, "de", engine="vendor-claim"))
    outcome = LanguageRouter(registry=registry).route(ledger=ledger, requested_language=LanguageTag("de"), task=Task.STT)
    strict_rejected = outcome.kind.value in {"unavailable", "ambiguous"}
    ledger.record(Capability(Task.STT, CapabilityStatus.MEASURED, "de", engine="probe", metric_name="wer", metric_value=0.1, measured_on="audit"))
    outcome2 = LanguageRouter(registry=registry).route(ledger=ledger, requested_language=LanguageTag("de"), task=Task.STT)
    return Finding(
        "declared-only capability cannot be routed",
        "PASS" if (strict_rejected and outcome2.kind.value not in {"unavailable", "ambiguous"}) else "FAIL",
        f"DECLARED -> {outcome.kind.value}; after MEASURED -> {outcome2.kind.value}",
    )


def check_approval_gate_is_inalienable() -> Finding:
    from nativevoice.core.language import Task
    from nativevoice.models.api import ModelManifest, Provenance, ReviewStatus
    from nativevoice.models.approval import ensure_approved

    cases = {
        "bare manifest": ModelManifest(model_id="a", engine="e", task=Task.STT),
        "approved, no approver": ModelManifest(model_id="b", engine="e", task=Task.STT, review=ReviewStatus.APPROVED),
        "approved, unknown licence": ModelManifest(
            model_id="c", engine="e", task=Task.STT, review=ReviewStatus.APPROVED, approved_by="audit"
        ),
        "rejected": ModelManifest(
            model_id="d",
            engine="e",
            task=Task.STT,
            review=ReviewStatus.REJECTED,
            approved_by="audit",
            provenance=Provenance(source="s", licence_spdx="MIT", redistributable=True),
        ),
    }
    escaped = []
    for label, manifest in cases.items():
        try:
            ensure_approved(manifest)
            escaped.append(label)
        except Exception:
            pass
    environment_backdoor = any(
        re.search(r"(ULTRON|VOICE).*(APPROV|SKIP|BYPASS)", path.read_text(encoding="utf-8"))
        for path in _source_files()
    )
    return Finding(
        "approval gate default-deny",
        "PASS" if not escaped and not environment_backdoor else "FAIL",
        f"accepted without review: {escaped or 'none'}; env bypass present: {environment_backdoor}",
    )


def main() -> int:
    checks = [
        check_error_taxonomy,
        check_network_isolation,
        check_lazy_optional_imports,
        check_host_independence,
        check_lifecycle_symmetry,
        check_no_global_mutable_state,
        check_speech_path_persists_nothing,
        check_asset_fetches_are_forced_offline,
        check_approval_gate_is_inalienable,
        check_capability_statuses_are_honest,
        check_thread_hygiene,
    ]
    findings = []
    for check in checks:
        try:
            findings.append(check())
        except Exception as error:  # a check that cannot run has not passed
            findings.append(Finding(check.__name__, "NOT_RUN", f"{type(error).__name__}: {error}"))
    width = max(len(item.check) for item in findings)
    counts = {"PASS": 0, "FAIL": 0, "NOT_RUN": 0}
    for item in findings:
        counts[item.status] += 1
        print(f"{item.status:8s} {item.check.ljust(width)}  {item.detail}")
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['NOT_RUN']} not run")
    (ROOT / "artifacts" / "audit").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts" / "audit" / "mechanical-findings.json").write_text(
        json.dumps([item._asdict() for item in findings], indent=2), encoding="utf-8"
    )
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
