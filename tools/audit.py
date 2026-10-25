"""Static and runtime audit checks over the whole engine.

An audit that is only reading produces impressions. These are mechanical checks,
each with a finding or a number, so a claim of "audited" can be re-run and
compared after a fix. Run from the project root:

    python tools/audit.py

Exits non-zero if any check FAILs. NOT RUN is reported separately from FAIL: a
check that could not execute has not passed, and pretending otherwise is the
specific self-deception this project is built to avoid.

Writing is not part of checking. A plain run reports and creates **nothing**, because
the tool's own output area is committed evidence and an ordinary verification must not
dirty it. Four explicit destinations exist, and they are mutually exclusive by design:

    --out PATH            write the findings to a destination outside the repository's
                          protected evidence areas (`artifacts/` and `evidence/`); the
                          directory must exist, an existing file at that destination is
                          replaced, and nothing about it is re-checked before the write
    --check-artefact PATH compare the findings against a committed findings record and
                          report the row-level drift, changing nothing; drift exits non-zero
    --publish PATH        overwrite an existing committed record in artifacts/audit/, with
                          --yes, and only while its bytes still equal its HEAD version
    --publish-new PATH    create a record that does not exist yet, with --yes, as a direct
                          child of artifacts/audit/ named mechanical-findings-YYYY-MM-DD.json

Overwrite and creation are separate flags and neither falls back into the other:
`--publish` refuses an absent target, `--publish-new` refuses a target that exists. Both
print the resolved destination, the findings summary and the proposed content digest, and
write nothing until `--yes`. Nothing is written without one of these, and no check runs
until the requested destination has been accepted: a typo must not cost a full run, and a
refusal must never have executed anything. Paths are resolved before they are judged, so
`..`, a relative spelling and a symlink are all refused on what they point at rather than
on how they were typed. Publication is a decision with a printed before/after, never a
side effect of looking.

The naming convention is enforced, not suggested, because the artefact area is evidence:
a file there whose name says nothing about when it was taken makes two findings records
unreadable against each other, and `x.json` beside the committed record is a file nobody
can explain. The date is the revision; the undated `mechanical-findings.json` remains the
one record `--publish` overwrites and can never be minted by `--publish-new`.

Each write closes its own check-then-write window as far as the filesystem allows. A new
record is created with `O_CREAT | O_EXCL`, so a file that appeared after validation makes
the creation fail instead of being followed. An overwrite re-reads both the working bytes
and the `HEAD` bytes immediately before writing and refuses if either moved; the remaining
window there is the instant between that read and the write's own open, which no check
inside this process can close - the recoverable guarantee is that the refused bytes are
still in `HEAD`, not that the file was never touched.

`--out` is the deliberate exception, not an oversight: it publishes nothing, names no
evidence and is not compared against any baseline, so it neither re-validates its
destination before writing nor refuses a file that appeared or changed there during the
run. An existing destination is replaced. The protection that matters for it is the one
applied before the run - a destination inside `artifacts/` or `evidence/` is refused, so
what can be replaced is only a file the operator chose outside the evidence areas.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io as _io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import date
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

#: The only sites allowed to raise a builtin `TypeError`, keyed `path-relative-to-src::function`.
#:
#: `redact()` refuses a non-string because a `Path` handed to it would serialise as an
#: un-redacted `PurePath` repr and publish the account name — a contract about the
#: argument's type, which no domain error states better and which
#: `tests/test_evidence_paths.py::test_redact_refuses_a_path_object` pins. Nothing else
#: qualifies: a builtin raised anywhere other than these keys is an offender, any other
#: builtin category is an offender even at these keys, and a key whose guard has gone is
#: itself an offender, so an exception can never outlive the reason it was granted.
ARGUMENT_TYPE_GUARDS = {"diagnostics/evidence_paths.py::redact"}


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


def _raise_sites(tree: ast.AST):
    """Yield `(enclosing_function_or_None, raise_node)` for every `raise Name(...)` site.

    Parents are tracked on purpose: the taxonomy exception is granted per guarding
    function, and `ast.walk` throws exactly that relationship away.
    """

    def walk(node: ast.AST, function: str | None):
        for child in ast.iter_child_nodes(node):
            inside = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else function
            if isinstance(child, ast.Raise) and isinstance(child.exc, ast.Call):
                yield function, child
            yield from walk(child, inside)

    yield from walk(tree, None)


def check_error_taxonomy() -> Finding:
    """Library code raises our types, so a host can branch on category not message.

    A builtin `TypeError` is admitted only at a reviewed argument-type guard named in
    ARGUMENT_TYPE_GUARDS, and admitted sites stay visible in the finding.
    """
    allowed = _our_exception_names() | {"SystemExit", "StopIteration", "KeyboardInterrupt"}
    offenders: list[str] = []
    admitted: list[str] = []
    matched: set[str] = set()
    for path in _source_files():
        relative = path.relative_to(SRC).as_posix()
        for function, node in _raise_sites(_tree(path)):
            name = ast.unparse(node.exc.func)
            if "." in name or name.startswith("self"):
                continue
            if name in allowed:
                continue
            guard = f"{relative}::{function}"
            if name == "TypeError" and function is not None and guard in ARGUMENT_TYPE_GUARDS:
                admitted.append(f"{guard}:{node.lineno}")
                matched.add(guard)
                continue
            offenders.append(f"{path.name}:{node.lineno}:{name}")
    offenders.extend(f"audit.py:0:stale-exception:{guard}" for guard in sorted(ARGUMENT_TYPE_GUARDS - matched))
    detail = f"{len(offenders)} raise site(s) outside core.errors" + (f": {offenders[:6]}" if offenders else "")
    if admitted:
        detail += f"; {len(admitted)} admitted argument-type guard(s): {admitted}"
    return Finding("error taxonomy", "PASS" if not offenders else "FAIL", detail)


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


def check_measurement_assets_can_never_reach_the_product_path() -> Finding:
    """A research asset stays off the shipped path, override included.

    Approval answers "has a human signed this off"; scope answers "may this be
    served by the product path". They used to be one flag, so a permission written
    for local benchmarking became product routing the moment someone set
    `review=approved` - `redistributable=False` satisfies the approval gate by
    design. Both scopes are written into one directory with byte-identical assets,
    review state and provenance, so the only variable is `use_scope`, and the
    research asset must be refused by discovery, by `register()` and by the load
    seam while the product asset loads. The seam is also tested alone, by writing
    the manifest into an index discovery had rejected, so a bypass of the registry
    cannot silently re-open it. The positive control matters as much as the
    refusals: without it a gate that simply broke every load would report a pass.

    Two guards with nearly the same name own different axes and are proved apart,
    because reading them as one thing is how the boundary gets broken in review:
    :func:`ensure_product_scope` is the scope-only half the load seam calls, which no
    override widens, while :func:`ensure_product_eligible` is the combined
    approval-plus-scope reporting form that the seam deliberately does not call.
    """
    import hashlib
    import tempfile

    from nativevoice.config.paths import data_root, measurement_data_root
    from nativevoice.core.errors import ModelNotApproved
    from nativevoice.core.language import Task
    from nativevoice.models.api import Asset, AssetKind, AssetScope, ModelManifest, Provenance, ReviewStatus
    from nativevoice.models.approval import (
        ApprovalGate,
        ensure_product_eligible,
        ensure_product_scope,
    )
    from nativevoice.models.lifecycle import ModelManager
    from nativevoice.models.manifest_io import save_manifest
    from nativevoice.models.registry import ModelRegistry

    payload = b"audit-identical-weights"
    record = Asset(
        kind=AssetKind.MODEL,
        relative_path="weights.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    # redistributable=False on purpose: that is the real shape of the unresolved
    # research assets, and it must not be the thing that saves the boundary.
    signed_off = Provenance(source="audit", licence_spdx="MIT", redistributable=False)
    common = dict(engine="stub", task=Task.STT, assets=(record,), provenance=signed_off,
                  review=ReviewStatus.APPROVED, approved_by="audit", approved_on="audit")
    product = ModelManifest(model_id="audit-product", **common)
    research = ModelManifest(model_id="audit-research", use_scope=AssetScope.MEASUREMENT, **common)

    escapes: list[str] = []

    def refused(build, label: str) -> None:
        try:
            build()
        except ModelNotApproved as error:
            if "use scope" not in str(error):
                escapes.append(f"{label}: refused for the wrong reason - {error}")
        except Exception as error:
            escapes.append(f"{label}: raised {type(error).__name__} instead of a scope refusal")
        else:
            escapes.append(f"{label}: accepted")

    with tempfile.TemporaryDirectory() as working:
        for manifest in (product, research):
            save_manifest(manifest, Path(working) / f"{manifest.model_id}.json")
            destination = Path(working) / manifest.model_id / "weights.bin"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        shipped = ModelRegistry(working)
        research_registry = ModelRegistry(working, scope=AssetScope.MEASUREMENT)
        if research.model_id in shipped.ids():
            escapes.append("discovery indexed the research asset")
        if not any("use scope" in issue.reason for issue in shipped.issues):
            escapes.append("the research asset was dropped silently, not reported")
        if research.model_id not in research_registry.ids():
            escapes.append("the measurement registry does not serve its own scope")
        refused(lambda: shipped.register(research), "register()")
        # Proved separately on purpose: this is the scope-only half the load seam
        # calls, and the next line is the combined reporting form it does not.
        refused(lambda: ensure_product_scope(research), "scope-only seam guard")
        refused(lambda: ensure_product_eligible(research), "combined eligibility report")

        gate = ApprovalGate()
        gate.override(research, model_id=research.model_id, reason="audit: approved for local measurement")
        # The gate must be open here, or this proves nothing: the whole gap is that
        # an approved-but-not-redistributable asset passes approval, and scope is
        # what stops it from going further.
        try:
            gate.check(research)
        except Exception as error:
            escapes.append(f"control: the approval gate refused it too ({error}), so scope proved nothing")
        refused(
            lambda: ModelManager(research_registry, lambda m, b: object(), approvals=gate).load(research.model_id),
            "load seam under an active override",
        )
        # The last seam has to hold on its own: copying the manifest into an index
        # that discovery already rejected bypasses both `_index()` checks, and the
        # bytes on disk are digest-correct, so only the load gate can refuse it.
        injected = ModelRegistry(working)
        injected.manifests[research.model_id] = research
        refused(
            lambda: ModelManager(injected, lambda m, b: object(), approvals=gate).load(research.model_id),
            "hand-injected manifest with intact assets",
        )
        try:
            ModelManager(shipped, lambda m, b: object()).load(product.model_id)
        except Exception as error:
            escapes.append(f"control: the product asset could not be loaded - {type(error).__name__}: {error}")

    # Resolved, not spelled. Two roots can carry different names and still be the same
    # tree through a junction, a symlink or a per-user redirect, and Windows' Store
    # build redirects LOCALAPPDATA in exactly that way, so a lexical comparison proves
    # only that the strings differ. `resolve()` defaults to strict=False: a scope that
    # has never been provisioned is compared without its directory being created.
    shipped_root = data_root().resolve()
    research_root = measurement_data_root().resolve()
    overlapping = (
        research_root == shipped_root
        or shipped_root in research_root.parents
        or research_root in shipped_root.parents
    )
    if overlapping:
        escapes.append(
            f"the measurement root is not a separate tree once resolved: {research_root} vs {shipped_root}"
        )
    detail = (
        f"escapes: {escapes}"
        if escapes
        else "identical assets: product scope loaded; research scope refused by index, register(), "
        "the scope-only seam guard, the combined eligibility report and the load seam - override "
        f"included, injection included; roots distinct as spelled and after resolution ({research_root.name})"
    )
    return Finding("measurement assets cannot reach the product path", "PASS" if not escapes else "FAIL", detail)


def write_findings(findings: list[Finding], out: Path) -> Path:
    """Publish the findings table as committed evidence, with home paths redacted.

    A PASS detail names a root by its last component only, but the checks that fail quote
    the resolved paths to show what collided, and on a real machine those strings carry the
    account that ran the tool. The artefact is the published form, so the redaction belongs
    here; the console print above still shows the operator the paths to act on.
    """
    from nativevoice.diagnostics.evidence_paths import write_json

    return write_json(out, [item._asdict() for item in findings])


CHECKS = (
    check_error_taxonomy,
    check_network_isolation,
    check_lazy_optional_imports,
    check_host_independence,
    check_lifecycle_symmetry,
    check_no_global_mutable_state,
    check_speech_path_persists_nothing,
    check_asset_fetches_are_forced_offline,
    check_approval_gate_is_inalienable,
    check_measurement_assets_can_never_reach_the_product_path,
    check_capability_statuses_are_honest,
    check_thread_hygiene,
)

#: The only area `--publish` may write into. Committed evidence lives here, so publication
#: is confined to it and a path outside it is refused rather than quietly accepted.
PUBLISHED_AREA = (ROOT / "artifacts" / "audit").resolve()

#: Every part of the repository that holds committed evidence. `--out` may not write into
#: any of it: an audit of the current tree is not a publication, and the corpus, benchmark,
#: device, evaluation, platform, provenance and attribution records are as much evidence as
#: the findings table is. `--publish` is confined further, to `artifacts/audit/` only.
PROTECTED_AREAS = ((ROOT / "artifacts").resolve(), (ROOT / "evidence").resolve())

#: The exact shape of a published findings record, so a malformed reference is refused
#: rather than compared as if it were evidence. Anything else would either crash or, worse,
#: report a plausible-looking drift: `json.loads("{}")` gives a dict, and iterating a dict
#: yields its keys, so an empty object would read as "no rows recorded".
REFERENCE_KEYS = ("check", "status", "detail")
REFERENCE_STATUSES = ("PASS", "FAIL", "NOT_RUN")


def _inside_protected(resolved: Path) -> bool:
    return any(area == resolved or area in resolved.parents for area in PROTECTED_AREAS)


def _protected_names() -> str:
    return ", ".join(sorted(area.relative_to(ROOT).as_posix() + "/" for area in PROTECTED_AREAS))


def _out_refusal(given: Path, resolved: Path) -> str | None:
    """Why this `--out` destination is not allowed, or None when it is.

    The path is resolved before the comparison, so `../`, a relative spelling and a symlink
    all get the same answer as an absolute one. Refusing the whole evidence area, rather
    than only the findings directory, is what makes the claim in the help text true.
    """
    if given.is_symlink():
        return "is a symlink; a destination must be a real file"
    if _inside_protected(resolved):
        return f"resolves inside the protected evidence area ({_protected_names()}); use --publish to write there deliberately"
    if resolved.exists() and not resolved.is_file():
        return "resolves to something that is not a regular file"
    if not resolved.parent.is_dir():
        return "its directory does not exist (this tool does not create directories for you)"
    return None


def _validate_rows(loaded: object) -> str | None:
    """Why this decoded JSON is not a findings record, or None when it is.

    One rule set, used by both readers of a findings record: the `--check-artefact` file and
    the committed bytes `--publish` compares against. Validating only one of them left the
    publisher parsing untrusted structure with `json.loads` inline, where a bad row raised
    mid-run and an empty object produced a confident nonsense drift report. Duplicate names
    are refused because the comparison keys rows by name, so a repeated name would silently
    collapse one row into another and hide the very drift being looked for.
    """
    if not isinstance(loaded, list):
        return f"the top level must be a list of findings, found {type(loaded).__name__}"
    seen: set[str] = set()
    for index, row in enumerate(loaded):
        if not isinstance(row, dict):
            return f"row {index} must be an object, found {type(row).__name__}"
        missing = [key for key in REFERENCE_KEYS if key not in row]
        extra = sorted(key for key in row if key not in REFERENCE_KEYS)
        if missing:
            return f"row {index} is missing {missing}"
        if extra:
            return f"row {index} carries unexpected keys {extra}"
        if not isinstance(row["check"], str) or not isinstance(row["detail"], str):
            return f"row {index} must have string `check` and `detail`"
        if row["status"] not in REFERENCE_STATUSES:
            return f"row {index} has status {row['status']!r}, expected one of {list(REFERENCE_STATUSES)}"
        if row["check"] in seen:
            return f"row {index} repeats check {row['check']!r}; a finding name must be unique"
        seen.add(row["check"])
    return None


def _reference_rows(path: Path) -> tuple[list[dict] | None, str | None]:
    """Read a findings record from a file, or explain in one line why it is not one.

    Structural validation belongs here rather than in the comparison, because a reference
    that merely parses is not evidence: `["a"]` would raise while indexing a string, and
    `[{"check":"x"}]` would raise on a missing key. A tool that reports on the repository
    must refuse cleanly instead of dumping a traceback where a message belongs.
    """
    if not path.is_file():
        return None, "not a readable file"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return None, f"it is not readable JSON ({error})"
    problem = _validate_rows(loaded)
    if problem is not None:
        return None, problem
    return loaded, None


def _run_checks(checks) -> list[Finding]:
    findings = []
    for check in checks:
        try:
            findings.append(check())
        except Exception as error:  # a check that cannot run has not passed
            findings.append(Finding(check.__name__, "NOT_RUN", f"{type(error).__name__}: {error}"))
    return findings


def _report(findings: list[Finding]) -> tuple[int, int, int]:
    """Print the table the operator acts on, and return its status counts.

    An empty list is reported as an empty run rather than raising: `max()` over no findings
    is a `ValueError` from the printing helper, and a tool that reports on the repository
    should not turn "nothing ran" into a crash.
    """
    width = max((len(item.check) for item in findings), default=0)
    counts = {"PASS": 0, "FAIL": 0, "NOT_RUN": 0}
    for item in findings:
        counts[item.status] += 1
        print(f"{item.status:8s} {item.check.ljust(width)}  {item.detail}")
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['NOT_RUN']} not run")
    return counts["PASS"], counts["FAIL"], counts["NOT_RUN"]


def _rows(findings: list[Finding]) -> list[dict]:
    """The findings exactly as they would be published: same order, same redaction.

    Comparing raw `Finding` objects to a committed artefact would report drift for every
    FAIL row, because the artefact stores redacted paths and the console keeps the real
    ones. Normalising through the same serializer the publisher uses is what makes a
    comparison mean a change rather than mean a privacy rule.
    """
    from nativevoice.diagnostics.evidence_paths import scrub

    return scrub([item._asdict() for item in findings])


def _delta(recorded: list[dict], computed: list[dict]) -> list[str]:
    """Row-level drift between a committed artefact and what the checks now report."""
    remaining = {row["check"]: row for row in recorded}
    lines: list[str] = []
    for row in computed:
        previous = remaining.pop(row["check"], None)
        if previous is None:
            lines.append(f"+ {row['check']}: {row['status']} (not recorded in the artefact)")
        elif previous["status"] != row["status"]:
            lines.append(f"~ {row['check']}: {previous['status']} -> {row['status']}")
        elif previous["detail"] != row["detail"]:
            lines.append(f"~ {row['check']}: detail differs")
    for name, row in remaining.items():
        lines.append(f"- {name}: {row['status']} (recorded, no longer computed)")
    return lines


def _committed_bytes(resolved: Path) -> bytes | None:
    """The bytes `HEAD` holds for this path, or None when it holds none or git is absent."""
    if shutil.which("git") is None:
        return None
    relative = resolved.relative_to(ROOT).as_posix()
    shown = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT, capture_output=True)
    return shown.stdout if shown.returncode == 0 else None


def _publication_refusal(given: Path, resolved: Path) -> str | None:
    """Why this target may not be published, or None when it may.

    `resolve()` alone is not a guard: a symlink planted under the artefact area can name a
    file somewhere else entirely, and writing through it would edit evidence nobody typed.
    So the link itself is refused, the *resolved* parent must be exactly the artefact area
    (no sub-directory games, no `..` escape), and an existing target must be a real file.
    The given path is kept only to name the symlink; everything after that works on the
    resolved one, because a relative `--publish artifacts/audit/x.json` still has to be
    comparable with `ROOT`, which is absolute.
    """
    if given.is_symlink():
        return "is a symlink; a published artefact is a real committed file"
    if resolved.parent != PUBLISHED_AREA:
        return f"does not resolve directly inside {PUBLISHED_AREA.relative_to(ROOT)}/"
    if resolved.exists() and not resolved.is_file():
        return "resolves to something that is not a regular file"
    return None


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _proposed_payload(findings: list[Finding]) -> bytes:
    """The exact content `write_findings` will produce, before line-ending translation.

    The publisher hands this same text to `write_json`, which then writes it in the
    platform's text mode - on Windows that means CRLF, so the file on disk hashes
    differently from what the repository will store after normalisation. The operator
    comparing evidence against a commit needs the normalised digest, so that is the one
    printed, and it is labelled as such rather than passed off as the file's hash.
    """
    from nativevoice.diagnostics.evidence_paths import dumps

    return dumps([item._asdict() for item in findings]).encode("utf-8")


#: The only file name `--publish-new` may create: the committed record's stem, a real
#: calendar date, `.json`, nothing else. A convention, not decoration — the directory is
#: the evidence area, so an arbitrary name (`x.json`, `final-v2.json`) would put an
#: unexplained file beside published evidence and make the set of findings revisions
#: unreadable to anyone comparing two of them. The undated `mechanical-findings.json`
#: stays the committed record that `--publish` overwrites, and cannot be created here.
NEW_RECORD_FORMAT = "mechanical-findings-YYYY-MM-DD.json"
NEW_RECORD_NAME = re.compile(r"^mechanical-findings-(\d{4})-(\d{2})-(\d{2})\.json$")


def _new_name_refusal(name: str) -> str | None:
    """Why this file name is not a publishable new record, or None when it is."""
    match = NEW_RECORD_NAME.match(name)
    if match is None:
        return f"is not a new-record file name; the convention is {NEW_RECORD_FORMAT}"
    try:
        date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError as error:
        return f"names a date that does not exist ({error}) in the convention {NEW_RECORD_FORMAT}"
    return None


def _publish_new_refusal(given: Path, resolved: Path) -> str | None:
    """Why a new-record destination is not allowed, or None when it is.

    Creation is confined to a direct child of the artefact area, must follow the dated
    naming convention, must not already exist, and is judged on the resolved path so a
    `..` or a symlink cannot name somewhere else. Refusing an existing target is the
    point: this operation cannot overwrite anything, so there is no way for it to
    degrade into the update path and no fallback between the two.
    """
    if given.is_symlink():
        return "is a symlink; a new findings record is created as a real file"
    if os.path.lexists(resolved) and not resolved.exists():
        return "already exists as a broken link or entry; refusing to write beside it"
    if resolved.parent != PUBLISHED_AREA:
        return f"does not resolve directly inside {PUBLISHED_AREA.relative_to(ROOT)}/"
    problem = _new_name_refusal(resolved.name)
    if problem is not None:
        return problem
    if resolved.exists():
        return ("already exists; --publish-new only creates, use --publish to overwrite a "
                "committed record whose bytes still match HEAD")
    if not resolved.parent.is_dir():
        return "its directory does not exist"
    return None


def _create_exclusive(target: Path, payload: bytes) -> str | None:
    """Create `target` with the given content, failing if it exists. Returns a refusal text.

    `O_EXCL` is the point: the existence check in `_publish_new_refusal` happens before the
    checks run, and an hour of audit is a wide window for something else to appear. Checking
    then writing through the ordinary text-mode writer would follow that file; exclusive
    creation makes the race the filesystem's problem and answers with a refusal. The payload
    is translated the way text mode would, so an exclusive create and `write_json` produce
    byte-identical records on the same machine.
    """
    data = payload.replace(b"\n", os.linesep.encode()) if os.linesep != "\n" else payload
    try:
        descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    except FileExistsError:
        return "already exists (created after validation); refusing to overwrite"
    except OSError as error:
        return f"could not be created exclusively ({error})"
    owned = False
    try:
        handle = os.fdopen(descriptor, "wb")
        owned = True
        with handle:
            handle.write(data)
    except OSError as error:
        if not owned:
            # `fdopen` failed, so the descriptor is still this function's to close; leaving it
            # open would hold the file locked on Windows and make the cleanup below fail.
            try:
                os.close(descriptor)
            except OSError:
                pass
        # The descriptor was opened with O_EXCL, so this file did not exist a moment ago and
        # is ours; leaving half a record in the evidence area would be worse than removing it.
        try:
            target.unlink()
            outcome = "removed"
        except OSError:
            outcome = "LEFT IN PLACE, remove it by hand"
        return f"could not be written ({error}); the partial file was {outcome}"
    return None


def _baseline_moved(resolved: Path, committed: bytes) -> str | None:
    """Why this overwrite must not proceed, judged on the bytes as they are right now.

    The pre-run validation compared the working file with `HEAD`. Checks take minutes, and
    a `git checkout`, an editor save or another publication can move either side inside that
    window; writing then would bury an edit nobody reviewed or publish against a baseline
    that is no longer the one the operator approved. So both sides are re-read here, at the
    last point before the write, and either drift is a refusal.
    """
    if not resolved.is_file():
        return "the target disappeared after validation; refusing to write it blind"
    if resolved.read_bytes() != committed:
        return ("its working bytes no longer match HEAD; the file changed during the run, "
                "so this write would bury that change")
    now_committed = _committed_bytes(resolved)
    if now_committed != committed:
        return "its HEAD baseline changed during the run; the approved before-state is gone"
    return None


def main(argv: list[str] | None = None, checks=CHECKS) -> int:
    parser = argparse.ArgumentParser(
        prog="tools/audit.py",
        description="Mechanical audit checks over the engine. A plain run writes nothing.",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--out", type=Path, metavar="PATH",
                             help="write the findings outside the protected evidence area "
                                  "(replaces an existing file there; not revalidated before the write)")
    destination.add_argument("--publish", type=Path, metavar="PATH",
                             help="overwrite an existing committed findings record (needs --yes)")
    destination.add_argument("--publish-new", type=Path, metavar="PATH", dest="publish_new",
                             help=f"create a new findings record in artifacts/audit/ named "
                                  f"{NEW_RECORD_FORMAT}, on a real date (needs --yes)")
    destination.add_argument("--check-artefact", type=Path, metavar="PATH",
                             help="report drift against a committed findings record without writing it")
    parser.add_argument("--yes", action="store_true", help="confirm a --publish or --publish-new write")
    args = parser.parse_args([] if argv is None else argv)

    if args.yes and args.publish is None and args.publish_new is None:
        parser.error("--yes only means something together with --publish or --publish-new")

    # Every destination is validated before a single check runs. The runtime checks open
    # threads and drive whole conversations, so a mistyped path must not be able to cost an
    # operator a full run — and a refused write must never have executed anything at all.
    out_target = None
    if args.out is not None:
        out_target = args.out.resolve()
        reason = _out_refusal(args.out, out_target)
        if reason is not None:
            print(f"REFUSED --out {args.out}: {reason}")
            return 1

    recorded = None
    if args.check_artefact is not None:
        recorded, reason = _reference_rows(args.check_artefact)
        if reason is not None:
            print(f"REFUSED --check-artefact {args.check_artefact}: {reason}")
            return 1

    publish_target = committed = committed_rows = None
    if args.publish is not None:
        reason = _publication_refusal(args.publish, args.publish.resolve())
        if reason is not None:
            print(f"REFUSED --publish {args.publish}: {reason}")
            return 1
        publish_target = args.publish.resolve()
        if not publish_target.is_file():
            # Publication overwrites a committed findings record and nothing else. Creating a
            # brand-new artefact is a different decision with its own review, and leaving it
            # permitted here meant a mistyped file name silently minted a new file inside the
            # protected area — `--yes` turns a typo into evidence.
            print(f"REFUSED --publish {args.publish}: it is not an existing committed findings record;"
                  " publishing a new dated revision is a separate decision")
            return 1
        committed = _committed_bytes(publish_target)
        if committed is None:
            print(f"REFUSED --publish {args.publish}: git is unresolvable or the path is not committed, so there is no baseline to check against")
            return 1
        if publish_target.read_bytes() != committed:
            print(f"REFUSED --publish {args.publish}: its bytes differ from HEAD; refusing to bury a local edit")
            return 1
        # The baseline is parsed and validated here, before any check runs, with the same
        # rules as `--check-artefact`. Doing it after the run meant a malformed committed
        # record crashed a full audit at the last step, and `{}` was read as "no rows
        # recorded" — a confident wrong answer rather than a refusal.
        try:
            committed_rows = json.loads(committed.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            print(f"REFUSED --publish {args.publish}: the committed bytes are not readable JSON ({error})")
            return 1
        problem = _validate_rows(committed_rows)
        if problem is not None:
            print(f"REFUSED --publish {args.publish}: the committed baseline is not a valid findings record ({problem})")
            return 1

    new_target = None
    if args.publish_new is not None:
        new_target = args.publish_new.resolve()
        reason = _publish_new_refusal(args.publish_new, new_target)
        if reason is not None:
            print(f"REFUSED --publish-new {args.publish_new}: {reason}")
            return 1

    findings = _run_checks(checks)
    _passed, failed, _not_run = _report(findings)
    exit_code = 1 if failed else 0
    rows = _rows(findings)

    if out_target is not None:
        print(f"wrote {write_findings(findings, out_target)}")

    if recorded is not None:
        drift = _delta(recorded, rows)
        print(f"\nartefact comparison against {args.check_artefact}: {'no drift' if not drift else f'{len(drift)} drift item(s)'}")
        for line in drift:
            print("  " + line)
        if drift:
            exit_code = 1

    if publish_target is not None or new_target is not None:
        writing_to = new_target if new_target is not None else publish_target
        label = writing_to.relative_to(ROOT).as_posix()
        # The findings about to be written are checked against the same rules a committed
        # record is held to. They are produced by the checks, so this is the last point
        # before a write rather than before the run — and a record that fails it is refused
        # rather than published, even though every check passed.
        problem = _validate_rows(rows)
        if problem is not None:
            print(f"\nREFUSED write to {label}: the computed findings are not a valid record ({problem})")
            return 1
        payload = _proposed_payload(findings)
        summary = ", ".join(f"{sum(1 for item in findings if item.status == s)} {s}"
                           for s in ("PASS", "FAIL", "NOT_RUN"))
        print(f"\n{'create' if new_target is not None else 'overwrite'} target: {label}")
        print(f"  resolved destination: {writing_to}")
        print(f"  findings: {len(rows)} ({summary})")
        print(f"  proposed content sha256 (LF-normalised, as the repository will store it): {_sha_bytes(payload)}")
        if not args.yes:
            if publish_target is not None:
                before = _delta(committed_rows, rows)
                print(f"  drift against the committed bytes: {len(before)} item(s)")
                for line in before:
                    print("    " + line)
            print("  this was a dry run: nothing was written; re-run with --yes to publish")
            return exit_code
        if new_target is not None:
            raced = _create_exclusive(new_target, payload)
            if raced is not None:
                print(f"\nREFUSED create {label}: {raced}")
                return 1
        else:
            # The last honest look before an overwrite: the baseline the operator approved
            # is re-read here, minutes after the pre-run check, and a move on either side
            # stops the write.
            moved = _baseline_moved(publish_target, committed)
            if moved is not None:
                print(f"\nREFUSED write to {label}: {moved}")
                return 1
            write_findings(findings, publish_target)
        print(f"  published ({'created' if new_target is not None else 'overwritten'}): "
              f"sha256 {_sha(writing_to)} on disk (text-mode bytes; the normalised digest above is what git records)")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
