"""Tests for the STT orchestration layer.

These exercise real routing/resampling/cancellation/retraction behaviour using
purpose-built stub engines. The `faster_whisper` real-inference path is present
but marked ``requires_approved_asset`` and skipped by default: the CT2 weights on
this machine are unreviewed third-party assets and are NOT RUN for acceptance.
"""

from __future__ import annotations

import array
import builtins
import os
import sys

import pytest

from fixtures import sine
from nativevoice.core.audio import AudioBuffer, AudioFormat, Recognition
from nativevoice.core.cancel import CancellationToken
from nativevoice.core.errors import (
    CapabilityUnavailable,
    CancellationRequested,
    EngineUnavailable,
    LanguageUnsupported,
    ModelNotApproved,
    ModelNotInstalled,
)
from nativevoice.core.events import TranscriptionEvent, TranscriptionKind
from nativevoice.core.language import Capability, CapabilityStatus, LanguageTag, Task
from nativevoice.language import CapabilityLedger, LanguageRouter
from nativevoice.stt import (
    DeterministicEchoEngine,
    EngineRegistry,
    FasterWhisperAdapter,
    TranscriptReducer,
    TranscriptionService,
)
from nativevoice.stt.adapters.optional_backend import ABSENT, BROKEN, import_state
from nativevoice.stt.api import TranscriptionRequest, TranscriptionResult
from nativevoice.stt.text_context import MAX_PROMPT_CHARS, build_prompt, dedupe, sanitize

requires_approved_asset = pytest.mark.requires_approved_asset


def _measured(language, engine):
    return Capability(
        task=Task.STT,
        status=CapabilityStatus.MEASURED,
        language=language,
        engine=engine,
        metric_name="wer",
        metric_value=0.03,
        model_id=f"{engine}-m",
    )


class StubEngine:
    """Non-streaming engine; records what it is asked to do."""

    supports_streaming = False

    def __init__(self, engine_id="stub", rate=16000, langs=("en", "hi"), status=CapabilityStatus.MEASURED):
        self.engine_id = engine_id
        self.model_id = f"{engine_id}-model"
        self.format = AudioFormat(rate, 1)
        self.langs = langs
        self.status = status
        self.received_rates: list[int] = []
        self.received_languages: list = []
        self.transcript = "hello there"

    def capabilities(self):
        return tuple(
            _measured(lang, self.engine_id) if self.status is CapabilityStatus.MEASURED
            else Capability(task=Task.STT, status=self.status, language=lang, engine=self.engine_id)
            for lang in self.langs
        )

    def warm(self):
        return None

    def transcribe(self, request):
        self.received_rates.append(request.audio.format.sample_rate)
        self.received_languages.append(request.language.code if request.language else None)
        return Recognition(text=self.transcript, language="en", confidence=0.91)

    def close(self):
        return None


class StreamEngine:
    """Streaming engine that demonstrates a retract and records invocations."""

    supports_streaming = True

    def __init__(self, engine_id="stream", partial_count=6):
        self.engine_id = engine_id
        self.model_id = f"{engine_id}-model"
        self.format = AudioFormat(16000, 1)
        self.invoked = 0
        self.partial_count = partial_count

    def capabilities(self):
        return (_measured("en", self.engine_id),)

    def warm(self):
        return None

    def transcribe(self, request):
        raise AssertionError("streaming engine should use stream()")

    def stream(self, chunks, language=None):
        self.invoked += 1
        for _ in chunks:
            pass
        for index in range(self.partial_count):
            yield TranscriptionEvent(TranscriptionKind.PARTIAL, f"partial{index}", "u")
        yield TranscriptionEvent(TranscriptionKind.FINAL, "the final transcript", "u", is_final=True)

    def close(self):
        return None


def _service(engines, status_by=None):
    registry = EngineRegistry()
    for engine in engines:
        registry.register(engine, engine.capabilities())
    return TranscriptionService(registry, LanguageRouter(), registry.ledger), registry


# --- selection / refusal -----------------------------------------------------
def test_no_engines_raises_engine_unavailable():
    registry = EngineRegistry()
    assert not registry.has_engines()
    service = TranscriptionService(registry, LanguageRouter(), registry.ledger)
    with pytest.raises(EngineUnavailable):
        service.transcribe(TranscriptionRequest(audio=sine(rate=16000), language=LanguageTag("en")))


def test_unsupported_language_raises_before_invoking_engine():
    engine = StubEngine()
    service, _ = _service([engine])
    with pytest.raises(LanguageUnsupported):
        service.transcribe(TranscriptionRequest(audio=sine(rate=16000), language=LanguageTag("bn")))
    assert engine.received_rates == []  # the engine was never invoked


def test_stream_unsupported_does_not_invoke_engine():
    engine = StreamEngine()
    service, _ = _service([engine])
    with pytest.raises(LanguageUnsupported):
        list(service.stream(iter([sine(rate=16000)]), LanguageTag("bn")))
    assert engine.invoked == 0


def test_declared_only_engine_is_not_selected_in_strict_mode():
    engine = StubEngine(langs=("en",), status=CapabilityStatus.DECLARED)
    registry = EngineRegistry()
    registry.register(engine, engine.capabilities())
    assert registry.best_for("en", strict=True) is None
    assert registry.best_for("en", strict=False) is not None
    service = TranscriptionService(registry, LanguageRouter(), registry.ledger, strict=True)
    with pytest.raises(LanguageUnsupported):
        service.transcribe(TranscriptionRequest(audio=sine(rate=16000), language=LanguageTag("en")))


# --- happy path / resample / timings ----------------------------------------
def test_transcribe_resamples_to_engine_rate():
    engine = StubEngine(rate=16000)
    service, _ = _service([engine])
    result = service.transcribe(TranscriptionRequest(audio=sine(rate=44100, seconds=1.0), language=LanguageTag("en")))
    assert engine.received_rates == [16000]  # 44.1k input became the engine rate
    assert result.text == "hello there"
    assert result.engine == engine.engine_id
    assert result.language.code == "en"
    assert result.confidence == 0.91


def test_transcribe_attaches_timings():
    engine = StubEngine()
    service, _ = _service([engine])
    result = service.transcribe(TranscriptionRequest(audio=sine(rate=16000, seconds=0.5), language=LanguageTag("en")))
    assert "total_ms" in result.timings
    assert "audio_duration_s" in result.timings
    assert result.timings["capability_status"] == "measured"


def test_transcribe_truncates_overlong_audio():
    engine = StubEngine()
    service, _ = _service([engine])
    long_audio = sine(rate=16000, seconds=5.0)
    result = service.transcribe(TranscriptionRequest(audio=long_audio, language=LanguageTag("en"), max_duration_s=1.0))
    assert result.truncated is True


def test_auto_detection_defers_language_to_engine():
    engine = StubEngine(langs=("en",))
    service, _ = _service([engine])
    result = service.transcribe(TranscriptionRequest(audio=sine(rate=16000), language=None, allow_auto_detection=True))
    assert engine.received_languages == [None]
    assert result.text == "hello there"


def test_empty_engine_output_raises_capability_unavailable():
    engine = StubEngine()
    engine.transcript = "   "
    service, _ = _service([engine])
    with pytest.raises(CapabilityUnavailable):
        service.transcribe(TranscriptionRequest(audio=sine(rate=16000), language=LanguageTag("en")))


def test_language_capability_reports_no_model_for_unknown():
    engine = StubEngine(langs=("en",))
    service, _ = _service([engine])
    assert service.language_capability(LanguageTag("bn")).status is CapabilityStatus.NO_MODEL
    assert service.language_capability(LanguageTag("en")).status is CapabilityStatus.MEASURED


# --- reducer -----------------------------------------------------------------
def test_reducer_replaces_partial_does_not_append():
    reducer = TranscriptReducer()
    assert reducer.feed(_partial("Hel")) == "Hel"
    assert reducer.feed(_partial("Hello")) == "Hello"  # replaced, not "HelHello"


def test_reducer_retract_clears_stale_partial():
    reducer = TranscriptReducer()
    reducer.feed(_partial("Hello"))
    assert reducer.feed(TranscriptionEvent(TranscriptionKind.RETRACT, "", "u")) == ""
    assert reducer.retracts == 1
    assert reducer.feed(_partial("Goodbye")) == "Goodbye"


def test_reducer_finalizes_and_keeps_committed():
    reducer = TranscriptReducer()
    reducer.feed(TranscriptionEvent(TranscriptionKind.FINAL, "first.", "u", is_final=True))
    reducer.feed(_partial("sec"))
    assert reducer.visible() == "first. sec"


def _partial(text):
    return TranscriptionEvent(TranscriptionKind.PARTIAL, text, "u")


# --- streaming through the service ------------------------------------------
def test_stream_forwards_partials_and_emits_reduced_final():
    engine = StreamEngine(partial_count=3)
    service, _ = _service([engine])
    events = list(service.stream(iter([sine(rate=16000), sine(rate=16000)]), LanguageTag("en")))
    kinds = [event.kind for event in events]
    assert kinds.count(TranscriptionKind.PARTIAL) == 3
    assert events[-1].kind is TranscriptionKind.FINAL
    assert events[-1].text == "the final transcript"
    assert events[-1].sequence > events[0].sequence  # monotonically numbered


def test_stream_uses_retract_engine_and_marks_retracts():
    class RetractStream(StreamEngine):
        def stream(self, chunks, language=None):
            self.invoked += 1
            for _ in chunks:
                pass
            yield _partial("Hel")
            yield TranscriptionEvent(TranscriptionKind.RETRACT, "", "u")
            yield _partial("Goodbye")
            yield TranscriptionEvent(TranscriptionKind.FINAL, "Goodbye", "u", is_final=True)

    engine = RetractStream()
    service, _ = _service([engine])
    events = list(service.stream(iter([sine(rate=16000)]), LanguageTag("en")))
    assert events[-1].kind is TranscriptionKind.FINAL
    assert events[-1].text == "Goodbye"
    assert events[-1].metadata["retracts"] == 1


def test_cancellation_mid_stream():
    engine = StreamEngine(partial_count=20)
    service, _ = _service([engine])
    token = CancellationToken()
    emitted = []
    with pytest.raises(CancellationRequested):
        for index, event in enumerate(service.stream(iter([sine(rate=16000)]), LanguageTag("en"), token=token)):
            emitted.append(event)
            if index == 2:
                token.cancel("user barge-in")
    assert len(emitted) == 3  # we never emitted the event after cancellation


def test_stream_falls_back_to_transcribe_for_non_streaming_engine():
    engine = StubEngine()
    service, _ = _service([engine])
    events = list(service.stream(iter([sine(rate=16000), sine(rate=16000)]), LanguageTag("en")))
    assert len(events) == 1
    assert events[0].kind is TranscriptionKind.FINAL
    assert events[0].text == "hello there"


# --- deterministic echo fixture ---------------------------------------------
def test_echo_engine_cannot_satisfy_real_routing():
    echo = DeterministicEchoEngine()
    registry = EngineRegistry()
    registry.register(echo, echo.capabilities())
    for code in ("en", "hi", "bn"):
        assert registry.best_for(code, strict=True) is None
        assert registry.best_for(code, strict=False) is None  # UNSUPPORTED is never usable


def test_echo_engine_refused_by_service_but_works_directly():
    echo = DeterministicEchoEngine()
    registry = EngineRegistry()
    registry.register(echo, echo.capabilities())
    service = TranscriptionService(registry, LanguageRouter(), registry.ledger)
    with pytest.raises(LanguageUnsupported):
        service.transcribe(TranscriptionRequest(audio=sine(rate=16000), language=LanguageTag("en")))
    assert echo.invocations == []
    # used directly (bypassing routing) it returns a deterministic transcript
    buffer = sine(rate=16000, seconds=0.25)
    assert echo.transcribe(TranscriptionRequest(audio=buffer)).text.startswith("echo frames=")


# --- faster-whisper adapter --------------------------------------------------
def test_whisper_refuses_without_approval():
    adapter = FasterWhisperAdapter(model_name="base", approved=False)
    with pytest.raises(ModelNotApproved) as exc:
        adapter.transcribe(TranscriptionRequest(audio=sine(rate=16000), language=LanguageTag("en")))
    assert exc.value.details.get("review") == "unreviewed"
    with pytest.raises(ModelNotApproved):
        adapter.warm()


def test_whisper_names_missing_pieces_when_authorized():
    state = _whisper_backend_state()
    if state == ABSENT:
        pytest.skip(
            "NOT RUN: faster-whisper is not installed on this machine, so the expected-path "
            "branch below is unreachable - the package-absent path is tested separately"
        )
    assert state == "importable", (
        "faster-whisper is present but will not import. This test is about a missing model "
        "directory, and a broken installation must not be reported as one."
    )
    adapter = FasterWhisperAdapter(model_name="definitely-not-installed-xyz", approved=True)
    with pytest.raises(ModelNotInstalled) as exc:
        adapter.transcribe(TranscriptionRequest(audio=sine(rate=16000), language=LanguageTag("en")))
    assert "expected_path" in exc.value.details
    assert exc.value.details["model_id"] == "faster-whisper-definitely-not-installed-xyz"


class TestOptionalBackendImportStates:
    """An absent optional package and a broken one are two different failures.

    Both arrive as an import error and both were reported as "not installed", which
    sends a host off to reinstall a package that is already there while the real fault
    - a wheel that loads and raises - stays invisible. These go through the import
    machinery rather than the installed package, so they run and fail identically
    whether or not faster-whisper is present here.
    """

    @pytest.mark.parametrize(
        "error, expected",
        [
            (ModuleNotFoundError("No module named 'faster_whisper'", name="faster_whisper"), ABSENT),
            (ModuleNotFoundError("No module named 'ctranslate2'", name="ctranslate2"), BROKEN),
            (ModuleNotFoundError("No module named 'faster_whisper.feature_extractor'", name="faster_whisper.feature_extractor"), BROKEN),
            (ModuleNotFoundError("a finder raised without naming anything"), BROKEN),
            (ImportError("numpy.dtype size changed, may indicate binary incompatibility"), BROKEN),
            (OSError("DLL load failed while importing _ext"), BROKEN),
        ],
    )
    def test_the_classifier_names_the_two_states_apart(self, error, expected):
        assert import_state("faster_whisper", error) == expected

    def test_absent_package_is_reported_as_an_installation_gap(self, monkeypatch):
        _raise_on_import(monkeypatch, ModuleNotFoundError("No module named 'faster_whisper'", name="faster_whisper"))
        adapter = FasterWhisperAdapter(model_name="base", approved=True)
        with pytest.raises(ModelNotInstalled) as exc:
            adapter._import_backend()
        assert exc.value.details["import_state"] == ABSENT
        assert exc.value.args[0] == "faster-whisper backend package is not installed"

    def test_a_package_that_will_not_load_is_not_reported_as_absent(self, monkeypatch):
        _raise_on_import(monkeypatch, OSError("DLL load failed while importing _ext"))
        adapter = FasterWhisperAdapter(model_name="base", approved=True)
        with pytest.raises(ModelNotInstalled) as exc:
            adapter._import_backend()
        assert exc.value.details["import_state"] == BROKEN
        assert exc.value.args[0] == "faster-whisper backend package is present but failed to import"

    def test_poisoning_the_module_table_presents_as_absent(self, monkeypatch):
        """What the suite's existing `sys.modules` simulation actually proves.

        `test_measurement.py` blocks the backend with `setitem(sys.modules, ..., None)`;
        measured on CPython 3.12.10 and 3.14.7 that raises `ModuleNotFoundError` naming the
        package, so the adapter reports an installation gap rather than a broken load. Pinned
        here so nobody reads that control as evidence about a corrupt wheel - and so a future
        interpreter that changes it fails a named test instead of silently changing the
        meaning of a guard-ordering assertion.
        """
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        adapter = FasterWhisperAdapter(model_name="base", approved=True)
        with pytest.raises(ModelNotInstalled) as exc:
            adapter._import_backend()
        assert exc.value.details["import_state"] == ABSENT

    def test_the_two_states_do_not_collapse_into_one_message(self, monkeypatch):
        """The point of the split: a host branching on the payload gets different answers."""
        adapter = FasterWhisperAdapter(model_name="base", approved=True)
        messages = {}
        for state, error in (
            (ABSENT, ModuleNotFoundError("No module named 'faster_whisper'", name="faster_whisper")),
            (BROKEN, ImportError("binary built against a different numpy")),
        ):
            _raise_on_import(monkeypatch, error)
            with pytest.raises(ModelNotInstalled) as exc:
                adapter._import_backend()
            messages[state] = (exc.value.args[0], exc.value.details["import_state"])
        assert messages[ABSENT] != messages[BROKEN]


def _raise_on_import(monkeypatch, error: BaseException) -> None:
    """Make `import faster_whisper` inside the adapter raise `error`, nothing else changes."""
    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "faster_whisper":
            raise error
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)


def _whisper_backend_state() -> str:
    """Ask the shipped classifier, so a skip can never disagree with the adapter.

    `pytest.importorskip` would skip here too, but it skips on any ModuleNotFoundError -
    including one naming a dependency of an installed-but-broken package - which is the
    confusion these tests exist to prevent. "importable" is the third answer.
    """
    try:
        FasterWhisperAdapter(model_name="base", approved=True)._import_backend()
    except ModelNotInstalled as exc:
        return str(exc.details["import_state"])
    return "importable"


def test_whisper_is_unreviewed_and_declared_only():
    adapter = FasterWhisperAdapter(model_name="base")
    manifest = adapter.manifest()
    assert manifest.review.value == "unreviewed"
    assert "to be verified" in manifest.provenance.licence_spdx
    assert manifest.provenance.source.startswith("Systran/faster-whisper")
    assert all(cap.status is CapabilityStatus.DECLARED for cap in adapter.capabilities())


def test_whisper_caps_are_not_selected_in_strict_routing():
    adapter = FasterWhisperAdapter(model_name="base")
    registry = EngineRegistry()
    registry.register(adapter, adapter.capabilities())
    assert registry.best_for("en", strict=True) is None
    assert registry.best_for("en", strict=False) is not None


@requires_approved_asset
@pytest.mark.skipif(
    not os.environ.get("ULTRON_RUN_APPROVED_ASSET"),
    reason="NOT RUN for acceptance: real CT2 inference on an unreviewed third-party asset",
)
def test_real_whisper_inference_requires_approved_asset():
    # Only runs when explicitly opted in AND a genuinely approved asset exists.
    adapter = FasterWhisperAdapter(model_name="base", approved=True)
    result = adapter.transcribe(TranscriptionRequest(audio=sine(rate=16000, seconds=1.0)))
    assert isinstance(result, TranscriptionResult)


# --- text context ------------------------------------------------------------
def test_build_prompt_dedupes_and_preserves_order():
    prompt = build_prompt(hint_terms=["alpha", "beta", "Alpha", "beta", ""], context="")
    assert prompt == "Terms: alpha, beta"


def test_build_prompt_strips_control_characters():
    assert sanitize("fo\x01o\x7f bar") == "foo bar"
    assert "fo\x01o" not in build_prompt(["fo\x01o"])
    assert not dedupe(["a\x00", ""])[1:2]


def test_build_prompt_is_hard_capped():
    terms = [f"term{i}" for i in range(500)]
    prompt = build_prompt(terms)
    assert len(prompt) <= MAX_PROMPT_CHARS
    small = build_prompt(terms, max_chars=40)
    assert len(small) <= 40


def test_build_prompt_empty_when_nothing():
    assert build_prompt([], "") == ""


# --- registry describe -------------------------------------------------------
def test_describe_reports_installed_and_missing():
    empty = EngineRegistry().describe()
    assert "no engines installed" in empty
    engine = StubEngine()
    registry = EngineRegistry()
    registry.register(engine, engine.capabilities())
    text = registry.describe()
    assert engine.engine_id in text
    assert "tts" in text.lower()
