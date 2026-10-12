"""Recognition subsystem: contracts, orchestration, engine registry, adapters.

Heavy third-party runtimes appear only inside optional adapters and only after an
explicit approval gate; importing this package installs no engine and reads no
network or model cache.
"""

from .adapters.faster_whisper import FasterWhisperAdapter
from .adapters.faster_whisper_engine import FasterWhisperEngine
from .adapters.self_test import DeterministicEchoEngine
from .api import (
    TranscriptionEngine,
    TranscriptionRequest,
    TranscriptionResult,
    Transcriber,
)
from .engine_registry import EngineRegistry
from .orchestrator import (
    TranscriptReducer,
    TranscriptionService,
    result_from_recognition,
)
from .text_context import MAX_PROMPT_CHARS, MAX_TERMS, build_prompt

__all__ = [
    "DeterministicEchoEngine",
    "EngineRegistry",
    "FasterWhisperAdapter",
    "MAX_PROMPT_CHARS",
    "MAX_TERMS",
    "Transcriber",
    "TranscriptionEngine",
    "TranscriptionRequest",
    "TranscriptionResult",
    "TranscriptionService",
    "TranscriptReducer",
    "build_prompt",
    "result_from_recognition",
]
