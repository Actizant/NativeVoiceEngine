"""Text-to-speech front-end, routing and engine contracts.

Nothing in this package can produce speech today, and that is stated rather than
hidden: the pipeline, chunker and normaliser are complete and tested, the
capability registry is honest about the gap, and the only adapter that runs is the
plumbing self-test. A Bengali or Hindi request fails with the name of the missing
asset, which is the behaviour the project needs until a model exists.
"""

from __future__ import annotations

from .api import (
    Prosody,
    SpeechSynthesizer,
    SynthesisEngine,
    SynthesisOutcome,
    SynthesisRequest,
    VoiceReference,
)
from .chunking import TextChunk, TextChunker
from .engine_registry import EngineDescription, EngineRegistry
from .normalise import (
    DEFAULT_RULES,
    CurrencyWord,
    LanguageRules,
    NormalisationReport,
    NormaliseChange,
    NumberGrammar,
    SPOKEN_NUMBER_CEILING,
    TextNormaliser,
)
from .phonemiser import (
    GraphemeToPhonemeRegistry,
    PronunciationProvider,
    TransliterationProvider,
)
from .pipeline import DiagnosticsLedger, SpeechPipeline

__all__ = [
    "DEFAULT_RULES",
    "SPOKEN_NUMBER_CEILING",
    "CurrencyWord",
    "DiagnosticsLedger",
    "EngineDescription",
    "EngineRegistry",
    "GraphemeToPhonemeRegistry",
    "LanguageRules",
    "NormaliseChange",
    "NormalisationReport",
    "NumberGrammar",
    "Prosody",
    "PronunciationProvider",
    "SpeechPipeline",
    "SpeechSynthesizer",
    "SynthesisEngine",
    "SynthesisOutcome",
    "SynthesisRequest",
    "TextChunk",
    "TextChunker",
    "TextNormaliser",
    "TransliterationProvider",
    "VoiceReference",
]
