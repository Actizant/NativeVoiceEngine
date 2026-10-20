"""Synthesis adapters.

``self_test`` verifies the pipeline's transport with scripted buffers and never
speaks a real language. ``sherpa_kokoro`` is the real engine: it renders audio for
an approved, integrity-verified Kokoro asset, and it declares a language
``MEASURED`` only once an evaluation run has produced a number for it.
"""

from .self_test import PLUMBING_LANGUAGE, ScriptedSynthesisEngine
from .sherpa_kokoro import ENGINE_ID as SHERPA_KOKORO_ENGINE_ID
from .sherpa_kokoro import SherpaKokoroEngine

__all__ = [
    "PLUMBING_LANGUAGE",
    "SHERPA_KOKORO_ENGINE_ID",
    "ScriptedSynthesisEngine",
    "SherpaKokoroEngine",
]
