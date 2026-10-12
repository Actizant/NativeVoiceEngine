"""Synthesis adapters.

``self_test`` verifies the pipeline's transport with scripted buffers and
``pocket_reference`` documents the target architecture and refuses - neither
speaks. ``sherpa_kokoro`` is the real engine: it renders audio for an approved,
integrity-verified Kokoro asset, and it declares a language ``MEASURED`` only once
an evaluation run has produced a number for it.
"""

from .pocket_reference import ENGINE_ID as POCKET_REFERENCE_ENGINE_ID
from .pocket_reference import PocketReferenceAdapter, development_plan
from .self_test import PLUMBING_LANGUAGE, ScriptedSynthesisEngine
from .sherpa_kokoro import ENGINE_ID as SHERPA_KOKORO_ENGINE_ID
from .sherpa_kokoro import SherpaKokoroEngine

__all__ = [
    "PLUMBING_LANGUAGE",
    "POCKET_REFERENCE_ENGINE_ID",
    "SHERPA_KOKORO_ENGINE_ID",
    "PocketReferenceAdapter",
    "ScriptedSynthesisEngine",
    "SherpaKokoroEngine",
    "development_plan",
]
