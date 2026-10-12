"""Recognition engine adapters.

Only the deterministic plumbing fixture is imported eagerly; the optional
CT2 Whisper adapter is exposed here but performs no heavy import or IO at module
load time.
"""

from .faster_whisper import FasterWhisperAdapter
from .faster_whisper_engine import FasterWhisperEngine
from .self_test import DeterministicEchoEngine

__all__ = ["DeterministicEchoEngine", "FasterWhisperAdapter", "FasterWhisperEngine"]
