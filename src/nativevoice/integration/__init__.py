"""Integration seam: the host contract and the façade a host consumes."""

from .adapter import VoiceInterface
from .contract import HostReply, VoiceEngine, VoiceHost

__all__ = ["HostReply", "VoiceEngine", "VoiceHost", "VoiceInterface"]
