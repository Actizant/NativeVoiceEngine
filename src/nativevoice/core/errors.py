"""Error taxonomy.

Every failure the voice stack can report is a member of this tree so that
callers (and the host integration adapter) can branch on category rather than
on message text. Errors carry structured fields, not prose-only payloads.
"""

from __future__ import annotations

from typing import Any


class VoiceError(Exception):
    """Base class for every error raised by this package."""

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details = details

    def __str__(self) -> str:
        base = super().__str__()
        if not self.details:
            return base
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(self.details.items()))
        return f"{base} ({rendered})"


class ConfigurationError(VoiceError, ValueError):
    """A config file, profile selection, model parameter or argument is invalid.

    Also a `ValueError`, because argument validation was already raising that and a
    host that catches it must not break when the taxonomy gets richer.
    """


class CapabilityUnavailable(VoiceError):
    """The feature exists in the interface but is not implemented or installed.

    This is the error that keeps an unimplemented capability from being reported
    as a working one. Anything that cannot genuinely do the task raises this
    instead of returning empty or synthetic success.
    """


class EngineUnavailable(CapabilityUnavailable):
    """No engine instance can serve this request right now."""


class ModelNotInstalled(CapabilityUnavailable):
    """A recognised manifest has no asset present on disk."""


class ModelNotApproved(CapabilityUnavailable):
    """An asset is present but its provenance/licence review has not passed.

    Enforced at load time by `nativevoice.models.registry`; there is no
    ambient-approval or default-allow path.
    """


class IntegrityError(VoiceError):
    """A model asset does not match its recorded digest or size."""


class ResourceLimitExceeded(VoiceError):
    """The requested model/configuration does not fit the active budget.

    `explanation` must state the measured numbers on both sides so the user can
    act on it rather than guess.
    """

    def __init__(self, message: str, explanation: str = "", **details: Any) -> None:
        super().__init__(message, **details)
        self.explanation = explanation

    def __str__(self) -> str:
        # VoiceError.__str__ renders the structured details, which buries the one
        # sentence a user can act on. Lead with it, and keep details uncluttered.
        base = super().__str__()
        if not self.explanation:
            return base
        return f"{self.args[0]}: {self.explanation}" + (
            f" ({', '.join(f'{k}={v!r}' for k, v in sorted(self.details.items()))})" if self.details else ""
        )


class DeviceError(VoiceError):
    """An audio input/output device failed."""


class DeviceDisconnected(DeviceError):
    """The device vanished, or its stream callback stopped arriving."""


class LanguageUnsupported(CapabilityUnavailable):
    """No engine can serve this language for this task.

    Distinct from a generic failure: the caller is expected to fall back to
    text, or report the limitation, rather than synthesise unrelated speech.
    """


class CancellationRequested(VoiceError):
    """Raised when a cancelled token is observed at a yield boundary."""


class EngineTimeout(VoiceError):
    """An engine exceeded its configured deadline."""


class EngineCrashed(VoiceError):
    """An engine process or runtime died and the host had to contain it."""


class StorageError(VoiceError):
    """Local persistence (profile, config, cache) could not be completed."""


class ProfileInvalid(VoiceError):
    """A voice profile failed consent, integrity or quality validation."""
