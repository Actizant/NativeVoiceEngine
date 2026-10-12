"""Language identity and per-task capability.

The reference study made the central trap obvious: OpenWhispr's 61-entry registry
advertises per-engine support flags but has no Bengali entry at all, and Pocket
TTS ships seven languages while its README implies a general tool. A registry
that records only "supported: true|false" would reproduce that failure.

So capability is a status enum plus provenance, and the statuses distinguish
"measured on this machine" from "the vendor says so" from "no model installed".
Routing policy can then be strict about which of those it will act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .errors import ConfigurationError

BCP47_PRIMARY_SUBTAGS = ("zh-Hans", "zh-Hant")


class Task(str, Enum):
    STT = "stt"
    TTS = "tts"
    LID = "lid"
    VOICE_CLONE = "voice_clone"


class CapabilityStatus(str, Enum):
    MEASURED = "measured"
    """Evaluated here with a reported metric. The only status acceptance may cite."""

    DECLARED = "declared"
    """Asserted by an upstream model card or vendor, not verified by us."""

    MEASURED_FAILED = "measured_failed"
    """Attempted, measured, and the measurement did not clear the bar.

    Kept distinct from UNSUPPORTED, which means "refused, never tried". Collapsing
    them destroys the number that tells an engineer whether the gap is a missing
    asset or a bad one, and the distinction only survives if the status is routable-
    false in both cases - which it is.
    """

    PARTIAL = "partial"
    """Works with known gaps; `limitations` must say what they are."""

    UNSUPPORTED = "unsupported"
    """The installed engine cannot do this task for this language."""

    NO_MODEL = "no_model"
    """Infrastructure exists, no approved model asset is installed."""


@dataclass(frozen=True, slots=True)
class LanguageTag:
    """A pragmatic language tag: ISO-639 code, optional script and region."""

    code: str
    script: str = ""
    region: str = ""

    def __post_init__(self) -> None:
        if not self.code or not self.code.isascii():
            raise ConfigurationError("language code must be non-empty ASCII", code=self.code)
        lowered = self.code.lower()
        if lowered != self.code:
            object.__setattr__(self, "code", lowered)

    @classmethod
    def parse(cls, tag: str) -> "LanguageTag":
        """Split a BCP-47-ish tag into code / script / region.

        Subtags are classified by shape, not position: a 4-letter title-case piece is
        a script, a 2-letter or 3-digit piece is a region. Position-based parsing lost
        the region on tags like `de-DE-1901`, where the meaningful region sits in the
        middle and an orphan variant follows it.
        """
        for primary in BCP47_PRIMARY_SUBTAGS:
            if tag.lower().startswith(primary.lower()):
                remainder = tag[len(primary) :].lstrip("-")
                return cls(primary.split("-")[0], script=primary.split("-")[1], region=remainder)
        parts = [piece for piece in tag.replace("_", "-").split("-") if piece]
        code = parts[0].lower()
        script = ""
        region = ""
        for piece in parts[1:]:
            if len(piece) == 4 and piece.isalpha():
                script = piece.title()
            elif len(piece) == 2 and piece.isalpha():
                region = region or piece.upper()
            elif len(piece) == 3 and piece.isdigit():
                continue  # a variant subtag (de-DE-1901); not a region
        return cls(code, script, region)

    @property
    def bcp47(self) -> str:
        pieces = [self.code]
        if self.script:
            pieces.append(self.script)
        if self.region:
            pieces.append(self.region)
        return "-".join(pieces)

    def root(self) -> "LanguageTag":
        return LanguageTag(self.code)


@dataclass(frozen=True, slots=True)
class Capability:
    task: Task
    status: CapabilityStatus
    language: str
    engine: str = ""
    metric_name: str = ""
    metric_value: float | None = None
    measured_on: str = ""
    model_id: str = ""
    limitations: tuple[str, ...] = ()
    evidence: str = ""

    @property
    def usable_for_automatic_routing(self) -> bool:
        return self.status in (CapabilityStatus.MEASURED, CapabilityStatus.PARTIAL)

    def counts_as_evidence(self) -> bool:
        """Whether a real measurement backs this claim - including a failed one."""
        return self.status in (CapabilityStatus.MEASURED, CapabilityStatus.MEASURED_FAILED) and bool(
            self.metric_name
        ) and self.metric_value is not None


@dataclass(frozen=True, slots=True)
class LanguageProfile:
    """Static description of a language's writing/phonetic properties.

    These properties drive text normalisation and G2P choices, so they belong
    beside the capability registry rather than inside an engine.
    """

    code: str
    english_name: str
    native_name: str = ""
    script: str = ""
    rtl: bool = False
    iso_level: str = "1"
    space_delimited: bool = True
    notes: str = ""


@dataclass(slots=True)
class LanguageDecision:
    """Result of automatic spoken-language identification."""

    language: LanguageTag
    confidence: float
    alternatives: tuple[tuple[str, float], ...] = ()
    method: str = ""
    is_calibrated: bool = False
    duration_s: float = 0.0
    metadata: dict = field(default_factory=dict)

    def below(self, threshold: float) -> bool:
        return self.confidence < threshold
