"""Evaluation sample manifests, and the wall between synthetic audio and quality.

WHY THIS FILE EXISTS AT ALL. The single most tempting form of self-deception in a
voice project is to score a tone generator or a synthetic envelope as though it
were speech, print a WER, and move on. Nothing downstream can tell that the
number is meaningless, because the arithmetic itself is correct. So the rule is
enforced here, at the object that carries the audio: a sample records its licence
and consent basis, `usable` is derived from those fields rather than asserted, and
samples tagged `synthetic=True` are refused by `eval.transcription.evaluate_pairs`
before a single metric is computed.

The built-in fixtures are provided for plumbing tests only: they exercise file
resolution, grouping, latency measurement and report rendering without pretending
to be a corpus.
"""

from __future__ import annotations

import array
import json
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..audio.wav import read_wav
from ..core.audio import AudioBuffer, AudioFormat
from ..core.errors import CapabilityUnavailable, ConfigurationError
from ..core.language import LanguageTag

SYNTHETIC_REFUSAL = "synthetic fixtures cannot be scored for quality"


@dataclass(frozen=True, slots=True)
class EvalSample:
    """One utterance: where it is, what it says, and on what terms we may use it."""

    path: str
    language: str
    transcript: str
    licence: str = ""
    consent_basis: str = ""
    synthetic: bool = False
    sample_id: str = ""
    speaker: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.language:
            raise ConfigurationError("evaluation sample needs a language", path=self.path)
        if not self.sample_id:
            object.__setattr__(self, "sample_id", self.path or self.language)

    @property
    def usable(self) -> bool:
        """Licence AND consent present, and not a synthetic stand-in.

        Derived, never stored: an out-of-date `usable` flag on disk is worse than
        no flag, because it reads like a decision that was made deliberately.
        """
        return bool(self.licence and self.consent_basis) and not self.synthetic

    @property
    def unusable_reasons(self) -> tuple[str, ...]:
        reasons = []
        if not self.licence:
            reasons.append("licence missing")
        if not self.consent_basis:
            reasons.append("consent basis missing")
        if self.synthetic:
            reasons.append(SYNTHETIC_REFUSAL)
        return tuple(reasons)

    @property
    def tag(self) -> LanguageTag:
        return LanguageTag.parse(self.language)

    def as_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "path": self.path,
            "language": self.language,
            "transcript": self.transcript,
            "licence": self.licence,
            "consent_basis": self.consent_basis,
            "synthetic": self.synthetic,
            "usable": self.usable,
            "unusable_reasons": list(self.unusable_reasons),
        }


@dataclass(slots=True)
class SampleManifest:
    """A loaded corpus descriptor. `path` is empty for in-memory built-in fixtures."""

    name: str
    samples: tuple[EvalSample, ...] = ()
    source: str = ""
    notes: str = ""
    path: str = ""

    def __len__(self) -> int:
        return len(self.samples)

    @classmethod
    def load(cls, path: str) -> "SampleManifest":
        """Read a JSON manifest: {"name", "source", "samples": [ {...} ]}.

        Unknown keys are rejected rather than ignored, because a typo in `licence`
        would otherwise silently downgrade a real corpus to `usable=False` - or, if
        the typo were in `synthetic`, promote a tone to a quality corpus.
        """
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except OSError as exc:
            raise ConfigurationError("evaluation manifest unreadable", path=path, error=str(exc)) from exc
        except json.JSONDecodeError as exc:
            raise ConfigurationError("evaluation manifest is not JSON", path=path, error=str(exc)) from exc
        known = set(_SAMPLE_FIELDS)
        samples = []
        for raw in payload.get("samples", ()):
            extra = set(raw) - known
            if extra:
                raise ConfigurationError(
                    "unknown sample manifest key", keys=sorted(extra), path=path
                )
            samples.append(EvalSample(**raw))
        return cls(
            name=str(payload.get("name", "")) or _stem(path),
            samples=tuple(samples),
            source=str(payload.get("source", "")),
            notes=str(payload.get("notes", "")),
            path=path,
        )

    def languages(self) -> tuple[str, ...]:
        seen: list[str] = []
        for sample in self.samples:
            if sample.language not in seen:
                seen.append(sample.language)
        return tuple(seen)

    def by_language(self, language: str) -> tuple[EvalSample, ...]:
        root = LanguageTag.parse(language).code
        return tuple(s for s in self.samples if s.tag.code == root)

    def quality_usable(self) -> tuple[EvalSample, ...]:
        return tuple(s for s in self.samples if s.usable)

    def blocked(self) -> tuple[EvalSample, ...]:
        return tuple(s for s in self.samples if not s.usable)

    def gaps(self) -> dict:
        """Per-language counts of what we can and cannot score. Report material."""
        result = {}
        for language in self.languages():
            group = self.by_language(language)
            result[language] = {
                "samples": len(group),
                "usable": sum(1 for s in group if s.usable),
                "synthetic": sum(1 for s in group if s.synthetic),
                "licence_missing": sum(1 for s in group if not s.licence),
                "consent_missing": sum(1 for s in group if not s.consent_basis),
            }
        return result

    def audio(self, sample: EvalSample) -> AudioBuffer:
        if sample.synthetic:
            return _synthetic_buffer(sample)
        return read_wav(sample.path)

    def require_quality_corpus(self) -> None:
        """Raise unless at least one sample is licence+consent cleared and real."""
        if self.quality_usable():
            return
        raise CapabilityUnavailable(
            "no usable evaluation samples",
            manifest=self.name,
            samples=len(self.samples),
            blocked_by=sorted({r for s in self.samples for r in s.unusable_reasons}) or ["no samples"],
        )


_SAMPLE_FIELDS = tuple(EvalSample.__dataclass_fields__)

_TONE = "synthetic+tone"
_ENVELOPE = "synthetic+envelope"


def _synthetic_buffer(sample: EvalSample) -> AudioBuffer:
    """Regenerate a built-in fixture deterministically from its `sample_id`."""
    seed = _stable_seed(sample.sample_id)
    if sample.notes.startswith(_ENVELOPE):
        return _tone_envelope(seed=seed)
    return _pure_tone(freq=180.0 + (seed % 7) * 40.0, seconds=0.3, seed=seed)


def _stable_seed(text: str) -> int:
    """FNV-1a over the id: `hash()` is salted per process, which would make fixtures
    differ between runs and therefore make measured comparisons non-reproducible."""
    value = 0x811C9DC5
    for byte in text.encode("utf-8"):
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


def _pure_tone(freq: float, seconds: float, seed: int, rate: int = 16000) -> AudioBuffer:
    count = int(rate * seconds)
    phase = (seed % 64) / 64.0 * math.tau
    samples = array.array(
        "h",
        (int(0.3 * 32767 * math.sin(math.tau * freq * index / rate + phase)) for index in range(count)),
    )
    return AudioBuffer(samples, AudioFormat(rate, 1))


def _tone_envelope(seed: int, rate: int = 16000, seconds: float = 0.6) -> AudioBuffer:
    """Amplitude-burst envelope: shaped like an utterance in time, a tone in content."""
    count = int(rate * seconds)
    phase = (seed % 32) / 32.0 * math.tau
    samples = array.array("h")
    for index in range(count):
        moment = index / rate
        gate = 1.0 if (int(moment * 4) % 2 == 0) else 0.0
        value = math.sin(math.tau * 200.0 * moment + phase) + 0.4 * math.sin(math.tau * 400.0 * moment)
        samples.append(int(max(-1.0, min(1.0, value * gate * 0.25)) * 32767))
    return AudioBuffer(samples, AudioFormat(rate, 1))


def _stem(path: str) -> str:
    base = path.replace("\\", "/").rsplit("/", 1)[-1]
    return base[:-5] if base.endswith(".json") else base


def builtin_fixtures(languages: Sequence[str] = ("en", "hi", "bn")) -> SampleManifest:
    """Synthetic tones/envelopes for plumbing tests. Every entry is tagged.

    These exist so that file resolution, per-language grouping, latency harnesses
    and report rendering can be exercised offline and deterministically. They
    contain no speech, so `eval.transcription.evaluate_pairs` refuses them and
    `require_quality_corpus` raises on this manifest by construction.
    """
    samples = []
    for index, language in enumerate(languages):
        samples.append(
            EvalSample(
                path=f"{_TONE}://{language}/tone{index}",
                language=language,
                transcript="",
                licence="",
                consent_basis="",
                synthetic=True,
                sample_id=f"tone-{language}-{index}",
                notes=_TONE,
            )
        )
        samples.append(
            EvalSample(
                path=f"{_ENVELOPE}://{language}/burst{index}",
                language=language,
                transcript="placeholder transcript is not speech",
                licence="synthetic-no-licence-needed",
                consent_basis="synthetic-no-consent-needed",
                synthetic=True,
                sample_id=f"envelope-{language}-{index}",
                notes=_ENVELOPE,
            )
        )
    return SampleManifest(name="builtin-fixtures", samples=tuple(samples), source="nativevoice.eval.samples")


def synthetic_guard(samples: Iterable[EvalSample], activity: str) -> None:
    """Raise if any sample in `samples` is synthetic. Shared by scorers."""
    offenders = sorted({s.sample_id for s in samples if s.synthetic})
    if offenders:
        raise CapabilityUnavailable(
            SYNTHETIC_REFUSAL,
            activity=activity,
            synthetic_samples=offenders[:10],
            offending_count=len(offenders),
            required="a licensed, consented, real-speech corpus registered in a SampleManifest",
        )

