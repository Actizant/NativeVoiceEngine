"""Pronunciation: the interface, and an explicit record of the missing piece.

This module deliberately provides no pronunciation. The reference synthesis
architecture is grapheme-only (a 4000-entry SentencePiece over raw characters,
no phonemizer in the path), so a G2P stage is not required to use that model - but
it *is* required for any language whose written form does not map predictably to
sound, and for us that question is unanswerable today because no approved lexicon,
no transcription model and no Bengali or Hindi acoustic model exists at all.

A naive letter-reader would be the worst possible contribution here: it looks
like a working front-end, produces plausible-looking IPA, and silently
mispronounces every word the model then sings. So every path raises
`CapabilityUnavailable` with the name of the asset that is missing, and the
capability statuses are `NO_MODEL` so routing can see the gap rather than infer
it from an empty result.
"""

from __future__ import annotations

from typing import Protocol

from ..core.errors import CapabilityUnavailable
from ..core.language import Capability, CapabilityStatus, LanguageTag, Task

G2P_MISSING_ASSET = (
    "no approved grapheme-to-phoneme model or pronunciation lexicon exists for this project"
)
TRANSLITERATION_MISSING_ASSET = (
    "no approved transliteration asset exists; romanisation rules would have to be invented, "
    "and an invented transliteration is indistinguishable from a mispronunciation"
)

_NO_MODEL_LIMITATIONS = {
    "en": (
        "the reference architecture needs no G2P (grapheme-only SentencePiece), so an English "
        "G2P here would be unverified extra machinery, not a requirement"
    ),
    "hi": (
        "Hindi is community-only in the reference model and no approved Hindi acoustic model, "
        "lexicon or phoneme set exists here"
    ),
    "bn": (
        "Bengali is absent from the reference model's seven languages; no Bengali model asset, "
        "lexicon or phoneme set exists, so pronunciation cannot be claimed"
    ),
}


class PronunciationProvider(Protocol):
    """Something that turns text into a phoneme sequence for one language."""

    provider_id: str

    def capability(self, language: LanguageTag) -> Capability: ...

    def phonemes(self, text: str, language: LanguageTag) -> tuple[str, ...]: ...


def _code(language: LanguageTag | str) -> str:
    """Root language code, accepting either a tag or a bare code."""
    tag = language if isinstance(language, LanguageTag) else LanguageTag(language)
    return tag.root().code


class TransliterationProvider(Protocol):
    """Script conversion (e.g. Bengali -> Latin) as a separate, declared ability.

    Kept distinct from pronunciation on purpose: a romanisation is not a
    pronunciation, and conflating them lets an ASCII guess reach the model as if
    it were linguistics.
    """

    provider_id: str

    def transliterate(self, text: str, source: LanguageTag, target: LanguageTag) -> str: ...

    def capability(self, source: LanguageTag, target: LanguageTag) -> Capability: ...


class NoTransliteration:
    """The only transliteration we currently have: a refusal that names the gap."""

    provider_id = "none"

    def transliterate(self, text: str, source: LanguageTag, target: LanguageTag) -> str:
        raise CapabilityUnavailable(
            "transliteration is not implemented",
            source=source.bcp47,
            target=target.bcp47,
            reason=TRANSLITERATION_MISSING_ASSET,
        )

    def capability(self, source: LanguageTag, target: LanguageTag) -> Capability:
        return Capability(
            task=Task.TTS,
            status=CapabilityStatus.NO_MODEL,
            language=source.bcp47,
            engine=self.provider_id,
            limitations=(f"transliteration to {target.bcp47}: {TRANSLITERATION_MISSING_ASSET}",),
            evidence="no approved transliteration asset has been reviewed",
        )


class GraphemeToPhonemeRegistry:
    """Per-language declaration of what pronunciation machinery actually exists.

    Languages can be added as *codes with NO_MODEL* before any asset exists, so a
    capability query answers "we have not built this" instead of "this language is
    unknown" - the failure mode the reference study found in a 61-entry registry
    with no Bengali row at all.
    """

    def __init__(self, languages: tuple[str, ...] = ("en", "hi", "bn")) -> None:
        self._providers: dict[str, PronunciationProvider] = {}
        self._languages = {LanguageTag(code).root().code for code in languages}
        self.transliteration: TransliterationProvider = NoTransliteration()

    def languages(self) -> tuple[str, ...]:
        return tuple(sorted(self._languages))

    def declare(self, language: LanguageTag | str) -> None:
        """Register a language code as known-but-unmodelled."""
        self._languages.add(_code(language))

    def register(self, language: LanguageTag | str, provider: PronunciationProvider) -> None:
        """Attach a provider only once an approved asset backs it."""
        code = _code(language)
        if provider.capability(LanguageTag(code)).status not in (
            CapabilityStatus.MEASURED,
            CapabilityStatus.PARTIAL,
            CapabilityStatus.DECLARED,
        ):
            raise CapabilityUnavailable(
                "a pronunciation provider must declare a non-empty capability before it is registered",
                language=code,
            )
        self._languages.add(code)
        self._providers[code] = provider

    def installed(self, language: LanguageTag | str) -> bool:
        return _code(language) in self._providers

    def capability(self, language: LanguageTag | str) -> Capability:
        code = _code(language)
        provider = self._providers.get(code)
        if provider is not None:
            return provider.capability(LanguageTag(code))
        limitation = _NO_MODEL_LIMITATIONS.get(code)
        if limitation is None:
            limitation = (
                f"no {code} grapheme-to-phoneme asset has been authored, reviewed or approved"
            )
        return Capability(
            task=Task.TTS,
            status=CapabilityStatus.NO_MODEL,
            language=code,
            engine="grapheme-to-phoneme",
            limitations=(G2P_MISSING_ASSET, limitation),
            evidence="this project ships no approved G2P model or lexicon asset",
        )

    def phonemes(self, text: str, language: LanguageTag | str) -> tuple[str, ...]:
        code = _code(language)
        provider = self._providers.get(code)
        if provider is None:
            capability = self.capability(code)
            raise CapabilityUnavailable(
                "pronunciation is unavailable; synthesis would run on raw graphemes",
                language=code,
                reason=capability.limitations[0],
                detail=capability.limitations[-1],
            )
        return provider.phonemes(text, LanguageTag(code))

    def describe(self) -> tuple[Capability, ...]:
        return tuple(self.capability(code) for code in self.languages())
