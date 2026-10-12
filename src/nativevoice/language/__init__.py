"""Language identity, capability ledger, script/LID analysis, routing policy.

Split deliberately into identity (``LanguageRegistry``) and demonstrated
capability (``CapabilityLedger``) so that no language can be marked supported just
because it exists, and a language with no measurement reports ``NO_MODEL``.
"""

from .corpora import LEXICON_SCRIPTS, SAMPLE_SENTENCES, STOPWORDS
from .lid import (
    LexiconLanguageIdentifier,
    ScriptOnlyIdentifier,
    TransliteratedTextRisk,
    tokenize,
)
from .registry import (
    CapabilityLedger,
    LanguageNotFound,
    LanguageRegistry,
)
from .routing import (
    LanguageRouter,
    RoutingKind,
    RoutingOutcome,
)
from .scripts import (
    SCRIPT_RANGES,
    UNKNOWN_SCRIPT,
    analyze,
    char_script,
    detect_script,
    dominant_script,
    is_mixed,
    script_fractions,
)

__all__ = [
    "CapabilityLedger",
    "LEXICON_SCRIPTS",
    "LanguageNotFound",
    "LanguageRegistry",
    "LanguageRouter",
    "LexiconLanguageIdentifier",
    "RoutingKind",
    "RoutingOutcome",
    "SAMPLE_SENTENCES",
    "SCRIPT_RANGES",
    "STOPWORDS",
    "ScriptOnlyIdentifier",
    "UNKNOWN_SCRIPT",
    "TransliteratedTextRisk",
    "analyze",
    "char_script",
    "detect_script",
    "dominant_script",
    "is_mixed",
    "script_fractions",
    "tokenize",
]
