"""Language-aware text normalisation for synthesis front-ends.

Why this exists as data rather than code: the reference architecture (a
grapheme-only flow-matching model with no phonemizer) receives the exact
characters it will sing. Every "12.5%", "Mr." or "৳৫০" left un-expanded is a
token sequence the model was never trained to voice, and it degrades into
spelling or silence. So the rewrite knowledge lives in per-language rule tables
that a linguist can audit, and the only Python here is the machinery that
applies them.

Two behaviours come straight from the reference project's bug list:

* terminal punctuation is guaranteed, because a span with no final stop is read
  with a clipped or mispronounced last word;
* a number outside the authored range is never guessed. It stays as digits and
  the report says so, because inventing a spoken form for a number whose grammar
  we have not written down is exactly how a front-end ships a mispronunciation.

The Hindi and Bengali compound numerals (11-19, 21-99) are fused lexical forms,
not rule compositions. They are authored here from grammatical knowledge and
flagged `verified=False`, which the report surfaces as a limitation: they must be
checked against a licensed lexicon before any Bengali or Hindi voice ships.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from ..core.errors import ConfigurationError
from ..core.language import LanguageTag

CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b\u200c\ufeff]")
HTML_MARKUP = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>|&[a-zA-Z#0-9]+;")
WHITESPACE_RUN = re.compile(r"[ \t\r\n\v\f\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
NUMBER_TOKEN = re.compile(r"\d[\d.,]*|\d")
PERCENT_TOKEN = re.compile(r"(?P<value>\d[\d.,]*|\d)\s*[%％]")
SPOKEN_NUMBER_CEILING = 999

# Built from codepoints rather than literals: native digits survive no editor,
# font or copy-paste path intact, and a silently-mangled table would turn every
# Bengali numeral rewrite into a no-op that still looks right in review.
ASCII_DIGITS = "0123456789"
DEVANAGARI_DIGITS = "".join(chr(0x0966 + index) for index in range(10))
BENGALI_DIGITS = "".join(chr(0x09E6 + index) for index in range(10))


@dataclass(frozen=True, slots=True)
class NumberGrammar:
    """Spoken forms for one language's numerals.

    `decades` is keyed by the decade value (10, 20 ... 90). `exceptions` carries
    every form the composition rule cannot produce: English 11-19, and for Hindi
    and Bengali the whole fused 11-99 range. `composable` is False for languages
    where a missing exception must not be papered over with an ordered
    concatenation, because that concatenation would not be a real word.
    """

    zero: str
    ones: tuple[str, ...]
    decades: Mapping[int, str]
    exceptions: Mapping[int, str]
    hundred: str
    and_word: str
    point: str
    separator: str = " "
    composable: bool = True
    verified: bool = True


@dataclass(frozen=True, slots=True)
class CurrencyWord:
    singular: str
    plural: str


@dataclass(frozen=True, slots=True)
class LanguageRules:
    """Everything the normaliser needs about one language's written form."""

    code: str
    numbers: NumberGrammar | None
    native_digits: Mapping[str, str] = field(default_factory=dict)
    abbreviations: tuple[tuple[str, str, bool], ...] = ()
    currency: Mapping[str, CurrencyWord] = field(default_factory=dict)
    percent: str = ""
    terminal: str = "."
    terminators: tuple[str, ...] = (".", "!", "?")
    closing_tails: tuple[str, ...] = ('"', "'", "”", "’", ")", "]", "»")

    def limitations(self) -> tuple[str, ...]:
        notes = []
        if self.numbers is None:
            notes.append(f"{self.code}: no numeral grammar authored, digits are left as written")
        elif not self.numbers.verified:
            notes.append(
                f"{self.code}: fused numerals 11-99 are authored from grammatical knowledge, "
                "not verified against a licensed lexicon"
            )
        return tuple(notes)


_EN_NUMBERS = NumberGrammar(
    zero="zero",
    ones=("one", "two", "three", "four", "five", "six", "seven", "eight", "nine"),
    decades={
        10: "ten", 20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
        60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
    },
    exceptions={
        11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
        16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
    },
    hundred="hundred",
    and_word="and",
    point="point",
)

_HI_NUMBERS = NumberGrammar(
    zero="शून्य",
    ones=("एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ"),
    decades={
        10: "दस", 20: "बीस", 30: "तीस", 40: "चालीस", 50: "पचास",
        60: "साठ", 70: "सत्तर", 80: "अस्सी", 90: "नब्बे",
    },
    exceptions={
        11: "ग्यारह", 12: "बारह", 13: "तेरह", 14: "चौदह", 15: "पंद्रह",
        16: "सोलह", 17: "सत्रह", 18: "अठारह", 19: "उन्नीस",
        21: "इक्कीस", 22: "बाईस", 23: "तेईस", 24: "चौबीस", 25: "पच्चीस",
        26: "छब्बीस", 27: "सत्ताईस", 28: "अट्ठाईस", 29: "उनतीस",
        31: "इकतीस", 32: "बत्तीस", 33: "तैंतीस", 34: "चौंतीस", 35: "पैंतीस",
        36: "छत्तीस", 37: "सैंतीस", 38: "अड़तीस", 39: "उनतालीस",
        41: "इकतालीस", 42: "बयालीस", 43: "तैंतालीस", 44: "चौतालीस", 45: "पैंतालीस",
        46: "छियालीस", 47: "सत्तालीस", 48: "अड़तालीस", 49: "उनचास",
        51: "इक्यावन", 52: "बावन", 53: "तिरेठ", 54: "चौवन", 55: "पचानवे",
        56: "छप्पन", 57: "सत्तावन", 58: "अट्ठावन", 59: "उनसठ",
        61: "इकसठ", 62: "बासठ", 63: "तिरेसठ", 64: "चौंसठ", 65: "पैंसठ",
        66: "छियासठ", 67: "सड़सठ", 68: "अड़सठ", 69: "उनहत्तर",
        71: "इकहत्तर", 72: "बहत्तर", 73: "तहत्तर", 74: "चौहत्तर", 75: "पचहत्तर",
        76: "छिहत्तर", 77: "सतहत्तर", 78: "अठहत्तर", 79: "उन्यासी",
        81: "इक्यासी", 82: "बयासी", 83: "तिरसी", 84: "चौरासी", 85: "पचरासी",
        86: "छरासी", 87: "सत्तरासी", 88: "अट्ठरासी", 89: "नव्वासी",
        91: "इक्यानबे", 92: "बानबे", 93: "तिरेनबे", 94: "चौनबे", 95: "पचानबे",
        96: "छियानबे", 97: "सत्तनबे", 98: "अट्ठानबे", 99: "निन्यानबे",
    },
    hundred="सौ",
    and_word="",
    point="दशमलव",
    composable=False,
    verified=False,
)

_BN_NUMBERS = NumberGrammar(
    zero="শূন্য",
    ones=("এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়"),
    decades={
        10: "দশ", 20: "কুড়ি", 30: "তিরিশ", 40: "চল্লিশ", 50: "পঞ্চাশ",
        60: "ষাট", 70: "সত্তর", 80: "আশি", 90: "নব্বই",
    },
    exceptions={
        11: "এগারো", 12: "বারো", 13: "তেরো", 14: "চৌদ্দ", 15: "পনেরো",
        16: "ষোলো", 17: "সতেরো", 18: "আঠারো", 19: "উনিশ",
        21: "একুশ", 22: "বাইশ", 23: "তেইশ", 24: "চব্বিশ", 25: "পঁচিশ",
        26: "ছাব্বিশ", 27: "সাতাইশ", 28: "আটাইশ", 29: "ঊনতিশ",
        31: "একত্রিশ", 32: "বত্রিশ", 33: "তেত্রিশ", 34: "চৌত্রিশ", 35: "পঁয়ত্রিশ",
        36: "ছত্রিশ", 37: "সাতত্রিশ", 38: "আটত্রিশ", 39: "ঊনচল্লিশ",
        41: "একচল্লিশ", 42: "বিয়াল্লিশ", 43: "তেচল্লিশ", 44: "চৌচল্লিশ", 45: "পঁয়চল্লিশ",
        46: "ছয়চল্লিশ", 47: "সাতচল্লিশ", 48: "আটচল্লিশ", 49: "ঊনপঞ্চাশ",
        51: "একান্ন", 52: "বান্ন", 53: "তিরান্ন", 54: "চৌরান্ন", 55: "পঞ্চান্ন",
        56: "ছাপ্পান্ন", 57: "সাতান্ন", 58: "আটান্ন", 59: "ঊনষাট",
        61: "একষট্ঠি", 62: "বষট্ঠি", 63: "তেষট্ঠি", 64: "চৌষট্ঠি", 65: "পঁয়ষট্ঠি",
        66: "ছষট্ঠি", 67: "সাতষট্ঠি", 68: "আটষট্ঠি", 69: "ঊনহত্তর",
        71: "একাত্তর", 72: "বাত্তর", 73: "তত্তর", 74: "চত্তর", 75: "পঁচাত্তর",
        76: "ছাত্তর", 77: "সাতাত্তর", 78: "আটাত্তর", 79: "ঊনআশি",
        81: "একাশি", 82: "বিয়াশি", 83: "তিয়াশি", 84: "চুরাশি", 85: "পঁচাশি",
        86: "ছিয়াশি", 87: "সাতাশি", 88: "আটাশি", 89: "ঊনানব্বই",
        91: "একানব্বই", 92: "বানব্বই", 93: "তিরানব্বই", 94: "চৌরানব্বই", 95: "পচানব্বই",
        96: "ছিয়ানব্বই", 97: "সত্তানব্বই", 98: "আটানব্বই", 99: "নিরানব্বই",
    },
    hundred="শত",
    and_word="",
    point="দশমিক",
    composable=False,
    verified=False,
)

_EN_ABBREVIATIONS = (
    # (written form, spoken form, case sensitive). Titles and "No." stay
    # case-sensitive because their lowercase bodies are ordinary words: "no."
    # would otherwise become "number" at the end of a sentence, and "st." is left
    # out entirely because street/saint cannot be separated without an address
    # parser - guessing wrong is worse than letting the model read trained graphemes.
    ("Mr.", "mister", True),
    ("Mrs.", "missus", True),
    ("Ms.", "miss", True),
    ("Dr.", "doctor", True),
    ("Prof.", "professor", True),
    ("Rev.", "reverend", True),
    ("Jr.", "junior", True),
    ("Sr.", "senior", True),
    ("No.", "number", True),
    ("Nos.", "numbers", True),
    ("etc.", "et cetera", False),
    ("e.g.", "for example", False),
    ("i.e.", "that is", False),
    ("vs.", "versus", False),
    ("approx.", "approximately", False),
    ("dept.", "department", False),
    ("asst.", "assistant", False),
    ("ave.", "avenue", False),
    ("blvd.", "boulevard", False),
    ("rd.", "road", False),
    ("inc.", "incorporated", False),
    ("ltd.", "limited", False),
    ("www.", "world wide web", False),
    ("lbs.", "pounds", False),
    ("kg.", "kilograms", False),
)

_EN_CURRENCY = {
    "$": CurrencyWord("dollar", "dollars"),
    "€": CurrencyWord("euro", "euros"),
    "£": CurrencyWord("pound", "pounds"),
    "¥": CurrencyWord("yen", "yen"),
    "₹": CurrencyWord("rupee", "rupees"),
    "৳": CurrencyWord("taka", "taka"),
    "usd": CurrencyWord("us dollar", "us dollars"),
    "eur": CurrencyWord("euro", "euros"),
    "gbp": CurrencyWord("pound sterling", "pounds sterling"),
    "inr": CurrencyWord("rupee", "rupees"),
    "bdt": CurrencyWord("taka", "taka"),
    "jpy": CurrencyWord("yen", "yen"),
}

_HI_CURRENCY = {
    "$": CurrencyWord("डॉलर", "डॉलर"),
    "₹": CurrencyWord("रुपया", "रुपये"),
    "€": CurrencyWord("यूरो", "यूरो"),
    "£": CurrencyWord("पाउंड", "पाउंड"),
    "৳": CurrencyWord("टाका", "टाका"),
    "usd": CurrencyWord("अमेरिकी डॉलर", "अमेरिकी डॉलर"),
    "inr": CurrencyWord("रुपया", "रुपये"),
    "bdt": CurrencyWord("बांग्लादेशी टाका", "बांग्लादेशी टाका"),
}

_BN_CURRENCY = {
    "$": CurrencyWord("ডলার", "ডলার"),
    "৳": CurrencyWord("টাকা", "টাকা"),
    "₹": CurrencyWord("রুপি", "রুপি"),
    "€": CurrencyWord("ইউরো", "ইউরো"),
    "£": CurrencyWord("পাউন্ড", "পাউন্ড"),
    "usd": CurrencyWord("মার্কিন ডলার", "মার্কিন ডলার"),
    "inr": CurrencyWord("ভারতীয় রুপি", "ভারতীয় রুপি"),
    "bdt": CurrencyWord("বাংলাদেশী টাকা", "বাংলাদেশী টাকা"),
}

DEFAULT_RULES: Mapping[str, LanguageRules] = {
    "en": LanguageRules(
        code="en",
        numbers=_EN_NUMBERS,
        abbreviations=_EN_ABBREVIATIONS,
        currency=_EN_CURRENCY,
        percent="percent",
    ),
    "hi": LanguageRules(
        code="hi",
        numbers=_HI_NUMBERS,
        native_digits=dict(zip(DEVANAGARI_DIGITS, ASCII_DIGITS)),
        currency=_HI_CURRENCY,
        percent="प्रतिशत",
        terminal="।",
        terminators=("।", ".", "!", "?", "॥"),
    ),
    "bn": LanguageRules(
        code="bn",
        numbers=_BN_NUMBERS,
        native_digits=dict(zip(BENGALI_DIGITS, ASCII_DIGITS)),
        currency=_BN_CURRENCY,
        percent="শতকরা",
        terminal="।",
        terminators=("।", ".", "!", "?"),
    ),
}

_UNAUTHORED = LanguageRules(code="", numbers=None)


@dataclass(frozen=True, slots=True)
class NormaliseChange:
    """One rewrite actually performed, so a reviewer can audit the front-end."""

    kind: str
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class NormalisationReport:
    """The normalised text plus a full account of what was changed and refused."""

    text: str
    language: str
    changes: tuple[NormaliseChange, ...] = ()
    limitations: tuple[str, ...] = ()

    def changed(self) -> bool:
        return bool(self.changes)

    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted({change.kind for change in self.changes}))

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "changes": [
                {"kind": c.kind, "before": c.before, "after": c.after} for c in self.changes
            ],
            "limitations": list(self.limitations),
        }


class TextNormaliser:
    """Applies `LanguageRules` to raw text and reports every edit.

    Input that cannot be sung is refused rather than sanitised into something
    plausible: control characters and markup raise, because a synthesis engine
    handed `<p>` or a bell character has no trained interpretation for it.
    """

    def __init__(self, rules: Mapping[str, LanguageRules] | None = None) -> None:
        self.rules = dict(rules) if rules is not None else dict(DEFAULT_RULES)

    def rules_for(self, language: LanguageTag | str) -> LanguageRules:
        """The authored table for a language, or an explicit no-op table.

        A language with no table still gets whitespace collapse and a terminal
        stop; the report then states that digits were left as written. Inventing
        rewrites for a language we have not studied is how a front-end silently
        mangles it, so the gap is reported instead of filled.
        """
        code = language if isinstance(language, str) else language.root().code
        authored = self.rules.get(code)
        if authored is not None:
            return authored
        return replace(_UNAUTHORED, code=code)

    def has_rules(self, language: LanguageTag | str) -> bool:
        return self.rules_for(language).numbers is not None

    def number_words(self, value: int, grammar: NumberGrammar) -> str | None:
        """Spoken form for 0-999, or None when the grammar cannot produce one."""
        if value < 0 or value > SPOKEN_NUMBER_CEILING:
            return None
        return self._below_thousand(value, grammar)

    def _below_thousand(self, value: int, grammar: NumberGrammar) -> str:
        if value == 0:
            return grammar.zero
        pieces = []
        if value >= 100:
            pieces.append(grammar.ones[value // 100 - 1])
            pieces.append(grammar.hundred)
            value %= 100
            if value and grammar.and_word:
                pieces.append(grammar.and_word)
        if value:
            word = self._below_hundred(value, grammar)
            if word is None:
                return None
            pieces.append(word)
        return grammar.separator.join(pieces)

    def _below_hundred(self, value: int, grammar: NumberGrammar) -> str | None:
        if value in grammar.exceptions:
            return grammar.exceptions[value]
        if value in grammar.decades:
            return grammar.decades[value]
        if value < 10:
            return grammar.ones[value - 1]
        decade = value - value % 10
        if not grammar.composable or decade not in grammar.decades:
            return None
        return grammar.decades[decade] + grammar.separator + grammar.ones[value % 10 - 1]

    def normalise(self, text: str, language: LanguageTag | str) -> NormalisationReport:
        code = language if isinstance(language, str) else language.root().code
        self._reject_unsayable(text)
        rules = self.rules_for(code)
        changes: list[NormaliseChange] = []
        limitations = list(rules.limitations())

        current = WHITESPACE_RUN.sub(" ", text).strip()
        if current != text:
            changes.append(NormaliseChange("whitespace", text, current))

        if rules.native_digits:
            current = self._translate_digits(current, rules, changes)

        current = self._expand_abbreviations(current, rules, changes)
        current = self._expand_currency(current, rules, changes)
        current = self._expand_percent(current, rules, changes)
        current = self._expand_numbers(current, rules, changes, limitations)
        current = WHITESPACE_RUN.sub(" ", current).strip()
        current = self._guarantee_terminal(current, rules, changes)

        return NormalisationReport(current, code, tuple(changes), tuple(limitations))

    def normalised_text(self, text: str, language: LanguageTag | str) -> str:
        return self.normalise(text, language).text

    @staticmethod
    def _reject_unsayable(text: str) -> None:
        control = CONTROL_CHARACTERS.search(text)
        if control:
            raise ConfigurationError(
                "text contains a control character a synthesizer cannot voice",
                position=control.start(),
                codepoint=f"U+{ord(control.group()):04X}",
            )
        markup = HTML_MARKUP.search(text)
        if markup:
            raise ConfigurationError(
                "text contains markup; pass plain text to the synthesizer",
                found=markup.group(),
            )

    @staticmethod
    def _translate_digits(text: str, rules: LanguageRules, changes: list[NormaliseChange]) -> str:
        if not any(character in rules.native_digits for character in text):
            return text
        table = {ord(source): target for source, target in rules.native_digits.items()}
        translated = text.translate(table)
        changes.append(NormaliseChange("native_digits", text, translated))
        return translated

    def _expand_abbreviations(
        self, text: str, rules: LanguageRules, changes: list[NormaliseChange]
    ) -> str:
        for source, replacement, case_sensitive in rules.abbreviations:
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(
                r"(?<![0-9A-Za-z])" + re.escape(source) + r"(?![0-9A-Za-z])", flags
            )
            for match in pattern.finditer(text):
                if match.group().lower() == replacement.lower():
                    continue
                changes.append(NormaliseChange("abbreviation", match.group(), replacement))
            text = pattern.sub(replacement, text)
        return text

    def _expand_currency(self, text: str, rules: LanguageRules, changes: list[NormaliseChange]) -> str:
        if rules.numbers is None or not rules.currency:
            return text
        single = [key for key in rules.currency if len(key) == 1]
        codes = sorted((key for key in rules.currency if len(key) > 1), key=len, reverse=True)
        if single:
            symbols = "".join(re.escape(key) for key in single)
            text = self._rewrite_amounts(
                text,
                re.compile(rf"(?P<unit>[{symbols}])\s*(?P<value>\d[\d.,]*|\d)"),
                rules,
                changes,
                "currency",
            )
        if codes:
            alternatives = "|".join(re.escape(key) for key in codes)
            text = self._rewrite_amounts(
                text,
                re.compile(
                    rf"(?P<value>\d[\d.,]*|\d)\s*(?P<unit>{alternatives})\b", re.IGNORECASE
                ),
                rules,
                changes,
                "currency",
            )
        return text

    def _expand_percent(self, text: str, rules: LanguageRules, changes: list[NormaliseChange]) -> str:
        if rules.numbers is None or not rules.percent:
            return text
        return self._rewrite_amounts(text, PERCENT_TOKEN, rules, changes, "percent")

    def _rewrite_amounts(
        self,
        text: str,
        pattern: re.Pattern,
        rules: LanguageRules,
        changes: list[NormaliseChange],
        kind: str,
    ) -> str:
        out = []
        cursor = 0
        for match in pattern.finditer(text):
            words, _ = self._spoken_amount(match.group("value"), rules.numbers)
            unit = self._unit_phrase(match, rules, kind, match.group("value"))
            if words is None or unit is None:
                continue
            phrase = f"{words} {unit}"
            out.append(text[cursor : match.start()])
            out.append(phrase)
            changes.append(NormaliseChange(kind, match.group(), phrase))
            cursor = match.end()
        out.append(text[cursor :])
        return "".join(out)

    def _unit_phrase(self, match, rules: LanguageRules, kind: str, value: str) -> str | None:
        if kind == "percent":
            return rules.percent or None
        key = (match.groupdict().get("unit") or "").lower()
        currency = rules.currency.get(key)
        if currency is None:
            return None
        return currency.singular if value.rstrip(".,;:") in {"1", "1.0", "01"} else currency.plural

    def _expand_numbers(
        self, text: str, rules: LanguageRules, changes: list[NormaliseChange], limitations: list[str]
    ) -> str:
        if rules.numbers is None:
            if NUMBER_TOKEN.search(text):
                limitations.append("digits left unexpanded: no numeral grammar for this language")
            return text
        grammar = rules.numbers
        out = []
        cursor = 0
        for match in NUMBER_TOKEN.finditer(text):
            token = match.group()
            words, failure = self._spoken_amount(token, grammar)
            if words is None:
                if failure == "above_range":
                    limitations.append(
                        f"number {token} exceeds the authored spoken range "
                        f"(0-{SPOKEN_NUMBER_CEILING}); digits left as written"
                    )
                else:
                    limitations.append(
                        f"number {token} has no authored form in {rules.code}; "
                        "digits left as written"
                    )
                continue
            out.append(text[cursor : match.start()])
            out.append(words)
            changes.append(NormaliseChange("number", token, words))
            cursor = match.end()
        out.append(text[cursor :])
        return "".join(out)

    def _spoken_amount(self, token: str, grammar: NumberGrammar) -> tuple[str | None, str]:
        """Words for one numeric token, plus the reason when there are none.

        Fractions are read digit by digit after the decimal word, which is how a
        listener expects "12.5" and the only form that never invents a name for a
        quantity the grammar does not cover.
        """
        cleaned = token.rstrip(".,;:")
        integer_text, _, fraction_text = cleaned.partition(".")
        integer_text = integer_text.replace(",", "")
        if not integer_text.isdigit():
            return None, "no_authored_form"
        value = int(integer_text)
        if value > SPOKEN_NUMBER_CEILING:
            return None, "above_range"
        words = self._below_thousand(value, grammar)
        if words is None:
            return None, "no_authored_form"
        if fraction_text:
            if not fraction_text.isdigit():
                return None, "no_authored_form"
            spoken_digits = grammar.separator.join(
                self._digit_word(int(digit), grammar) for digit in fraction_text
            )
            words = f"{words}{grammar.separator}{grammar.point}{grammar.separator}{spoken_digits}"
        return words, ""

    @staticmethod
    def _digit_word(digit: int, grammar: NumberGrammar) -> str:
        return grammar.zero if digit == 0 else grammar.ones[digit - 1]

    @staticmethod
    def _guarantee_terminal(text: str, rules: LanguageRules, changes: list[NormaliseChange]) -> str:
        """Close every span with a stop.

        The reference model clips or mispronounces the final word of a span that
        has no terminal punctuation, so a dangling comma, semicolon or colon is
        *replaced* rather than followed by a stop: "list, " would otherwise end on
        the exact comma-intonation that keeps the sentence open.
        """
        if not text:
            return text
        tail = text[-1]
        if tail in rules.terminators:
            return text
        if tail in rules.closing_tails and len(text) >= 2 and text[-2] in rules.terminators:
            return text
        if tail in ",;:—–-":
            closed = text[:-1] + rules.terminal
            changes.append(NormaliseChange("terminal_punctuation", text, closed))
            return closed
        closed = text + rules.terminal
        changes.append(NormaliseChange("terminal_punctuation", text, closed))
        return closed
