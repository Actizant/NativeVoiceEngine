"""Language identity (registry) kept strictly separate from capability (ledger).

The reference study made the trap concrete: a 61-entry registry that stored a
per-engine ``supported: true`` boolean silently reproduced upstream's gaps -
there was no Bengali row at all, so Bengali could never be supported no matter
what the model could actually do. Identity and capability were conflated into a
single optimistic flag.

So this module splits them. ``LanguageRegistry`` records *what a language is*
(its ISO code, English name, native name, primary script) and nothing about
whether we can transcribe it. ``CapabilityLedger`` records *what has been shown*
per (task, language, engine), and the honest default when nothing has been
recorded is ``NO_MODEL`` - never ``supported``. The ledger is seeded with zero
measured entries so that no language can report success we have not earned.
"""

from __future__ import annotations

from collections import defaultdict

from ..core.errors import CapabilityUnavailable
from ..core.language import (
    Capability,
    CapabilityStatus,
    LanguageProfile,
    LanguageTag,
    Task,
)

# (code, english name, native name, primary script, rtl) - identity only.
# Exactly 61 languages, deliberately including a Bengali and a Hindi row with the
# correct native names. Every Indic script has at least one representative, plus
# the corpora and routing-referenced languages, so identity never silently omits
# a language the way the reference registry silently omitted Bengali.
_PROFILES: tuple[tuple[str, str, str, str, bool], ...] = (
    ("en", "English", "English", "Latin", False),
    ("hi", "Hindi", "हिन्दी", "Devanagari", False),
    ("bn", "Bengali", "বাংলা", "Bengali", False),
    ("mr", "Marathi", "मराठी", "Devanagari", False),
    ("ne", "Nepali", "नेपाली", "Devanagari", False),
    ("gu", "Gujarati", "ગુજરાતી", "Gujarati", False),
    ("pa", "Punjabi", "ਪੰਜਾਬੀ", "Gurmukhi", False),
    ("or", "Odia", "ଓଡ଼ିଆ", "Odia", False),
    ("ta", "Tamil", "தமிழ்", "Tamil", False),
    ("te", "Telugu", "తెలుగు", "Telugu", False),
    ("kn", "Kannada", "ಕನ್ನಡ", "Kannada", False),
    ("ml", "Malayalam", "മലയാളം", "Malayalam", False),
    ("si", "Sinhala", "සිංහල", "Sinhala", False),
    ("ur", "Urdu", "اردو", "Arabic", True),
    ("ar", "Arabic", "العربية", "Arabic", True),
    ("fa", "Persian", "فارسی", "Arabic", True),
    ("ps", "Pashto", "پښتو", "Arabic", True),
    ("he", "Hebrew", "עברית", "Hebrew", True),
    ("zh", "Chinese", "中文", "Han", False),
    ("ja", "Japanese", "日本語", "Hiragana", False),
    ("ko", "Korean", "한국어", "Hangul", False),
    ("ru", "Russian", "Русский", "Cyrillic", False),
    ("uk", "Ukrainian", "Українська", "Cyrillic", False),
    ("be", "Belarusian", "Беларуская", "Cyrillic", False),
    ("sr", "Serbian", "Српски", "Cyrillic", False),
    ("bg", "Bulgarian", "Български", "Cyrillic", False),
    ("el", "Greek", "Ελληνικά", "Greek", False),
    ("th", "Thai", "ไทย", "Thai", False),
    ("lo", "Lao", "ລາວ", "Lao", False),
    ("my", "Burmese", "မြန်မာ", "Myanmar", False),
    ("km", "Khmer", "ខ្មែរ", "Khmer", False),
    ("ka", "Georgian", "ქართული", "Georgian", False),
    ("hy", "Armenian", "Հայերեն", "Armenian", False),
    ("am", "Amharic", "አማርኛ", "Ethiopic", False),
    ("es", "Spanish", "Español", "Latin", False),
    ("fr", "French", "Français", "Latin", False),
    ("de", "German", "Deutsch", "Latin", False),
    ("it", "Italian", "Italiano", "Latin", False),
    ("pt", "Portuguese", "Português", "Latin", False),
    ("nl", "Dutch", "Nederlands", "Latin", False),
    ("sv", "Swedish", "Svenska", "Latin", False),
    ("no", "Norwegian", "Norsk", "Latin", False),
    ("da", "Danish", "Dansk", "Latin", False),
    ("pl", "Polish", "Polski", "Latin", False),
    ("cs", "Czech", "Čeština", "Latin", False),
    ("sk", "Slovak", "Slovenčina", "Latin", False),
    ("ro", "Romanian", "Română", "Latin", False),
    ("hu", "Hungarian", "Magyar", "Latin", False),
    ("lt", "Lithuanian", "Lietuvių", "Latin", False),
    ("lv", "Latvian", "Latviešu", "Latin", False),
    ("hr", "Croatian", "Hrvatski", "Latin", False),
    ("tr", "Turkish", "Türkçe", "Latin", False),
    ("vi", "Vietnamese", "Tiếng Việt", "Latin", False),
    ("id", "Indonesian", "Bahasa Indonesia", "Latin", False),
    ("ms", "Malay", "Bahasa Melayu", "Latin", False),
    ("tl", "Tagalog", "Tagalog", "Latin", False),
    ("sw", "Swahili", "Kiswahili", "Latin", False),
    ("ha", "Hausa", "Hausa", "Latin", False),
    ("af", "Afrikaans", "Afrikaans", "Latin", False),
    ("yo", "Yoruba", "Yoruba", "Latin", False),
    ("zu", "Zulu", "isiZulu", "Latin", False),
)

_NOT_SPACE_DELIMITED = {"zh", "ja"}


class LanguageNotFound(LookupError):
    """Requested code is not one of the registered identities."""

    def __init__(self, code: str) -> None:
        super().__init__(f"unknown language code {code!r}")
        self.code = code


def _code_of(tag: str | LanguageTag) -> str:
    if isinstance(tag, LanguageTag):
        return tag.code
    return tag.split("-")[0].split("_")[0].lower()


class LanguageRegistry:
    """Immutable catalogue of language identity. It answers *what* a language is."""

    __slots__ = ("_profiles",)

    def __init__(self, profiles: list[LanguageProfile] | None = None) -> None:
        if profiles is None:
            profiles = [
                LanguageProfile(
                    code=code,
                    english_name=english,
                    native_name=native,
                    script=script,
                    rtl=rtl,
                    iso_level="1",
                    space_delimited=code not in _NOT_SPACE_DELIMITED,
                    notes=("Mixed Han/Kana/Japanese syllabaries." if code == "ja" else
                           "Han logographic; no spaces." if code == "zh" else ""),
                )
                for code, english, native, script, rtl in _PROFILES
            ]
        self._profiles = {profile.code: profile for profile in profiles}

    def lookup(self, tag: str | LanguageTag) -> LanguageProfile:
        code = _code_of(tag)
        profile = self._profiles.get(code)
        if profile is None:
            raise LanguageNotFound(code)
        return profile

    def known(self, tag: str | LanguageTag) -> bool:
        return _code_of(tag) in self._profiles

    def all_codes(self) -> list[str]:
        return sorted(self._profiles)

    def profiles(self) -> list[LanguageProfile]:
        return [self._profiles[code] for code in self.all_codes()]

    def codes_for_script(self, script: str) -> list[str]:
        return sorted(code for code, p in self._profiles.items() if p.script == script)

    def __len__(self) -> int:
        return len(self._profiles)


# Authority ordering. Highest rank is the strongest statement of *evidence*, so
# `best` returns the most trustworthy record, not merely the most optimistic one.
# Routing applies its own usability policy on top (see routing.py) because PARTIAL
# may be usable while an unverified DECLARED is not - authority rank and usability
# are deliberately different questions.
_STATUS_RANK: dict[CapabilityStatus, int] = {
    CapabilityStatus.MEASURED: 6,
    CapabilityStatus.DECLARED: 5,
    CapabilityStatus.PARTIAL: 4,
    CapabilityStatus.MEASURED_FAILED: 3,
    CapabilityStatus.UNSUPPORTED: 2,
    CapabilityStatus.NO_MODEL: 1,
}


class CapabilityLedger:
    """What has been *shown* per (task, language, engine). Empty by construction."""

    __slots__ = ("_records", "_status_by_key")

    def __init__(self) -> None:
        # key -> Capability, keeping the highest-authority record per key.
        self._records: dict[tuple[str, str, str], list[Capability]] = defaultdict(list)
        self._status_by_key: dict[tuple[str, str, str], CapabilityStatus] = {}

    def record(self, capability: Capability) -> None:
        key = self._key(capability.task, capability.language, capability.engine)
        self._records[key].append(capability)
        existing = self._status_by_key.get(key)
        if existing is None or _STATUS_RANK[capability.status] >= _STATUS_RANK[existing]:
            self._status_by_key[key] = capability.status

    @staticmethod
    def _key(task: Task | str, language: str | LanguageTag, engine: str) -> tuple[str, str, str]:
        return (
            task.value if isinstance(task, Task) else str(task),
            _code_of(language),
            engine or "",
        )

    def capabilities_for(self, task: Task | str, language: str | LanguageTag) -> list[Capability]:
        """All recorded capabilities for (task, language), strongest first."""
        code = _code_of(language)
        out: list[Capability] = []
        for (recorded_task, recorded_lang, _engine), caps in self._records.items():
            if recorded_task == (task.value if isinstance(task, Task) else str(task)) and recorded_lang == code:
                out.extend(caps)
        out.sort(key=lambda cap: -_STATUS_RANK[cap.status])
        return out

    def best(self, task: Task | str, language: str | LanguageTag) -> Capability:
        code = _code_of(language)
        ranked = self.capabilities_for(task, code)
        if ranked:
            return ranked[0]
        # The honesty invariant: a language we have never recorded reports
        # NO_MODEL, never a defaulted "supported".
        return Capability(
            task=task if isinstance(task, Task) else Task(task),
            status=CapabilityStatus.NO_MODEL,
            language=code,
        )

    def usable(
        self,
        task: Task | str,
        language: str | LanguageTag,
        *,
        allow_declared: bool = False,
    ) -> Capability | None:
        """Strongest capability we may act on, honouring strict routing policy.

        MEASURED and PARTIAL are always usable. DECLARED is usable only when
        ``allow_declared`` is set - a vendor claim is not proof. UNSUPPORTED and
        NO_MODEL are never usable.
        """
        for cap in self.capabilities_for(task, language):
            if cap.status in (CapabilityStatus.MEASURED, CapabilityStatus.PARTIAL):
                return cap
            if cap.status is CapabilityStatus.DECLARED and allow_declared:
                return cap
        return None

    def matrix(self, task: Task | str) -> dict[str, dict[str, str]]:
        """Rows are languages, columns are engines, cells are capability status.

        Only languages with at least one record appear as rows; engines are keyed
        by engine id. A recorded-but-unverified engine is visible here as
        ``declared`` rather than being quietly dropped.
        """
        task_value = task.value if isinstance(task, Task) else str(task)
        rows: dict[str, dict[str, str]] = {}
        for (recorded_task, code, engine), _caps in self._records.items():
            if recorded_task != task_value:
                continue
            rows.setdefault(code, {})[engine] = self.best_for_engine(task_value, code, engine)
        return rows

    def best_for_engine(self, task: Task | str, language: str | LanguageTag, engine: str) -> str:
        cap = self.get(task, language, engine)
        return cap.status.value if cap is not None else CapabilityStatus.NO_MODEL.value

    def get(
        self,
        task: Task | str,
        language: str | LanguageTag,
        engine: str,
    ) -> Capability | None:
        key = self._key(task, language, engine)
        caps = self._records.get(key)
        if not caps:
            return None
        return max(caps, key=lambda cap: _STATUS_RANK[cap.status])

    def engines_for(self, task: Task | str) -> list[str]:
        task_value = task.value if isinstance(task, Task) else str(task)
        return sorted({engine for (t, _code, engine) in self._records if t == task_value and engine})

    def recorded_languages(self, task: Task | str) -> list[str]:
        task_value = task.value if isinstance(task, Task) else str(task)
        return sorted({code for (t, code, _e) in self._records if t == task_value})

    def report(self, task: Task | str, registry: LanguageRegistry | None = None) -> str:
        """Sorted text table of recorded capabilities. Honest about emptiness."""
        task_value = task.value if isinstance(task, Task) else str(task)
        header = ("Language", "Code", "Engine", "Status", "Metric", "Evidence")
        rows: list[tuple[str, str, str, str, str, str]] = []
        for code in self.recorded_languages(task_value):
            english = registry.lookup(code).english_name if registry and registry.known(code) else code
            for cap in self.capabilities_for(task_value, code):
                metric = ""
                if cap.metric_name and cap.metric_value is not None:
                    metric = f"{cap.metric_name}={cap.metric_value:.3f}"
                rows.append(
                    (
                        english,
                        code,
                        cap.engine or "(none)",
                        cap.status.value,
                        metric,
                        cap.model_id,
                    )
                )
        if not rows:
            body = [f"(no capabilities recorded for task {task_value!r}; every language reports no_model)"]
        else:
            rows.sort(key=lambda row: (row[0].lower(), row[1], row[2]))
            widths = [len(col) for col in header]
            for row in rows:
                widths = [max(w, len(v)) for w, v in zip(widths, row)]
            fmt = "  ".join(f"{{:<{w}}}" for w in widths)
            separator = "  ".join("-" * w for w in widths)
            body = [fmt.format(*header), separator, *(fmt.format(*row) for row in rows)]
        return "\n".join([f"Capability report - task={task_value}", *body])

    def assert_usable(self, capability: Capability, *, allow_declared: bool = False) -> None:
        if capability.status in (CapabilityStatus.MEASURED, CapabilityStatus.PARTIAL):
            return
        if capability.status is CapabilityStatus.DECLARED and allow_declared:
            return
        raise CapabilityUnavailable(
            "capability is not usable for automatic routing",
            status=capability.status.value,
            language=capability.language,
            engine=capability.engine,
        )
