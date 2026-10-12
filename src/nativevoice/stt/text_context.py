"""Hot-word / context prompts, capped and sanitised.

Reference lesson: an unbounded hot-word prompt once broke a cloud transcription
endpoint somewhere near ~1000 parts. Whisper-family engines also only read a
fixed context window (~224 tokens). So we never forward an unbounded term list:
the builder dedupes, strips control characters, hard-caps the term count and the
total character budget, and documents *why* the cap exists so nobody removes it
thinking it is a nuisance.
"""

from __future__ import annotations

from ..core.errors import ConfigurationError

# The Whisper prompt window is ~224 tokens; ~900 characters is a conservative,
# well-under-1000 character ceiling that stays inside it on every script.
MAX_PROMPT_CHARS = 900
MAX_TERMS = 64


def sanitize(text: str) -> str:
    """Strip control characters (the ones that crash endpoints) from a fragment."""
    return "".join(char for char in text if not (ord(char) < 0x20 or ord(char) == 0x7F))


def dedupe(terms) -> list[str]:
    """Case-insensitive dedupe preserving first-seen order; drops empties."""
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        cleaned = sanitize(str(term)).strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def build_prompt(
    hint_terms=(),
    context: str = "",
    *,
    max_chars: int = MAX_PROMPT_CHARS,
    max_terms: int = MAX_TERMS,
) -> str:
    """Render hint terms and free-text context into one engine prompt string.

    The result is guaranteed to be at most ``max_chars`` long, so a caller can
    never smuggle an oversized prompt into an engine that has a fixed window.
    """
    if max_chars <= 0:
        raise ConfigurationError("max_chars must be positive", max_chars=max_chars)
    if max_terms <= 0:
        raise ConfigurationError("max_terms must be positive", max_terms=max_terms)

    context_clean = sanitize(context).strip()
    terms = dedupe(hint_terms)[:max_terms]

    pieces: list[str] = []
    if context_clean:
        pieces.append(context_clean)
    if terms:
        pieces.append("Terms: " + ", ".join(terms))

    prompt = "\n".join(pieces).strip()
    if len(prompt) <= max_chars:
        return prompt

    # Budget-aware truncation: keep the context, then fill the rest with whole
    # terms only. A term is never emitted half-written into the window.
    remaining = max_chars
    result: list[str] = []
    if context_clean:
        head = context_clean[: min(len(context_clean), max_chars - len("Terms: ") - 6)]
        result.append(head)
        remaining -= len(head) + 1
    prefix = "Terms: "
    included: list[str] = []
    budget = remaining - len(prefix)
    for term in terms:
        candidate_len = len(term) + (2 if included else 0)
        if budget - candidate_len < 0:
            break
        included.append(term)
        budget -= candidate_len
    if included:
        result.append(prefix + ", ".join(included))
    return "\n".join(result).strip()[:max_chars]
