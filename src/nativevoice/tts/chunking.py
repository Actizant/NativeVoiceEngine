"""Lossless text chunking under a per-request token budget.

The reference architecture generates each text chunk independently, which is the
documented cause of cross-chunk prosody drift: pitch and pace reset at every
boundary. The budget is therefore a compromise we control deliberately - small
enough to stay inside what the model was trained on, large enough that ordinary
sentences are never split, and always reported so the session can warn about
drift instead of hiding it.

Three guarantees drive the implementation:

* **lossless**: chunks are index slices of the input, so `"".join(chunk.text)`
  reconstructs the input exactly. No character is ever dropped, added or reordered,
  because a synthesis front-end that edits text silently cannot be audited when a
  word is mispronounced.
* **no false boundaries**: decimals ("3.14"), ellipses and initialisms
  ("U.S.A.") do not end a sentence. Each would make the model start a fresh
  prosodic contour inside a number.
* **never drop the oversized case**: a sentence that cannot be split inside its
  budget is emitted whole and reported, rather than truncated mid-word.

The token counter is a cheap whitespace/punctuation estimate, not the model's
tokenizer: the reference uses a 4000-entry SentencePiece whose real counts are
unavailable here without its weights. It over- or under-counts by a factor we
cannot measure until a model exists, so the budget is treated as approximate and
`TextChunker.budget_utilisation()` exposes the estimate for logging.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

TERMINATORS = ".!?।॥"
CLOSING_MARKS = "\"'”’)]»"
SUBBOUNDARIES = ",;:—–"
ELLIPSIS = re.compile(r"\.{2,}|…+")
DECIMAL = re.compile(r"(?<=\d)[.,](?=\d)|(?<=\d),(?=\d\d\d\b)")
INITIALISM = re.compile(r"(?:[A-Za-z]\.){2,}")
# Only the forms whose dot is not a sentence end; the normaliser already expands
# most abbreviations, so this list is the safety net for text that skipped it.
ABBREVIATIONS = re.compile(
    r"\b(?:mr|mrs|ms|dr|prof|rev|st|jr|sr|vs|etc|approx|no|nos|dept|inc|ltd|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"e\.g|i\.e|u\.s|u\.k|a\.m|p\.m|fig|vol|ed|eds|trans|p|pp)\.",
    re.IGNORECASE,
)
WORD_LIKE = re.compile(r"[0-9A-Za-z\u00C0-\u024F\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF"
                       r"\u0900-\u097F\u0980-\u09FF\u0E00-\u0E7F\u3040-\u30FF\u4E00-\u9FFF]+")


class TextChunk:
    """One synthesis request's worth of text, and how full its budget is."""

    __slots__ = ("index", "text", "tokens", "starts_sentence", "is_split")

    def __init__(self, index: int, text: str, tokens: int, starts_sentence: bool, is_split: bool):
        self.index = index
        self.text = text
        self.tokens = tokens
        self.starts_sentence = starts_sentence
        self.is_split = is_split

    def __repr__(self) -> str:
        return f"TextChunk(index={self.index}, tokens={self.tokens}, text={self.text!r})"


class TextChunker:
    """Splits normalised text into budgeted, independently-generable chunks."""

    def __init__(self, max_tokens: int = 50) -> None:
        if max_tokens < 1:
            raise ConfigurationError("max_tokens must be positive")
        self.max_tokens = max_tokens

    def tokens(self, text: str) -> int:
        """Whitespace/punctuation token estimate - see module docstring for why."""
        return len(WORD_LIKE.findall(text))

    def budget_utilisation(self, chunks: Sequence[TextChunk]) -> float:
        """Mean tokens-per-chunk over `max_tokens`; >0.95 means drift-prone splits."""
        if not chunks:
            return 0.0
        return sum(chunk.tokens for chunk in chunks) / (len(chunks) * self.max_tokens)

    def sentences(self, text: str) -> tuple[str, ...]:
        """Sentences whose dot-ambiguities are not treated as boundaries."""
        protected = self._protected_offsets(text)
        pieces: list[str] = []
        start = 0
        position = 0
        length = len(text)
        while position < length:
            if text[position] in TERMINATORS and position not in protected:
                end = position + 1
                while end < length and text[end] in TERMINATORS + CLOSING_MARKS:
                    end += 1
                if end < length and not text[end].isspace():
                    position = end
                    continue
                pieces.append(text[start:end])
                start = end
                position = end
                continue
            position += 1
        if start < length:
            pieces.append(text[start:])
        return tuple(piece for piece in pieces if piece)

    def chunk(self, text: str) -> tuple[TextChunk, ...]:
        """Pack whole sentences together, splitting only what exceeds the budget.

        Consecutive sentences sharing a request is what keeps prosody continuous
        across them; a sentence that must be split loses that continuity, and the
        resulting chunk carries `is_split` so the caller can report drift risk.
        """
        units: list[tuple[str, bool, bool]] = []
        for sentence in self.sentences(text):
            for position, (piece, split) in enumerate(self._fit(sentence)):
                units.append((piece, split, position == 0))

        grouped: list[tuple[str, bool, bool]] = []
        buffer = ""
        buffer_split = False
        buffer_starts = True
        for piece, split, starts in units:
            if not piece:
                continue
            if buffer and self.tokens(buffer + piece) > self.max_tokens:
                grouped.append((buffer, buffer_split, buffer_starts))
                buffer, buffer_split, buffer_starts = piece, split, starts
                continue
            if buffer:
                buffer += piece
                buffer_split = buffer_split or split
            else:
                buffer, buffer_split, buffer_starts = piece, split, starts
        if buffer:
            grouped.append((buffer, buffer_split, buffer_starts))
        return tuple(
            TextChunk(index, piece, self.tokens(piece.strip()), starts, split)
            for index, (piece, split, starts) in enumerate(grouped)
        )

    def _fit(self, sentence: str) -> list[tuple[str, bool]]:
        """Split an oversized sentence on commas/semicolons, never on a fixed count."""
        if self.tokens(sentence) <= self.max_tokens:
            return [(sentence, False)]
        pieces: list[tuple[str, bool]] = []
        queue: list[str] = [sentence]
        while queue:
            current = queue.pop(0)
            if self.tokens(current) <= self.max_tokens:
                pieces.append((current, False))
                continue
            parts = self._split_subboundaries(current)
            if len(parts) < 2:
                pieces.append((current, True))
                continue
            queue = parts + queue
        return self._merge_small(pieces)

    def _merge_small(self, pieces: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
        """Re-join sub-pieces that a comma split left too short to sound natural."""
        out: list[tuple[str, bool]] = []
        for piece, split in pieces:
            if out and self.tokens(out[-1][0] + piece) <= self.max_tokens:
                previous, was_split = out[-1]
                out[-1] = (previous + piece, was_split or split)
            else:
                out.append((piece, split))
        return out

    @staticmethod
    def _split_subboundaries(text: str) -> list[str]:
        parts: list[str] = []
        start = 0
        for index, character in enumerate(text):
            if character in SUBBOUNDARIES and index + 1 < len(text) and text[index + 1].isspace():
                parts.append(text[start : index + 1])
                start = index + 1
        parts.append(text[start:])
        return [part for part in parts if part]

    @staticmethod
    def _protected_offsets(text: str) -> set[int]:
        protected: set[int] = set()
        for pattern in (DECIMAL, ELLIPSIS, INITIALISM, ABBREVIATIONS):
            for match in pattern.finditer(text):
                protected.update(range(match.start(), match.end()))
        return protected

    @staticmethod
    def reconstruct(chunks: Sequence[TextChunk]) -> str:
        """Exact reconstruction; equality with the input is the losslessness test."""
        return "".join(chunk.text for chunk in chunks)
from ..core.errors import ConfigurationError
