"""Correction-level tokenization with Unicode punctuation and source offsets.

``tokenize_with_offsets`` retains every inter-token boundary and its original
character offsets for audits or span-aware consumers.  The model-facing
``tokenize_text`` returns only token text; ``render_tokens`` uses a documented
canonical news-style join policy for label replay.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_TOKEN_PATTERN = re.compile(
    r"https?://[^\s<>\[\]{}\"“”]+"
    r"|(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,/]\d+)+)"
    r"|(?:\w+(?:/\w+)+)"
    r"|(?:\w+(?:[’'][\w]+)*(?:[-‐‑]\w+(?:[’'][\w]+)*)*)"
    r"|[^\w\s]",
    re.UNICODE | re.IGNORECASE,
)
_CLOSING = frozenset(".,!?;:%)]}»”’\"'")
_OPENING = frozenset("([{«“‘\"'")
_HYPHENS = frozenset("-‐‑")
_DASHES = frozenset("–—")
_TIGHT_BOTH = frozenset("/\\")


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """A token and its exact source span, including the boundary before it."""

    text: str
    start: int
    end: int
    gap_before: str


@dataclass(frozen=True, slots=True)
class TokenizedText:
    """Lossless tokenization of a source string."""

    original: str
    spans: tuple[TokenSpan, ...]
    leading_whitespace: str
    trailing_whitespace: str

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(span.text for span in self.spans)

    def reconstruct(self) -> str:
        if not self.spans:
            return self.original
        return "".join(span.gap_before + span.text for span in self.spans) + self.trailing_whitespace


def tokenize_with_offsets(text: str) -> TokenizedText:
    """Return tokens with exact offsets and whitespace boundaries.

    Numeric decimals, grouped numbers, fractions, URLs, apostrophized words,
    Filipino hyphenated forms, and alphanumeric date/range forms remain intact.
    Em/en dashes surrounded by whitespace remain separate punctuation tokens.
    """
    matches = tuple(_TOKEN_PATTERN.finditer(text))
    spans: list[TokenSpan] = []
    cursor = 0
    for match in matches:
        gap = text[cursor:match.start()]
        spans.append(TokenSpan(match.group(0), match.start(), match.end(), gap))
        cursor = match.end()
    if not matches:
        return TokenizedText(text, (), text, "")
    leading = text[:matches[0].start()]
    trailing = text[matches[-1].end():]
    return TokenizedText(text, tuple(spans), leading, trailing)


def tokenize_text(text: str) -> tuple[str, ...]:
    """Tokenize into correction words and punctuation without losing offsets upstream."""
    return tokenize_with_offsets(text).tokens


def _is_open_quote(token: str, quote_state: dict[str, bool]) -> bool:
    if token in {"“", "‘", "«"}:
        return True
    if token in {"”", "’", "»"}:
        return False
    if token in {'"', "'"}:
        quote_state[token] = not quote_state.get(token, False)
        return quote_state[token]
    return False


def render_tokens(tokens: Iterable[str]) -> str:
    """Render tokens with tight punctuation and quote-aware sentence spacing.

    Paired quotes and brackets hug their contents; closing punctuation stays
    with the preceding token; slash and hyphen marks stay tight; em/en dashes
    separate clauses with a space unless embedded in an alphanumeric token.
    This gives stable replay for decimals, headlines, quotations, and dates.
    """
    rendered = ""
    previous = ""
    previous_was_quote_opener = False
    quote_state: dict[str, bool] = {}
    for raw_token in tokens:
        token = str(raw_token)
        if not token:
            continue
        is_open_quote = _is_open_quote(token, quote_state)
        tight = (
            not rendered
            or token in _CLOSING
            or token in _HYPHENS
            or token in _TIGHT_BOTH
            or is_open_quote
            or previous in _OPENING - frozenset({'"', "'"})
            or previous_was_quote_opener
            or previous in _HYPHENS
            or previous in _TIGHT_BOTH
        )
        if token in _DASHES:
            # A whitespace-delimited dash is editorial punctuation. Numeric
            # ranges are normally retained as one token by the lexer.
            tight = False
        rendered += ("" if tight else " ") + token
        previous = token
        previous_was_quote_opener = is_open_quote
    return rendered


def canonicalize_text(text: str) -> str:
    """Return the canonical rendering used by exact-replay admission checks."""
    return render_tokens(tokenize_text(text))
