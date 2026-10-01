"""Conservative sentence segmentation that retains exact source boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .tokenizer import tokenize_with_offsets


_COMMON_ABBREVIATIONS = frozenset(
    {
        "atbp",
        "dr",
        "e.g",
        "etc",
        "gng",
        "i.e",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "jr",
        "vs",
    }
)
_INITIALISM_AT_END = re.compile(r"(?:[A-Za-z][.]){2,}$")
_CLOSING_AFTER_TERMINAL = frozenset(")]}\"'") | frozenset(map(chr, (0x00BB, 0x201D, 0x2019)))


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    """A sentence-like model unit with offsets into the original paragraph."""

    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SentenceLayout:
    """Sentence spans plus the exact ``n + 1`` gaps around and between them."""

    sentences: tuple[SentenceSpan, ...]
    gaps: tuple[str, ...]


def _period_is_terminal(text: str, index: int) -> bool:
    """Treat decimals, in-word periods, abbreviations, and initialisms cautiously."""
    previous = text[index - 1] if index else ""
    following = text[index + 1] if index + 1 < len(text) else ""

    if previous.isdigit() and following.isdigit():
        return False
    if previous.isalnum() and following.isalnum():
        return False

    prefix = text[max(0, index - 12) : index + 1]
    if _INITIALISM_AT_END.search(prefix):
        return False

    word_start = index
    while word_start > 0 and (text[word_start - 1].isalpha() or text[word_start - 1] == "."):
        word_start -= 1
    preceding_word = text[word_start:index].strip(".").casefold()
    return preceding_word not in _COMMON_ABBREVIATIONS


def _sentence_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    url_spans = tuple(
        (token.start, token.end)
        for token in tokenize_with_offsets(text).spans
        if token.text.casefold().startswith(("http://", "https://"))
    )
    url_index = 0
    sentence_start: int | None = None
    index = 0

    while index < len(text):
        while url_index < len(url_spans) and url_spans[url_index][1] <= index:
            url_index += 1
        if (
            url_index < len(url_spans)
            and url_spans[url_index][0] <= index < url_spans[url_index][1]
        ):
            if sentence_start is None:
                sentence_start = index
            url_end = url_spans[url_index][1]
            terminal = text[url_end - 1] if url_end > index else ""
            has_terminal = terminal in "?!"
            if terminal == ".":
                has_terminal = _period_is_terminal(text, url_end - 1)
            if (
                has_terminal
                and sentence_start is not None
                and (url_end == len(text) or text[url_end].isspace())
            ):
                spans.append((sentence_start, url_end))
                sentence_start = None
            index = url_end
            continue

        character = text[index]
        if character in "\r\n":
            if sentence_start is not None:
                end = index
                while end > sentence_start and text[end - 1].isspace():
                    end -= 1
                if end > sentence_start:
                    spans.append((sentence_start, end))
                sentence_start = None
            index += 1
            continue

        if character.isspace():
            index += 1
            continue

        if sentence_start is None:
            sentence_start = index

        if character in ".?!" and (character != "." or _period_is_terminal(text, index)):
            end = index + 1
            while end < len(text) and text[end] in ".?!":
                end += 1
            while end < len(text) and text[end] in _CLOSING_AFTER_TERMINAL:
                end += 1
            if end < len(text) and not text[end].isspace():
                index = end
                continue
            spans.append((sentence_start, end))
            sentence_start = None
            index = end
            continue

        index += 1

    if sentence_start is not None:
        end = len(text)
        while end > sentence_start and text[end - 1].isspace():
            end -= 1
        if end > sentence_start:
            spans.append((sentence_start, end))

    return tuple(spans)


def segment_sentences(text: str) -> SentenceLayout:
    """Split at common terminal punctuation and line breaks without losing text.

    Period handling is deliberately conservative: decimals, periods inside
    words, a short abbreviation list (including ``etc.`` and ``atbp.``), and
    short initialisms remain in their surrounding model unit. Newlines split
    nonempty units even when no terminal punctuation is present. All whitespace
    and punctuation remain represented exactly by sentence slices and gaps.
    """
    if not isinstance(text, str):
        raise TypeError("sentence segmentation input must be a string")

    spans = _sentence_spans(text)
    sentences: list[SentenceSpan] = []
    gaps: list[str] = []
    cursor = 0
    for start, end in spans:
        gaps.append(text[cursor:start])
        sentences.append(SentenceSpan(text=text[start:end], start=start, end=end))
        cursor = end
    gaps.append(text[cursor:])
    return SentenceLayout(tuple(sentences), tuple(gaps))


def reassemble_sentences(sentence_texts: tuple[str, ...] | list[str], gaps: tuple[str, ...] | list[str]) -> str:
    """Replace sentence spans while retaining every original gap verbatim."""
    if len(gaps) != len(sentence_texts) + 1:
        raise ValueError("sentence layouts require exactly one more gap than sentence")
    pieces = [gaps[0]]
    for index, sentence in enumerate(sentence_texts):
        pieces.extend((sentence, gaps[index + 1]))
    return "".join(pieces)
