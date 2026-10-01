from __future__ import annotations

import pytest

from app.services.gec_checkpoint.segmentation import (
    reassemble_sentences,
    segment_sentences,
)


@pytest.mark.parametrize(
    ("text", "sentences", "gaps"),
    [
        (
            "ako ay umuwi. kumain ako.",
            ("ako ay umuwi.", "kumain ako."),
            ("", " ", ""),
        ),
        (
            "kumain ka ba? oo.",
            ("kumain ka ba?", "oo."),
            ("", " ", ""),
        ),
        (
            "umalis siya! bumalik siya.",
            ("umalis siya!", "bumalik siya."),
            ("", " ", ""),
        ),
    ],
)
def test_terminal_punctuation_stays_with_its_sentence(text, sentences, gaps):
    layout = segment_sentences(text)

    assert tuple(sentence.text for sentence in layout.sentences) == sentences
    assert layout.gaps == gaps
    assert reassemble_sentences(sentences, layout.gaps) == text


def test_newlines_and_all_spacing_are_preserved_as_exact_gaps():
    text = "  kumain ako.\r\n\tako ay umuwi\n  umalis siya!  "
    layout = segment_sentences(text)

    assert tuple(sentence.text for sentence in layout.sentences) == (
        "kumain ako.",
        "ako ay umuwi",
        "umalis siya!",
    )
    assert layout.gaps == ("  ", "\r\n\t", "\n  ", "  ")
    assert reassemble_sentences(
        tuple(sentence.text for sentence in layout.sentences), layout.gaps
    ) == text


def test_abbreviations_and_initialisms_do_not_split_at_every_period():
    text = "etc. atbp. dr. santos. U.S. news."
    layout = segment_sentences(text)

    assert tuple(sentence.text for sentence in layout.sentences) == (
        "etc. atbp. dr. santos.",
        "U.S. news.",
    )
    assert reassemble_sentences(
        tuple(sentence.text for sentence in layout.sentences), layout.gaps
    ) == text


def test_closing_brackets_and_quotes_remain_attached_to_terminal_punctuation():
    text = '(kumusta?) "oo!" [tapos.]'
    layout = segment_sentences(text)

    assert tuple(sentence.text for sentence in layout.sentences) == (
        "(kumusta?)",
        '"oo!"',
        "[tapos.]",
    )
    assert layout.gaps == ("", " ", " ", "")
    assert reassemble_sentences(
        tuple(sentence.text for sentence in layout.sentences), layout.gaps
    ) == text


def test_text_without_sentence_content_round_trips_unchanged():
    text = " \t\r\n  "
    layout = segment_sentences(text)

    assert layout.sentences == ()
    assert reassemble_sentences((), layout.gaps) == text
