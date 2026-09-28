from __future__ import annotations

from types import SimpleNamespace

from app.schemas.schemas import NormalizationItem
from app.services.alignment import compose_suggestions
from app.services.filly_pipeline import FillyPipeline


def _change(text, start, end, replacement, iteration, tag="$REPLACE"):
    return SimpleNamespace(
        original=text[start:end],
        replacement=replacement,
        correction=replacement,
        start=start,
        end=end,
        tag=tag,
        label=tag,
        confidence=0.87,
        iteration=iteration,
    )


def _make_pass(text, edits, iteration):
    output = text
    for edit in reversed(edits):
        output = output[: edit.start] + edit.replacement + output[edit.end :]
    return SimpleNamespace(
        iteration=iteration,
        input_text=text,
        output_text=output,
        model_invoked=True,
        input_tokens=(),
        output_tokens=(),
        labels=(),
        changes=tuple(edits),
    )


class _FakeNormalizer:
    def __init__(self, normalized_text, changes):
        self.normalized_text = normalized_text
        self.changes = changes

    def normalize_text(self, _text):
        return self.normalized_text, list(self.changes)


class _FakeGEC:
    def __init__(self, pass_edit_batches):
        self.pass_edit_batches = pass_edit_batches
        self.calls = []

    def correct_iteratively(self, text, *, iterations):
        self.calls.append((text, iterations))
        assert iterations == len(self.pass_edit_batches)
        current = text
        passes = []
        for iteration, edits in enumerate(self.pass_edit_batches, start=1):
            stage_pass = _make_pass(current, edits(current, iteration), iteration)
            passes.append(stage_pass)
            current = stage_pass.output_text
        changes = tuple(change for stage_pass in passes for change in stage_pass.changes)
        return SimpleNamespace(
            original_text=text,
            corrected_text=current,
            iterations=len(passes),
            iteration_outputs=tuple(stage_pass.output_text for stage_pass in passes),
            passes=tuple(passes),
            changes=changes,
            metadata={"test_double": True},
        )


def _empty_batch(_text, _iteration):
    return []


def test_pipeline_maps_overlapping_normalization_and_gec_edits_through_five_passes():
    original = "🙂 aq z"
    normalized = "🙂 ako z"
    normalizer = _FakeNormalizer(
        normalized,
        [NormalizationItem(word="aq", suggestion="ako", start=2, end=4)],
    )

    def first_pass(text, iteration):
        return [
            _change(text, 2, 5, "aking", iteration, "$REPLACE_AKO"),
            _change(text, 6, 7, "", iteration, "$DELETE"),
        ]

    gec = _FakeGEC([first_pass, _empty_batch, _empty_batch, _empty_batch, _empty_batch])
    result = FillyPipeline(normalizer, gec, iterations=5).analyze(original)

    assert gec.calls == [(normalized, 5)]
    assert [stage.input_text for stage in result.gec.passes] == [
        normalized,
        "🙂 aking ",
        "🙂 aking ",
        "🙂 aking ",
        "🙂 aking ",
    ]
    assert [stage.output_text for stage in result.gec.passes] == [
        "🙂 aking ",
        "🙂 aking ",
        "🙂 aking ",
        "🙂 aking ",
        "🙂 aking ",
    ]
    assert result.corrected_text == "🙂 aking "
    assert [
        (item.start, item.end, item.original, item.replacement, item.source, item.tag)
        for item in result.suggestions
    ] == [
        (2, 4, "aq", "aking", "combined", "$REPLACE_AKO"),
        (5, 6, "z", "", "gec", "$DELETE"),
    ]

    accepted_all = original
    for item in sorted(result.suggestions, key=lambda item: (item.start, item.end), reverse=True):
        assert accepted_all[item.start : item.end] == item.original
        accepted_all = accepted_all[: item.start] + item.replacement + accepted_all[item.end :]
    assert accepted_all == result.corrected_text


def test_insertions_deletions_and_length_changes_keep_original_codepoint_offsets():
    original = "🙂 aq end"
    normalized = "🙂 ako end"
    normalization = [NormalizationItem(word="aq", suggestion="ako", start=2, end=4)]

    def first_pass(text, iteration):
        return [
            _change(text, 5, 5, "'y", iteration, "$APPEND_Y"),
            _change(text, 6, 9, "", iteration, "$DELETE"),
        ]

    gec = _FakeGEC([first_pass, _empty_batch, _empty_batch, _empty_batch, _empty_batch])
    result = FillyPipeline(_FakeNormalizer(normalized, normalization), gec, iterations=5).analyze(original)

    assert result.corrected_text == "🙂 ako'y "
    assert [
        (item.start, item.end, item.original, item.replacement, item.source, item.gec_iteration)
        for item in result.suggestions
    ] == [
        (2, 4, "aq", "ako'y", "combined", 1),
        (5, 8, "end", "", "gec", 1),
    ]
    accepted_all = original
    for item in sorted(result.suggestions, key=lambda item: (item.start, item.end), reverse=True):
        accepted_all = accepted_all[: item.start] + item.replacement + accepted_all[item.end :]
    assert accepted_all == result.corrected_text


def test_adjacent_edits_compose_into_one_safe_original_surface_suggestion():
    suggestions = compose_suggestions(
        original_text="ab",
        normalized_text="xy",
        corrected_text="xy",
        normalization_changes=[
            SimpleNamespace(word="a", suggestion="x", start=0, end=1),
            SimpleNamespace(word="b", suggestion="y", start=1, end=2),
        ],
        gec_passes=(),
    )

    assert [
        (item.start, item.end, item.original, item.replacement, item.source)
        for item in suggestions
    ] == [(0, 2, "ab", "xy", "normalization")]


def test_pure_gec_insertion_at_unicode_text_end_maps_to_original_boundary():
    original = "🙂 hi"
    text = original
    passes = []
    for iteration in range(1, 6):
        edits = [_change(text, len(text), len(text), "!", iteration, "$APPEND_PUNCT")] if iteration == 1 else []
        stage_pass = _make_pass(text, edits, iteration)
        passes.append(stage_pass)
        text = stage_pass.output_text

    suggestions = compose_suggestions(original, original, text, [], passes)

    assert [
        (item.start, item.end, item.original, item.replacement, item.source)
        for item in suggestions
    ] == [(4, 4, "", "!", "gec")]


def test_gec_edit_undone_by_later_pass_does_not_change_suggestion_provenance():
    original = "aq"
    normalized = "ako"
    first = _make_pass(normalized, [_change(normalized, 0, 3, "x", 1)], 1)
    second = _make_pass(first.output_text, [_change(first.output_text, 0, 1, "ako", 2)], 2)
    passes = [first, second]
    for iteration in range(3, 6):
        passes.append(_make_pass(passes[-1].output_text, [], iteration))

    suggestions = compose_suggestions(
        original,
        normalized,
        normalized,
        [NormalizationItem(word="aq", suggestion="ako", start=0, end=2)],
        passes,
    )

    assert [
        (item.start, item.end, item.original, item.replacement, item.source, item.tag)
        for item in suggestions
    ] == [(0, 2, "aq", "ako", "normalization", "NORMALIZATION")]

