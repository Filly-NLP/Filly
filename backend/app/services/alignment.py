"""Map normalization and iterative GEC edits back to original-text suggestions.

All offsets in this module are Python Unicode code-point offsets. Suggestions
group the recorded edits by their original source spans and use the final
corrected text for each group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from app.schemas.schemas import SuggestionResponse


Span = tuple[int, int]


@dataclass(frozen=True, slots=True)
class _Edit:
    start: int
    end: int
    original: str
    replacement: str
    source: str
    tag: str = ""
    confidence: float | None = None
    iteration: int | None = None


@dataclass(frozen=True, slots=True)
class _Provenance:
    start: int
    end: int
    tag: str = ""
    confidence: float | None = None
    iteration: int | None = None


class _AlignmentError(ValueError):
    """Raised when a supplied edit trace cannot be replayed exactly."""


def _union(spans: Iterable[Span]) -> Span:
    spans = tuple(spans)
    if not spans:
        return (0, 0)
    return min(start for start, _end in spans), max(end for _start, end in spans)


class _TextAlignment:
    """Track the original source span for every current character/boundary."""

    def __init__(self, text: str):
        self.text = text
        self.characters: list[Span] = [(index, index + 1) for index in range(len(text))]
        self.boundaries: list[Span] = [(index, index) for index in range(len(text) + 1)]

    def copy(self) -> _TextAlignment:
        result = _TextAlignment("")
        result.text = self.text
        result.characters = self.characters.copy()
        result.boundaries = self.boundaries.copy()
        return result

    def source_span(self, start: int, end: int) -> Span:
        if start < 0 or end < start or end > len(self.text):
            raise _AlignmentError("edit offset is outside its stage input")
        if start == end:
            return self.boundaries[start]
        return _union((*self.characters[start:end], *self.boundaries[start : end + 1]))

    def target_span(self, source_start: int, source_end: int) -> Span:
        """Find the final output range descended from an original source span."""
        matching_characters = []
        for index, (char_start, char_end) in enumerate(self.characters):
            if char_start == char_end:
                matches = source_start <= char_start <= source_end
            elif source_start == source_end:
                matches = False
            else:
                matches = char_start < source_end and source_start < char_end
            if matches:
                matching_characters.append(index)
        if matching_characters:
            return min(matching_characters), max(matching_characters) + 1

        # A deleted source range has no descendant character. Its collapsed
        # boundary retains the full removed span so the replacement is empty.
        for index, boundary_span in enumerate(self.boundaries):
            if boundary_span == (source_start, source_end):
                return index, index
        raise _AlignmentError("final text has no aligned range for a changed source span")

    def apply(self, edits: list[_Edit]) -> list[_Provenance]:
        """Replay one non-overlapping edit batch and return mapped edit spans."""
        previous_start = -1
        previous_end = -1
        provenance: list[_Provenance] = []
        for edit in edits:
            if edit.start < previous_end or edit.start == previous_start:
                raise _AlignmentError("stage edits overlap or share an ambiguous start")
            if edit.start < 0 or edit.end < edit.start or edit.end > len(self.text):
                raise _AlignmentError("stage edit has an invalid offset")
            if self.text[edit.start : edit.end] != edit.original:
                raise _AlignmentError("stage edit surface does not match its input text")
            source_start, source_end = self.source_span(edit.start, edit.end)
            provenance.append(
                _Provenance(
                    start=source_start,
                    end=source_end,
                    tag=edit.tag,
                    confidence=edit.confidence,
                    iteration=edit.iteration,
                )
            )
            previous_start, previous_end = edit.start, edit.end

        for edit in reversed(edits):
            source_span = self.source_span(edit.start, edit.end)
            start_boundary = self.boundaries[edit.start]
            end_boundary = self.boundaries[edit.end]
            if edit.replacement:
                inserted_boundaries = (
                    [start_boundary]
                    + [source_span] * (len(edit.replacement) - 1)
                    + [end_boundary]
                )
                self.boundaries = (
                    self.boundaries[: edit.start]
                    + inserted_boundaries
                    + self.boundaries[edit.end + 1 :]
                )
                self.characters = (
                    self.characters[: edit.start]
                    + [source_span] * len(edit.replacement)
                    + self.characters[edit.end :]
                )
            else:
                collapsed = _union(
                    (
                        source_span,
                        *self.boundaries[edit.start : edit.end + 1],
                    )
                )
                self.boundaries = (
                    self.boundaries[: edit.start]
                    + [collapsed]
                    + self.boundaries[edit.end + 1 :]
                )
                self.characters = self.characters[: edit.start] + self.characters[edit.end :]
            self.text = self.text[: edit.start] + edit.replacement + self.text[edit.end :]

        if len(self.boundaries) != len(self.text) + 1 or len(self.characters) != len(self.text):
            raise _AlignmentError("internal source alignment is inconsistent")
        return provenance


def _normalization_edits(changes: Iterable[Any]) -> list[_Edit]:
    edits = []
    for change in changes:
        edits.append(
            _Edit(
                start=change.start,
                end=change.end,
                original=change.word,
                replacement=change.suggestion,
                source="normalization",
                tag="NORMALIZATION",
            )
        )
    return edits


def _gec_edits(changes: Iterable[Any]) -> list[_Edit]:
    edits = []
    for change in changes:
        edits.append(
            _Edit(
                start=change.start,
                end=change.end,
                original=change.original,
                replacement=change.replacement,
                source="gec",
                tag=change.tag,
                confidence=change.confidence,
                iteration=change.iteration,
            )
        )
    return edits


@dataclass(slots=True)
class _EditGroup:
    start: int
    end: int
    normalization: list[_Provenance]
    gec: list[_Provenance]


def _group_provenance(
    normalization: list[_Provenance], gec: list[_Provenance]
) -> list[_EditGroup]:
    events = [(edit.start, edit.end, "normalization", edit) for edit in normalization]
    events.extend((edit.start, edit.end, "gec", edit) for edit in gec)
    events.sort(key=lambda event: (event[0], event[1], event[2]))
    groups: list[_EditGroup] = []
    for start, end, source, edit in events:
        if not groups or start > groups[-1].end:
            group = _EditGroup(start=start, end=end, normalization=[], gec=[])
            groups.append(group)
        else:
            group = groups[-1]
            group.end = max(group.end, end)
        (group.normalization if source == "normalization" else group.gec).append(edit)
    return groups


def compose_suggestions(
    original_text: str,
    normalized_text: str,
    corrected_text: str,
    normalization_changes: Iterable[Any],
    gec_passes: Iterable[Any],
) -> list[SuggestionResponse]:
    """Compose verified stage edits into non-overlapping original-text edits.

    If a stage trace cannot be replayed exactly, its suggestions are omitted.
    A verified normalization trace can still produce normalization suggestions
    when a GEC trace is incomplete or inconsistent.
    """
    try:
        normalization_edits = _normalization_edits(normalization_changes)
    except (AttributeError, TypeError):
        return []

    alignment = _TextAlignment(original_text)
    try:
        normalization_provenance = alignment.apply(normalization_edits)
        if alignment.text != normalized_text:
            return []
    except _AlignmentError:
        return []
    normalized_alignment = alignment.copy()

    gec_provenance: list[_Provenance] = []
    try:
        for stage_pass in gec_passes:
            if stage_pass.input_text != alignment.text:
                raise _AlignmentError("GEC pass input does not equal the prior stage output")
            pass_edits = _gec_edits(stage_pass.changes)
            mapped = alignment.apply(pass_edits)
            if alignment.text != stage_pass.output_text:
                raise _AlignmentError("GEC pass output does not match replayed edits")
            gec_provenance.extend(mapped)
        if alignment.text != corrected_text:
            raise _AlignmentError("corrected text does not equal the final GEC pass output")
    except (AttributeError, TypeError, _AlignmentError):
        # Do not attach GEC provenance to text that the trace cannot prove.
        alignment = normalized_alignment.copy()
        gec_provenance = []
        corrected_text = normalized_text

    suggestions: list[SuggestionResponse] = []
    counters = {"normalization": 0, "gec": 0, "combined": 0}
    for group in _group_provenance(normalization_provenance, gec_provenance):
        start, end = group.start, group.end
        normalization_hits = group.normalization
        gec_hits = group.gec
        try:
            replacement_start, replacement_end = alignment.target_span(start, end)
        except _AlignmentError:
            continue
        original = original_text[start:end]
        replacement = alignment.text[replacement_start:replacement_end]
        try:
            normalized_start, normalized_end = normalized_alignment.target_span(start, end)
        except _AlignmentError:
            continue
        normalized_replacement = normalized_text[normalized_start:normalized_end]
        gec_contributed = bool(gec_hits and replacement != normalized_replacement)

        if normalization_hits and gec_contributed:
            source = "combined"
            source_tag = "GRAMMAR"
        elif normalization_hits:
            source = "normalization"
            source_tag = "NORMALIZATION"
        elif gec_contributed:
            source = "gec"
            source_tag = "GRAMMAR"
        else:
            continue

        if original == replacement:
            # A later pass may undo an earlier edit. There is then no current
            # action to offer even though the research trace records the edit.
            continue

        if gec_contributed:
            specific_tags = sorted({edit.tag for edit in gec_hits if edit.tag})
            if len(specific_tags) == 1:
                source_tag = specific_tags[0]

        confidence: float | None = None
        gec_iteration: int | None = None
        if gec_contributed:
            latest_iteration = max(
                (edit.iteration for edit in gec_hits if edit.iteration is not None),
                default=None,
            )
            latest_hits = [
                edit for edit in gec_hits if edit.iteration == latest_iteration
            ] if latest_iteration is not None else gec_hits
            confidences = [edit.confidence for edit in latest_hits if edit.confidence is not None]
            confidence = max(confidences) if confidences else None
            gec_iteration = latest_iteration

        counters[source] += 1
        suggestions.append(
            SuggestionResponse(
                id=f"{source}-{counters[source]}",
                start=start,
                end=end,
                original=original,
                replacement=replacement,
                source=source,
                tag=source_tag,
                confidence=confidence,
                gec_iteration=gec_iteration,
            )
        )

    return suggestions
