"""Concrete FILLY/Balarila labels and deterministic correction replay.

Abstract GEG labels remain row metadata.  Classifier labels for operations
whose result depends on the source encode the required concrete details after
``@`` (merge width, split output, hyphen position, or target verb surface).
Unknown abstract operations fail closed; no label is guessed as KEEP.
"""

from __future__ import annotations

import re
from urllib.parse import quote, unquote_to_bytes

from .balarila_registry import TABLE3_TARGET_TAGS
from .tokenizer import TokenizedText, render_tokens, tokenize_text, tokenize_with_offsets

KEEP = "$KEEP"
DELETE = "$DELETE"

_FIXED_REPLACEMENTS = {
    "$REPLACE_nang": "nang", "$REPLACE_ng": "ng", "$REPLACE_daw": "daw",
    "$REPLACE_din": "din", "$REPLACE_dito": "dito", "$REPLACE_diyan": "diyan",
    "$REPLACE_doon": "doon", "$REPLACE_raw": "raw", "$REPLACE_rin": "rin",
    "$REPLACE_roon": "roon", "$REPLACE_rito": "rito", "$REPLACE_riyan": "riyan",
    "$REPLACE_nila": "nila", "$REPLACE_niya": "niya", "$REPLACE_sila": "sila",
    "$REPLACE_siya": "siya",
}
_PUNCTUATION = {"PERIOD": ".", "EMARK": "!", "QMARK": "?"}
_PUNCTUATION_REPLACEMENTS = {value: key for key, value in _PUNCTUATION.items()}
_PUNCTUATION_TOKENS = frozenset(".,!?;:%")
_VERB_TAGS = tuple(sorted(TABLE3_TARGET_TAGS, key=len, reverse=True))


class UnsupportedEditLabel(ValueError):
    """A label cannot be deterministically executed against its source."""


def _encoded_tokens(tokens: tuple[str, ...] | list[str]) -> str:
    if not tokens or any(not token for token in tokens):
        raise ValueError("a concrete transform payload must contain non-empty tokens")
    return quote(" ".join(tokens), safe="-_.~")


def encode_concrete_label(kind: str, payload_tokens: tuple[str, ...] | list[str]) -> str:
    """Encode concrete REPLACE/APPEND surfaces with a stable URL-safe payload."""
    if kind not in {"REPLACE", "APPEND"}:
        raise ValueError("kind must be REPLACE or APPEND")
    payload = _encoded_tokens(list(payload_tokens))
    # Never let the manuscript's generic placeholder become a model class.
    if kind == "APPEND" and payload == "t1":
        payload = "%74%31"
    return f"${kind}_{payload}"


def _decode_tokens(encoded: str, *, label: str) -> tuple[str, ...]:
    if not encoded:
        raise UnsupportedEditLabel(f"{label!r} has no concrete payload")
    text = _decode_url_component(encoded, label=label)
    tokens = tokenize_text(text)
    if not tokens:
        raise UnsupportedEditLabel(f"{label!r} decodes to an empty payload")
    return tokens


def _decode_url_component(encoded: str, *, label: str) -> str:
    if re.search(r"%(?![0-9A-Fa-f]{2})", encoded):
        raise UnsupportedEditLabel(f"{label!r} has a malformed percent-encoded payload")
    try:
        return unquote_to_bytes(encoded).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise UnsupportedEditLabel(f"{label!r} has an invalid UTF-8 payload") from exc


def _case_capital(token: str) -> str:
    return token[:1].upper() + token[1:]


def _replacement_gap_payload(label: str, output_tokens: tuple[str, ...]) -> tuple[str, ...] | None:
    marker = "@gaps:"
    if marker not in label:
        if "@" in label:
            raise UnsupportedEditLabel(f"{label!r} has an unknown replacement parameter")
        return None
    if label.count(marker) != 1:
        raise UnsupportedEditLabel(f"{label!r} has an invalid replacement-gap payload")
    encoded = label.split(marker, 1)[1]
    gaps = tuple(_decode_url_component(part, label=label) for part in encoded.split(","))
    if len(gaps) != len(output_tokens) + 1 or any(gap and not gap.isspace() for gap in gaps):
        raise UnsupportedEditLabel(f"{label!r} has an invalid replacement-gap payload")
    return gaps


def _punctuation_gap_payload(label: str) -> tuple[str, str] | None:
    marker = "@gap:"
    if marker not in label:
        if "@" in label:
            raise UnsupportedEditLabel(f"{label!r} has an unknown punctuation-add parameter")
        return None
    if label.count(marker) != 1:
        raise UnsupportedEditLabel(f"{label!r} has an invalid punctuation-gap payload")
    encoded = label.split(marker, 1)[1]
    before, separator, after = encoded.partition(",")
    if not separator:
        raise UnsupportedEditLabel(f"{label!r} has an invalid punctuation-gap payload")
    gap_before = _decode_url_component(before, label=label)
    gap_after = _decode_url_component(after, label=label)
    if any(char and not char.isspace() for char in (gap_before, gap_after)):
        raise UnsupportedEditLabel(f"{label!r} punctuation-gap payload is not whitespace")
    return gap_before, gap_after


def _is_punctuation(token: str) -> bool:
    return len(token) == 1 and token in _PUNCTUATION_TOKENS


def derive_label_for_output(source_token: str, output_tokens: tuple[str, ...]) -> str:
    """Choose identity, exact case transform, fixed label, then fallback.

    Pair-level structural transforms and metadata-conditioned morphology are
    selected in :mod:`align`, where adjacent source tokens and row metadata are
    available.
    """
    if output_tokens == (source_token,):
        return KEEP
    if not output_tokens:
        return DELETE
    if len(output_tokens) == 1:
        target = output_tokens[0]
        if target == _case_capital(source_token) and target != source_token:
            return "$TRANSFORM_CASE_CAPITAL"
        if target == source_token.lower() and target != source_token:
            return "$TRANSFORM_CASE_LOWER"
        for label, surface in _FIXED_REPLACEMENTS.items():
            if target == surface:
                return label
        if _is_punctuation(source_token) and target in _PUNCTUATION_REPLACEMENTS:
            return f"$CHANGE_PUNC_{_PUNCTUATION_REPLACEMENTS[target]}"
        if "-" in target and target.replace("-", "") == source_token:
            position = target.index("-")
            return f"$TRANSFORM_INSERT_HYPHEN@{position}"
        if "-" in source_token and target == source_token.replace("-", ""):
            return "$MERGE_HYPHEN@0"
    if output_tokens and output_tokens[0] == source_token and len(output_tokens) > 1:
        suffix = output_tokens[1:]
        if len(suffix) == 1 and suffix[0] in _PUNCTUATION_REPLACEMENTS:
            return f"$ADD_PUNC_{_PUNCTUATION_REPLACEMENTS[suffix[0]]}"
        if suffix == ("-",):
            return "$TRANSFORM_INSERT_HYPHEN@-1"
        return encode_concrete_label("APPEND", suffix)
    if len(output_tokens) > 1 and "-" in source_token:
        if tuple(part for part in re.split(r"[-‐‑–—]", source_token) if part) == output_tokens:
            return "$TRANSFORM_SPLIT_HYPHEN"
    return encode_concrete_label("REPLACE", output_tokens)


def _parse_int(value: str, *, label: str) -> int:
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value) is None:
        raise UnsupportedEditLabel(f"{label!r} has an invalid integer parameter")
    return int(value)


def _apply_single(source_token: str, label: str) -> tuple[str, ...]:
    if label == KEEP:
        return (source_token,)
    if label == DELETE:
        return ()
    if label == "$APPEND_t1":
        raise UnsupportedEditLabel("$APPEND_t1 is abstract; use a concrete encoded append label")
    if label in _FIXED_REPLACEMENTS:
        return (_FIXED_REPLACEMENTS[label],)
    if label == "$TRANSFORM_CASE_CAPITAL":
        return (_case_capital(source_token),)
    if label == "$TRANSFORM_CASE_LOWER":
        return (source_token.lower(),)
    if label == "$TRANSFORM_SPLIT_HYPHEN":
        parts = tuple(part for part in re.split(r"[-‐‑–—]", source_token) if part)
        if len(parts) < 2:
            raise UnsupportedEditLabel(f"{label} requires a hyphenated source token")
        return parts
    if label.startswith("$TRANSFORM_INSERT_HYPHEN@"):
        if label.count("@") != 1:
            raise UnsupportedEditLabel(f"{label!r} has an invalid integer parameter")
        position = _parse_int(label.rsplit("@", 1)[1], label=label)
        if position == -1:
            return (source_token, "-")
        if position <= 0 or position >= len(source_token):
            raise UnsupportedEditLabel(f"{label} requires an interior insertion position")
        return (source_token[:position] + "-" + source_token[position:],)
    if label.startswith("$TRANSFORM_SPLIT_SPACE@"):
        output = _decode_tokens(label.split("@", 1)[1], label=label)
        if len(output) < 2 or "".join(output) != source_token:
            raise UnsupportedEditLabel(f"{label} output must split the source token exactly")
        return output
    if label.startswith("$MERGE_HYPHEN@") or label.startswith("$MERGE_SPACE@"):
        if label.count("@") != 1:
            raise UnsupportedEditLabel(f"{label!r} has an invalid merge parameter")
        width = _parse_int(label.rsplit("@", 1)[1], label=label)
        if label.startswith("$MERGE_HYPHEN@") and width == 0:
            if "-" not in source_token:
                raise UnsupportedEditLabel(f"{label} requires an internally hyphenated token")
            return (source_token.replace("-", ""),)
        if width < 2:
            raise UnsupportedEditLabel(f"{label} has an invalid merge width")
        raise UnsupportedEditLabel(f"{label} requires neighboring source tokens")
    for tag in _VERB_TAGS:
        prefix = f"{tag}@"
        if label.startswith(prefix):
            output = _decode_tokens(label[len(prefix):], label=label)
            if len(output) != 1:
                raise UnsupportedEditLabel(f"{label} must encode one concrete verb surface")
            return output
    if label.startswith("$ADD_PUNC_"):
        name = label.removeprefix("$ADD_PUNC_").split("@", 1)[0]
        if name not in _PUNCTUATION:
            raise UnsupportedEditLabel(f"unknown punctuation label {label!r}")
        _punctuation_gap_payload(label)
        return (source_token, _PUNCTUATION[name])
    if label.startswith("$CHANGE_PUNC_"):
        name = label.removeprefix("$CHANGE_PUNC_")
        if name not in _PUNCTUATION or not _is_punctuation(source_token):
            raise UnsupportedEditLabel(f"{label} requires a recognized punctuation token")
        return (_PUNCTUATION[name],)
    if label.startswith("$REPLACE_"):
        payload = label.removeprefix("$REPLACE_")
        if "@gaps:" in payload:
            payload = payload.split("@gaps:", 1)[0]
        output = _decode_tokens(payload, label=label)
        _replacement_gap_payload(label, output)
        return output
    if label.startswith("$APPEND_"):
        suffix = _decode_tokens(label.removeprefix("$APPEND_"), label=label)
        return (source_token, *suffix)
    raise UnsupportedEditLabel(f"unknown or non-executable edit label {label!r}")


def apply_label(source_token: str, label: str) -> tuple[str, ...]:
    """Replay a label that does not require neighboring source tokens."""
    return _apply_single(source_token, label)


def validate_label_form(label: str) -> None:
    """Validate that a vocabulary label has a well-formed executable form.

    This checks class syntax and encoded payload shape only. Preconditions
    involving neighboring tokens or a particular source surface remain replay
    time checks.
    """
    if not isinstance(label, str) or not label:
        raise UnsupportedEditLabel("an edit label must be a non-empty string")
    if label in {KEEP, DELETE, "$TRANSFORM_CASE_CAPITAL", "$TRANSFORM_CASE_LOWER",
                 "$TRANSFORM_SPLIT_HYPHEN", *_FIXED_REPLACEMENTS}:
        return
    if label.startswith("$TRANSFORM_INSERT_HYPHEN@"):
        if label.count("@") != 1:
            raise UnsupportedEditLabel(f"{label!r} has an invalid integer parameter")
        position = _parse_int(label.rsplit("@", 1)[1], label=label)
        if position != -1 and position <= 0:
            raise UnsupportedEditLabel(f"{label!r} has an invalid hyphen insertion position")
        return
    if label.startswith("$TRANSFORM_SPLIT_SPACE@"):
        encoded = label.split("@", 1)[1]
        if "@" in encoded:
            raise UnsupportedEditLabel(f"{label!r} has a malformed split payload")
        output = _decode_tokens(encoded, label=label)
        if len(output) < 2:
            raise UnsupportedEditLabel(f"{label!r} must encode at least two split tokens")
        return
    if label.startswith("$MERGE_HYPHEN@") or label.startswith("$MERGE_SPACE@"):
        if label.count("@") != 1:
            raise UnsupportedEditLabel(f"{label!r} has an invalid merge parameter")
        width = _parse_int(label.rsplit("@", 1)[1], label=label)
        if label.startswith("$MERGE_HYPHEN@"):
            if width != 0 and width < 2:
                raise UnsupportedEditLabel(f"{label!r} has an invalid merge width")
        elif width < 2:
            raise UnsupportedEditLabel(f"{label!r} has an invalid merge width")
        return
    if any(label.startswith(f"{tag}@") for tag in _VERB_TAGS):
        encoded = label.split("@", 1)[1]
        if "@" in encoded:
            raise UnsupportedEditLabel(f"{label!r} has a malformed verb payload")
        if len(_decode_tokens(encoded, label=label)) != 1:
            raise UnsupportedEditLabel(f"{label!r} must encode one concrete verb surface")
        return
    if label.startswith("$ADD_PUNC_"):
        name = label.removeprefix("$ADD_PUNC_").split("@", 1)[0]
        if name not in _PUNCTUATION:
            raise UnsupportedEditLabel(f"unknown punctuation label {label!r}")
        _punctuation_gap_payload(label)
        return
    if label.startswith("$CHANGE_PUNC_"):
        name = label.removeprefix("$CHANGE_PUNC_")
        if name not in _PUNCTUATION:
            raise UnsupportedEditLabel(f"unknown punctuation label {label!r}")
        return
    if label.startswith("$REPLACE_"):
        payload = label.removeprefix("$REPLACE_")
        if "@gaps:" in payload:
            encoded, _gap_payload = payload.split("@gaps:", 1)
            if "@" in encoded:
                raise UnsupportedEditLabel(f"{label!r} has a malformed replacement payload")
            payload = encoded
        elif "@" in payload:
            raise UnsupportedEditLabel(f"{label!r} has an unknown replacement parameter")
        output = _decode_tokens(payload, label=label)
        _replacement_gap_payload(label, output)
        return
    if label.startswith("$APPEND_"):
        encoded = label.removeprefix("$APPEND_")
        if "@" in encoded:
            raise UnsupportedEditLabel(f"{label!r} has a malformed append payload")
        _decode_tokens(encoded, label=label)
        if label == "$APPEND_t1":
            raise UnsupportedEditLabel("$APPEND_t1 is abstract; use a concrete encoded append label")
        return
    raise UnsupportedEditLabel(f"unknown or non-executable edit label {label!r}")


def _label_segments(
    source_tokens: tuple[str, ...], labels: tuple[str, ...]
) -> list[tuple[int, int, tuple[str, ...], str]]:
    """Return source-indexed label operations for token and raw-text replay."""
    if len(source_tokens) != len(labels):
        raise ValueError(
            f"expected one label per source token, got {len(source_tokens)} tokens and {len(labels)} labels"
        )
    segments: list[tuple[int, int, tuple[str, ...], str]] = []
    index = 0
    while index < len(source_tokens):
        token = source_tokens[index]
        label = labels[index]
        if label.startswith("$MERGE_HYPHEN@") or label.startswith("$MERGE_SPACE@"):
            if label.count("@") != 1:
                raise UnsupportedEditLabel(f"{label!r} has an invalid merge parameter")
            width = _parse_int(label.rsplit("@", 1)[1], label=label)
            if width == 0 and label.startswith("$MERGE_HYPHEN@"):
                output_tokens = _apply_single(token, label)
                segments.append((index, index + 1, output_tokens, "replace"))
                index += 1
                continue
            if width < 2 or index + width > len(source_tokens):
                raise UnsupportedEditLabel(f"{label} merge width is outside the source token sequence")
            if any(labels[index + offset] != DELETE for offset in range(1, width)):
                raise UnsupportedEditLabel(f"{label} requires DELETE labels for consumed neighbor tokens")
            parts = source_tokens[index:index + width]
            joiner = "-" if label.startswith("$MERGE_HYPHEN@") else ""
            if any(not part or not any(char.isalnum() for char in part) for part in parts):
                raise UnsupportedEditLabel(f"{label} may merge only non-punctuation source tokens")
            segments.append((index, index + width, (joiner.join(parts),), "replace"))
            index += width
            continue
        if label == DELETE:
            start = index
            while index < len(labels) and labels[index] == DELETE:
                index += 1
            segments.append((start, index, (), "delete"))
            continue
        output_tokens = _apply_single(token, label)
        if output_tokens == (token,):
            kind = "keep"
        elif (
            output_tokens
            and output_tokens[0] == token
            and len(output_tokens) > 1
            and (label.startswith(("$APPEND_", "$ADD_PUNC_"))
                 or label == "$TRANSFORM_INSERT_HYPHEN@-1")
        ):
            kind = (
                "append_punctuation_gap"
                if label.startswith("$ADD_PUNC_") and _punctuation_gap_payload(label) is not None
                else "append"
            )
        else:
            kind = "replace"
        if label.startswith("$REPLACE_") and _replacement_gap_payload(label, output_tokens) is not None:
            kind = "replace_with_gaps"
        segments.append((index, index + 1, output_tokens, kind))
        index += 1
    return segments


def apply_labels(source_tokens: tuple[str, ...], labels: tuple[str, ...]) -> tuple[str, ...]:
    """Apply one label per source token, including bounded merge operations."""
    output: list[str] = []
    for _start, _end, segment_tokens, _kind in _label_segments(source_tokens, labels):
        output.extend(segment_tokens)
    return tuple(output)


def reconstruct_text(source_tokens: tuple[str, ...], labels: tuple[str, ...]) -> str:
    """Render token-only output canonically for explicitly canonical contexts.

    Raw sentence replay must use :func:`apply_labels_to_text` so unchanged
    source gaps are retained.
    """
    return render_tokens(apply_labels(source_tokens, labels))


def apply_labels_to_text(source_text: str, labels: tuple[str, ...] | list[str]) -> str:
    """Apply labels to raw text while copying every untouched source slice.

    Token payloads introduced by edits use ``render_tokens`` only inside the
    edited span. Existing source gaps remain byte-for-codepoint identical.
    Deletions consume the separator before a deleted middle/end span, or the
    following separator for a deleted prefix. Insertions retain the original
    gap before the next source token. A leading insertion has no correction-
    token slot in the current model and is rejected explicitly.
    """
    _tokenized, patches = _raw_text_patches(source_text, tuple(labels))
    result = source_text
    for start, end, replacement, _token_start, _token_end, _label in reversed(patches):
        result = result[:start] + replacement + result[end:]
    return result


def _raw_text_patches(
    source_text: str, labels: tuple[str, ...]
) -> tuple[TokenizedText, tuple[tuple[int, int, str, int, int, str], ...]]:
    """Build the exact source patches used by replay and inference edit records.

    Each patch carries its affected source-token range and leading classifier
    label so callers can report the same atomic edit that replay applies.
    """
    tokenized = tokenize_with_offsets(source_text)
    source_tokens = tokenized.tokens
    segments = _label_segments(source_tokens, labels)
    if not source_tokens:
        return tokenized, ()

    patches: list[tuple[int, int, str, int, int, str]] = []
    suppressed_deletions: set[int] = set()

    def extend_through_following_deletions(next_index: int) -> tuple[int, int]:
        while next_index < len(source_tokens) and labels[next_index] == DELETE:
            suppressed_deletions.add(next_index)
            next_index += 1
        end = (
            tokenized.spans[next_index].start
            if next_index < len(source_tokens)
            else len(source_text)
        )
        return end, next_index

    for i1, i2, output_tokens, kind in segments:
        if kind == "keep":
            continue
        spans = tokenized.spans
        if kind == "append":
            inserted = output_tokens[1:]
            left_token = source_tokens[i1]
            rendered_left = render_tokens((left_token,))
            rendered_pair = render_tokens((left_token, *inserted))
            if not rendered_pair.startswith(rendered_left):
                raise UnsupportedEditLabel(
                    "appended surface cannot be separated from its source token deterministically"
                )
            boundary_text = rendered_pair[len(rendered_left):]
            patches.append((spans[i1].end, spans[i1].end, boundary_text, i1, i2, labels[i1]))
            continue

        if kind == "append_punctuation_gap":
            gaps = _punctuation_gap_payload(labels[i1])
            if gaps is None or len(output_tokens) != 2:
                raise UnsupportedEditLabel("concrete punctuation-gap label has an invalid operation shape")
            gap_before, gap_after = gaps
            start = spans[i1].end
            end, affected_end = extend_through_following_deletions(i2)
            patches.append((start, end, gap_before + output_tokens[-1] + gap_after,
                            i1, affected_end, labels[i1]))
            continue

        if kind == "replace_with_gaps":
            if i1 != 0 or i2 != 1:
                raise UnsupportedEditLabel("replacement-gap labels are supported only on the first token slot")
            gaps = _replacement_gap_payload(labels[i1], output_tokens)
            if gaps is None:
                raise UnsupportedEditLabel("concrete replacement-gap label has no gap payload")
            start = 0
            end, affected_end = extend_through_following_deletions(i2)
            replacement = "".join(
                gaps[offset] + token for offset, token in enumerate(output_tokens)
            ) + gaps[-1]
            patches.append((start, end, replacement, i1, affected_end, labels[i1]))
            continue

        if kind == "delete":
            if all(index in suppressed_deletions for index in range(i1, i2)):
                continue
            if any(index in suppressed_deletions for index in range(i1, i2)):
                raise UnsupportedEditLabel("a raw gap edit only partially covers a deletion run")
            if i1 == 0:
                start = spans[0].start
                end = spans[i2].start if i2 < len(spans) else spans[i2 - 1].end
            else:
                start = spans[i1 - 1].end
                end = spans[i2 - 1].end
            patches.append((start, end, "", i1, i2, labels[i1]))
            continue

        start = spans[i1].start
        end = spans[i2 - 1].end
        replacement = render_tokens(output_tokens)
        patches.append((start, end, replacement, i1, i2, labels[i1]))

    ordered = sorted(patches, key=lambda patch: (patch[0], patch[1]))
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1] or (
            previous[0] == previous[1] == current[0] == current[1]
        ):
            raise UnsupportedEditLabel(
                f"overlapping raw-text edits at source spans {previous[:2]} and {current[:2]}"
            )
    return tokenized, tuple(ordered)
