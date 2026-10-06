"""CSV loading and fail-fast schema validation for research evaluation."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INPUT_ALIASES = ("input", "source", "source_text", "text", "noisy_text", "original")
TARGET_ALIASES = ("target", "target_text", "reference", "gold", "expected")
CORRECTION_ALIASES = (
    "gec_correction",
    "correction",
    "correction_tag",
    "correction_tags",
    "gec_tag",
    "expected_gec_tag",
    "generation_tag",
    "error_type",
)
TYPE_ALIASES = (
    "unnormalized_type",
    "normalization_type",
    "noise_type",
    "informal_type",
    "category",
)


class DatasetError(ValueError):
    """Raised when a CSV cannot be interpreted without guessing."""


@dataclass(frozen=True)
class Dataset:
    path: Path
    headers: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    columns: dict[str, str | None]


def canonical_type(value: str) -> str:
    """Fold case and superficial separators while preserving other labels."""
    label = re.sub(r"\s+", " ", re.sub(r"[-_]", " ", value.strip())).casefold()
    return {
        "abbreviations": "abbreviation",
        "spelling variations": "spelling variation",
    }.get(label, label)


def _resolve_column(
    headers: list[str],
    *,
    field: str,
    aliases: tuple[str, ...],
    override: str | None,
    required: bool,
) -> str | None:
    if override is not None:
        wanted = override.strip().casefold()
        matches = [header for header in headers if header.strip().casefold() == wanted]
        if len(matches) != 1:
            raise DatasetError(
                f"--{field}-column {override!r} does not identify exactly one CSV column; "
                f"available columns: {', '.join(headers)}"
            )
        return matches[0]

    accepted = {alias.casefold() for alias in aliases}
    matches = [header for header in headers if header.strip().casefold() in accepted]
    if len(matches) > 1:
        raise DatasetError(
            f"ambiguous {field} columns: {', '.join(matches)}. "
            f"Choose one explicitly with --{field}-column."
        )
    if matches:
        return matches[0]
    if required:
        raise DatasetError(
            f"could not find a {field} column; accepted aliases: {', '.join(aliases)}. "
            f"Available columns: {', '.join(headers)}"
        )
    return None


def load_dataset(
    path: str | Path,
    *,
    input_column: str | None = None,
    target_column: str | None = None,
    correction_column: str | None = None,
    type_column: str | None = None,
) -> Dataset:
    """Read and validate every CSV record before any model is initialized."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DatasetError(f"dataset CSV does not exist or is not a file: {source}")

    try:
        stream = source.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise DatasetError(f"could not open dataset CSV {source}: {exc}") from exc

    try:
        with stream:
            reader = csv.reader(stream, strict=True)
            try:
                raw_headers = next(reader)
            except StopIteration:
                raw_headers = []
            if not raw_headers:
                raise DatasetError("dataset CSV is empty or has no header row")
            if any(not header.strip() for header in raw_headers):
                raise DatasetError("dataset CSV contains an empty column name")
            headers = list(raw_headers)
            folded = [header.strip().casefold() for header in headers]
            if len(set(folded)) != len(folded):
                raise DatasetError("dataset CSV has duplicate column names (case-insensitive)")

            columns = {
                "input": _resolve_column(
                    headers, field="input", aliases=INPUT_ALIASES,
                    override=input_column, required=True,
                ),
                "target": _resolve_column(
                    headers, field="target", aliases=TARGET_ALIASES,
                    override=target_column, required=True,
                ),
                "correction": _resolve_column(
                    headers, field="correction", aliases=CORRECTION_ALIASES,
                    override=correction_column, required=False,
                ),
                "type": _resolve_column(
                    headers, field="type", aliases=TYPE_ALIASES,
                    override=type_column, required=False,
                ),
            }

            rows: list[dict[str, Any]] = []
            for record_number, values in enumerate(reader, start=2):
                if not values:
                    raise DatasetError(f"CSV record {record_number} is a blank data row")
                if len(values) != len(headers):
                    relation = "more" if len(values) > len(headers) else "fewer"
                    raise DatasetError(
                        f"CSV record {record_number} has {relation} values than the header columns"
                    )
                record = dict(zip(headers, values))

                input_value = record[columns["input"]]  # type: ignore[index]
                target_value = record[columns["target"]]  # type: ignore[index]
                missing: list[str] = []
                if not input_value.strip():
                    missing.append("input")
                if not target_value.strip():
                    missing.append("target")
                if missing:
                    raise DatasetError(
                        f"CSV record {record_number} has missing or blank "
                        f"{', '.join(missing)} value(s)"
                    )

                raw_type = record[columns["type"]] if columns["type"] else ""
                normalized_type = canonical_type(raw_type) if raw_type.strip() else ""
                if ";" in raw_type:
                    raise DatasetError(
                        f"CSV record {record_number} has multiple normalization types. "
                        "Use the single dominant category for each sample."
                    )
                if normalized_type == "overall":
                    raise DatasetError(
                        f"CSV record {record_number} uses reserved type label 'overall'"
                    )

                # Keep the complete source record separately so aliases with
                # distinct meanings (for example generation_tag and
                # expected_gec_tag) survive prediction export unchanged.
                source_metadata = {header: record[header] for header in headers}
                rows.append(
                    {
                        "row_id": len(rows) + 1,
                        "input": input_value,
                        "target": target_value,
                        "gec_correction": (
                            record[columns["correction"]]
                            if columns["correction"] is not None else None
                        ),
                        "unnormalized_type": raw_type if raw_type.strip() else None,
                        "types": (normalized_type,) if normalized_type else (),
                        "metadata": source_metadata,
                    }
                )
            if not rows:
                raise DatasetError("dataset CSV contains no data rows")
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetError(f"could not read dataset CSV {source}: {exc}") from exc

    return Dataset(source, tuple(headers), tuple(rows), columns)
