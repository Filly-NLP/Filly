"""Build deterministic FILLY Filipino normalizer artifacts from paired data."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = PROJECT_ROOT / "backend" / "normalization" / "data" / "train_pairs.csv"
DEFAULT_BASE_VOCABULARY = (
    PROJECT_ROOT / "backend" / "normalization" / "data" / "base_vocabulary.txt"
)
DEFAULT_CURATED_SOURCE = (
    PROJECT_ROOT / "backend" / "normalization" / "data" / "curated_colloquial_source.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "backend" / "artifacts" / "normalizer"

# This workbook is supplied source data. The runtime and artifact builder use
# the checked-in CSV extraction so neither needs to open Excel files.
CURATED_SOURCE_WORKBOOK = "N-GRAM_DATASET.xlsx"
CURATED_SOURCE_WORKBOOK_SHA256 = (
    "0dbda0dd4f143b2a29525319db9439de4f459aa72dcb0d024be9940abd34b727"
)

# This provenance identifies the supplied archive from which the committed
# training pairs and independent Tagalog word list were extracted.
SOURCE_ARCHIVE = "efficient-spelling-normalization-filipino-main (1).zip"
SOURCE_ARCHIVE_SHA256 = "f001e49b9e7854925068774f0c600e1a83b9d00dc895a49377e648041e494b7c"
TRAINING_MEMBER = "efficient-spelling-normalization-filipino-main/data/train_words.csv"
VOCABULARY_MEMBER = (
    "efficient-spelling-normalization-filipino-main/TagalogStemmerPython/output/with_info.txt"
)
VOCABULARY_LICENSE = "backend/normalization/data/LICENSE.TagalogStemmer.txt"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _curated_form_key(value: str) -> str:
    """Canonicalize a whole-form key while retaining punctuation."""
    return " ".join(value.strip().split()).casefold()


def build_curated_mappings(
    source_path: Path,
) -> tuple[bytes, dict[str, object]]:
    """Build whole-form rules from the source-sheet CSV without choosing conflicts."""
    try:
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            expected_headers = {"ID", "Informal", "Normalized", "Category"}
            if reader.fieldnames is None or not expected_headers.issubset(reader.fieldnames):
                raise ValueError(
                    f"Curated source must contain columns {sorted(expected_headers)}: {source_path}"
                )
            source_rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"Could not read curated source {source_path}: {exc}") from exc

    grouped: dict[str, dict[str, list[tuple[str, str]]]] = {}
    slang_row_count = 0
    for row_number, row in enumerate(source_rows, start=2):
        if str(row.get("Category", "")).strip().casefold() != "slang":
            continue
        slang_row_count += 1
        source = str(row.get("Informal", "")).strip()
        target = str(row.get("Normalized", "")).strip()
        source_id = str(row.get("ID", "")).strip()
        source_key = _curated_form_key(source)
        target_key = _curated_form_key(target)
        if not source_key or not target_key or not source_id:
            raise ValueError(
                f"Empty ID, Informal, or Normalized value in curated source row {row_number}"
            )
        grouped.setdefault(source_key, {}).setdefault(target_key, []).append((source_id, target))

    mappings: dict[str, dict[str, object]] = {}
    ambiguous: dict[str, list[dict[str, object]]] = {}
    duplicate_mapping_count = 0
    duplicate_source_row_count = 0
    for source_key in sorted(grouped):
        targets = grouped[source_key]
        if len(targets) > 1:
            ambiguous[source_key] = [
                {
                    "normalized": records[0][1],
                    "source_ids": [source_id for source_id, _target in records],
                }
                for _target_key, records in sorted(targets.items())
            ]
            continue

        records = next(iter(targets.values()))
        if len(records) > 1:
            duplicate_mapping_count += 1
            duplicate_source_row_count += len(records) - 1
        mappings[source_key] = {
            "normalized": records[0][1],
            "source_id": records[0][0],
            "source_ids": [source_id for source_id, _target in records],
            "category": "slang",
        }

    if ambiguous:
        conflict_summary = "; ".join(
            f"{source}: "
            + ", ".join(
                f"{row['normalized']} (IDs {', '.join(row['source_ids'])})"
                for row in targets
            )
            for source, targets in sorted(ambiguous.items())
        )
        warnings.warn(
            "Ambiguous curated whole-form mappings are excluded from the direct rules: "
            + conflict_summary,
            RuntimeWarning,
            stacklevel=2,
        )

    payload = {
        "schema_version": 1,
        "mapping_type": "curated_whole_form_normalization_rules",
        "mappings": mappings,
        "ambiguous": ambiguous,
    }
    content = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    stats: dict[str, object] = {
        "source_row_count": len(source_rows),
        "populated_mapping_row_count": sum(
            any(str(row.get(field, "")).strip() for field in ("Informal", "Normalized", "Category"))
            for row in source_rows
        ),
        "slang_row_count": slang_row_count,
        "unique_slang_input_count": len(grouped),
        "accepted_mapping_count": len(mappings),
        "duplicate_mapping_count": duplicate_mapping_count,
        "duplicate_source_row_count": duplicate_source_row_count,
        "ambiguous_mapping_count": len(ambiguous),
        "ambiguous_source_row_count": sum(
            len(target["source_ids"])
            for targets in ambiguous.values()
            for target in targets
        ),
        "excluded_ambiguous_input_count": len(ambiguous),
        "source_path": source_path,
        "source_sha256": sha256_bytes(source_path.read_bytes()),
        "artifact_sha256": sha256_bytes(content),
    }
    return content, stats


def _read_csv_pairs(path: Path) -> list[tuple[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"Could not read paired training data {path}: {exc}") from exc

    if not rows:
        raise ValueError(f"Paired training data is empty: {path}")
    if len(rows[0]) == 2 and rows[0][0].strip().lower() in {"input", "source", "wrong"}:
        rows = rows[1:]

    pairs: list[tuple[str, str]] = []
    for line_number, row in enumerate(rows, start=1):
        if len(row) != 2:
            raise ValueError(f"Expected two CSV columns in {path} at row {line_number}")
        source, target = (field.strip() for field in row)
        if not source or not target:
            raise ValueError(f"Empty source or target in {path} at row {line_number}")
        pairs.append((source.lower(), target.lower()))
    if not pairs:
        raise ValueError(f"Paired training data has no usable rows: {path}")
    return pairs


def _read_base_vocabulary(path: Path) -> set[str]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # The source Tagalog stemmer writes its word list as Latin-1.
        text = raw.decode("latin-1")
    words: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = ast.literal_eval(line)
            word = record["word"]
        except (SyntaxError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Invalid Tagalog vocabulary record in {path} at line {line_number}") from exc
        if not isinstance(word, str) or not word.strip():
            raise ValueError(f"Invalid word in {path} at line {line_number}")
        words.add(word.strip().lower())
    return words


def collect_rules(source: str, target: str, ngram_size: int) -> dict[str, list[str]]:
    """Reproduce ``utils.collect_rules`` from the supplied research archive."""
    rules: dict[str, list[str]] = {}
    source_pointer = 0
    target_pointer = 0
    while source_pointer < len(source) and target_pointer < len(target):
        source_ngram = source[source_pointer : source_pointer + ngram_size]
        target_ngram = target[target_pointer : target_pointer + ngram_size]
        if source_ngram == target_ngram:
            source_pointer += ngram_size
            target_pointer += ngram_size
        else:
            source_pointer += 1
            target_pointer += ngram_size
        rules.setdefault(source_ngram, []).append(target_ngram)
    return rules


def induce_rules(
    pairs: Iterable[tuple[str, str]], max_ngram: int = 2
) -> dict[str, dict[str, float]]:
    """Induce n-gram replacement distributions from training pairs only."""
    observations: dict[str, Counter[str]] = {}
    for source, target in pairs:
        for ngram_size in range(2, max_ngram + 1):
            for source_ngram, target_ngrams in collect_rules(source, target, ngram_size).items():
                counter = observations.setdefault(source_ngram, Counter())
                counter.update(target_ngrams)

    # The reference algorithm includes each source n-gram as an unchanged
    # candidate before normalizing replacement counts into rule weights.
    for source_ngram, counter in observations.items():
        counter[source_ngram] += 1

    rules: dict[str, dict[str, float]] = {}
    for source_ngram, counter in observations.items():
        total = sum(counter.values())
        rules[source_ngram] = {
            target_ngram: count / total for target_ngram, count in counter.items()
        }
    return rules


def build_artifact_bytes(
    pairs_path: Path,
    base_vocabulary_path: Path,
    curated_source_path: Path = DEFAULT_CURATED_SOURCE,
    source_workbook_sha256: str | None = CURATED_SOURCE_WORKBOOK_SHA256,
    max_ngram: int = 2,
    candidate_cutoff: int = 100,
) -> dict[str, bytes]:
    if max_ngram < 2:
        raise ValueError("max_ngram must be at least 2")
    if candidate_cutoff < 1:
        raise ValueError("candidate_cutoff must be positive")

    pairs = _read_csv_pairs(pairs_path)
    base_vocabulary = _read_base_vocabulary(base_vocabulary_path)
    learned_vocabulary = {
        token
        for _source, target in pairs
        for token in target.split()
        if token
    }
    vocabulary = sorted(base_vocabulary | learned_vocabulary)
    rules = induce_rules(pairs, max_ngram=max_ngram)
    curated_bytes, curated_stats = build_curated_mappings(curated_source_path)

    rules_payload = {
        "schema_version": 1,
        "algorithm": "character_n_gram_rules_with_damerau_levenshtein_ranking",
        "max_ngram": max_ngram,
        "candidate_cutoff": candidate_cutoff,
        "rules": rules,
    }
    rules_bytes = (json.dumps(rules_payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    vocabulary_bytes = ("\n".join(vocabulary) + "\n").encode("utf-8")

    def display_path(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(resolved)

    metadata = {
        "schema_version": 2,
        "algorithm": (
            "automatic character n-gram rules with Damerau-Levenshtein ranking, "
            "supplemented by curated whole-form normalization rules"
        ),
        "algorithm_source": {
            "archive": SOURCE_ARCHIVE,
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "implementation": "main_algo.py and utils.py (N-Grams + DLD V1)",
            "training_pairs_member": TRAINING_MEMBER,
            "base_vocabulary_member": VOCABULARY_MEMBER,
        },
        "parameters": {
            "min_ngram": 2,
            "max_ngram": max_ngram,
            "candidate_cutoff": candidate_cutoff,
            "candidate_ranking": "unrestricted Damerau-Levenshtein distance",
            "candidate_vocabulary_filter": True,
        },
        "training_data": {
            "path": display_path(pairs_path),
            "sha256": sha256_bytes(pairs_path.read_bytes()),
            "pair_count": len(pairs),
            "induced_source_ngram_count": len(rules),
            "induced_replacement_count": sum(len(values) for values in rules.values()),
            "vocabulary_target_token_count": len(learned_vocabulary),
        },
        "base_vocabulary": {
            "path": display_path(base_vocabulary_path),
            "sha256": sha256_bytes(base_vocabulary_path.read_bytes()),
            "entry_count": len(base_vocabulary),
            "provenance_license": VOCABULARY_LICENSE,
        },
        "vocabulary_count": len(vocabulary),
        "curated_source": {
            "workbook_filename": (
                CURATED_SOURCE_WORKBOOK if source_workbook_sha256 is not None else None
            ),
            "workbook_sha256": source_workbook_sha256,
            "extracted_path": display_path(curated_source_path),
            "extracted_sha256": curated_stats["source_sha256"],
            "source_row_count": curated_stats["source_row_count"],
            "populated_mapping_row_count": curated_stats["populated_mapping_row_count"],
            "id_only_row_count": (
                curated_stats["source_row_count"] - curated_stats["populated_mapping_row_count"]
            ),
            "slang_row_count": curated_stats["slang_row_count"],
            "unique_slang_input_count": curated_stats["unique_slang_input_count"],
            "accepted_mapping_count": curated_stats["accepted_mapping_count"],
            "duplicate_mapping_count": curated_stats["duplicate_mapping_count"],
            "duplicate_source_row_count": curated_stats["duplicate_source_row_count"],
            "ambiguous_mapping_count": curated_stats["ambiguous_mapping_count"],
            "ambiguous_source_row_count": curated_stats["ambiguous_source_row_count"],
            "excluded_ambiguous_input_count": curated_stats["excluded_ambiguous_input_count"],
            "build_timestamp": None,
        },
        "artifacts": {
            "rules.json": {"sha256": sha256_bytes(rules_bytes)},
            "vocabulary.txt": {"sha256": sha256_bytes(vocabulary_bytes)},
            "curated_mappings.json": {"sha256": sha256_bytes(curated_bytes)},
        },
    }
    metadata_bytes = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return {
        "rules.json": rules_bytes,
        "vocabulary.txt": vocabulary_bytes,
        "curated_mappings.json": curated_bytes,
        "metadata.json": metadata_bytes,
    }


def write_artifacts(artifacts: dict[str, bytes], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in artifacts.items():
        target = output_dir / filename
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--base-vocabulary", type=Path, default=DEFAULT_BASE_VOCABULARY)
    parser.add_argument("--curated-source", type=Path, default=DEFAULT_CURATED_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-ngram", type=int, default=2)
    parser.add_argument("--candidate-cutoff", type=int, default=100)
    args = parser.parse_args()

    artifacts = build_artifact_bytes(
        args.pairs,
        args.base_vocabulary,
        curated_source_path=args.curated_source,
        max_ngram=args.max_ngram,
        candidate_cutoff=args.candidate_cutoff,
    )
    write_artifacts(artifacts, args.output)
    metadata = json.loads(artifacts["metadata.json"])
    print(
        "Built normalizer artifacts: "
        f"{metadata['training_data']['pair_count']} training pairs, "
        f"{metadata['training_data']['induced_source_ngram_count']} rules, "
        f"{metadata['vocabulary_count']} vocabulary entries, "
        f"{metadata['curated_source']['accepted_mapping_count']} curated mappings, "
        f"{metadata['curated_source']['ambiguous_mapping_count']} ambiguous forms excluded "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
