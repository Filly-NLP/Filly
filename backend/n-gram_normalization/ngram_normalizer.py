"""Reusable training and inference for the repository's N-gram normalizer."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from utils import collate_dict, collate_max, collect_rules, generate_candidates, choose_top_k


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "train_words.csv"
DEFAULT_VOCAB = PROJECT_ROOT / "TagalogStemmerPython" / "output" / "with_info.txt"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "ngram_rules.json"
PAIR_COLUMNS = ("Input", "Target")


def load_pairs_csv(path: str | Path) -> list[tuple[str, str]]:
    """Load Input/Target pairs, accepting the legacy headerless test files."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Training data is empty: {path}")

    has_header = tuple(cell.strip() for cell in rows[0]) == PAIR_COLUMNS
    data_rows = rows[1:] if has_header else rows
    pairs: list[tuple[str, str]] = []
    for line_number, row in enumerate(data_rows, start=2 if has_header else 1):
        if len(row) != 2:
            raise ValueError(
                f"{path}:{line_number} must have exactly two CSV fields; found {len(row)}"
            )
        source, target = (cell.strip() for cell in row)
        if not source or not target:
            raise ValueError(f"{path}:{line_number} contains a blank Input or Target")
        pairs.append((source, target))
    return pairs


def load_vocabulary(path: str | Path = DEFAULT_VOCAB) -> set[str]:
    """Load the word field from the bundled Tagalog stemmer resource."""
    vocabulary: set[str] = set()
    with Path(path).open("r", encoding="latin1") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                item = ast.literal_eval(line.strip())
                word = str(item["word"]).strip().lower()
            except (SyntaxError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid vocabulary record at line {line_number}: {path}") from exc
            if word:
                vocabulary.add(word)
    return vocabulary


def train_model(
    pairs: Iterable[tuple[str, str]],
    *,
    max_sub: int = 2,
    vocabulary_path: str | Path | None = None,
    source_path: str | Path | None = None,
) -> dict:
    """Generate frequency-ranked rules using the original repository algorithm."""
    if max_sub < 2:
        raise ValueError("max_sub must be at least 2")

    normalized_pairs = [(source.lower(), target.lower()) for source, target in pairs]
    if not normalized_pairs:
        raise ValueError("At least one training pair is required")

    raw_rules: dict[str, list[str]] = {}
    exact_counts: dict[str, Counter[str]] = defaultdict(Counter)
    target_terms: set[str] = set()

    for source, target in normalized_pairs:
        exact_counts[source][target] += 1
        target_terms.add(target)
        target_terms.update(target.split())
        for substring_length in range(2, max_sub + 1):
            raw_rules = collate_dict(
                raw_rules,
                collect_rules(source, target, substring_length, substring_length),
            )

    for key in raw_rules:
        raw_rules[key].append(key)
    weighted_rules = collate_max(raw_rules, "test")

    exact_rules = {
        source: [target for target, _ in counts.most_common()]
        for source, counts in exact_counts.items()
    }
    vocabulary = load_vocabulary(vocabulary_path) if vocabulary_path is not None else set()
    vocabulary.update(target_terms)

    source_sha256 = None
    if source_path is not None:
        source_sha256 = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()

    return {
        "format_version": 1,
        "algorithm": "character-ngram-rules+damerau-levenshtein",
        "max_sub": max_sub,
        "training_rows": len(normalized_pairs),
        "training_data_sha256": source_sha256,
        "external_vocabulary": Path(vocabulary_path).name if vocabulary_path else None,
        "rules": weighted_rules,
        "exact_rules": exact_rules,
        "vocabulary": sorted(vocabulary),
    }


def save_model(model: dict, path: str | Path = DEFAULT_MODEL) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(model, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def load_model(path: str | Path = DEFAULT_MODEL) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    if model.get("format_version") != 1:
        raise ValueError(f"Unsupported model format in {path}")
    return model


_TOKEN_BOUNDARY = re.compile(r"^([^\w']*)(.*?)([^\w']*)$", flags=re.UNICODE)


def normalize_word(word: str, model: dict, *, cutoff: int = 100) -> str:
    """Normalize one token, using learned exact rules then N-gram candidates."""
    if not word:
        return word
    match = _TOKEN_BOUNDARY.match(word)
    if match is None:
        return word
    prefix, core, suffix = match.groups()
    if not core:
        return word

    lowered = core.lower()
    exact = model["exact_rules"].get(lowered)
    if exact:
        normalized = exact[0]
    elif lowered in model["vocabulary"]:
        normalized = lowered
    else:
        candidates = generate_candidates(
            lowered,
            model["rules"],
            int(model["max_sub"]),
            cutoff,
        )
        normalized = choose_top_k(
            candidates,
            lowered,
            set(model["vocabulary"]),
            1,
            True,
        )[0]

    if core[:1].isupper():
        normalized = normalized[:1].upper() + normalized[1:]
    return prefix + normalized + suffix


def normalize_text(text: str, model: dict, *, cutoff: int = 100) -> str:
    """Normalize every non-whitespace token while retaining original spacing."""
    pieces = re.split(r"(\s+)", text)
    return "".join(
        piece if not piece or piece.isspace() else normalize_word(piece, model, cutoff=cutoff)
        for piece in pieces
    )
