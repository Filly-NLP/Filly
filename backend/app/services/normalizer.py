"""Filipino character n-gram spelling normalizer.

The rule induction and candidate recursion follow the N-Gram + DLD V1 code in
``efficient-spelling-normalization-filipino-main.zip``. Runtime resources are
built by ``backend/scripts/train_normalizer.py`` and loaded once per process.
Offsets returned here are Python Unicode code-point offsets into the original
input text.
"""

from __future__ import annotations

import json
import hashlib
import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.schemas.schemas import NormalizationItem

logger = logging.getLogger(__name__)

_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "normalizer"
_URL_RE = re.compile(r"(?i)(?:https?://|ftp://|www\.)\S+")
_EMAIL_RE = re.compile(r"(?i)[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+")


def _resolve_artifact_paths(
    artifact_dir: Path | str,
    rules_path: Path | str | None,
    vocabulary_path: Path | str | None,
    metadata_path: Path | str | None,
    curated_mappings_path: Path | str | None,
) -> tuple[Path, Path, Path, Path]:
    artifact_dir = Path(artifact_dir)
    rules = Path(rules_path) if rules_path is not None else artifact_dir / "rules.json"
    vocabulary = (
        Path(vocabulary_path) if vocabulary_path is not None else artifact_dir / "vocabulary.txt"
    )
    if metadata_path is not None:
        metadata = Path(metadata_path)
    elif rules_path is not None:
        metadata = rules.parent / "metadata.json"
    elif vocabulary_path is not None:
        metadata = vocabulary.parent / "metadata.json"
    else:
        metadata = artifact_dir / "metadata.json"
    curated_mappings = (
        Path(curated_mappings_path)
        if curated_mappings_path is not None
        else rules.parent / "curated_mappings.json"
    )
    return rules.resolve(), vocabulary.resolve(), metadata.resolve(), curated_mappings.resolve()


def damerau_levenshtein_distance(left: str, right: str) -> int:
    """Return the unrestricted Damerau–Levenshtein distance between strings."""
    left_len, right_len = len(left), len(right)
    max_distance = left_len + right_len
    matrix = [[0] * (right_len + 2) for _ in range(left_len + 2)]
    matrix[0][0] = max_distance
    for i in range(left_len + 1):
        matrix[i + 1][0] = max_distance
        matrix[i + 1][1] = i
    for j in range(right_len + 1):
        matrix[0][j + 1] = max_distance
        matrix[1][j + 1] = j

    last_seen: dict[str, int] = {}
    for i in range(1, left_len + 1):
        last_match_column = 0
        for j in range(1, right_len + 1):
            prior_row = last_seen.get(right[j - 1], 0)
            prior_column = last_match_column
            substitution_cost = 1
            if left[i - 1] == right[j - 1]:
                substitution_cost = 0
                last_match_column = j

            matrix[i + 1][j + 1] = min(
                matrix[i][j] + substitution_cost,
                matrix[i + 1][j] + 1,
                matrix[i][j + 1] + 1,
                matrix[prior_row][prior_column]
                + (i - prior_row - 1)
                + 1
                + (j - prior_column - 1),
            )
        last_seen[left[i - 1]] = i
    return matrix[left_len + 1][right_len + 1]


def _apply_case(source: str, suggestion: str) -> str:
    if source.isupper():
        return suggestion.upper()
    if source and source[0].isupper():
        return suggestion[0].upper() + suggestion[1:]
    return suggestion


def _is_word_letter(character: str) -> bool:
    category = unicodedata.category(character)
    return category.startswith("L") or category.startswith("M")


def _curated_form_key(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def _protected_spans(text: str) -> list[tuple[int, int]]:
    spans = [match.span() for pattern in (_URL_RE, _EMAIL_RE) for match in pattern.finditer(text)]
    return sorted(spans)


class FilipinoNormalizer:
    """Normalize text with curated whole-form rules and automatic N-Gram + DLD."""

    def __init__(
        self,
        artifact_dir: Path | str = _ARTIFACT_DIR,
        *,
        rules_path: Path | str | None = None,
        vocabulary_path: Path | str | None = None,
        metadata_path: Path | str | None = None,
        curated_mappings_path: Path | str | None = None,
        max_edit_distance: int | None = 2,
        use_curated_mappings: bool = True,
    ):
        if max_edit_distance is not None and max_edit_distance < 0:
            raise ValueError("max_edit_distance must be non-negative or None")
        if type(use_curated_mappings) is not bool:
            raise ValueError("use_curated_mappings must be a bool")
        self.max_edit_distance = max_edit_distance
        self.use_curated_mappings = use_curated_mappings
        rules_path, vocabulary_path, metadata_path, curated_mappings_path = _resolve_artifact_paths(
            artifact_dir,
            rules_path,
            vocabulary_path,
            metadata_path,
            curated_mappings_path,
        )
        self._resource_signature = (
            rules_path,
            vocabulary_path,
            metadata_path,
            curated_mappings_path,
        )
        artifact_dir = Path(artifact_dir)

        required_paths = (metadata_path, rules_path, vocabulary_path, curated_mappings_path)
        if any(not path.is_file() for path in required_paths):
            raise FileNotFoundError(
                f"Normalizer artifacts are incomplete in {artifact_dir}; "
                "build them with backend/scripts/train_normalizer.py"
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 2:
            raise ValueError(f"Unsupported normalizer artifact schema in {metadata_path}")
        for filename, path in (
            ("rules.json", rules_path),
            ("vocabulary.txt", vocabulary_path),
            ("curated_mappings.json", curated_mappings_path),
        ):
            expected_hash = metadata.get("artifacts", {}).get(filename, {}).get("sha256")
            if not expected_hash:
                raise ValueError(f"Missing {filename} checksum in {metadata_path}")
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"Normalizer artifact checksum mismatch: {path}")

        payload = json.loads(rules_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported rule schema in {rules_path}")
        self.max_ngram = int(payload["max_ngram"])
        self.candidate_cutoff = int(payload["candidate_cutoff"])
        self.rules: dict[str, dict[str, float]] = payload["rules"]
        self.vocabulary = {
            line.strip().lower()
            for line in vocabulary_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        curated_payload = json.loads(curated_mappings_path.read_text(encoding="utf-8"))
        if (
            curated_payload.get("schema_version") != 1
            or curated_payload.get("mapping_type") != "curated_whole_form_normalization_rules"
            or not isinstance(curated_payload.get("mappings"), dict)
            or not isinstance(curated_payload.get("ambiguous"), dict)
        ):
            raise ValueError(f"Invalid curated mapping resource in {curated_mappings_path}")
        self.curated_mappings: dict[str, dict[str, object]] = curated_payload["mappings"]
        self.curated_ambiguities: dict[str, list[dict[str, object]]] = curated_payload["ambiguous"]
        self._curated_pattern = self._compile_curated_pattern(self.curated_mappings)
        if not self.rules or not self.vocabulary or not self.curated_mappings:
            raise ValueError(f"Normalizer artifacts contain no usable rules or vocabulary: {artifact_dir}")

    @staticmethod
    def _compile_curated_pattern(mappings: dict[str, dict[str, object]]) -> re.Pattern[str] | None:
        if not mappings:
            return None
        alternatives = []
        for form in sorted(mappings, key=lambda value: (-len(value), value)):
            parts = form.split(" ")
            alternatives.append(r"[ \t]+".join(re.escape(part) for part in parts))
        left_boundary = r"(?<!\w)(?<!\w['’])(?<!\w[-‐‑‒–—])"
        right_boundary = r"(?!\w)(?!['’]\w)(?![-‐‑‒–—]\w)"
        return re.compile(
            left_boundary + "(?:" + "|".join(alternatives) + ")" + right_boundary,
            re.IGNORECASE,
        )

    @staticmethod
    def _collect_candidates(
        word: str,
        rules: dict[str, dict[str, float]],
        max_ngram: int,
        cutoff: int,
    ) -> dict[str, float]:
        """Port the source archive's recursive candidate-generation procedure."""

        @lru_cache(maxsize=None)
        def generate(remaining: str) -> tuple[tuple[str, float], ...]:
            if remaining == "":
                return (("", 1.0),)
            if len(remaining) == 1:
                if remaining in rules:
                    return tuple(rules[remaining].items())
                return ((remaining, 1.0), ("", 1.0))

            result: dict[str, float] = {}
            for ngram_size in range(2, max_ngram + 1):
                if len(remaining) < ngram_size:
                    continue
                current = remaining[:ngram_size]
                if current in rules:
                    replacements = rules[current]
                    for replacement, rule_weight in replacements.items():
                        suffix = remaining[ngram_size:] if replacement == current else remaining[1:]
                        next_candidates = dict(generate(suffix))
                        next_candidates[""] = 1.0
                        for next_part, next_weight in next_candidates.items():
                            result[replacement + next_part] = rule_weight * next_weight
                else:
                    next_candidates = dict(generate(remaining[1:]))
                    for next_part, next_weight in next_candidates.items():
                        result[remaining[0] + next_part] = next_weight

            if not result:
                result[remaining] = 1.0
            cutoff_values = sorted(result.values(), reverse=True)[:cutoff]
            threshold = cutoff_values[-1]
            result = {candidate: weight for candidate, weight in result.items() if weight >= threshold}
            return tuple(result.items())

        return dict(generate(word))

    def _normalize_automatic(self, word: str) -> tuple[str, Optional[int]]:
        lowered = word.lower()
        if len(lowered) < 2 or len(lowered) > 128 or lowered in self.vocabulary:
            return word, None

        candidates = self._collect_candidates(
            lowered,
            self.rules,
            self.max_ngram,
            self.candidate_cutoff,
        )
        matches = [
            candidate
            for candidate in candidates
            if candidate.strip()
            and all(part.lower() in self.vocabulary for part in candidate.strip().split())
        ]
        if not matches:
            # The research implementation falls back to the entire vocabulary
            # here. For production text, that can rewrite unrelated unknowns;
            # an unsupported token therefore stays unchanged.
            return word, None

        # ``min`` is stable, so tied distances keep candidate insertion order
        # while each candidate's distance is computed only once.
        best = matches[0]
        distance = damerau_levenshtein_distance(lowered, best)
        for candidate in matches[1:]:
            candidate_distance = damerau_levenshtein_distance(lowered, candidate)
            if candidate_distance < distance:
                best, distance = candidate, candidate_distance
        if best == lowered or (
            self.max_edit_distance is not None and distance > self.max_edit_distance
        ):
            return word, None
        return best.strip(), distance

    def _normalize_lowercase(self, word: str) -> tuple[str, Optional[int]]:
        if self.use_curated_mappings:
            mapping = self.curated_mappings.get(_curated_form_key(word))
            if mapping is not None:
                return str(mapping["normalized"]), None
        return self._normalize_automatic(word)

    def explain_word(self, word: str) -> dict[str, object]:
        """Describe the winning whole-form or automatic normalization path."""
        key = _curated_form_key(word)
        mapping = self.curated_mappings.get(key)
        is_ambiguous = key in self.curated_ambiguities
        if self.use_curated_mappings and mapping is not None:
            strategy = "curated_rule"
            status = "matched"
            source_id = mapping.get("source_id")
        else:
            strategy = "ngram_dld"
            status = "ambiguous" if is_ambiguous else (
                "available_disabled" if mapping is not None else "absent"
            )
            source_id = None
        output, _distance = self._normalize_lowercase(word)
        return {
            "input": word,
            "strategy": strategy,
            "source_id": source_id,
            "output": _apply_case(word, output),
            "curated_rule_status": status,
        }

    def normalize_word(self, word: str) -> str:
        """Return one normalized word, preserving its leading capitalization."""
        normalized, _distance = self._normalize_lowercase(word)
        return _apply_case(word, normalized)

    def normalize_text(self, text: str) -> tuple[str, list[NormalizationItem]]:
        """Normalize eligible words, returning text and original-text edits.

        URLs and email addresses are protected as complete spans. Digits and
        underscores attached to a word also make that token ineligible. The
        returned offsets are Python code-point offsets into ``text``.
        """
        if not text:
            return text, []

        protected = _protected_spans(text)
        changes: list[NormalizationItem] = []
        curated_spans: list[tuple[int, int]] = []
        if self.use_curated_mappings and self._curated_pattern is not None:
            for match in self._curated_pattern.finditer(text):
                start, end = match.span()
                if any(start < protected_end and end > protected_start for protected_start, protected_end in protected):
                    continue
                original = match.group()
                key = _curated_form_key(original)
                mapping = self.curated_mappings.get(key)
                if mapping is None:
                    continue
                suggestion = _apply_case(original, str(mapping["normalized"]))
                curated_spans.append((start, end))
                if suggestion != original:
                    changes.append(
                        NormalizationItem(
                            word=original,
                            suggestion=suggestion,
                            start=start,
                            end=end,
                            type="normalization",
                            confidence=0.0,
                            category="spelling_variation",
                            strategy="curated_rule",
                            source_id=str(mapping["source_id"]),
                        )
                    )
        index = 0
        while index < len(text):
            if not text[index].isalpha():
                index += 1
                continue
            start = index
            index += 1
            while index < len(text) and _is_word_letter(text[index]):
                index += 1
            end = index

            if any(start < curated_end and end > curated_start for curated_start, curated_end in curated_spans):
                continue

            if (start > 0 and (text[start - 1].isdigit() or text[start - 1] == "_")) or (
                end < len(text) and (text[end].isdigit() or text[end] == "_")
            ):
                continue
            if any(start < protected_end and end > protected_start for protected_start, protected_end in protected):
                continue

            original = text[start:end]
            # Curated rules are applied only by the complete-form matcher
            # above; this token pass is strictly automatic N-Gram + DLD.
            normalized, _distance = self._normalize_automatic(original)
            suggestion = _apply_case(original, normalized)
            if suggestion != original:
                changes.append(
                    NormalizationItem(
                        word=original,
                        suggestion=suggestion,
                        start=start,
                        end=end,
                        type="normalization",
                        confidence=0.0,
                        category="spelling_variation",
                        strategy="ngram_dld",
                    )
                )

        changes.sort(key=lambda change: (change.start, change.end))
        normalized_text = text
        for change in reversed(changes):
            normalized_text = (
                normalized_text[: change.start]
                + change.suggestion
                + normalized_text[change.end :]
            )
        return normalized_text, changes

    def normalize(self, text: str) -> list[NormalizationItem]:
        """Compatibility API returning suggestions with source offsets."""
        _normalized_text, changes = self.normalize_text(text)
        return changes


# Older imports may still use this class name.
FilNormalizer = FilipinoNormalizer

_normalizer: Optional[FilipinoNormalizer] = None


def get_normalizer(
    *,
    rules_path: Path | str | None = None,
    vocabulary_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
    curated_mappings_path: Path | str | None = None,
    artifact_dir: Path | str = _ARTIFACT_DIR,
    max_edit_distance: int | None = 2,
) -> FilipinoNormalizer:
    global _normalizer
    requested_paths = _resolve_artifact_paths(
        artifact_dir,
        rules_path,
        vocabulary_path,
        metadata_path,
        curated_mappings_path,
    )
    requested_signature = (*requested_paths, max_edit_distance)
    if _normalizer is None:
        _normalizer = FilipinoNormalizer(
            artifact_dir,
            rules_path=rules_path,
            vocabulary_path=vocabulary_path,
            metadata_path=metadata_path,
            curated_mappings_path=curated_mappings_path,
            max_edit_distance=max_edit_distance,
        )
        logger.info(
            "Filipino normalizer initialized with %d automatic rules, %d curated rules, and %d vocabulary entries",
            len(_normalizer.rules),
            len(_normalizer.curated_mappings),
            len(_normalizer.vocabulary),
        )
    elif (
        any(value is not None for value in (rules_path, vocabulary_path, metadata_path, curated_mappings_path))
        or Path(artifact_dir).resolve() != _ARTIFACT_DIR.resolve()
        or max_edit_distance != 2
    ):
        if requested_signature != (*_normalizer._resource_signature, _normalizer.max_edit_distance):
            raise RuntimeError("Normalizer is already loaded with different resource paths")
    return _normalizer
