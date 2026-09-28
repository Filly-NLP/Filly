"""Matched ranking A/B for FILLY's standalone normalizer.

Both strategies run in one process against the same artifacts and the ordered
source inputs from the archive validation split. Each repetition alternates
which strategy runs first. No target strings are passed to either normalizer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
DEFAULT_ARCHIVE = PROJECT_ROOT / "efficient-spelling-normalization-filipino-main (1).zip"
DEFAULT_ARTIFACT_DIR = BACKEND_ROOT / "artifacts" / "normalizer"
DEFAULT_TRAINING_PAIRS = BACKEND_ROOT / "normalization" / "data" / "train_pairs.csv"
DEFAULT_SIZES = (1, 100, 1_000, 10_000)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: Any) -> str:
    packed = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(packed.encode("utf-8"))


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
            check=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _validation_sources(archive_path: Path, training_path: Path) -> tuple[list[str], dict[str, Any]]:
    sys.path.insert(0, str(BACKEND_ROOT))
    from scripts.evaluate_normalizer import (
        deterministic_validation_test_split,
        filter_training_overlaps,
        load_archive_gold,
        load_training_pairs,
        runtime_workload_metadata,
    )

    all_pairs = load_archive_gold(archive_path)
    training_pairs = load_training_pairs(training_path)
    nonoverlap, _excluded, overlap_counts = filter_training_overlaps(all_pairs, training_pairs)
    validation, reserved_test, partition = deterministic_validation_test_split(nonoverlap)
    sources = [pair.source for pair in validation]
    metadata = {
        "split": "validation",
        "split_seed": partition["seed"],
        "validation_row_count": len(validation),
        "reserved_test_row_count": len(reserved_test),
        "validation_source_workload": runtime_workload_metadata(sources),
        "overlap_counts": overlap_counts,
        "validation_rows_sha256": partition["validation_rows_sha256"],
        "reserved_test_rows_accessed_for_partition_only": True,
        "reserved_test_sources_passed_to_normalizer": False,
    }
    return sources, metadata


def _sentence_payload(normalized: str, changes: list[Any]) -> dict[str, Any]:
    return {
        "text": normalized,
        "changes": [
            {
                "word": item.word,
                "suggestion": item.suggestion,
                "start": item.start,
                "end": item.end,
                "type": item.type,
                "confidence": item.confidence,
                "category": item.category,
            }
            for item in changes
        ],
    }


def _make_legacy_class(base: type, distance: Any) -> type:
    class SortedRankingNormalizer(base):
        """Reference ranking using a stable full sort by unrestricted DLD."""

        def _normalize_lowercase(self, word: str) -> tuple[str, int | None]:
            lowered = word.lower()
            if len(lowered) < 2 or len(lowered) > 128 or lowered in self.vocabulary:
                return word, None

            candidates = self._collect_candidates(
                lowered, self.rules, self.max_ngram, self.candidate_cutoff
            )
            matches = [
                candidate
                for candidate in candidates
                if candidate.strip()
                and all(part.lower() in self.vocabulary for part in candidate.strip().split())
            ]
            if not matches:
                return word, None

            ranked = sorted(
                (
                    (distance(lowered, candidate), index, candidate)
                    for index, candidate in enumerate(matches)
                ),
                key=lambda row: (row[0], row[1]),
            )
            best_distance, _index, best = ranked[0]
            if best == lowered or (
                self.max_edit_distance is not None
                and best_distance > self.max_edit_distance
            ):
                return word, None
            return best.strip(), best_distance

    SortedRankingNormalizer.__name__ = "SortedRankingNormalizer"
    return SortedRankingNormalizer


def _workload(normalizer: Any, kind: str, words: list[str]) -> Any:
    if kind == "word":
        return [normalizer.normalize_word(word) for word in words]
    return _sentence_payload(*normalizer.normalize_text(" ".join(words)))


def _timed_workload(normalizer: Any, kind: str, words: list[str]) -> tuple[Any, float]:
    start = time.perf_counter_ns()
    output = _workload(normalizer, kind, words)
    return output, (time.perf_counter_ns() - start) / 1_000_000


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(BACKEND_ROOT))
    from app.services.normalizer import (
        FilipinoNormalizer,
        damerau_levenshtein_distance,
    )

    source_words, validation_metadata = _validation_sources(args.archive, args.training_pairs)
    legacy_type = _make_legacy_class(FilipinoNormalizer, damerau_levenshtein_distance)
    legacy = legacy_type(artifact_dir=args.artifact_dir, max_edit_distance=args.max_edit_distance)
    optimized = FilipinoNormalizer(
        artifact_dir=args.artifact_dir, max_edit_distance=args.max_edit_distance
    )

    workloads: dict[str, dict[str, Any]] = {"word": {}, "sentence": {}}
    all_output_fingerprints: dict[str, dict[str, str]] = {"legacy_sorted": {}, "optimized_first_min": {}}
    order_by_repeat = [
        ["legacy_sorted", "optimized_first_min"] if index % 2 == 0
        else ["optimized_first_min", "legacy_sorted"]
        for index in range(args.repeats)
    ]

    for kind in ("word", "sentence"):
        for size in DEFAULT_SIZES:
            words = [source_words[index % len(source_words)] for index in range(size)]
            # Warm both variants with the same inputs; alternate warmup order by workload.
            first, second = (legacy, optimized) if (size + len(kind)) % 2 == 0 else (optimized, legacy)
            _workload(first, kind, words)
            _workload(second, kind, words)

            samples: dict[str, list[float]] = {"legacy_sorted": [], "optimized_first_min": []}
            output_digests: dict[str, str] = {}
            for run_index, order in enumerate(order_by_repeat):
                outputs: dict[str, Any] = {}
                instances = {
                    "legacy_sorted": legacy,
                    "optimized_first_min": optimized,
                }
                for strategy in order:
                    output, elapsed_ms = _timed_workload(instances[strategy], kind, words)
                    outputs[strategy] = output
                    samples[strategy].append(elapsed_ms)
                if outputs["legacy_sorted"] != outputs["optimized_first_min"]:
                    raise AssertionError(
                        f"Full output parity failed for {kind}:{size}, repetition {run_index + 1}"
                    )
                output_digests = {
                    strategy: _digest(output) for strategy, output in outputs.items()
                }

            key = f"{kind}:{size}"
            all_output_fingerprints["legacy_sorted"][key] = output_digests["legacy_sorted"]
            all_output_fingerprints["optimized_first_min"][key] = output_digests["optimized_first_min"]
            legacy_median = statistics.median(samples["legacy_sorted"])
            optimized_median = statistics.median(samples["optimized_first_min"])
            workloads[kind][str(size)] = {
                "input_token_count": size,
                "alternating_order": order_by_repeat,
                "legacy_sorted_ms": [round(value, 4) for value in samples["legacy_sorted"]],
                "optimized_first_min_ms": [round(value, 4) for value in samples["optimized_first_min"]],
                "legacy_sorted_median_ms": round(legacy_median, 4),
                "optimized_first_min_median_ms": round(optimized_median, 4),
                "optimized_change_percent": round((optimized_median / legacy_median - 1) * 100, 3)
                if legacy_median else None,
                "optimized_speedup": round(legacy_median / optimized_median, 5)
                if optimized_median else None,
                "output_parity": output_digests["legacy_sorted"] == output_digests["optimized_first_min"],
            }

    validation_outputs = {
        "legacy_sorted": [legacy.normalize_word(word) for word in source_words],
        "optimized_first_min": [optimized.normalize_word(word) for word in source_words],
    }
    if validation_outputs["legacy_sorted"] != validation_outputs["optimized_first_min"]:
        raise AssertionError("Full validation source prediction parity failed")
    validation_fingerprints = {
        strategy: _digest(output) for strategy, output in validation_outputs.items()
    }
    if validation_fingerprints["legacy_sorted"] != validation_fingerprints["optimized_first_min"]:
        raise AssertionError("Validation source fingerprint parity failed")

    artifact_hashes = {
        name: _sha256_bytes((args.artifact_dir / name).read_bytes())
        for name in ("rules.json", "vocabulary.txt", "metadata.json")
    }
    module_path = Path(sys.modules["app.services.normalizer"].__file__).resolve()
    legacy_aggregate_ms = sum(
        row["legacy_sorted_median_ms"]
        for kind_rows in workloads.values() for row in kind_rows.values()
    )
    optimized_aggregate_ms = sum(
        row["optimized_first_min_median_ms"]
        for kind_rows in workloads.values() for row in kind_rows.values()
    )
    parity_passed = all(
        row["output_parity"]
        for kind_rows in workloads.values() for row in kind_rows.values()
    ) and validation_fingerprints["legacy_sorted"] == validation_fingerprints["optimized_first_min"]
    selected = (
        "optimized_first_min"
        if parity_passed and optimized_aggregate_ms < legacy_aggregate_ms
        else "legacy_sorted"
    )
    return {
        "schema_version": 1,
        "experiment": "legacy stable sorted DLD ranking vs optimized first-min DLD ranking",
        "normalizer_configuration": {
            "max_applied_damerau_levenshtein": args.max_edit_distance,
            "candidate_cutoff": optimized.candidate_cutoff,
            "max_ngram": optimized.max_ngram,
            "ranking_distance": "unrestricted Damerau-Levenshtein",
        },
        "measurement": {
            "processes": 1,
            "repetitions_per_workload_per_strategy": args.repeats,
            "warmup_passes_per_strategy_per_workload": 1,
            "run_order": "alternates every repetition; same process, code, artifacts, and inputs",
            "validation_gold_targets_passed_to_normalizer": False,
            "output_parity_checked_on_every_measured_repetition": True,
            "output_parity_checked_for_all_validation_source_rows": True,
        },
        "validation_workload": validation_metadata,
        "provenance": {
            "git_head": _git_head(),
            "python": sys.version,
            "platform": platform.platform(),
            "normalizer_source_sha256": _sha256_bytes(module_path.read_bytes()),
            "benchmark_source_sha256": _sha256_bytes(Path(__file__).read_bytes()),
            "archive_sha256": _sha256_bytes(args.archive.read_bytes()),
            "training_pairs_sha256": _sha256_bytes(args.training_pairs.read_bytes()),
            "artifacts_sha256": artifact_hashes,
            "validation_source_list_sha256": validation_metadata["validation_source_workload"]["ordered_words_sha256"],
        },
        "output_fingerprints": {
            "workloads": all_output_fingerprints,
            "validation_source_rows": validation_fingerprints,
        },
        "workloads": workloads,
        "production_ranking_decision": {
            "selected_strategy": selected,
            "basis": "lower sum of the eight matched per-workload median batch times; exact output parity is required",
            "legacy_sorted_aggregate_median_ms": round(legacy_aggregate_ms, 4),
            "optimized_first_min_aggregate_median_ms": round(optimized_aggregate_ms, 4),
            "optimized_aggregate_change_percent": round(
                (optimized_aggregate_ms / legacy_aggregate_ms - 1) * 100, 3
            ) if legacy_aggregate_ms else None,
            "full_output_parity": parity_passed,
            "scope": "these validation-split workloads and this host only; not a broad speedup claim",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--training-pairs", type=Path, default=DEFAULT_TRAINING_PAIRS)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--max-edit-distance", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args(argv)
    if args.repeats < 5:
        parser.error("--repeats must be at least 5 for a matched comparison")
    if args.max_edit_distance < 0:
        parser.error("--max-edit-distance must be non-negative")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing ranking report: {args.output}")

    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing ranking report: {args.output}")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Ranking comparison written to {args.output}")
    print(f"Validation output parity: {report['output_fingerprints']['validation_source_rows']['legacy_sorted'] == report['output_fingerprints']['validation_source_rows']['optimized_first_min']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
