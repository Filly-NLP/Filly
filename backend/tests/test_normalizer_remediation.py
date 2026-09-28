from __future__ import annotations

import argparse
import copy

import pytest

from app.services.normalizer import FilipinoNormalizer, damerau_levenshtein_distance
from scripts import evaluate_normalizer
from scripts.benchmark_normalizer_ranking import _make_legacy_class
from scripts.profile_normalizer_runtime import _validate_comparable_reports


def _profile_report(*, source_hash: str, repeats: int = 5) -> dict:
    sizes = [1, 100]
    fingerprints = {f"{kind}:{size}": f"{kind}-{size}" for kind in ("word", "sentence") for size in sizes}
    return {
        "schema_version": 2,
        "normalizer_configuration": {"max_edit_distance": 2, "candidate_cutoff": 100, "max_ngram": 3},
        "benchmark_configuration": {
            "sizes": sizes,
            "repeats": repeats,
            "profile_words": 1000,
            "warmup_passes_per_workload": 1,
        },
        "measurement_scope": {
            "normalizer_only": True,
            "gec_called": False,
            "reserved_test_accessed": False,
            "source": "backend/normalization/data/train_pairs.csv source column only",
            "first_call_definition": "pre-workload warmup",
            "warm_definition": "one untimed full workload pass followed by timed repetitions",
        },
        "provenance": {
            "python": "same-python",
            "platform": "same-platform",
            "training_pairs_sha256": "train",
            "rules_sha256": "rules",
            "vocabulary_sha256": "vocabulary",
            "metadata_sha256": "metadata",
            "workload_word_list_sha256": "workload",
            "normalizer_source_sha256": source_hash,
            "git_head": "head",
        },
        "workloads": {
            kind: {
                str(size): {
                    "token_count": size,
                    "warm_repetitions": repeats,
                    "warm_batch_ms": [1.0] * repeats,
                    "warm_median_ms": 1.0,
                }
                for size in sizes
            }
            for kind in ("word", "sentence")
        },
        "output_fingerprints": fingerprints,
    }


def test_runtime_comparison_allows_code_revision_but_requires_matched_measurements():
    baseline = _profile_report(source_hash="before")
    current = _profile_report(source_hash="after")

    _validate_comparable_reports(baseline, current)

    mismatched_repeats = copy.deepcopy(current)
    mismatched_repeats["benchmark_configuration"]["repeats"] = 7
    with pytest.raises(ValueError, match="benchmark configuration"):
        _validate_comparable_reports(baseline, mismatched_repeats)

    inconsistent_samples = copy.deepcopy(current)
    inconsistent_samples["workloads"]["word"]["100"]["warm_repetitions"] = 3
    with pytest.raises(ValueError, match="repeat counts"):
        _validate_comparable_reports(baseline, inconsistent_samples)

    mismatched_workload = copy.deepcopy(current)
    mismatched_workload["provenance"]["workload_word_list_sha256"] = "other-workload"
    with pytest.raises(ValueError, match="provenance"):
        _validate_comparable_reports(baseline, mismatched_workload)

    old_schema = copy.deepcopy(baseline)
    old_schema["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version 2"):
        _validate_comparable_reports(old_schema, current)


class _DiagnosticNormalizer:
    vocabulary = {"bb", "cc", "ab"}
    rules = {}
    max_ngram = 3
    candidate_cutoff = 100
    max_edit_distance = 1

    def __init__(self, candidates: dict[str, float]):
        self.candidates = candidates

    def _collect_candidates(self, *_args):
        return self.candidates


def test_diagnostics_distinguish_expected_candidate_absent_and_distance_gated(monkeypatch):
    monkeypatch.setattr(evaluate_normalizer, "_damerau_levenshtein_distance", damerau_levenshtein_distance)
    absent_normalizer = _DiagnosticNormalizer({"cc": 1.0})
    # Bind the fake candidate set without involving the artifact loader.
    monkeypatch.setattr(
        absent_normalizer,
        "_collect_candidates",
        lambda *_args: absent_normalizer.candidates,
    )
    absent = evaluate_normalizer._candidate_diagnostics(absent_normalizer, "aa", "bb", "aa")
    assert absent["expected_candidate_status"] == "absent_from_generated_candidates"
    assert absent["failure_type"] == "expected_candidate_not_generated"

    gated_normalizer = _DiagnosticNormalizer({"bb": 1.0, "ab": 1.0})
    monkeypatch.setattr(
        gated_normalizer,
        "_collect_candidates",
        lambda *_args: gated_normalizer.candidates,
    )
    gated = evaluate_normalizer._candidate_diagnostics(gated_normalizer, "aa", "bb", "ab")
    assert gated["expected_candidate_status"] == "distance_gated"
    assert gated["expected_candidate_damerau_levenshtein"] == 2
    assert gated["maximum_applied_damerau_levenshtein"] == 1
    assert gated["failure_type"] == "expected_candidate_distance_gated"


def test_uncapped_baseline_cli_value_is_explicit_and_nonnegative_values_stay_supported():
    assert evaluate_normalizer._parse_max_edit_distance("none") is None
    assert evaluate_normalizer._parse_max_edit_distance("UNLIMITED") is None
    assert evaluate_normalizer._parse_max_edit_distance("2") == 2
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
        evaluate_normalizer._parse_max_edit_distance("-1")


def test_sorted_and_first_min_ranking_have_full_word_and_sentence_output_parity():
    legacy_type = _make_legacy_class(FilipinoNormalizer, damerau_levenshtein_distance)
    legacy = legacy_type(max_edit_distance=2)
    optimized = FilipinoNormalizer(max_edit_distance=2)
    words = ["aq", "kse", "skul", "nandun", "musta", "QZXV", "AKO"]

    assert [legacy.normalize_word(word) for word in words] == [
        optimized.normalize_word(word) for word in words
    ]
    text = "AQ, kse! skul nandun; QZXV."
    legacy_sentence = legacy.normalize_text(text)
    optimized_sentence = optimized.normalize_text(text)
    assert legacy_sentence[0] == optimized_sentence[0]
    assert [item.model_dump() for item in legacy_sentence[1]] == [
        item.model_dump() for item in optimized_sentence[1]
    ]
