"""Check final-output diagnostics count collateral damage and incomplete traces."""

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "duplicate_summary", Path(__file__).resolve().parents[1] / "scripts" / "summarize_duplicate_diagnostics.py"
)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def fixtures():
    cases = [
        {"case_id": "positive", "case_type": "new_duplicate_corruption",
         "source_text": "Bumili siya siya ng libro.", "target_text": "Bumili siya ng libro.",
         "duplicate_source_span": {"text": "siya"}, "target_class": "function"},
        {"case_id": "control", "case_type": "identity_repeat",
         "source_text": "Araw-araw siyang masaya.", "target_text": "Araw-araw siyang masaya."},
    ]
    predictions = [
        {**case, "passes": [{"output_text": case["target_text"]} for _ in range(5)]}
        for case in cases
    ]
    return cases, predictions


def test_collateral_edits_reduce_precision_without_erasing_correct_duplicate_removal():
    cases, predictions = fixtures()
    predictions[0]["passes"][1]["output_text"] = "Bumili siya ng libro.."
    predictions[1]["passes"][0]["output_text"] = "Araw siyang masaya."
    result, records = summary.summarize(cases, predictions, Counter({"siya": 10}))
    first = result["by_pass"]["1"]["all_cases"]
    assert (first["tp"], first["fp"], first["fn"]) == (1, 1, 0)
    assert first["diagnostic_err"] == 0
    second = result["by_pass"]["2"]["duplicate_corruptions"]
    assert (second["tp"], second["fp"], second["fn"]) == (1, 1, 0)
    assert second["exact_count"] == 0
    assert next(row for row in records if row["case_id"] == "positive" and row["pass"] == 2)["duplicate_removed"]
    assert result["by_pass"]["1"]["duplicate_token_coverage=seen"]["n"] == 1


def test_incomplete_or_duplicate_predictions_cannot_silently_change_denominators():
    cases, predictions = fixtures()
    with pytest.raises(ValueError, match="IDs differ"):
        summary.summarize(cases, predictions[:1], Counter())
    with pytest.raises(ValueError, match="duplicate case IDs"):
        summary.summarize(cases, predictions + predictions[:1], Counter())
    predictions[0]["passes"] = predictions[0]["passes"][:4]
    with pytest.raises(ValueError, match="incomplete five-pass"):
        summary.summarize(cases, predictions, Counter())


def test_changed_source_cannot_be_scored_against_unrelated_gold():
    cases, predictions = fixtures()
    predictions[0]["source_text"] = "Ibang pangungusap."
    with pytest.raises(ValueError, match="source/target mismatch"):
        summary.summarize(cases, predictions, Counter())
