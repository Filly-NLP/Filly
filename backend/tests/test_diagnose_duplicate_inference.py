from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_duplicate_inference.py"
SPEC = importlib.util.spec_from_file_location("diagnose_duplicate_inference", SCRIPT)
diagnose = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(diagnose)


def _row(pair_id: str, text: str, *, token_index: int = 1, split: str = "dev") -> dict:
    return {
        "pair_id": pair_id,
        "clean_id": f"clean-{pair_id}",
        "split": split,
        "dataset_stage": "dataset2_stage3",
        "is_errorful": False,
        "source_text": text,
        "target_text": text,
        "source_corpus": "fixture",
        "publisher": "fixture",
        "source_doc_id": f"doc-{pair_id}",
    }


def test_function_word_taxonomy_is_case_insensitive_and_stable():
    assert "ang" in diagnose.FUNCTION_WORDS
    assert "the" in diagnose.FUNCTION_WORDS
    assert "salamat" not in diagnose.FUNCTION_WORDS


def test_candidate_injection_preserves_source_text_and_marks_one_repeat():
    row = _row("candidate", "Maraming tao sa bahay.")
    candidates = diagnose.discover_corruption_candidates([row], seed=17)
    candidate = candidates["interior_function"][0]
    source = candidate["source_text"]
    target = candidate["target_text"]
    inserted = candidate["duplicate_source_span"]
    assert source[:inserted["start"] - 1] + source[inserted["end"]:] == target
    assert diagnose._has_adjacent_repeat(source)
    assert not diagnose._has_adjacent_repeat(target)
    assert candidate["rank"] == diagnose._rank(17, "interior_function", candidate["candidate_key"])


def test_clean_identity_gate_rejects_non_dev_and_changed_pairs():
    good = _row("a", "Ang bata.")
    assert diagnose._clean_identity(good)
    assert not diagnose._clean_identity(_row("b", "Ang bata.", split="synthetic_test"))
    changed = _row("c", "Ang bata.")
    changed["target_text"] = "Ang batang."
    assert not diagnose._clean_identity(changed)


def test_selection_is_sha_ranked_and_respects_length_and_clean_id_quotas():
    candidates = {
        "interior_function": [
            {"candidate_key": f"k{i}", "rank": "ignored", "length_bucket": "short",
             "provenance": {"clean_id": f"clean-{i}"}, "group": "interior_function",
             "source_text": f"Ang bata{i} bata{i}.", "target_text": f"Ang bata{i}.",
             "original_source_span": {"start": 4, "end": 9, "text": f"bata{i}"},
             "duplicate_source_span": {"start": 10, "end": 15, "text": f"bata{i}"}}
            for i in range(5)
        ]
    }
    quotas = {"interior_function": {"short": 2}}
    first = diagnose._select_candidates(candidates, seed=42, quotas=quotas)
    second = diagnose._select_candidates(candidates, seed=42, quotas=quotas)
    assert [item["candidate_key"] for item in first] == [item["candidate_key"] for item in second]
    assert len({item["provenance"]["clean_id"] for item in first}) == 2
    assert all(item["length_bucket"] == "short" for item in first)
    assert all(item["canonical_alignment"]["reconstructed_text"] == item["target_text"] for item in first)


def test_summary_keeps_identity_controls_and_duplicate_corruptions_separate():
    predictions = [
        {"case_type": "new_duplicate_corruption", "source_text": "Ang bata bata.", "target_text": "Ang bata.",
         "passes": [{"output_text": "Ang bata."}, {"output_text": "Ang bata bata."}]},
        {"case_type": "identity_repeat", "source_text": "Sige sige.", "target_text": "Sige sige.",
         "passes": [{"output_text": "Sige sige."}, {"output_text": "Sige."}]},
    ]
    summary = diagnose.summarize_predictions(predictions)
    assert summary["case_counts"] == {"new_duplicate_corruption": 1, "identity_repeat": 1}
    assert summary["by_pass"][0]["groups"]["new_duplicate_corruption"]["exact_matches"] == 1
    assert summary["by_pass"][1]["groups"]["new_duplicate_corruption"]["exact_matches"] == 0
    assert summary["by_pass"][1]["groups"]["identity_repeat"]["exact_matches"] == 0


def test_run_manifest_rejects_changed_case_bytes_and_non_dev_provenance(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"case_id":"fixture"}\n', encoding="utf-8")
    path = tmp_path / "manifest.json"
    manifest = {
        "source": {"sha256": diagnose.EXPECTED_DEV_SHA256, "split": "dev",
                   "dataset_stage": "dataset2_stage3", "rows": 50_400},
        "checkpoint_reference": {"sha256": diagnose.EXPECTED_CHECKPOINT_SHA256},
        "case_file": cases.name, "cases_sha256": diagnose.sha256_file(cases),
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    diagnose.validate_prepared_cases(cases, path)
    cases.write_text('{"case_id":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="filename/hash mismatch"):
        diagnose.validate_prepared_cases(cases, path)
    manifest["source"]["split"] = "synthetic_test"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned Stage-3 DEV"):
        diagnose.validate_prepared_cases(cases, path)
