"""Acceptance decision tests use only synthetic metric records, no model runtime."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_duplicate_candidate.py"
SPEC = importlib.util.spec_from_file_location("compare_duplicate_candidate", SCRIPT)
compare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compare)


def metric(*, n=100, precision=0.96, recall=0.4, f0_5=0.8, err=0.5, fp=2, delete_fp=1):
    return {
        "n": n,
        "precision": precision,
        "recall": recall,
        "f0_5": f0_5,
        "diagnostic_err": err,
        "fp": fp,
        "deletion_only": {"fp": delete_fp},
        "exact_count": n // 2,
    }


def results(
    *, stress_recall=0.4, unseen_recall=0.1, content_recall=0.1,
    historical_recall=0.4, official_f05=0.8, official_err=0.5,
    stress_precision=0.96, identity_delete_fp=1, natural_fp=2,
    family_f05=0.8, stress_p5_recall=None, stress_p5_precision=None,
    stress_p5_f05=0.8, unseen_p5_recall=None, content_p5_recall=None,
):
    stress = {}
    for pass_number in ("1", "5"):
        is_last = pass_number == "5"
        stress[pass_number] = {
            "duplicate_corruptions": metric(
                precision=(stress_p5_precision if is_last and stress_p5_precision is not None else stress_precision),
                recall=(stress_p5_recall if is_last and stress_p5_recall is not None else stress_recall),
                f0_5=(stress_p5_f05 if is_last else 0.8),
            ),
            "duplicate_token_coverage=unseen": metric(
                n=60,
                recall=(unseen_p5_recall if is_last and unseen_p5_recall is not None else unseen_recall),
            ),
            "target_class=content": metric(
                n=50,
                recall=(content_p5_recall if is_last and content_p5_recall is not None else content_recall),
            ),
            "target_class=function": metric(n=50, recall=0.5),
            "identity_controls": metric(
                n=20, fp=identity_delete_fp, delete_fp=identity_delete_fp
            ),
        }
    official = {}
    for pass_number in ("1", "5"):
        official[pass_number] = {
            "official_stage3_dev": metric(
                n=50_400, f0_5=official_f05, err=official_err, precision=0.9, recall=0.8
            ),
            "historical_duplicate_dev": metric(n=1364, recall=historical_recall),
            "natural_repetition_identity": metric(n=199, fp=natural_fp, delete_fp=natural_fp),
            "identity_controls": metric(n=25_500, fp=10),
            "family:punctuation": metric(n=100, f0_5=family_f05),
        }
    return {
        "official_stage3_dev": {"by_pass": official},
        "frozen_duplicate_stress": {"by_pass": stress},
    }


def test_strong_duplicate_gain_with_stable_gec_returns_canonical_retraining_decision():
    baseline = results(stress_recall=0.4, unseen_recall=0, content_recall=0)
    candidate = results(
        stress_recall=0.47,
        unseen_recall=0.15,
        content_recall=0.18,
        historical_recall=0.43,
    )

    decision = compare.classify_acceptance(baseline, candidate)

    assert decision == "PROCEED TO CANONICAL RETRAINING"
    assert decision in compare.DECISIONS


def test_small_or_incomplete_generalization_gain_requests_another_bounded_experiment():
    baseline = results(stress_recall=0.4, unseen_recall=0, content_recall=0)
    candidate = results(stress_recall=0.43, unseen_recall=0.04, content_recall=0.03)

    decision = compare.classify_acceptance(baseline, candidate)

    assert decision == "RUN ANOTHER BOUNDED EXPERIMENT"
    assert decision in compare.DECISIONS


def test_large_overall_forgetting_returns_do_not_proceed():
    baseline = results()
    candidate = results(
        stress_recall=0.6,
        unseen_recall=0.3,
        content_recall=0.3,
        official_f05=0.76,
    )

    decision = compare.classify_acceptance(baseline, candidate)

    assert decision == "DO NOT PROCEED"
    assert decision in compare.DECISIONS


def test_identity_false_deletion_growth_can_veto_even_with_duplicate_recall_gain():
    baseline = results(identity_delete_fp=1)
    candidate = results(
        stress_recall=0.6,
        unseen_recall=0.3,
        content_recall=0.3,
        identity_delete_fp=5,
    )

    assert compare.classify_acceptance(baseline, candidate) == "DO NOT PROCEED"


def test_pass_five_duplicate_precision_collapse_blocks_canonical_retraining():
    baseline = results(stress_recall=0.4, unseen_recall=0, content_recall=0)
    candidate = results(
        stress_recall=0.47,
        unseen_recall=0.15,
        content_recall=0.18,
        stress_p5_recall=0.47,
        stress_p5_precision=0.75,
        unseen_p5_recall=0.15,
        content_p5_recall=0.18,
    )

    assert compare.classify_acceptance(baseline, candidate) == "RUN ANOTHER BOUNDED EXPERIMENT"


def test_pass_five_duplicate_recall_and_f05_collapse_vetoes_candidate():
    baseline = results(stress_recall=0.4, unseen_recall=0, content_recall=0)
    candidate = results(
        stress_recall=0.47,
        unseen_recall=0.15,
        content_recall=0.18,
        stress_p5_recall=0.1,
        stress_p5_f05=0.4,
        unseen_p5_recall=0.15,
        content_p5_recall=0.18,
    )

    assert compare.classify_acceptance(baseline, candidate) == "DO NOT PROCEED"


def test_official_evaluator_agrees_with_tagged_final_output_groups_and_uses_historical_coverage():
    cases = [
        {
            "case_id": "duplicate",
            "pair_id": "duplicate",
            "source_text": "si si",
            "target_text": "si",
            "is_errorful": True,
            "tags": ["$DELETE"],
            "families": ["duplicate_word"],
            "groups": [
                "official_stage3_dev", "official_errorful", "historical_duplicate_dev",
                "family:duplicate_word", "tag:$DELETE",
            ],
            "operation": {"type": "duplicate", "correct": "si"},
            "is_duplicate": True,
            "duplicate_token": "si",
        },
        {
            "case_id": "identity",
            "pair_id": "identity",
            "source_text": "Araw-araw siya.",
            "target_text": "Araw-araw siya.",
            "is_errorful": False,
            "tags": [],
            "families": [],
            "groups": ["official_stage3_dev", "identity_controls"],
            "operation": {},
            "is_duplicate": False,
            "duplicate_token": "",
        },
    ]
    predictions = [
        {
            "case_id": case["case_id"],
            "source_text": case["source_text"],
            "target_text": case["target_text"],
            "passes": [{"output_text": case["target_text"]} for _ in range(5)],
        }
        for case in cases
    ]

    report = compare.summarize_official_dev(
        cases,
        predictions,
        compare.Counter({"si": 10}),
        compare.Counter({"sa": 4}),
    )

    assert report["by_pass"]["1"]["historical_duplicate_dev"]["recall"] == 1.0
    assert report["by_pass"]["1"]["historical_duplicate_dev"]["exact_rate"] == 1.0
    assert report["by_pass"]["1"]["duplicate_token_coverage=seen"]["n"] == 1
    assert report["by_pass"]["1"]["augmentation_token_coverage=unseen"]["n"] == 1
    assert report["official_evaluator"]["1"]["overall"]["f0_5"] == 1.0
    assert report["official_evaluator"]["1"]["duplicate word"]["n"] == 1


def test_hash_report_requires_and_reads_candidate_specific_digest():
    expected = "a" * 64
    with tempfile.TemporaryDirectory() as temporary_directory:
        report = Path(temporary_directory) / "hash_report.json"
        report.write_text(
            json.dumps({"candidate_checkpoint_sha256": expected}),
            encoding="utf-8",
        )

        assert compare._candidate_sha_from_hash_report(report) == expected
