"""Score saved development stress traces without loading a model or test rows."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPORTS = BACKEND / "reports"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.evaluation.engine import _load_edit_metrics


def metrics(rows: list[dict]) -> dict:
    _, prf = _load_edit_metrics()
    totals = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn")}
    initial = sum(row["gold_edit_count"] for row in rows)
    residual = sum(row["residual_edits"] for row in rows)
    return {
        "n": len(rows), **prf(**totals),
        "deletion_only": prf(**{key: sum(row[f"delete_{key}"] for row in rows) for key in ("tp", "fp", "fn")}),
        "gold_edit_count": initial,
        "predicted_edit_count": sum(row["predicted_edit_count"] for row in rows),
        "exact_count": sum(row["exact"] for row in rows),
        "residual_edits": residual,
        "diagnostic_err": 1 - residual / initial if initial else None,
    }


def summarize(cases: list[dict], predictions: list[dict], train_counts: Counter) -> tuple[dict, list[dict]]:
    """Reject incomplete/mismatched traces, then count exact final-output edits."""
    case_map = {case["case_id"]: case for case in cases}
    prediction_map = {row["case_id"]: row for row in predictions}
    if len(case_map) != len(cases) or len(prediction_map) != len(predictions):
        raise ValueError("duplicate case IDs")
    if case_map.keys() != prediction_map.keys():
        raise ValueError("case/prediction IDs differ; incomplete inference must not receive a score")
    score, _ = _load_edit_metrics()
    edit_sets = importlib.import_module(score.__module__)._final_output_edit_sets
    records = []
    for case_id, case in case_map.items():
        prediction = prediction_map[case_id]
        if any(case[field] != prediction[field] for field in ("source_text", "target_text")):
            raise ValueError(f"source/target mismatch for {case_id}")
        passes = prediction.get("passes", [])
        if len(passes) != 5 or prediction.get("error"):
            raise ValueError(f"incomplete five-pass trace for {case_id}")
        positive = case["case_type"] == "new_duplicate_corruption"
        word = case.get("duplicate_source_span", {}).get("text", "") if positive else ""
        focus = prediction.get("topk_focus", [])
        piece_counts = [item["subword_piece_count"] for item in focus]
        pieces = max(piece_counts, default=0)
        piece_bucket = "1" if pieces == 1 else "2" if pieces == 2 else "3+" if pieces else "unobserved"
        known = "seen" if word.casefold() in train_counts else "unseen"
        for number, stage in enumerate(passes, start=1):
            output = stage["output_text"]
            counts = score(case["source_text"], case["target_text"], output)
            gold_deletes = Counter({key: n for key, n in edit_sets(case["source_text"], case["target_text"]).items() if key[1] == "delete"})
            predicted_deletes = Counter({key: n for key, n in edit_sets(case["source_text"], output).items() if key[1] == "delete"})
            residual = score(output, case["target_text"], output)["gold_edit_count"]
            exact = output == case["target_text"]
            if positive and counts["gold_edit_count"] != 1:
                raise ValueError(f"stress corruption {case_id} is not a single gold edit")
            records.append({
                "case_id": case_id, "pass": number, "case_type": case["case_type"],
                "stratum": case.get("stratum", ""), "target_class": case.get("target_class", "control"),
                "length_bucket": case.get("length_bucket", ""),
                "sentence_token_count": case.get("sentence_token_count", ""),
                "punctuation_context": case.get("punctuation_context", ""),
                "duplicate_token": word, "duplicate_train_occurrences": train_counts[word.casefold()],
                "duplicate_token_coverage": known if positive else "control",
                "subword_piece_count": pieces, "subword_bucket": piece_bucket,
                "both_copies_same_piece_count": len(piece_counts) == 2 and len(set(piece_counts)) == 1 if positive else None,
                "source_text": case["source_text"], "target_text": case["target_text"],
                "prediction": output, "exact": exact, **counts,
                "delete_tp": sum((gold_deletes & predicted_deletes).values()),
                "delete_fp": sum((predicted_deletes - gold_deletes).values()),
                "delete_fn": sum((gold_deletes - predicted_deletes).values()),
                "residual_edits": residual,
                "duplicate_removed": bool(positive and counts["tp"] == 1),
                "p1_any_detection_gate_block": any(item.get("gate_blocked_delete", False) for item in focus),
                "p1_any_keep_bonus_block": any(item.get("keep_bonus_suppressed_raw_delete", False) for item in focus),
                "p1_delete_outside_both_top5": bool(focus) and all(item.get("delete_raw_rank") is None for item in focus),
                "p1_topk": json.dumps(focus, ensure_ascii=False),
                "provenance": json.dumps(case.get("provenance", {}), ensure_ascii=False),
            })
    grouped = defaultdict(list)
    for row in records:
        number = str(row["pass"])
        grouped[(number, "all_cases")].append(row)
        positive = row["case_type"] == "new_duplicate_corruption"
        grouped[(number, "duplicate_corruptions" if positive else "identity_controls")].append(row)
        if positive:
            for field in ("target_class", "stratum", "length_bucket", "subword_bucket", "duplicate_token_coverage", "punctuation_context"):
                grouped[(number, f"{field}={row[field]}")].append(row)
    counts = {
        number: {group: metrics(rows) for (p, group), rows in sorted(grouped.items()) if p == number}
        for number in ("1", "2", "3", "4", "5")
    }
    p1_misses = [row for row in records if row["pass"] == 1 and row["case_type"] == "new_duplicate_corruption" and row["fn"]]
    return {
        "schema_version": 1,
        "scope": "new DEV stress suite only; no final-test inference or production change",
        "training_coverage_scope": "Stage 3 TRAIN duplicate row counts, casefolded forms; Stage 2 parent exposure audited separately in duplicate_data_report.json",
        "scoring": "exact source-relative final-output edits; no tag-location credit required",
        "deletion_only_note": "Secondary operation-specific view excludes collateral substitutions/insertions; primary category-row view includes them.",
        "err_definition": "1 - direct token edits(prediction,TARGET) / token edits(SOURCE,TARGET); diagnostic, not thesis ERR",
        "case_count": len(cases), "by_pass": counts,
        "p1_missed_duplicate_sites": len(p1_misses),
        "p1_miss_indicators_not_disjoint": {
            "detection_gate_would_block_delete": sum(row["p1_any_detection_gate_block"] for row in p1_misses),
            "keep_bonus_blocks_raw_delete": sum(row["p1_any_keep_bonus_block"] for row in p1_misses),
            "delete_outside_top5_at_both_copies": sum(row["p1_delete_outside_both_top5"] for row in p1_misses),
        },
        "limitations": [
            "Grouped precision is conditional on that diagnostic group, not natural prevalence.",
            "Subword, lexical, position, and length groups are correlated; differences do not prove causes.",
            "Identity repeats are corpus-preservation controls, not expert-validated intentional repetition gold.",
            "The same-output duplicate deletion slot is interchangeable; either copy may be deleted correctly.",
        ],
    }, records


def main() -> None:
    case_path = REPORTS / "duplicate_inference_stress_cases.jsonl"
    prediction_path = REPORTS / "duplicate_inference_predictions.jsonl"
    distribution_path = REPORTS / "duplicate_data_duplicate_token_distribution.csv"
    facts_path = REPORTS / "duplicate_inference_facts.json"
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    if facts.get("status") != "COMPLETED" or facts.get("limited_benchmark"):
        raise ValueError("only a complete, full stress run may be summarized")
    for path, key in ((case_path, "cases_sha256"), (prediction_path, "predictions_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != facts[key]:
            raise ValueError(f"saved run hash mismatch: {path.name}")
    cases = [json.loads(line) for line in case_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    predictions = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_counts = Counter()
    with distribution_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["view"] == "train_stage3" and row["statistic"] == "token_form":
                train_counts[row["value"].casefold()] += int(row["count"])
    summary, records = summarize(cases, predictions, train_counts)
    summary["source_sha256"] = {
        str(path.relative_to(BACKEND)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (case_path, prediction_path, distribution_path, facts_path)
    }
    (REPORTS / "duplicate_word_stress_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, selected in (
        ("duplicate_word_stress_results.csv", records),
        ("duplicate_word_failures.csv", [row for row in records if not row["exact"]]),
    ):
        with (REPORTS / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(selected)
    print(json.dumps({p: groups["duplicate_corruptions"] for p, groups in summary["by_pass"].items()}, indent=2))


if __name__ == "__main__":
    main()
