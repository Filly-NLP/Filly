"""Run a paired local comparison of the current Stage-3 checkpoint and a candidate.

The command is intended for post-Colab evaluation only. It uses the native
``GECService`` checkpoint loader/decoder, scores final outputs with FILLY's
authoritative edit scorer, and reads only the pinned Stage-3 DEV parquet plus
the already-sealed duplicate stress case file. It never opens TEST data.

Example (run from the repository root with the GEC environment Python):

    python Filly/backend/scripts/compare_duplicate_candidate.py compare \
      --baseline-checkpoint Filly/best.pt \
      --candidate-checkpoint Filly/backend/reports/colab_download/<RUN_ID>/candidate_stage3_update100.pt \
      --candidate-hash-report Filly/backend/reports/colab_download/<RUN_ID>/hash_report.json \
      --output-dir Filly/backend/reports/duplicate_candidate_comparison
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


BACKEND_ROOT = Path(__file__).resolve().parents[1]
FILLY_ROOT = BACKEND_ROOT.parent
WORKSPACE_ROOT = FILLY_ROOT.parent
DEFAULT_DEV = (
    WORKSPACE_ROOT
    / "Grammar-Error-Correction"
    / "defense_prereqs"
    / "gec_cpu"
    / "production-data"
    / "dev"
    / "dataset2_stage3.parquet"
)
DEFAULT_STRESS_CASES = BACKEND_ROOT / "reports" / "duplicate_inference_stress_cases.jsonl"
DEFAULT_STRESS_MANIFEST = BACKEND_ROOT / "reports" / "duplicate_inference_stress_manifest.json"
DEFAULT_HISTORICAL_TRAIN_DISTRIBUTION = (
    BACKEND_ROOT / "reports" / "duplicate_data_duplicate_token_distribution.csv"
)
DEFAULT_TRAIN_DUPLICATES = (
    WORKSPACE_ROOT
    / "Grammar-Error-Correction"
    / "outputs"
    / "duplicate_bounded_v1"
    / "duplicate_bounded_v1.parquet"
)
DEFAULT_OUTPUT = BACKEND_ROOT / "reports" / "duplicate_candidate_comparison"
PASSES = tuple(str(value) for value in range(1, 6))
DECISIONS = frozenset(
    {
        "PROCEED TO CANONICAL RETRAINING",
        "RUN ANOTHER BOUNDED EXPERIMENT",
        "DO NOT PROCEED",
    }
)
ACCEPTANCE_THRESHOLDS = {
    "proceed_min_duplicate_recall_gain_stress_p1_p5": 0.05,
    "proceed_min_duplicate_precision_stress_p1_p5": 0.95,
    "proceed_max_duplicate_f0_5_decline_stress_p1_p5": 0.01,
    "proceed_max_historical_duplicate_dev_recall_decline_p1_p5": 0.01,
    "proceed_min_unseen_recall_gain_stress_p1_p5": 0.05,
    "proceed_min_content_recall_gain_stress_p1_p5": 0.05,
    "proceed_max_duplicate_exact_rate_decline_stress_p1_p5": 0.01,
    "proceed_max_stress_identity_false_positive_increase_p1_p5": 2,
    "proceed_max_official_identity_fp_rate_increase_p1_p5": 0.001,
    "proceed_min_official_f0_5_and_err_delta_p1_p5": -0.005,
    "proceed_min_major_family_f0_5_delta_p1_p5_n30": -0.01,
    "proceed_max_natural_repetition_identity_fp_rate_increase_p1_p5": 0.005,
    "proceed_max_stress_identity_false_delete_increase_p1_p5": 2,
    "veto_min_official_f0_5_or_err_delta_p1_p5": -0.02,
    "veto_min_family_f0_5_delta_p1_p5_n50": -0.05,
    "veto_min_duplicate_f0_5_delta_p1_p5": -0.05,
    "veto_max_stress_identity_false_positive_increase_p1_p5": 3,
    "veto_max_stress_identity_false_delete_increase_p1_p5": 3,
    "veto_max_official_identity_fp_rate_increase_p1_p5": 0.01,
    "veto_max_natural_repetition_identity_fp_rate_increase_p1_p5": 0.05,
}
WORD_RE = re.compile(r"[\w]+(?:['’][\w]+)*(?:[-‐‑][\w]+(?:['’][\w]+)*)*", re.UNICODE)

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts import diagnose_duplicate_inference as diagnose  # noqa: E402
from scripts import summarize_duplicate_diagnostics as diagnostic_summary  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluator_code_fingerprint() -> dict[str, dict[str, str]]:
    paths = {
        "comparison_command": BACKEND_ROOT / "scripts" / "compare_duplicate_candidate.py",
        "duplicate_diagnostic_runner": BACKEND_ROOT / "scripts" / "diagnose_duplicate_inference.py",
        "duplicate_diagnostic_summarizer": BACKEND_ROOT / "scripts" / "summarize_duplicate_diagnostics.py",
        "official_evaluation_engine": BACKEND_ROOT / "app" / "evaluation" / "engine.py",
        "native_gec_service": BACKEND_ROOT / "app" / "services" / "gec.py",
        "native_predictor": BACKEND_ROOT / "app" / "services" / "gec_checkpoint" / "predictor.py",
        "native_model": BACKEND_ROOT / "app" / "services" / "gec_checkpoint" / "model.py",
        "native_tokenizer": BACKEND_ROOT / "app" / "services" / "gec_checkpoint" / "tokenizer.py",
        "native_segmentation": BACKEND_ROOT / "app" / "services" / "gec_checkpoint" / "segmentation.py",
        "native_transforms": BACKEND_ROOT / "app" / "services" / "gec_checkpoint" / "transforms.py",
        "authoritative_edit_scorer": WORKSPACE_ROOT / "Grammar-Error-Correction" / "src" / "filly_gec" / "evaluation" / "edit_metrics.py",
    }
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(value)
    return rows


def _read_parquet_rows(path: Path, *, required_sha256: str | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    lowered = str(path).casefold().replace("\\", "/")
    if "synthetic_test" in lowered or path.name.casefold() == "final.parquet":
        raise ValueError("refusing TEST or synthetic-test input; only Stage-3 DEV is allowed")
    actual_hash = sha256_file(path)
    if required_sha256 is not None and actual_hash != required_sha256:
        raise ValueError(
            f"parquet SHA-256 mismatch for {path}: expected {required_sha256}, got {actual_hash}"
        )
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("reading frozen Parquet inputs requires pyarrow") from exc
    return parquet.read_table(path).to_pylist()


def _json_object(value: Any, *, field: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} is not valid JSON") from exc
        if isinstance(parsed, dict):
            return parsed
    return {}


def _engine_category_name(value: str) -> str:
    return " ".join(value.strip().replace("_", " ").replace("-", " ").split()).casefold()


def load_official_dev(path: Path) -> list[dict[str, Any]]:
    """Load and validate the frozen 50,400-row official Stage-3 DEV view."""
    rows = _read_parquet_rows(path, required_sha256=diagnose.EXPECTED_DEV_SHA256)
    if len(rows) != 50_400:
        raise ValueError(f"Stage-3 DEV must contain 50,400 rows, got {len(rows)}")
    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("split") != "dev" or row.get("dataset_stage") != "dataset2_stage3":
            raise ValueError("official Stage-3 DEV input contains rows from another split or stage")
        pair_id = row.get("pair_id")
        source, target = row.get("source_text"), row.get("target_text")
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
            raise ValueError("Stage-3 DEV pair IDs must be non-empty and unique")
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError(f"Stage-3 DEV row {pair_id} has invalid source/target text")
        seen.add(pair_id)
        tags = tuple(str(value) for value in (row.get("correction_tags") or ()))
        families = tuple(str(value) for value in (row.get("error_families") or ()))
        if bool(row.get("is_errorful")) != bool(tags):
            raise ValueError(f"Stage-3 DEV row {pair_id} has inconsistent error/tag metadata")
        operation = _json_object(row.get("generation_operation"), field="generation_operation")
        is_duplicate = "duplicate_word" in families or "$DELETE" in tags
        identity = not bool(row.get("is_errorful"))
        groups = {"official_stage3_dev"}
        groups.add("official_errorful" if not identity else "identity_controls")
        groups.update(f"family:{value}" for value in families)
        groups.update(f"tag:{value}" for value in tags)
        if is_duplicate:
            groups.add("historical_duplicate_dev")
        if identity and _has_adjacent_repeat(source):
            groups.add("natural_repetition_identity")
        prepared.append(
            {
                "case_id": f"stage3-dev:{pair_id}",
                "pair_id": pair_id,
                "source_text": source,
                "target_text": target,
                "is_errorful": not identity,
                "tags": list(tags),
                "families": list(families),
                "groups": sorted(groups),
                "operation": operation,
                "is_duplicate": is_duplicate,
                "duplicate_token": _duplicate_word(row, operation) if is_duplicate else "",
            }
        )
    if not any(row["is_duplicate"] for row in prepared):
        raise ValueError("official Stage-3 DEV contains no registered duplicate-word rows")
    return prepared


def load_stress_cases(cases_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    """Validate and read the already-sealed 100+20 stress suite."""
    diagnose.validate_prepared_cases(cases_path, manifest_path)
    cases = _read_jsonl(cases_path)
    if len(cases) != 120:
        raise ValueError(f"frozen duplicate stress suite must contain 120 rows, got {len(cases)}")
    case_ids = [case.get("case_id") for case in cases]
    if any(not isinstance(value, str) or not value for value in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("frozen stress case IDs must be unique non-empty strings")
    positives = [case for case in cases if case.get("case_type") == "new_duplicate_corruption"]
    controls = [case for case in cases if case.get("case_type", "").startswith("identity_")]
    if len(positives) != 100 or len(controls) != 20:
        raise ValueError("frozen stress suite must contain 100 corruptions and 20 identity controls")
    for case in cases:
        provenance = case.get("provenance", {})
        if provenance.get("split") != "dev" or provenance.get("dataset_stage") != "dataset2_stage3":
            raise ValueError(f"stress case {case['case_id']} is not bound to Stage-3 DEV")
    return cases


def load_training_duplicate_counts(path: Path) -> Counter[str]:
    """Read the 15-form parent exposure from the audited Stage-3 TRAIN distribution."""
    if not path.is_file():
        raise FileNotFoundError(path)
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("view") == "train_stage3" and row.get("statistic") == "token_form":
                form = row.get("value", "").strip().casefold()
                if form:
                    counts[form] += int(row["count"])
    if not counts:
        raise ValueError("historical Stage-3 TRAIN distribution contains no duplicate token forms")
    return counts


def load_augmentation_duplicate_counts(path: Path) -> Counter[str]:
    """Read vocabulary added by the frozen 1,280-row bounded augmentation."""
    rows = _read_parquet_rows(path)
    counts: Counter[str] = Counter()
    for row in rows:
        operation = _json_object(row.get("generation_operation"), field="generation_operation")
        if operation.get("type") != "duplicate":
            continue
        word = operation.get("duplicated_word_casefold") or operation.get("duplicated_word")
        if isinstance(word, str) and word.strip():
            counts[word.casefold()] += 1
    if not counts:
        raise ValueError("frozen training duplicate artifact contains no duplicate-word provenance")
    return counts


def _duplicate_word(row: Mapping[str, Any], operation: Mapping[str, Any]) -> str:
    word = operation.get("correct") or operation.get("duplicated_word")
    if isinstance(word, str) and word.strip():
        return word.strip()
    # Legacy Stage-3 operations store source/target spans instead of a dedicated
    # duplicate-word key. The exact target span is the gold retained word.
    spans = row.get("target_spans") or []
    for span in spans:
        if isinstance(span, Mapping) and isinstance(span.get("text"), str) and span["text"].strip():
            return span["text"].strip()
    return ""


def _has_adjacent_repeat(text: str) -> bool:
    words = [match.group(0).casefold() for match in WORD_RE.finditer(text)]
    return any(left == right for left, right in zip(words, words[1:]))


def _official_word_groups(
    row: Mapping[str, Any],
    historical_train_counts: Counter[str],
    augmentation_counts: Counter[str],
) -> set[str]:
    groups = set(row["groups"])
    if not row.get("is_duplicate"):
        return groups
    word = str(row.get("duplicate_token", "")).casefold()
    groups.add(
        f"duplicate_token_coverage={'seen' if word in historical_train_counts else 'unseen'}"
    )
    groups.add(
        f"augmentation_token_coverage={'seen' if word in augmentation_counts else 'unseen'}"
    )
    groups.add(
        f"target_class={'function' if word in diagnose.FUNCTION_WORDS else 'content'}"
    )
    operation = row.get("operation", {})
    position = operation.get("duplicate_position") or operation.get("position")
    if position in {"initial", "interior", "final"}:
        groups.add(f"duplicate_position={position}")
    elif position in {"first", "last"}:
        groups.add(f"duplicate_position={'initial' if position == 'first' else 'final'}")
    return groups


def _score_one(source: str, target: str, output: str) -> dict[str, Any]:
    final_output_edit_counts, _prf = diagnostic_summary._load_edit_metrics()
    scorer_module = importlib.import_module(final_output_edit_counts.__module__)
    edit_sets = scorer_module._final_output_edit_sets
    counts = final_output_edit_counts(source, target, output)
    gold = Counter({key: value for key, value in edit_sets(source, target).items() if key[1] == "delete"})
    predicted = Counter({key: value for key, value in edit_sets(source, output).items() if key[1] == "delete"})
    residual = final_output_edit_counts(output, target, output)["gold_edit_count"]
    return {
        **counts,
        "exact": output == target,
        "residual_edits": residual,
        "delete_tp": sum((gold & predicted).values()),
        "delete_fp": sum((predicted - gold).values()),
        "delete_fn": sum((gold - predicted).values()),
    }


def summarize_official_dev(
    cases: list[dict[str, Any]], predictions: list[dict[str, Any]],
    historical_train_counts: Counter[str], augmentation_counts: Counter[str],
) -> dict[str, Any]:
    """Aggregate final-output metrics by pass and actual DEV tags/families."""
    if len(cases) != len(predictions):
        raise ValueError("official DEV prediction count differs from its frozen row count")
    if [case["case_id"] for case in cases] != [row.get("case_id") for row in predictions]:
        raise ValueError("official DEV prediction IDs/order do not match the frozen rows")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    official_rows: list[dict[str, Any]] = []
    outputs_by_pass: dict[str, list[str]] = {pass_number: [] for pass_number in PASSES}
    for case, prediction in zip(cases, predictions):
        if prediction.get("source_text") != case["source_text"] or prediction.get("target_text") != case["target_text"]:
            raise ValueError(f"source/target mismatch for {case['case_id']}")
        outputs = prediction.get("passes", [])
        if len(outputs) != 5 or prediction.get("error"):
            raise ValueError(f"incomplete five-pass trace for {case['case_id']}")
        groups = _official_word_groups(case, historical_train_counts, augmentation_counts)
        if len(case["families"]) > 1:
            raise ValueError(f"official Stage-3 row {case['case_id']} has multiple error families")
        official_rows.append(
            {
                "row_id": len(official_rows) + 1,
                "input": case["source_text"],
                "target": case["target_text"],
                "types": list(case["families"]),
                "metadata": {"pair_id": case["pair_id"], "correction_tags": case["tags"]},
            }
        )
        for pass_number, pass_row in enumerate(outputs, 1):
            output = pass_row.get("output_text")
            if not isinstance(output, str):
                raise ValueError(f"pass {pass_number} has no output text for {case['case_id']}")
            outputs_by_pass[str(pass_number)].append(output)
            score = _score_one(case["source_text"], case["target_text"], output)
            score.update({"case_id": case["case_id"], "output_text": output})
            for group in groups:
                grouped[(str(pass_number), group)].append(score)
    # Reuse FILLY's standard evaluation engine on the already-computed outputs.
    # This validates that the same source-relative metrics/ERR used in normal
    # official DEV reports agree with the finer tag and diagnostic group views.
    from app.evaluation.engine import evaluate as official_evaluate

    official_engine_by_pass: dict[str, dict[str, Any]] = {}
    for pass_number in PASSES:
        cached_outputs = outputs_by_pass[pass_number]
        cursor = iter(zip(official_rows, cached_outputs))

        def cached_gec(text: str) -> str:
            try:
                row, output = next(cursor)
            except StopIteration as exc:
                raise ValueError("official evaluator requested more predictions than were cached") from exc
            if text != row["input"]:
                raise ValueError("official evaluator row order differs from cached native GEC predictions")
            return output

        official_result = official_evaluate(official_rows, "off", cached_gec)
        official_groups = {
            item["group"]: item
            for item in official_result["metrics"]
            if item["mode"] == "without_normalization"
        }
        if next(cursor, None) is not None:
            raise ValueError("official evaluator consumed fewer rows than were cached")
        for group_name, local_group in (
            ("overall", "official_stage3_dev"),
            *[
                (_engine_category_name(name), f"family:{name}")
                for name in sorted({family for row in cases for family in row["families"]})
            ],
        ):
            official_metric = official_groups.get(group_name)
            local_rows = grouped.get((pass_number, local_group), [])
            local_metric = diagnostic_summary.metrics(local_rows) if local_rows else None
            if official_metric is None or local_metric is None:
                raise ValueError(f"official evaluator did not return expected DEV group {group_name!r}")
            expected = {
                "n": local_metric["n"],
                "tp": local_metric["tp"],
                "fp": local_metric["fp"],
                "fn": local_metric["fn"],
            }
            observed = {key: int(official_metric[key]) for key in expected}
            if observed != expected:
                raise ValueError(f"official evaluator counts disagree for pass {pass_number}, group {group_name}")
            if abs(float(official_metric["f05"]) - float(local_metric["f0_5"])) > 1e-12:
                raise ValueError(f"official evaluator F0.5 disagrees for pass {pass_number}, group {group_name}")
            official_err = official_metric["final_system_error_reduction_rate"]
            if official_err is not None and abs(float(official_err) - float(local_metric["diagnostic_err"])) > 1e-12:
                raise ValueError(f"official evaluator ERR disagrees for pass {pass_number}, group {group_name}")
        official_engine_by_pass[pass_number] = {
            group: {
                "n": item["n"], "tp": item["tp"], "fp": item["fp"], "fn": item["fn"],
                "precision": item["precision"], "recall": item["recall"], "f0_5": item["f05"],
                "final_system_error_reduction_rate": item["final_system_error_reduction_rate"],
            }
            for group, item in official_groups.items()
        }

    by_pass = {
        pass_number: {
            group: _with_exact_rate(diagnostic_summary.metrics(rows))
            for (number, group), rows in sorted(grouped.items())
            if number == pass_number
        }
        for pass_number in PASSES
    }
    return {
        "n": len(cases),
        "duplicate_rows": sum(case["is_duplicate"] for case in cases),
        "official_evaluator": official_engine_by_pass,
        "by_pass": by_pass,
    }


def _with_exact_rate(metrics: dict[str, Any]) -> dict[str, Any]:
    """Expose exact-target accuracy alongside the authoritative edit metrics."""
    metrics["exact_rate"] = (
        metrics["exact_count"] / metrics["n"] if metrics["n"] else None
    )
    return metrics


def _stress_focus_piece_records(service: Any, case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Count fast-tokenizer pieces at the two frozen duplicate focus spans."""
    focus_spans = case.get("focus_spans") or []
    if not focus_spans:
        return []
    from app.services.gec_checkpoint.tokenizer import tokenize_with_offsets

    tokenized = tokenize_with_offsets(case["source_text"])
    focus_indices: dict[int, Mapping[str, Any]] = {}
    for focus in focus_spans:
        for index, span in enumerate(tokenized.spans):
            if span.start == focus.get("start") and span.end == focus.get("end"):
                focus_indices[index] = focus
                break
    if not focus_indices:
        return []
    encoded = service.tokenizer(
        [list(tokenized.tokens)],
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    word_ids = encoded.word_ids(0)
    piece_counts: Counter[int] = Counter(index for index in word_ids if index is not None)
    return [
        {
            "token": tokenized.tokens[index],
            "subword_piece_count": int(piece_counts[index]),
            "focus_role": focus.get("role"),
        }
        for index, focus in sorted(focus_indices.items())
    ]


def _prediction_record(service: Any, case: Mapping[str, Any], *, include_focus: bool) -> dict[str, Any]:
    result = service.correct_iteratively(case["source_text"], iterations=5)
    if len(result.passes) != 5:
        raise RuntimeError(f"native GEC service returned {len(result.passes)} passes for {case['case_id']}")
    record = {
        "case_id": case["case_id"],
        "source_text": case["source_text"],
        "target_text": case["target_text"],
        "passes": [{"output_text": stage.output_text} for stage in result.passes],
    }
    if include_focus:
        record.update(
            {
                key: case.get(key)
                for key in (
                    "case_type", "stratum", "target_class", "length_bucket",
                    "duplicate_source_span", "focus_spans", "provenance",
                )
            }
        )
        record["topk_focus"] = _stress_focus_piece_records(service, case)
    return record


def _signature(service: Any, checkpoint_path: Path) -> dict[str, Any]:
    metadata = service.metadata
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path),
        "stage": metadata.get("stage"),
        "selected_best": metadata.get("selected_best"),
        "run_id": metadata.get("run_id"),
        "model_id": metadata.get("model_id"),
        "model_revision": metadata.get("model_revision"),
        "label_vocab_size": metadata.get("label_vocab_size"),
        "label_vocab_sha256": metadata.get("label_vocab_sha256"),
        "vocabulary_support_scope": metadata.get("vocabulary_support_scope"),
        "transformers_version": metadata.get("transformers_version"),
        "decoder": {
            "iterations": 5,
            "additional_keep_confidence": service.predictor.additional_confidence,
            "minimum_error_probability": service.predictor.min_error_probability,
        },
    }


def _require_candidate_compatibility(baseline: Mapping[str, Any], candidate: Mapping[str, Any], baseline_vocab: Mapping[str, int], candidate_vocab: Mapping[str, int]) -> None:
    if baseline.get("stage") != "stage3" or baseline.get("selected_best") is not True:
        raise ValueError("baseline must be the selected Stage-3 checkpoint")
    if candidate.get("stage") != "stage3" or candidate.get("selected_best") is not True:
        raise ValueError("candidate must be a finalized native Stage-3 checkpoint accepted by GECService")
    for key in (
        "model_id", "model_revision", "label_vocab_size", "label_vocab_sha256",
        "vocabulary_support_scope", "transformers_version", "decoder",
    ):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"candidate GEC checkpoint signature differs from baseline: {key}")
    if dict(baseline_vocab) != dict(candidate_vocab):
        raise ValueError("candidate label vocabulary differs from the current Stage-3 checkpoint")
    if baseline.get("sha256") == candidate.get("sha256"):
        raise ValueError("candidate checkpoint hash is identical to the current Stage-3 checkpoint")


def _run_checkpoint(
    label: str,
    checkpoint_path: Path,
    dev_cases: list[dict[str, Any]],
    stress_cases: list[dict[str, Any]],
    historical_train_counts: Counter[str],
    augmentation_counts: Counter[str],
    output_dir: Path,
    *,
    stress_cases_path: Path,
    expected_baseline_signature: Mapping[str, Any] | None = None,
    expected_baseline_vocab: Mapping[str, int] | None = None,
    expected_checkpoint_sha256: str | None = None,
    device: str,
    progress_every: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    os.environ.setdefault("GECTOR_LOCAL_FILES_ONLY", "true")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from app.services.gec import GECService

    service = GECService(checkpoint_path=checkpoint_path, device=device)
    signature = _signature(service, checkpoint_path)
    if expected_checkpoint_sha256 is not None and signature["sha256"] != expected_checkpoint_sha256:
        raise ValueError(f"{label} checkpoint hash changed between preflight and native GEC loading")
    if signature["stage"] != "stage3" or signature["selected_best"] is not True:
        raise ValueError(f"{label} checkpoint is not a finalized Stage-3 best checkpoint")
    if expected_baseline_signature is not None:
        _require_candidate_compatibility(
            expected_baseline_signature, signature,
            expected_baseline_vocab or {}, service.label_vocab,
        )
    if service.iterations < 1:
        raise ValueError("GECService reports an invalid active iteration setting")

    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    dev_path = label_dir / "stage3_dev_predictions.jsonl"
    started = time.perf_counter()
    dev_predictions: list[dict[str, Any]] = []
    with dev_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, case in enumerate(dev_cases, 1):
            record = _prediction_record(service, case, include_focus=False)
            dev_predictions.append(record)
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if index == 1 or index % progress_every == 0 or index == len(dev_cases):
                print(f"{label} official DEV {index}/{len(dev_cases)}", file=sys.stderr, flush=True)
    dev_summary = summarize_official_dev(
        dev_cases, dev_predictions, historical_train_counts, augmentation_counts
    )
    del dev_predictions

    stress_path = label_dir / "duplicate_stress_predictions.jsonl"
    stress_predictions: list[dict[str, Any]] = []
    with stress_path.open("w", encoding="utf-8", newline="\n") as stream:
        for case in stress_cases:
            record = _prediction_record(service, case, include_focus=True)
            record.update(
                {
                    "case_type": case["case_type"],
                    "stratum": case.get("stratum", ""),
                    "target_class": case.get("target_class", "control"),
                    "length_bucket": case.get("length_bucket", ""),
                    "duplicate_source_span": case.get("duplicate_source_span"),
                    "focus_spans": case.get("focus_spans", []),
                    "provenance": case.get("provenance", {}),
                }
            )
            stress_predictions.append(record)
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    stress_facts, stress_rows = diagnostic_summary.summarize(
        stress_cases, stress_predictions, historical_train_counts
    )
    for groups in stress_facts["by_pass"].values():
        for metrics in groups.values():
            _with_exact_rate(metrics)
    augmentation_facts, _augmentation_rows = diagnostic_summary.summarize(
        stress_cases, stress_predictions, augmentation_counts
    )
    stress_facts["augmentation_training_coverage_by_pass"] = {
        pass_number: {
            "seen": augmentation_facts["by_pass"][pass_number].get(
                "duplicate_token_coverage=seen"
            ),
            "unseen": augmentation_facts["by_pass"][pass_number].get(
                "duplicate_token_coverage=unseen"
            ),
        }
        for pass_number in PASSES
    }
    stress_facts["case_file_sha256"] = sha256_file(stress_cases_path)
    stress_facts["predictions_sha256"] = sha256_file(stress_path)
    (label_dir / "duplicate_stress_summary.json").write_text(
        json.dumps(stress_facts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    elapsed = time.perf_counter() - started
    result = {
        "checkpoint": signature,
        "official_stage3_dev": dev_summary,
        "frozen_duplicate_stress": stress_facts,
        "timing_seconds": elapsed,
        "outputs": {
            "stage3_dev_predictions": str(dev_path),
            "stage3_dev_predictions_sha256": sha256_file(dev_path),
            "duplicate_stress_predictions": str(stress_path),
            "duplicate_stress_predictions_sha256": sha256_file(stress_path),
            "duplicate_stress_summary": str(label_dir / "duplicate_stress_summary.json"),
        },
    }
    (label_dir / "facts.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    label_vocab = dict(service.label_vocab)
    del service, stress_predictions, stress_rows, _augmentation_rows
    gc.collect()
    return result, label_vocab


def _metric(result: Mapping[str, Any], suite: str, pass_number: str, group: str) -> Mapping[str, Any]:
    if suite == "official_stage3_dev":
        by_pass = result.get("official_stage3_dev", {}).get("by_pass", {})
    elif suite == "frozen_duplicate_stress":
        by_pass = result.get("frozen_duplicate_stress", {}).get("by_pass", {})
    else:
        raise ValueError(f"unknown suite {suite}")
    value = by_pass.get(pass_number, {}).get(group)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing metric {suite}/pass{pass_number}/{group}")
    return value


def _delta(candidate_value: Any, baseline_value: Any) -> float | None:
    if candidate_value is None or baseline_value is None:
        return None
    return float(candidate_value) - float(baseline_value)


def _exact_rate(metrics: Mapping[str, Any]) -> float | None:
    count = int(metrics.get("n", 0))
    if count <= 0:
        return None
    return int(metrics["exact_count"]) / count


def classify_acceptance(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    """Return exactly one bounded-experiment decision from measured comparisons.

    The decision is a screening result for whether canonical retraining is
    justified. It recomputes every baseline in the same run and never imports
    historical score constants.
    """
    passes = ("1", "5")
    severe = False
    all_pass_duplicate_recall_gain = True
    all_pass_duplicate_quality_stable = True
    all_pass_generalization = True
    all_pass_identity_stable = True
    all_pass_family_stable = True
    all_pass_official_stable = True
    all_pass_historical_duplicate_stable = True
    shared_family_groups = set.intersection(*(
        set(baseline["official_stage3_dev"]["by_pass"][pass_number])
        & set(candidate["official_stage3_dev"]["by_pass"][pass_number])
        for pass_number in passes
    ))

    for pass_number in passes:
        base_dup = _metric(baseline, "frozen_duplicate_stress", pass_number, "duplicate_corruptions")
        cand_dup = _metric(candidate, "frozen_duplicate_stress", pass_number, "duplicate_corruptions")
        recall_gain = float(cand_dup["recall"]) - float(base_dup["recall"])
        all_pass_duplicate_recall_gain &= recall_gain >= ACCEPTANCE_THRESHOLDS[
            "proceed_min_duplicate_recall_gain_stress_p1_p5"
        ]
        all_pass_duplicate_quality_stable &= (
            float(cand_dup["precision"]) >= ACCEPTANCE_THRESHOLDS["proceed_min_duplicate_precision_stress_p1_p5"]
            and float(cand_dup["f0_5"]) >= float(base_dup["f0_5"]) - ACCEPTANCE_THRESHOLDS[
                "proceed_max_duplicate_f0_5_decline_stress_p1_p5"
            ]
            and _exact_rate(cand_dup) is not None
            and _exact_rate(base_dup) is not None
            and float(_exact_rate(cand_dup)) >= float(_exact_rate(base_dup)) - ACCEPTANCE_THRESHOLDS[
                "proceed_max_duplicate_exact_rate_decline_stress_p1_p5"
            ]
        )
        if float(cand_dup["f0_5"]) - float(base_dup["f0_5"]) < ACCEPTANCE_THRESHOLDS[
            "veto_min_duplicate_f0_5_delta_p1_p5"
        ]:
            severe = True

        for group, threshold_key in (
            ("duplicate_token_coverage=unseen", "proceed_min_unseen_recall_gain_stress_p1_p5"),
            ("target_class=content", "proceed_min_content_recall_gain_stress_p1_p5"),
        ):
            old_group = _metric(baseline, "frozen_duplicate_stress", pass_number, group)
            new_group = _metric(candidate, "frozen_duplicate_stress", pass_number, group)
            all_pass_generalization &= (
                float(new_group["recall"]) - float(old_group["recall"])
                >= ACCEPTANCE_THRESHOLDS[threshold_key]
            )
        old_function = _metric(baseline, "frozen_duplicate_stress", pass_number, "target_class=function")
        new_function = _metric(candidate, "frozen_duplicate_stress", pass_number, "target_class=function")
        all_pass_generalization &= float(new_function["recall"]) >= float(old_function["recall"]) - 0.01

        base_dev = _metric(baseline, "official_stage3_dev", pass_number, "official_stage3_dev")
        cand_dev = _metric(candidate, "official_stage3_dev", pass_number, "official_stage3_dev")
        if base_dev.get("diagnostic_err") is None or cand_dev.get("diagnostic_err") is None:
            raise ValueError("official DEV F0.5 and ERR are required for acceptance classification")
        severe_delta = ACCEPTANCE_THRESHOLDS["veto_min_official_f0_5_or_err_delta_p1_p5"]
        if (
            float(cand_dev["f0_5"]) - float(base_dev["f0_5"]) < severe_delta
            or float(cand_dev["diagnostic_err"]) - float(base_dev["diagnostic_err"]) < severe_delta
        ):
            severe = True
        stable_delta = ACCEPTANCE_THRESHOLDS["proceed_min_official_f0_5_and_err_delta_p1_p5"]
        all_pass_official_stable &= (
            float(cand_dev["f0_5"]) >= float(base_dev["f0_5"]) + stable_delta
            and float(cand_dev["diagnostic_err"]) >= float(base_dev["diagnostic_err"]) + stable_delta
        )

        old_duplicate_dev = _metric(baseline, "official_stage3_dev", pass_number, "historical_duplicate_dev")
        new_duplicate_dev = _metric(candidate, "official_stage3_dev", pass_number, "historical_duplicate_dev")
        all_pass_historical_duplicate_stable &= (
            float(new_duplicate_dev["recall"]) >= float(old_duplicate_dev["recall"])
            - ACCEPTANCE_THRESHOLDS["proceed_max_historical_duplicate_dev_recall_decline_p1_p5"]
        )

        old_identity = _metric(baseline, "frozen_duplicate_stress", pass_number, "identity_controls")
        new_identity = _metric(candidate, "frozen_duplicate_stress", pass_number, "identity_controls")
        stress_fp_increase = int(new_identity["fp"]) - int(old_identity["fp"])
        delete_fp_increase = int(new_identity.get("deletion_only", {}).get("fp", 0)) - int(
            old_identity.get("deletion_only", {}).get("fp", 0)
        )
        all_pass_identity_stable &= (
            stress_fp_increase <= ACCEPTANCE_THRESHOLDS[
                "proceed_max_stress_identity_false_positive_increase_p1_p5"
            ]
            and delete_fp_increase <= ACCEPTANCE_THRESHOLDS[
                "proceed_max_stress_identity_false_delete_increase_p1_p5"
            ]
        )
        if (
            stress_fp_increase > ACCEPTANCE_THRESHOLDS[
                "veto_max_stress_identity_false_positive_increase_p1_p5"
            ]
            or delete_fp_increase > ACCEPTANCE_THRESHOLDS[
                "veto_max_stress_identity_false_delete_increase_p1_p5"
            ]
        ):
            severe = True

        old_official_identity = _metric(baseline, "official_stage3_dev", pass_number, "identity_controls")
        new_official_identity = _metric(candidate, "official_stage3_dev", pass_number, "identity_controls")
        old_identity_rate = float(old_official_identity["fp"]) / max(int(old_official_identity["n"]), 1)
        new_identity_rate = float(new_official_identity["fp"]) / max(int(new_official_identity["n"]), 1)
        identity_rate_delta = new_identity_rate - old_identity_rate
        all_pass_identity_stable &= identity_rate_delta <= ACCEPTANCE_THRESHOLDS[
            "proceed_max_official_identity_fp_rate_increase_p1_p5"
        ]
        if identity_rate_delta > ACCEPTANCE_THRESHOLDS["veto_max_official_identity_fp_rate_increase_p1_p5"]:
            severe = True

        old_repetition = _metric(baseline, "official_stage3_dev", pass_number, "natural_repetition_identity")
        new_repetition = _metric(candidate, "official_stage3_dev", pass_number, "natural_repetition_identity")
        old_repeat_rate = float(old_repetition["fp"]) / max(int(old_repetition["n"]), 1)
        new_repeat_rate = float(new_repetition["fp"]) / max(int(new_repetition["n"]), 1)
        repeat_rate_delta = new_repeat_rate - old_repeat_rate
        all_pass_identity_stable &= repeat_rate_delta <= ACCEPTANCE_THRESHOLDS[
            "proceed_max_natural_repetition_identity_fp_rate_increase_p1_p5"
        ]
        if repeat_rate_delta > ACCEPTANCE_THRESHOLDS[
            "veto_max_natural_repetition_identity_fp_rate_increase_p1_p5"
        ]:
            severe = True

        for group in shared_family_groups:
            if not group.startswith("family:"):
                continue
            old_family = _metric(baseline, "official_stage3_dev", pass_number, group)
            new_family = _metric(candidate, "official_stage3_dev", pass_number, group)
            family_delta = float(new_family["f0_5"]) - float(old_family["f0_5"])
            if int(old_family["n"]) >= 30:
                all_pass_family_stable &= family_delta >= ACCEPTANCE_THRESHOLDS[
                    "proceed_min_major_family_f0_5_delta_p1_p5_n30"
                ]
            if int(old_family["n"]) >= 50 and family_delta < ACCEPTANCE_THRESHOLDS[
                "veto_min_family_f0_5_delta_p1_p5_n50"
            ]:
                severe = True

    if severe:
        return "DO NOT PROCEED"

    if (
        all_pass_duplicate_recall_gain
        and all_pass_duplicate_quality_stable
        and all_pass_generalization
        and all_pass_official_stable
        and all_pass_historical_duplicate_stable
        and all_pass_identity_stable
        and all_pass_family_stable
    ):
        return "PROCEED TO CANONICAL RETRAINING"
    return "RUN ANOTHER BOUNDED EXPERIMENT"


def _candidate_sha_from_hash_report(path: Path) -> str:
    """Extract a candidate-specific digest from a downloaded Colab report."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates: set[str] = set()
    direct_keys = {
        "candidate_sha256", "candidate_checkpoint_sha256", "candidate_best_sha256",
        "candidate_stage3_update100_sha256",
    }

    def visit(value: Any, *, candidate_context: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized = str(key).casefold().replace("-", "_")
                context = candidate_context or "candidate" in normalized
                if normalized in direct_keys and isinstance(item, str):
                    candidates.add(item.casefold())
                elif context and normalized in {"sha256", "checkpoint_sha256", "file_sha256"} and isinstance(item, str):
                    candidates.add(item.casefold())
                elif context and normalized in {"path", "filename", "name", "artifact"} and isinstance(item, str):
                    if "candidate_stage3_update100" in item.casefold() or "candidate_best" in item.casefold():
                        sibling_digest = next(
                            (
                                sibling
                                for sibling_key, sibling in value.items()
                                if str(sibling_key).casefold().replace("-", "_")
                                in {"sha256", "checkpoint_sha256", "file_sha256"}
                                and isinstance(sibling, str)
                            ),
                            None,
                        )
                        if sibling_digest:
                            candidates.add(sibling_digest.casefold())
                visit(item, candidate_context=context)
        elif isinstance(value, list):
            for item in value:
                visit(item, candidate_context=candidate_context)

    visit(payload)
    valid = {value for value in candidates if re.fullmatch(r"[0-9a-f]{64}", value)}
    if len(valid) != 1:
        raise ValueError(
            "hash report must contain exactly one candidate SHA-256 (candidate_checkpoint.sha256)"
        )
    return next(iter(valid))


def paired_outcomes(baseline_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    """Summarize paired case outcomes from the two identical input runs."""
    output: dict[str, Any] = {}
    for suite, filename in (
        ("official_stage3_dev", "stage3_dev_predictions.jsonl"),
        ("frozen_duplicate_stress", "duplicate_stress_predictions.jsonl"),
    ):
        baseline_rows = _read_jsonl(baseline_dir / filename)
        candidate_rows = _read_jsonl(candidate_dir / filename)
        if [row.get("case_id") for row in baseline_rows] != [row.get("case_id") for row in candidate_rows]:
            raise ValueError(f"paired {suite} predictions have different case IDs/order")
        by_pass = {}
        for index, pass_number in enumerate(PASSES):
            exact_gains = exact_losses = changed = duplicate_tp_gains = duplicate_tp_losses = 0
            for before, after in zip(baseline_rows, candidate_rows):
                old_output = before["passes"][index]["output_text"]
                new_output = after["passes"][index]["output_text"]
                changed += old_output != new_output
                exact_gains += old_output != before["target_text"] and new_output == after["target_text"]
                exact_losses += old_output == before["target_text"] and new_output != after["target_text"]
                if suite == "frozen_duplicate_stress" and before.get("case_type") == "new_duplicate_corruption":
                    old_tp = _score_one(before["source_text"], before["target_text"], old_output)["tp"]
                    new_tp = _score_one(after["source_text"], after["target_text"], new_output)["tp"]
                    duplicate_tp_gains += old_tp == 0 and new_tp > 0
                    duplicate_tp_losses += old_tp > 0 and new_tp == 0
            by_pass[pass_number] = {
                "prediction_changed_cases": changed,
                "exact_target_gains": exact_gains,
                "exact_target_losses": exact_losses,
                "duplicate_cases_with_final_edit_tp_gain": duplicate_tp_gains,
                "duplicate_cases_with_final_edit_tp_loss": duplicate_tp_losses,
            }
        output[suite] = by_pass
    return output


def compare(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"comparison output directory is not empty: {output_dir}")
    baseline_path = args.baseline_checkpoint.resolve()
    baseline_sha = sha256_file(baseline_path)
    if baseline_sha != diagnose.EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "baseline checkpoint hash differs from the Stage-3 checkpoint pinned by the frozen stress manifest"
        )
    dev_cases = load_official_dev(args.dev_parquet.resolve())
    stress_cases = load_stress_cases(args.stress_cases.resolve(), args.stress_manifest.resolve())
    historical_train_counts = load_training_duplicate_counts(
        args.historical_train_distribution.resolve()
    )
    augmentation_counts = load_augmentation_duplicate_counts(args.training_duplicates.resolve())
    expected_candidate_sha = (args.candidate_sha256 or _candidate_sha_from_hash_report(
        args.candidate_hash_report.resolve()
    )).casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_candidate_sha):
        raise ValueError("expected candidate SHA-256 must contain exactly 64 hexadecimal characters")
    actual_candidate_sha = sha256_file(args.candidate_checkpoint.resolve())
    if actual_candidate_sha != expected_candidate_sha:
        raise ValueError(
            "candidate checkpoint SHA-256 does not match the downloaded Colab hash report"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline, baseline_vocab = _run_checkpoint(
        "baseline", baseline_path, dev_cases, stress_cases,
        historical_train_counts, augmentation_counts, output_dir,
        stress_cases_path=args.stress_cases.resolve(),
        expected_checkpoint_sha256=baseline_sha,
        device=args.device, progress_every=args.progress_every,
    )
    candidate, candidate_vocab = _run_checkpoint(
        "candidate", args.candidate_checkpoint.resolve(), dev_cases, stress_cases,
        historical_train_counts, augmentation_counts, output_dir,
        stress_cases_path=args.stress_cases.resolve(),
        expected_baseline_signature=baseline["checkpoint"],
        expected_baseline_vocab=baseline_vocab,
        expected_checkpoint_sha256=expected_candidate_sha,
        device=args.device, progress_every=args.progress_every,
    )
    _require_candidate_compatibility(baseline["checkpoint"], candidate["checkpoint"], baseline_vocab, candidate_vocab)
    paired = paired_outcomes(output_dir / "baseline", output_dir / "candidate")
    decision = classify_acceptance(baseline, candidate)
    result = {
        "schema_version": 1,
        "scope": "same-code paired post-Colab Stage-3 DEV and sealed duplicate diagnostic comparison; TEST not read",
        "inputs": {
            "official_stage3_dev": {
                "path": str(args.dev_parquet.resolve()),
                "sha256": sha256_file(args.dev_parquet.resolve()),
                "rows": len(dev_cases),
                "historical_duplicate_dev_rows": sum(case["is_duplicate"] for case in dev_cases),
            },
            "frozen_stress_cases": {
                "path": str(args.stress_cases.resolve()),
                "sha256": sha256_file(args.stress_cases.resolve()),
                "rows": len(stress_cases),
                "duplicate_corruptions": sum(case.get("case_type") == "new_duplicate_corruption" for case in stress_cases),
                "identity_controls": sum(case.get("case_type", "").startswith("identity_") for case in stress_cases),
            },
            "training_duplicate_vocabulary": {
                "path": str(args.historical_train_distribution.resolve()),
                "sha256": sha256_file(args.historical_train_distribution.resolve()),
                "unique_historical_forms": len(historical_train_counts),
                "coverage_definition": "Stage-3 TRAIN duplicate forms before bounded augmentation",
            },
            "bounded_augmentation_vocabulary": {
                "path": str(args.training_duplicates.resolve()),
                "sha256": sha256_file(args.training_duplicates.resolve()),
                "unique_forms": len(augmentation_counts),
            },
        },
        "baseline": baseline,
        "candidate": candidate,
        "candidate_hash_verification": {
            "source": "--candidate-sha256" if args.candidate_sha256 else str(args.candidate_hash_report.resolve()),
            "expected_sha256": expected_candidate_sha,
            "observed_sha256_before_native_load": actual_candidate_sha,
            "verified": True,
        },
        "evaluator_code_fingerprint": evaluator_code_fingerprint(),
        "paired_outcomes": paired,
        "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
        "acceptance_decision": decision,
    }
    report_path = output_dir / "comparison.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_metrics_csv(result, output_dir / "group_metrics.csv")
    print(decision)
    return result


def _write_metrics_csv(result: Mapping[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for model_name in ("baseline", "candidate"):
        model = result[model_name]
        for suite_name, suite_key in (
            ("official_stage3_dev", "official_stage3_dev"),
            ("frozen_duplicate_stress", "frozen_duplicate_stress"),
        ):
            suite = model[suite_key]
            for pass_number, groups in suite["by_pass"].items():
                for group, metrics in groups.items():
                    rows.append({
                        "checkpoint": model_name, "suite": suite_name, "pass": pass_number,
                        "group": group, **metrics,
                    })
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("compare", help="run current Stage-3 vs candidate paired evaluation")
    command.add_argument("--baseline-checkpoint", type=Path, required=True)
    command.add_argument("--candidate-checkpoint", type=Path, required=True)
    command.add_argument("--dev-parquet", type=Path, default=DEFAULT_DEV)
    command.add_argument("--stress-cases", type=Path, default=DEFAULT_STRESS_CASES)
    command.add_argument("--stress-manifest", type=Path, default=DEFAULT_STRESS_MANIFEST)
    command.add_argument("--historical-train-distribution", type=Path, default=DEFAULT_HISTORICAL_TRAIN_DISTRIBUTION)
    command.add_argument("--training-duplicates", type=Path, default=DEFAULT_TRAIN_DUPLICATES)
    command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    command.add_argument("--progress-every", type=int, default=1000)
    candidate_hash = command.add_mutually_exclusive_group(required=True)
    candidate_hash.add_argument("--candidate-sha256", help="expected SHA-256 copied from the downloaded Colab hash report")
    candidate_hash.add_argument("--candidate-hash-report", type=Path, help="downloaded Colab hash_report.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if not args.baseline_checkpoint.is_file():
        parser.error(f"baseline checkpoint does not exist: {args.baseline_checkpoint}")
    if not args.candidate_checkpoint.is_file():
        parser.error(f"candidate checkpoint does not exist: {args.candidate_checkpoint}")
    compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
