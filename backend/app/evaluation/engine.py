"""Evaluation metrics for the FILLY normalization and GEC pipeline.

The GEC precision, recall, and F0.5 values use the authoritative
``filly_gec.evaluation.edit_metrics`` implementation.  The manuscript's
normalization Error Reduction Rate is computed from edit counts needed to
transform each stage's text into the shared final target:

    ERR = (errors_before_normalization - errors_after_normalization)
          / errors_before_normalization * 100

The shared target is the CSV's expected final normalized and grammatical
text.  The evaluation file does not provide an independent normalization-only
gold target, so the normalization-stage ERR is an operational comparison to
that final target, not a standalone normalization accuracy measure.  A zero
baseline has an undefined percentage and is reported as ``None``.  The
separate ``final_system_error_reduction_rate`` uses the raw input and final
GEC output errors against that same target and is reported as a fraction.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


_MODE_OFF = "without_normalization"
_MODE_ON = "with_normalization"
_BASE_GROUPS = ("overall", "slang", "abbreviation", "spelling variation")


@lru_cache(maxsize=1)
def _load_edit_metrics() -> tuple[Callable[..., dict[str, Any]], Callable[..., dict[str, Any]]]:
    """Load the repository's GEC scorer, adding its adjacent source tree if needed."""
    module_name = "filly_gec.evaluation.edit_metrics"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name not in {"filly_gec", "filly_gec.evaluation", module_name}:
            raise
        source_dir = Path(__file__).resolve().parents[4] / "Grammar-Error-Correction" / "src"
        if not source_dir.is_dir():
            raise ModuleNotFoundError(
                "The authoritative FILLY GEC scorer is unavailable. Expected it at "
                f"{source_dir / 'filly_gec' / 'evaluation' / 'edit_metrics.py'}."
            ) from exc
        source_text = str(source_dir)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        importlib.invalidate_caches()
        module = importlib.import_module(module_name)
    return module.final_output_edit_counts, module._prf


def scorer_provenance() -> dict[str, str]:
    """Return the authoritative edit scorer's resolved path and SHA-256."""
    final_output_edit_counts, _ = _load_edit_metrics()
    scorer_path = Path(importlib.import_module(final_output_edit_counts.__module__).__file__).resolve()
    return {
        "path": str(scorer_path),
        "sha256": hashlib.sha256(scorer_path.read_bytes()).hexdigest(),
    }


def _canonical_type(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Normalization type must be a string, got {type(value).__name__}.")
    normalized = " ".join(value.strip().replace("_", " ").replace("-", " ").split())
    if not normalized:
        raise ValueError("Normalization types cannot be blank.")
    key = normalized.casefold()
    if key == "slang":
        return "slang"
    if key in {"abbreviation", "abbreviations"}:
        return "abbreviation"
    if key in {"spelling variation", "spelling variations"}:
        return "spelling variation"
    if ";" in normalized:
        raise ValueError("Each evaluation row must have one dominant normalization type.")
    if key == "overall":
        raise ValueError("'overall' is a reserved metrics group, not a normalization type.")
    return key


def _row_types(row: dict[str, Any]) -> tuple[str, ...]:
    raw_types = row.get("types")
    if raw_types is None:
        raw_type = row.get("unnormalized_type")
        raw_types = () if raw_type is None or not str(raw_type).strip() else (raw_type,)
    if not isinstance(raw_types, (tuple, list)):
        raise ValueError("Each row's 'types' value must be a tuple or list of type labels.")
    canonical = tuple(dict.fromkeys(_canonical_type(item) for item in raw_types))
    if len(canonical) > 1:
        raise ValueError(
            "Each evaluation row must have one dominant normalization type; "
            "the research dataset assigns one category per item."
        )
    return canonical


def _validate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise TypeError("rows must be a list of row dictionaries.")
    seen_ids: set[int] = set()
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"Evaluation row {index} must be a dictionary.")
        row_id = row.get("row_id")
        if type(row_id) is not int:
            raise ValueError(f"Evaluation row {index} must have an integer row_id.")
        if row_id in seen_ids:
            raise ValueError(f"Duplicate evaluation row_id: {row_id}.")
        seen_ids.add(row_id)
        for field in ("input", "target"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Evaluation row {row_id} has a missing or blank {field} value.")
        types = _row_types(row)
        metadata = row.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Evaluation row {row_id} metadata must be a dictionary.")
        validated.append({**row, "types": types, "metadata": dict(metadata)})
    return validated


def _error_count(source: str, target: str) -> int:
    final_output_edit_counts, _ = _load_edit_metrics()
    # Score an identity prediction from this stage's text; the scorer's gold
    # edit count is exactly the number of source-relative edits needed to reach
    # the target, without double-counting a wrong substitution as FP + FN.
    return int(final_output_edit_counts(source, target, source)["gold_edit_count"])


def _metric_row(
    mode: str,
    group: str,
    row_scores: list[dict[str, int]],
    row_errors: list[tuple[int, int, int]],
) -> dict[str, Any]:
    _, prf = _load_edit_metrics()
    tp = sum(score["tp"] for score in row_scores)
    fp = sum(score["fp"] for score in row_scores)
    fn = sum(score["fn"] for score in row_scores)
    scores = prf(tp, fp, fn)
    errors_before = sum(item[0] for item in row_errors)
    errors_after = sum(item[1] for item in row_errors)
    final_errors_after = sum(item[2] for item in row_errors)
    error_reduction_rate = (
        (errors_before - errors_after) / errors_before * 100.0 if errors_before else None
    )
    final_system_error_reduction_rate = (
        (errors_before - final_errors_after) / errors_before if errors_before else None
    )
    return {
        "mode": mode,
        "group": group,
        "n": len(row_scores),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(scores["precision"]),
        "recall": float(scores["recall"]),
        "f05": float(scores["f0_5"]),
        "errors_before": errors_before,
        "errors_after": errors_after,
        "error_reduction_rate": error_reduction_rate,
        "final_errors_after": final_errors_after,
        "final_system_error_reduction_rate": final_system_error_reduction_rate,
    }


def evaluate(
    rows: list[dict[str, Any]],
    normalization: str,
    gec: Callable[[str], str],
    normalizer: Callable[[str], str | None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate GEC with normalization off, on, or in both modes.

    Rows require ``row_id``, ``input``, and ``target``.  ``types`` is the
    zero-or-one-item tuple of dominant normalization categories; ``metadata``
    preserves the CSV's original fields.  Empty type tuples remain in overall
    metrics without being assigned to a normalization category.
    """
    if not isinstance(normalization, str):
        raise ValueError("normalization must be one of: off, on, both.")
    normalization = normalization.strip().lower()
    if normalization not in {"off", "on", "both"}:
        raise ValueError("normalization must be one of: off, on, both.")
    if not callable(gec):
        raise TypeError("gec must be callable.")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None.")
    if normalization in {"on", "both"} and not callable(normalizer):
        raise ValueError("normalizer must be provided when normalization is on or both.")

    checked_rows = _validate_rows(rows)
    modes = [_MODE_OFF] if normalization == "off" else [_MODE_ON]
    if normalization == "both":
        modes = [_MODE_OFF, _MODE_ON]
    total = len(checked_rows) * len(modes)
    completed = 0

    prediction_by_id: dict[int, dict[str, Any]] = {}
    score_rows: dict[str, dict[int, dict[str, Any]]] = {mode: {} for mode in modes}
    for row in checked_rows:
        row_id = row["row_id"]
        prediction = {
            "row_id": row_id,
            "input": row["input"],
            "target": row["target"],
            "gec_correction": row.get("gec_correction"),
            "unnormalized_type": row.get("unnormalized_type"),
            "types": row["types"],
            "metadata": dict(row["metadata"]),
        }
        prediction_by_id[row_id] = prediction

    for mode in modes:
        for row in checked_rows:
            raw_input = row["input"]
            target = row["target"]
            normalized_input: str | None = None
            gec_input = raw_input
            if mode == _MODE_ON:
                normalized_result = normalizer(raw_input)  # type: ignore[misc]
                if not isinstance(normalized_result, str):
                    raise TypeError(
                        f"Normalizer returned {type(normalized_result).__name__} for row "
                        f"{row['row_id']}; expected str."
                    )
                normalized_input = normalized_result
                gec_input = normalized_input

            output = gec(gec_input)
            if not isinstance(output, str):
                raise TypeError(
                    f"GEC returned {type(output).__name__} for row {row['row_id']}; expected str."
                )

            final_output_edit_counts, _ = _load_edit_metrics()
            quality = final_output_edit_counts(raw_input, target, output)
            errors_before = _error_count(raw_input, target)
            errors_after = (
                _error_count(normalized_input, target)
                if normalized_input is not None
                else errors_before
            )
            final_errors_after = _error_count(output, target)
            row_result = {
                "quality": {"tp": int(quality["tp"]), "fp": int(quality["fp"]), "fn": int(quality["fn"])},
                "errors_before": errors_before,
                "errors_after": errors_after,
                "final_errors_after": final_errors_after,
                "output": output,
            }
            score_rows[mode][row["row_id"]] = row_result
            prediction = prediction_by_id[row["row_id"]]
            if mode == _MODE_OFF:
                prediction["output_without_normalization"] = output
                prediction["exact_without_normalization"] = output == target
            else:
                prediction["normalized_input"] = normalized_input
                prediction["output_with_normalization"] = output
                prediction["exact_with_normalization"] = output == target

            completed += 1
            if progress is not None:
                progress(completed, total)

    group_order = list(_BASE_GROUPS)
    row_group: dict[int, str | None] = {}
    for row in checked_rows:
        types = row["types"]
        category = types[0] if types else None
        row_group[row["row_id"]] = category
        if category is not None and category not in group_order:
            group_order.append(category)

    metrics: list[dict[str, Any]] = []
    for mode in modes:
        for group in group_order:
            group_ids = [
                row["row_id"]
                for row in checked_rows
                if group == "overall" or row_group[row["row_id"]] == group
            ]
            group_scores = [score_rows[mode][row_id]["quality"] for row_id in group_ids]
            group_errors = [
                (
                    score_rows[mode][row_id]["errors_before"],
                    score_rows[mode][row_id]["errors_after"],
                    score_rows[mode][row_id]["final_errors_after"],
                )
                for row_id in group_ids
            ]
            metrics.append(_metric_row(mode, group, group_scores, group_errors))

    comparison: list[dict[str, Any]] = []
    if normalization == "both":
        without = {row["group"]: row for row in metrics if row["mode"] == _MODE_OFF}
        with_norm = {row["group"]: row for row in metrics if row["mode"] == _MODE_ON}
        for group in group_order:
            off = without[group]
            on = with_norm[group]
            comparison.append({
                "group": group,
                "delta_precision": on["precision"] - off["precision"],
                "delta_recall": on["recall"] - off["recall"],
                "delta_f05": on["f05"] - off["f05"],
                "delta_error_reduction_rate": (
                    on["error_reduction_rate"] - off["error_reduction_rate"]
                    if on["error_reduction_rate"] is not None
                    and off["error_reduction_rate"] is not None
                    else None
                ),
                "delta_final_system_error_reduction_rate": (
                    on["final_system_error_reduction_rate"]
                    - off["final_system_error_reduction_rate"]
                    if on["final_system_error_reduction_rate"] is not None
                    and off["final_system_error_reduction_rate"] is not None
                    else None
                ),
            })

    return {
        "metrics": metrics,
        "comparison": comparison,
        "predictions": [prediction_by_id[row["row_id"]] for row in checked_rows],
    }
