"""CSV, JSON, and console output for FILLY evaluation runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


METRIC_FIELDS = (
    "mode",
    "group",
    "n",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f05",
    "errors_before",
    "errors_after",
    "error_reduction_rate",
    "final_errors_after",
    "final_system_error_reduction_rate",
)
COMPARISON_FIELDS = (
    "group",
    "delta_precision",
    "delta_recall",
    "delta_f05",
    "delta_error_reduction_rate",
    "delta_final_system_error_reduction_rate",
)


def _prediction_fields(normalization: str) -> list[str]:
    fields = [
        "row_id",
        "input",
        "target",
        "gec_correction",
        "unnormalized_type",
        "types",
        "original_columns_json",
    ]
    if normalization in {"on", "both"}:
        fields.append("normalized_input")
    if normalization in {"off", "both"}:
        fields.extend(("output_without_normalization", "exact_without_normalization"))
    if normalization in {"on", "both"}:
        fields.extend(("output_with_normalization", "exact_with_normalization"))
    return fields


def _write_csv(path: Path, fieldnames: tuple[str, ...] | list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_reports(
    output_dir: str | Path,
    *,
    normalization: str,
    result: dict[str, list[dict[str, Any]]],
    provenance: dict[str, Any],
) -> dict[str, Path]:
    """Write stable machine-readable reports, preserving all source metadata."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "evaluation_metrics.csv"
    predictions_path = destination / "evaluation_predictions.csv"
    summary_path = destination / "evaluation_summary.json"

    metrics = result["metrics"]
    comparison = result.get("comparison", [])
    predictions = result["predictions"]
    _write_csv(metrics_path, METRIC_FIELDS, metrics)
    prediction_rows: list[dict[str, Any]] = []
    for row in predictions:
        prediction_rows.append(
            {
                **row,
                "types": ";".join(row.get("types", ())),
                "original_columns_json": json.dumps(
                    row.get("metadata", {}), ensure_ascii=False, sort_keys=True,
                ),
            }
        )
    _write_csv(predictions_path, _prediction_fields(normalization), prediction_rows)

    outputs = {
        "evaluation_summary": summary_path,
        "evaluation_metrics": metrics_path,
        "evaluation_predictions": predictions_path,
    }
    summary: dict[str, Any] = {
        "metadata": provenance,
        "metrics": metrics,
        "comparison": comparison,
        "units": {
            "precision_recall_f05": "fraction (0 to 1)",
            "error_reduction_rate": "percent; normalization stage before GEC",
            "final_system_error_reduction_rate": "fraction; raw input to final GEC output",
        },
    }
    if normalization == "both":
        comparison_path = destination / "evaluation_comparison.csv"
        _write_csv(comparison_path, COMPARISON_FIELDS, comparison)
        outputs["evaluation_comparison"] = comparison_path
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return outputs


def _value(value: Any, *, percent: bool = False, decimals: int = 3) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    return f"{number:.{decimals}f}%" if percent else f"{number:.{decimals}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    line = "| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |"
    divider = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(value.rjust(widths[i]) for i, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join((line, divider, *body))


def render_console(
    *,
    dataset_path: str | Path,
    rows: int,
    total_rows: int,
    normalization: str,
    result: dict[str, list[dict[str, Any]]],
    gec_passes: int,
    passes_source: str,
    production_passes: int,
    device_requested: str,
    device_used: str,
) -> str:
    """Return a compact, thesis-friendly metric summary."""
    lines = [
        "FILLY Evaluation",
        f"Dataset: {Path(dataset_path)}",
        f"Rows evaluated: {rows} / {total_rows}",
        f"GEC passes: {gec_passes} ({passes_source}; production setting: {production_passes})",
        f"Device: {device_used} (requested: {device_requested})",
    ]
    mode_titles = {
        "without_normalization": "WITHOUT NORMALIZATION",
        "with_normalization": "WITH NORMALIZATION",
    }
    requested_modes = {"off": ["without_normalization"], "on": ["with_normalization"]}
    requested_modes["both"] = ["without_normalization", "with_normalization"]
    for mode in requested_modes[normalization]:
        mode_rows = [row for row in result["metrics"] if row["mode"] == mode]
        lines.extend(("", f"Mode: {mode_titles[mode]}"))
        table_rows = [
            [
                str(row["group"]),
                str(row["n"]),
                str(row["tp"]),
                str(row["fp"]),
                str(row["fn"]),
                _value(row["precision"]),
                _value(row["recall"]),
                _value(row["f05"]),
                _value(row["error_reduction_rate"], percent=True, decimals=2),
                _value(row["final_system_error_reduction_rate"], decimals=4),
            ]
            for row in mode_rows
        ]
        lines.append(
            _table(
                ["Group", "N", "TP", "FP", "FN", "Precision", "Recall", "F0.5", "ERR (%)", "Final ERR (fraction)"],
                table_rows,
            )
        )
    if normalization == "both":
        lines.extend(("", "Normalization effect (with minus without)"))
        delta_rows = [
            [
                row["group"],
                _value(row["delta_precision"]),
                _value(row["delta_recall"]),
                _value(row["delta_f05"]),
                _value(row["delta_error_reduction_rate"], decimals=2),
                _value(row["delta_final_system_error_reduction_rate"], decimals=4),
            ]
            for row in result.get("comparison", [])
        ]
        lines.append(
            _table(
                ["Group", "Delta Precision", "Delta Recall", "Delta F0.5", "Delta ERR (pp)", "Delta Final ERR"],
                delta_rows,
            )
        )
    lines.extend(
        (
            "",
            "ERR (%) measures raw-to-normalized-stage error reduction before GEC, using the shared final target as its operational reference. "
            "Final ERR is a fraction from raw input to final GEC output.",
        )
    )
    return "\n".join(lines)
