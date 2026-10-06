"""Command-line orchestration for FILLY's research evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.dataset import DatasetError, load_dataset
from app.evaluation.engine import evaluate, scorer_provenance
from app.evaluation.reporting import render_console, write_reports


FILLY_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = FILLY_ROOT / "backend"


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python filly_cli.py",
        description="Evaluate FILLY with GEC only, normalization plus GEC, or both modes.",
        epilog=(
            "ERR is the manuscript's normalization-stage error reduction percentage: "
            "100 * (raw-to-target edits - normalized-to-target edits) / raw-to-target edits. "
            "The CSV target is the shared final-text reference. Final-system ERR is reported "
            "separately as a fraction from raw input to final GEC output."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluation = subparsers.add_parser(
        "evaluate",
        help="evaluate a CSV dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Validate the entire CSV, then evaluate the requested FILLY pipeline mode.",
        epilog=(
            "ERR (%) = 100 * (raw-to-target edit count - normalized-to-target edit count) "
            "/ raw-to-target edit count, measured before GEC. Final-system ERR is a separate "
            "fraction from raw input to final GEC output. Both use the CSV target as a shared "
            "operational reference."
        ),
    )
    evaluation.add_argument("dataset", help="CSV with input and target text columns")
    evaluation.add_argument(
        "--normalization",
        required=True,
        type=str.lower,
        choices=("on", "off", "both"),
        help="off runs GEC only; on runs normalization then GEC; both runs both paths",
    )
    evaluation.add_argument("--output-dir", help="directory for JSON and CSV reports")
    evaluation.add_argument("--input-column", help="explicit source-text column override")
    evaluation.add_argument("--target-column", help="explicit reference-text column override")
    evaluation.add_argument(
        "--correction-column", help="explicit optional correction metadata column override",
    )
    evaluation.add_argument("--type-column", help="explicit normalization category column override")
    evaluation.add_argument("--limit", type=_positive_int, help="evaluate the first N validated rows")
    evaluation.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), type=str.lower,
        help="inference device; defaults to backend Settings.DEVICE",
    )
    evaluation.add_argument(
        "--gec-passes", type=_positive_int,
        help="sequential GEC decoder passes (research default is one; production default is five)",
    )
    return parser


def _sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        if not candidate.is_file():
            return None
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _git_state(path: str | Path) -> dict[str, Any]:
    location = Path(path).resolve()
    try:
        root = subprocess.run(
            ["git", "-C", str(location), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", root, "status", "--short", "--untracked-files=no"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.splitlines()
        return {"root": root, "head": head, "dirty": bool(status), "status": status}
    except (OSError, subprocess.SubprocessError):
        return {"root": None, "head": None, "dirty": None, "status": None}


def _hash_paths(paths: tuple[Path, ...] | list[Path]) -> list[dict[str, str | None]]:
    records: list[dict[str, str | None]] = []
    for path in paths:
        resolved = Path(path).resolve()
        records.append({"path": str(resolved), "sha256": _sha256(resolved)})
    return records


def _tokenizer_assets(service: Any) -> list[dict[str, str | None]]:
    tokenizer = service.tokenizer
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    candidates: set[Path] = set()
    for key, value in init_kwargs.items():
        if (key.endswith("_file") or key == "tokenizer_file") and isinstance(value, (str, Path)):
            candidate = Path(value).expanduser()
            if candidate.is_file():
                candidates.add(candidate.resolve())
    return _hash_paths(sorted(candidates, key=lambda path: str(path)))


def _code_hashes() -> dict[str, Any]:
    from app.evaluation import cli as cli_module
    from app.evaluation import dataset as dataset_module
    from app.evaluation import engine as engine_module
    from app.evaluation import reporting as reporting_module

    paths = {
        "entrypoint": FILLY_ROOT / "filly_cli.py",
        "cli": Path(cli_module.__file__),
        "dataset": Path(dataset_module.__file__),
        "engine": Path(engine_module.__file__),
        "reporting": Path(reporting_module.__file__),
    }
    return {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in paths.items()}


def _build_provenance(
    *,
    dataset: Any,
    all_rows: int,
    rows: list[dict[str, Any]],
    normalization: str,
    passes: int,
    passes_source: str,
    device_request: str,
    settings: Any,
    gec_service: Any,
    normalizer: Any | None,
    columns: dict[str, str | None],
    limit: int | None,
    scorer: dict[str, str],
) -> dict[str, Any]:
    import app.services.gec as gec_module

    checkpoint_path = Path(gec_service.checkpoint_path).resolve()
    gec_source = Path(gec_module.__file__).resolve()
    # Use the actual service's package source file, even if its tokenizer is
    # represented by a dynamically generated tokenizer class.
    from app.services.gec_checkpoint import tokenizer as tokenizer_module

    normalizer_resources = None
    if normalizer is not None:
        resource_paths = tuple(Path(item) for item in normalizer._resource_signature)
        normalizer_resources = _hash_paths(resource_paths)
    scorer_path = Path(scorer["path"])
    scorer_tokenizer_module = importlib.import_module("filly_gec.tagging.tokenizer")
    scorer_tokenizer_path = Path(scorer_tokenizer_module.__file__).resolve()
    from app.services import normalizer as normalizer_module
    from app.services.gec_checkpoint import predictor as predictor_module
    from app.services.gec_checkpoint import segmentation as segmentation_module
    from app.services.gec_checkpoint import transforms as transforms_module

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(dataset.path),
            "sha256": _sha256(dataset.path),
            "rows_in_csv": all_rows,
            "rows_evaluated": len(rows),
            "limit": limit,
            "resolved_columns": columns,
        },
        "normalization_mode": normalization,
        "effective_configuration": {
            "gec_passes": passes,
            "gec_passes_source": passes_source,
            "production_configured_passes": settings.GECTOR_ITERATIONS,
            "device_requested": device_request,
            "device_used": str(gec_service.device),
            "max_input_length_setting": settings.MAX_INPUT_LENGTH,
            "gec_additional_keep_confidence": gec_module.ADDITIONAL_KEEP_CONFIDENCE,
            "gec_min_error_probability": gec_module.MIN_ERROR_PROBABILITY,
            "normalizer_max_edit_distance": (
                normalizer.max_edit_distance if normalizer is not None else None
            ),
            "normalizer_curated_mappings_enabled": (
                normalizer.use_curated_mappings if normalizer is not None else None
            ),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "service_metadata": dict(gec_service.metadata),
        },
        "normalizer_resources": normalizer_resources,
        "tokenizer_assets": _tokenizer_assets(gec_service),
        "scorer": {
            **scorer,
            "repository": _git_state(scorer_path.parent),
        },
        "scorer_tokenizer": {
            "path": str(scorer_tokenizer_path),
            "sha256": _sha256(scorer_tokenizer_path),
            "repository": _git_state(scorer_tokenizer_path.parent),
        },
        "repositories": {
            "filly": _git_state(FILLY_ROOT),
            "gec_service": _git_state(gec_source.parent),
        },
        "code_sha256": _code_hashes(),
        "runtime_sources": {
            "gec_service": {"path": str(gec_source), "sha256": _sha256(gec_source)},
            "tokenizer": {
                "path": str(Path(tokenizer_module.__file__).resolve()),
                "sha256": _sha256(tokenizer_module.__file__),
            },
            "decoder": {
                name: {"path": str(Path(module.__file__).resolve()), "sha256": _sha256(module.__file__)}
                for name, module in (
                    ("predictor", predictor_module),
                    ("transforms", transforms_module),
                    ("segmentation", segmentation_module),
                )
            },
            "normalizer": (
                {"path": str(Path(normalizer_module.__file__).resolve()), "sha256": _sha256(normalizer_module.__file__)}
                if normalizer is not None else None
            ),
        },
    }


def _evaluate(args: argparse.Namespace) -> int:
    # CSV resolution and validation happen before importing or loading either model.
    dataset = load_dataset(
        args.dataset,
        input_column=args.input_column,
        target_column=args.target_column,
        correction_column=args.correction_column,
        type_column=args.type_column,
    )
    all_rows = len(dataset.rows)
    rows = list(dataset.rows[: args.limit]) if args.limit else list(dataset.rows)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        if output_dir.exists() and not output_dir.is_dir():
            raise DatasetError(f"--output-dir is not a directory: {output_dir}")
        report_names = (
            "evaluation_summary.json",
            "evaluation_metrics.csv",
            "evaluation_predictions.csv",
            "evaluation_comparison.csv",
        )
        collisions = [
            str(output_dir / name)
            for name in report_names
            if (output_dir / name).resolve() == dataset.path
        ]
        if collisions:
            raise DatasetError(
                "output report path would overwrite the input dataset: " + ", ".join(collisions)
            )

    # Resolve scorer availability before loading the large neural checkpoint.
    scorer = scorer_provenance()

    from app.core.config import Settings

    env_file = BACKEND_ROOT / ".env"
    settings = Settings(_env_file=env_file if env_file.is_file() else None)
    pass_count = args.gec_passes if args.gec_passes is not None else 1
    pass_source = "command line" if args.gec_passes is not None else "research default"
    device_request = args.device or settings.DEVICE
    requested_device = None if device_request.lower() == "auto" else device_request.lower()

    from app.services.gec import GECService

    gec_service = GECService(
        checkpoint_path=settings.GECTOR_MODEL_PATH,
        device=requested_device,
    )
    normalizer_service = None
    if args.normalization in {"on", "both"}:
        from app.services.normalizer import get_normalizer

        normalizer_service = get_normalizer(
            rules_path=settings.NORMALIZER_RULES_PATH,
            vocabulary_path=settings.NORMALIZER_VOCAB_PATH,
        )

    def gec(text: str) -> str:
        return gec_service.correct_iteratively(text, iterations=pass_count).corrected_text

    normalizer = (
        (lambda text: normalizer_service.normalize_text(text)[0])
        if normalizer_service is not None
        else None
    )

    def progress(completed: int, total: int) -> None:
        if total >= 20:
            interval = max(1, total // 20)
            if completed % interval == 0 or completed == total:
                print(f"Evaluating {completed} / {total}", file=sys.stderr, end="\r", flush=True)
                if completed == total:
                    print(file=sys.stderr)

    result = evaluate(rows, args.normalization, gec, normalizer=normalizer, progress=progress)
    print(
        render_console(
            dataset_path=dataset.path,
            rows=len(rows),
            total_rows=all_rows,
            normalization=args.normalization,
            result=result,
            gec_passes=pass_count,
            passes_source=pass_source,
            production_passes=settings.GECTOR_ITERATIONS,
            device_requested=device_request,
            device_used=str(gec_service.device),
        )
    )
    if args.output_dir:
        provenance = _build_provenance(
            dataset=dataset,
            all_rows=all_rows,
            rows=rows,
            normalization=args.normalization,
            passes=pass_count,
            passes_source=pass_source,
            device_request=device_request,
            settings=settings,
            gec_service=gec_service,
            normalizer=normalizer_service,
            columns=dataset.columns,
            limit=args.limit,
            scorer=scorer,
        )
        outputs = write_reports(
            args.output_dir,
            normalization=args.normalization,
            result=result,
            provenance=provenance,
        )
        print("Reports:")
        for path in outputs.values():
            print(f"  {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            return _evaluate(args)
    except (DatasetError, OSError, RuntimeError, ValueError, TypeError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 2
