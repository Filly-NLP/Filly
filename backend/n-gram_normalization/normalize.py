"""Normalize Filipino text with FILLY's current artifact-backed normalizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = BACKEND_ROOT / "artifacts" / "normalizer"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.normalizer import FilipinoNormalizer  # noqa: E402


def _parse_max_edit_distance(value: str) -> int | None:
    if value.strip().casefold() in {"none", "uncapped", "unlimited"}:
        return None
    try:
        distance = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer or 'none'") from exc
    if distance < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer or 'none'")
    return distance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="+", help="Word or sentence to normalize")
    parser.add_argument(
        "--artifact-dir",
        "--model",
        dest="artifact_dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Current normalizer artifact directory (--model is a legacy alias)",
    )
    parser.add_argument(
        "--max-edit-distance",
        type=_parse_max_edit_distance,
        default=2,
        help="Maximum Damerau-Levenshtein distance for an applied correction (default: 2)",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        help="Legacy compatibility check; must match the candidate cutoff recorded in the artifacts",
    )
    args = parser.parse_args(argv)

    if args.artifact_dir.is_file():
        parser.error("--model now names an artifact directory, not the legacy single-file JSON model")

    normalizer = FilipinoNormalizer(
        args.artifact_dir,
        max_edit_distance=args.max_edit_distance,
    )
    if args.cutoff is not None and args.cutoff != normalizer.candidate_cutoff:
        parser.error(
            f"--cutoff {args.cutoff} does not match the artifact cutoff "
            f"{normalizer.candidate_cutoff}; rebuild artifacts to change it"
        )

    normalized, _changes = normalizer.normalize_text(" ".join(args.text))
    print(normalized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
