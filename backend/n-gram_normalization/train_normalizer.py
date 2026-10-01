"""Compatibility CLI for FILLY's deterministic normalizer artifact builder."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = BACKEND_ROOT / "scripts" / "train_normalizer.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", "--pairs", dest="pairs", type=Path, help="Input,Target CSV file")
    parser.add_argument("--output", type=Path, help="Output artifact directory")
    parser.add_argument("--max-sub", "--max-ngram", dest="max_ngram", type=int)
    parser.add_argument(
        "--vocab",
        "--base-vocabulary",
        dest="base_vocabulary",
        type=Path,
        help="One-word-per-line base vocabulary",
    )
    parser.add_argument(
        "--curated-source",
        type=Path,
        help="Extracted source-sheet CSV for curated whole-form normalization rules",
    )
    parser.add_argument("--candidate-cutoff", type=int)
    args = parser.parse_args(argv)

    engine_args: list[str] = []
    for option, value in (
        ("--pairs", args.pairs),
        ("--output", args.output),
        ("--max-ngram", args.max_ngram),
        ("--base-vocabulary", args.base_vocabulary),
        ("--curated-source", args.curated_source),
        ("--candidate-cutoff", args.candidate_cutoff),
    ):
        if value is not None:
            engine_args.extend((option, str(value)))

    completed = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), *engine_args],
        cwd=BACKEND_ROOT,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
