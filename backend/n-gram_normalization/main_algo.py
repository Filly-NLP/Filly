"""Compatibility entry point for FILLY's current normalizer evaluator.

The former random cross-validation implementation is intentionally not
reimplemented. Use the repository evaluator's deterministic validation/test
splits instead.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SCRIPT = BACKEND_ROOT / "scripts" / "evaluate_normalizer.py"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.normalizer import FilipinoNormalizer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Other evaluation options pass through to backend/scripts/evaluate_normalizer.py; run that script with --help for details.",
    )
    parser.add_argument(
        "--mode",
        choices=("test", "eval_cv"),
        default="test",
        help="Legacy mode; 'test' selects the current reserved test split, 'eval_cv' is unsupported",
    )
    parser.add_argument(
        "--use_dld",
        "--use-dld",
        choices=("True", "False", "true", "false"),
        default="True",
        help="Legacy switch; the current engine always ranks candidates by DLD",
    )
    parser.add_argument("--max_sub", "--max-sub", type=int, help="Check the max N-gram size recorded in artifacts")
    parser.add_argument("--cutoff", type=int, help="Check the candidate cutoff recorded in artifacts")
    args, evaluator_args = parser.parse_known_args(argv)

    if args.mode == "eval_cv":
        parser.error("legacy random eval_cv is unsupported; use --split validation or --split test")
    if args.use_dld.casefold() != "true":
        parser.error("the current normalizer always ranks candidates with Damerau-Levenshtein distance")

    if args.max_sub is not None or args.cutoff is not None:
        normalizer = FilipinoNormalizer()
        if args.max_sub is not None and args.max_sub != normalizer.max_ngram:
            parser.error(
                f"--max_sub {args.max_sub} does not match artifact max N-gram size {normalizer.max_ngram}"
            )
        if args.cutoff is not None and args.cutoff != normalizer.candidate_cutoff:
            parser.error(
                f"--cutoff {args.cutoff} does not match artifact candidate cutoff "
                f"{normalizer.candidate_cutoff}"
            )

    has_split = any(argument == "--split" or argument.startswith("--split=") for argument in evaluator_args)
    if not has_split:
        evaluator_args = ["--split", "test", *evaluator_args]

    completed = subprocess.run(
        [sys.executable, str(EVALUATOR_SCRIPT), *evaluator_args],
        cwd=BACKEND_ROOT,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
