"""Normalize Filipino words or text with a trained N-gram model."""

import argparse
from pathlib import Path

from ngram_normalizer import DEFAULT_MODEL, load_model, normalize_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="+", help="Word or sentence to normalize")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Trained JSON model")
    parser.add_argument("--cutoff", type=int, default=100, help="Candidate cutoff")
    args = parser.parse_args()

    text = " ".join(args.text)
    model = load_model(args.model)
    print(normalize_text(text, model, cutoff=args.cutoff))


if __name__ == "__main__":
    main()
