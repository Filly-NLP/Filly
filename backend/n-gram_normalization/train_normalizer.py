"""Train and save the Filipino N-gram normalization rules."""

import argparse
from pathlib import Path

from ngram_normalizer import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    DEFAULT_VOCAB,
    load_pairs_csv,
    save_model,
    train_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Input,Target CSV file")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL, help="Output JSON model")
    parser.add_argument("--max-sub", type=int, default=2, help="Maximum character N-gram size")
    parser.add_argument(
        "--vocab",
        type=Path,
        default=DEFAULT_VOCAB,
        help="Optional Tagalog vocabulary resource",
    )
    args = parser.parse_args()

    pairs = load_pairs_csv(args.data)
    vocabulary_path = args.vocab if args.vocab.is_file() else None
    if vocabulary_path is None:
        print(f"Warning: vocabulary file not found; using training targets only: {args.vocab}")
    model = train_model(
        pairs,
        max_sub=args.max_sub,
        vocabulary_path=vocabulary_path,
        source_path=args.data,
    )
    output = save_model(model, args.output)
    print(f"Trained on {model['training_rows']} pairs")
    print(f"Generated {len(model['rules'])} N-gram source rules")
    print(f"Saved model: {output.resolve()}")


if __name__ == "__main__":
    main()
