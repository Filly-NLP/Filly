from pathlib import Path

from scripts.evaluate_normalizer import (
    GoldPair,
    compute_pair_metrics,
    deterministic_validation_test_split,
    evaluate_valid_words,
    evaluate,
    filter_training_overlaps,
    load_archive_gold,
    load_valid_words,
    parse_gold_pairs,
    runtime_workload_metadata,
)


def test_csv_parser_supports_headered_and_headerless_gold_pairs():
    assert parse_gold_pairs("input,expected\nkc,kasi\n")[0] == GoldPair(2, "kc", "kasi")
    assert parse_gold_pairs("kc,kasi,ignored\n")[0] == GoldPair(1, "kc", "kasi")


def test_overlap_filter_excludes_exact_and_source_overlap_but_reports_target_overlap():
    training = [GoldPair(1, "kc", "kasi"), GoldPair(2, "foo", "bar")]
    held_out = [
        GoldPair(1, "kc", "kasi"),
        GoldPair(2, "KC", "kasi"),
        GoldPair(3, "new", "bar"),
    ]

    kept, excluded, counts = filter_training_overlaps(held_out, training)

    assert [(pair.source, pair.expected) for pair in kept] == [("new", "bar")]
    assert [row["row_number"] for row in excluded] == [1, 2]
    assert counts == {"exact_pair": 2, "source": 2, "target": 3, "excluded": 2}


def test_pair_metrics_use_explicit_changed_and_gold_need_denominators():
    pairs = [
        GoldPair(1, "a", "b"),
        GoldPair(2, "c", "d"),
        GoldPair(3, "e", "e"),
        GoldPair(4, "g", "h"),
    ]
    metrics = compute_pair_metrics(pairs, ["b", "c", "f", "i"])

    assert metrics["exact_match"] == {"correct": 1, "denominator": 4, "rate": 0.25}
    assert metrics["changed_word_precision"] == {"numerator": 1, "denominator": 3, "rate": 1 / 3}
    assert metrics["changed_word_recall"] == {"numerator": 1, "denominator": 3, "rate": 1 / 3}
    assert metrics["changed_decision_f1"] == 1 / 3
    assert metrics["false_positive_rate_on_identity_pairs"]["rate"] == 1.0
    assert metrics["unchanged_failure_rate"]["rate"] == 1 / 3
    assert metrics["wrong_replacement_rate"]["rate"] == 1 / 3


def test_false_positive_rate_uses_separate_valid_word_denominator():
    metrics = evaluate_valid_words(
        ["tama", "train-form", "valid"],
        ["tama", "changed", "bad"],
        training_sources={"train-form"},
    )

    assert metrics["train_source_overlaps_excluded"] == 1
    assert metrics["evaluated_word_count"] == 2
    assert metrics["false_positive_rate"] == {"numerator": 1, "denominator": 2, "rate": 0.5}


def test_archive_test_and_valid_words_are_independent_and_training_overlap_is_visible():
    project_root = Path(__file__).resolve().parents[2]
    archive = project_root / "efficient-spelling-normalization-filipino-main (1).zip"
    training_path = project_root / "backend" / "normalization" / "data" / "train_pairs.csv"

    gold = load_archive_gold(archive)
    from scripts.evaluate_normalizer import load_training_pairs

    kept, excluded, counts = filter_training_overlaps(gold, load_training_pairs(training_path))

    assert len(gold) == 100
    assert len(kept) == 98
    assert len(excluded) == 2
    assert counts["exact_pair"] == 2
    assert counts["source"] == 2
    assert counts["target"] == 39
    assert len(load_valid_words(archive)) == 6082


def test_partition_is_stable_78_20_and_runtime_workload_stays_full_ordered_set():
    pairs = [GoldPair(index, f"source-{index}", f"target-{index}") for index in range(1, 99)]

    validation, test, metadata = deterministic_validation_test_split(pairs, seed=42)
    validation_again, test_again, metadata_again = deterministic_validation_test_split(pairs, seed=42)

    assert len(validation) == 78
    assert len(test) == 20
    assert validation == validation_again
    assert test == test_again
    assert metadata == metadata_again
    assert metadata["seed"] == 42
    assert len(set(pair.row_number for pair in validation) & set(pair.row_number for pair in test)) == 0
    assert len(validation) + len(test) == len(pairs)
    full_workload = runtime_workload_metadata([pair.source for pair in pairs])
    assert full_workload["word_count"] == 98
    assert full_workload["gold_labels_passed_to_benchmark"] is False


def test_validation_report_records_dld_limit_and_leaves_reserved_test_out():
    report = evaluate(split="validation", max_edit_distance=2, include_runtime=False)
    evaluation = report["evaluation"]

    assert evaluation["evaluated_pair_count"] == 78
    assert evaluation["final_test_reserved_for_final_evaluation"] is True
    assert evaluation["is_final_test_evaluation"] is False
    assert evaluation["partition"]["test_rows_sha256"] == (
        "9df2b473c98f13ef8756466f304a62e6bfbc29d140eebbc0bc58e623aa07c05f"
    )
    assert evaluation["normalizer_configuration"]["maximum_applied_correction_distance"] == 2
    assert evaluation["metrics"]["exact_match"] == {
        "correct": 36,
        "denominator": 78,
        "rate": 36 / 78,
    }
