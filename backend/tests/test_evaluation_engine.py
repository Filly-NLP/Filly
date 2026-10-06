import pytest

from app.evaluation.engine import evaluate, scorer_provenance


def make_row(row_id, source, target, kind="slang", **metadata):
    return {
        "row_id": row_id,
        "input": source,
        "target": target,
        "gec_correction": metadata.get("gec_correction"),
        "unnormalized_type": kind,
        "types": () if kind is None else (kind,),
        "metadata": {key: value for key, value in metadata.items() if key != "gec_correction"},
    }


def metric_for(result, mode, group="overall"):
    return next(
        row for row in result["metrics"]
        if row["mode"] == mode and row["group"] == group
    )


def test_authoritative_edit_counts_and_aggregated_prf_match_hand_calculation():
    rows = [
        make_row(1, "I go", "I went", "slang"),
        make_row(2, "We see", "We saw", "abbreviation"),
        make_row(3, "They stay", "They stay", "spelling_variation"),
    ]
    predictions = {"I go": "I went", "We see": "We sees", "They stay": "They stay here"}

    result = evaluate(rows, "off", predictions.__getitem__)
    overall = metric_for(result, "without_normalization")

    assert (overall["n"], overall["tp"], overall["fp"], overall["fn"]) == (3, 1, 2, 1)
    assert overall["precision"] == pytest.approx(1 / 3)
    assert overall["recall"] == pytest.approx(1 / 2)
    assert overall["f05"] == pytest.approx(5 / 14)
    slang = metric_for(result, "without_normalization", "slang")
    abbreviation = metric_for(result, "without_normalization", "abbreviation")
    spelling = metric_for(result, "without_normalization", "spelling variation")
    assert (slang["tp"], slang["fp"], slang["fn"]) == (1, 0, 0)
    assert (abbreviation["tp"], abbreviation["fp"], abbreviation["fn"]) == (0, 1, 1)
    assert (spelling["tp"], spelling["fp"], spelling["fn"]) == (0, 1, 0)
    assert scorer_provenance()["path"].endswith("edit_metrics.py")
    assert len(scorer_provenance()["sha256"]) == 64


def test_group_metrics_include_expected_empty_categories_and_canonicalize_types():
    rows = [
        make_row(1, "raw one", "gold one", "slang"),
        make_row(2, "raw two", "gold two", "abbreviations"),
        make_row(3, "raw three", "gold three", "spelling-variations"),
        make_row(4, "raw four", "gold four", "Other Feature"),
        make_row(5, "raw five", "gold five", None),
    ]

    result = evaluate(rows, "off", lambda text: text)
    groups = {row["group"]: row for row in result["metrics"]}

    assert groups["overall"]["n"] == 5
    assert groups["slang"]["n"] == 1
    assert groups["abbreviation"]["n"] == 1
    assert groups["spelling variation"]["n"] == 1
    assert groups["other feature"]["n"] == 1
    assert "N0" not in groups
    assert groups["overall"]["errors_before"] == 5


def test_multi_label_and_reserved_overall_type_are_rejected():
    multi = make_row(1, "x", "y")
    multi["types"] = ("slang", "abbreviation")
    reserved = make_row(2, "x", "y", "overall")

    with pytest.raises(ValueError, match="one dominant"):
        evaluate([multi], "off", lambda text: text)
    with pytest.raises(ValueError, match="reserved"):
        evaluate([reserved], "off", lambda text: text)


def test_without_normalization_bypasses_normalizer_and_reports_zero_err():
    calls = []
    rows = [make_row(1, "I go", "I went")]
    result = evaluate(
        rows,
        "off",
        lambda text: calls.append(text) or "I went",
        normalizer=lambda text: pytest.fail("normalizer must not run in off mode"),
    )

    metric = metric_for(result, "without_normalization")
    prediction = result["predictions"][0]
    assert calls == ["I go"]
    assert metric["errors_before"] == metric["errors_after"] == 1
    assert metric["error_reduction_rate"] == 0
    assert metric["final_errors_after"] == 0
    assert metric["final_system_error_reduction_rate"] == 1
    assert "normalized_input" not in prediction
    assert prediction["output_without_normalization"] == "I went"
    assert prediction["exact_without_normalization"] is True


@pytest.mark.parametrize(
    ("normalized", "expected_after", "expected_err"),
    [("I went", 0, 100.0), ("I go", 1, 0.0), ("we leave", 2, -100.0)],
)
def test_normalization_err_uses_stage_to_target_edit_counts(normalized, expected_after, expected_err):
    rows = [make_row(1, "I go", "I went")]
    result = evaluate(rows, "on", lambda text: "I went", lambda text: normalized)
    metric = metric_for(result, "with_normalization")

    assert metric["errors_before"] == 1
    assert metric["errors_after"] == expected_after
    assert metric["error_reduction_rate"] == expected_err
    assert metric["final_errors_after"] == 0
    assert metric["final_system_error_reduction_rate"] == 1
    assert result["predictions"][0]["normalized_input"] == normalized


def test_zero_error_baseline_has_undefined_err_and_final_system_rate():
    rows = [make_row(1, "already right", "already right")]
    result = evaluate(rows, "both", lambda text: text, lambda text: "changed")
    off = metric_for(result, "without_normalization")
    on = metric_for(result, "with_normalization")

    assert off["errors_before"] == off["errors_after"] == 0
    assert off["error_reduction_rate"] is None
    assert off["final_system_error_reduction_rate"] is None
    assert on["errors_after"] == 2
    assert on["error_reduction_rate"] is None
    assert on["final_system_error_reduction_rate"] is None


def test_error_reduction_uses_ratio_of_summed_counts_not_mean_of_row_rates():
    rows = [
        make_row(1, "a", "b"),
        make_row(2, "one two three", "four five six"),
    ]
    normalized = {"a": "b", "one two three": "one two three"}
    result = evaluate(rows, "on", lambda text: text, normalized.__getitem__)
    overall = metric_for(result, "with_normalization")

    # Row ERRs are 100% and 0%; the pooled count formula is (4 - 3) / 4 = 25%.
    assert overall["errors_before"] == 4
    assert overall["errors_after"] == 3
    assert overall["error_reduction_rate"] == 25
    assert overall["final_errors_after"] == 3
    assert overall["final_system_error_reduction_rate"] == pytest.approx(0.25)


def test_wrong_single_substitution_counts_as_one_final_error_not_two():
    rows = [make_row(1, "I go", "I went")]
    result = evaluate(rows, "off", lambda text: "I goes")
    overall = metric_for(result, "without_normalization")

    assert (overall["tp"], overall["fp"], overall["fn"]) == (0, 1, 1)
    assert overall["errors_before"] == 1
    assert overall["final_errors_after"] == 1
    assert overall["final_system_error_reduction_rate"] == 0


def test_both_mode_uses_same_rows_preserves_original_metadata_and_returns_deltas():
    rows = [make_row(17, "raw", "gold", "spelling variation", custom_header="kept")]
    progress = []
    gec_inputs = []
    result = evaluate(
        rows,
        "both",
        lambda text: gec_inputs.append(text) or ("gold" if text == "gold" else "raw"),
        lambda text: "gold",
        progress=lambda completed, total: progress.append((completed, total)),
    )
    prediction = result["predictions"][0]
    comparison = {row["group"]: row for row in result["comparison"]}

    assert gec_inputs == ["raw", "gold"]
    assert progress == [(1, 2), (2, 2)]
    assert prediction["metadata"] == {"custom_header": "kept"}
    assert prediction["row_id"] == 17
    assert prediction["input"] == "raw"
    assert prediction["target"] == "gold"
    assert prediction["normalized_input"] == "gold"
    assert prediction["output_without_normalization"] == "raw"
    assert prediction["output_with_normalization"] == "gold"
    assert prediction["exact_without_normalization"] is False
    assert prediction["exact_with_normalization"] is True
    assert comparison["overall"]["delta_f05"] > 0
    assert comparison["overall"]["delta_error_reduction_rate"] == 100
    assert comparison["overall"]["delta_final_system_error_reduction_rate"] == 1


def test_invalid_normalizer_result_and_type_semicolon_fail_fast():
    rows = [make_row(1, "x", "y")]
    with pytest.raises(TypeError, match="expected str"):
        evaluate(rows, "on", lambda text: text, lambda text: None)

    semicolon_type = make_row(2, "x", "y", "slang;abbreviation")
    with pytest.raises(ValueError, match="one dominant"):
        evaluate([semicolon_type], "off", lambda text: text)
