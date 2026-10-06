from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.evaluation.dataset import DatasetError, load_dataset


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def test_canonical_columns_and_normalized_group_labels(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "canonical.csv",
        ["input", "target", "unnormalized_type"],
        [["raw one", "target one", "Spelling-Variation"], ["raw two", "target two", "ABBREVIATIONS"]],
    )

    dataset = load_dataset(path)

    assert [row["row_id"] for row in dataset.rows] == [1, 2]
    assert dataset.rows[0]["types"] == ("spelling variation",)
    assert dataset.rows[0]["unnormalized_type"] == "Spelling-Variation"
    assert dataset.rows[1]["types"] == ("abbreviation",)


def test_aliases_are_case_insensitive_and_correction_tags_are_supported(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "aliases.csv",
        ["SOURCE_TEXT", "Target_Text", "Correction_Tags", "Noise_Type"],
        [["raw", "target", "[$REPLACE_x]", "slang"]],
    )

    dataset = load_dataset(path)

    assert dataset.columns == {
        "input": "SOURCE_TEXT",
        "target": "Target_Text",
        "correction": "Correction_Tags",
        "type": "Noise_Type",
    }
    assert dataset.rows[0]["gec_correction"] == "[$REPLACE_x]"


def test_simultaneous_correction_aliases_require_explicit_override(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "ambiguous.csv",
        ["source_text", "target_text", "generation_tag", "expected_gec_tag"],
        [["raw", "target", "operation-x", "$REPLACE_y"]],
    )

    with pytest.raises(DatasetError, match="ambiguous correction columns"):
        load_dataset(path)

    dataset = load_dataset(path, correction_column="expected_gec_tag")
    assert dataset.rows[0]["gec_correction"] == "$REPLACE_y"
    assert dataset.rows[0]["metadata"] == {
        "source_text": "raw",
        "target_text": "target",
        "generation_tag": "operation-x",
        "expected_gec_tag": "$REPLACE_y",
    }


@pytest.mark.parametrize(
    ("headers", "values", "missing"),
    [
        (["target"], ["gold"], "input"),
        (["input"], ["raw"], "target"),
        (["input", "target"], ["", "gold"], "input"),
        (["input", "target"], ["raw", "  "], "target"),
    ],
)
def test_required_columns_and_nonblank_values_fail(tmp_path: Path, headers, values, missing) -> None:
    path = _write_csv(tmp_path / "invalid.csv", headers, [values])
    with pytest.raises(DatasetError, match=missing):
        load_dataset(path)


def test_explicit_input_and_target_overrides_work(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "override.csv",
        ["raw_sentence", "expected_sentence", "category"],
        [["raw", "gold", "slang"]],
    )

    dataset = load_dataset(
        path,
        input_column="raw_sentence",
        target_column="expected_sentence",
    )

    assert dataset.rows[0]["input"] == "raw"
    assert dataset.rows[0]["target"] == "gold"


def test_all_explicit_column_overrides_work(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "all_overrides.csv",
        ["raw_sentence", "final_sentence", "chosen_tag", "dominant_feature"],
        [["raw", "gold", "correction-x", "Slang"]],
    )

    dataset = load_dataset(
        path,
        input_column="raw_sentence",
        target_column="final_sentence",
        correction_column="chosen_tag",
        type_column="dominant_feature",
    )

    assert dataset.rows[0]["gec_correction"] == "correction-x"
    assert dataset.rows[0]["types"] == ("slang",)


@pytest.mark.parametrize("type_value", ["slang;abbreviation", "overall"])
def test_multiple_or_reserved_category_fails(tmp_path: Path, type_value: str) -> None:
    path = _write_csv(
        tmp_path / "types.csv",
        ["input", "target", "unnormalized_type"],
        [["raw", "gold", type_value]],
    )
    with pytest.raises(DatasetError):
        load_dataset(path)


def test_blank_and_malformed_records_are_not_silently_skipped(tmp_path: Path) -> None:
    path = tmp_path / "blank.csv"
    path.write_text("input,target\nraw,gold\n\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="blank data row"):
        load_dataset(path)

    path.write_text('input,target\n"raw,gold\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="could not read dataset CSV"):
        load_dataset(path)


def test_optional_correction_metadata_and_type_are_allowed(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "optional.csv", ["input", "target"], [["raw", "gold"]])
    dataset = load_dataset(path)
    assert dataset.rows[0]["gec_correction"] is None
    assert dataset.rows[0]["types"] == ()


def test_actual_cli_both_mode_smoke_writes_paired_outputs_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from app.evaluation import cli
    from app.services import gec as gec_module
    from app.services import normalizer as normalizer_module

    checkpoint = tmp_path / "small-checkpoint-fixture.pt"
    checkpoint.write_bytes(b"test fixture only")
    monkeypatch.setenv("GECTOR_MODEL_PATH", str(checkpoint))

    class FakeGEC:
        def __init__(self, checkpoint_path, *, device):
            self.checkpoint_path = Path(checkpoint_path)
            self.device = "cpu" if device is None else device
            self.metadata = {"readiness_status": "TEST_FIXTURE"}
            self.tokenizer = SimpleNamespace(init_kwargs={})

        def correct_iteratively(self, text: str, *, iterations: int):
            assert iterations == 1
            return SimpleNamespace(corrected_text=text)

    class FakeNormalizer:
        _resource_signature = tuple(
            tmp_path / name for name in ("rules.json", "vocabulary.txt", "metadata.json", "curated_mappings.json")
        )
        max_edit_distance = 2
        use_curated_mappings = True

        def normalize_text(self, text: str):
            return {"raw": "gold"}.get(text, text), []

    for resource in FakeNormalizer._resource_signature:
        resource.write_text("fixture", encoding="utf-8")
    fake_normalizer = FakeNormalizer()
    monkeypatch.setattr(gec_module, "GECService", FakeGEC)
    monkeypatch.setattr(normalizer_module, "get_normalizer", lambda **_kwargs: fake_normalizer)

    source = _write_csv(
        tmp_path / "paired.csv",
        ["input", "target", "expected_gec_tag", "generation_tag", "unnormalized_type"],
        [
            ["raw", "gold", "$REPLACE_example", "generated_example", "slang"],
            ["identity", "identity", "", "identity", "spelling_variation"],
        ],
    )
    out = tmp_path / "reports"

    result = cli.main(
        [
            "evaluate",
            str(source),
            "--normalization",
            "both",
            "--output-dir",
            str(out),
            "--correction-column",
            "expected_gec_tag",
            "--limit",
            "1",
        ]
    )

    assert result == 0
    stdout = capsys.readouterr().out
    assert "Rows evaluated: 1 / 2" in stdout
    with (out / "evaluation_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        prediction = next(csv.DictReader(stream))
    assert prediction["output_without_normalization"] == "raw"
    assert prediction["normalized_input"] == "gold"
    assert prediction["output_with_normalization"] == "gold"
    original = json.loads(prediction["original_columns_json"])
    assert original["generation_tag"] == "generated_example"
    assert original["expected_gec_tag"] == "$REPLACE_example"

    summary = json.loads((out / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary["metadata"]["dataset"]["rows_in_csv"] == 2
    assert summary["metadata"]["dataset"]["rows_evaluated"] == 1
    assert summary["metadata"]["checkpoint"]["sha256"]
    assert summary["metadata"]["scorer"]["sha256"]
    assert summary["metadata"]["scorer_tokenizer"]["sha256"]
    assert (out / "evaluation_comparison.csv").is_file()


def test_cli_limit_still_validates_later_rows_before_scorer_or_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.evaluation import cli

    source = _write_csv(
        tmp_path / "invalid_after_limit.csv",
        ["input", "target"],
        [["valid", "gold"], ["", "invalid"]],
    )
    monkeypatch.setattr(cli, "scorer_provenance", lambda: pytest.fail("scorer loaded before CSV validation"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["evaluate", str(source), "--normalization", "off", "--limit", "1"])
    assert exc.value.code == 2


@pytest.mark.parametrize("mode", ["off", "on"])
def test_single_mode_loads_only_requested_path_and_exports_relevant_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    from app.evaluation import cli
    from app.services import gec as gec_module
    from app.services import normalizer as normalizer_module

    checkpoint = tmp_path / "single-mode-fixture.pt"
    checkpoint.write_bytes(b"test fixture only")
    monkeypatch.setenv("GECTOR_MODEL_PATH", str(checkpoint))
    service_instances = []
    pass_counts = []

    class FakeGEC:
        def __init__(self, checkpoint_path, *, device):
            self.checkpoint_path = Path(checkpoint_path)
            self.device = "cpu" if device is None else device
            self.metadata = {"readiness_status": "TEST_FIXTURE"}
            self.tokenizer = SimpleNamespace(init_kwargs={})
            service_instances.append(self)

        def correct_iteratively(self, text: str, *, iterations: int):
            pass_counts.append(iterations)
            return SimpleNamespace(corrected_text=text)

    normalizer_calls = []

    class FakeNormalizer:
        _resource_signature = ()
        max_edit_distance = 2
        use_curated_mappings = True

        def normalize_text(self, text: str):
            normalizer_calls.append(text)
            return text, []

    monkeypatch.setattr(gec_module, "GECService", FakeGEC)
    if mode == "on":
        monkeypatch.setattr(normalizer_module, "get_normalizer", lambda **_kwargs: FakeNormalizer())
    else:
        monkeypatch.setattr(
            normalizer_module,
            "get_normalizer",
            lambda **_kwargs: pytest.fail("off mode initialized the normalizer"),
        )

    source = _write_csv(
        tmp_path / f"single-{mode}.csv",
        ["input", "target"],
        [["raw one", "target one"], ["raw two", "target two"]],
    )
    out = tmp_path / f"single-{mode}-reports"
    args = [
        "evaluate", str(source), "--normalization", mode, "--output-dir", str(out),
    ]
    if mode == "off":
        args.extend(("--gec-passes", "3"))

    assert cli.main(args) == 0
    assert len(service_instances) == 1
    assert pass_counts == ([3, 3] if mode == "off" else [1, 1])
    assert len(normalizer_calls) == (2 if mode == "on" else 0)

    with (out / "evaluation_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        first = next(reader)
        fields = reader.fieldnames or []
    if mode == "off":
        assert "normalized_input" not in fields
        assert "output_with_normalization" not in fields
        assert first["output_without_normalization"] == "raw one"
        summary = json.loads((out / "evaluation_summary.json").read_text(encoding="utf-8"))
        assert summary["metadata"]["effective_configuration"]["gec_passes"] == 3
    else:
        assert "normalized_input" in fields
        assert "output_with_normalization" in fields
        assert "output_without_normalization" not in fields
        assert first["output_with_normalization"] == "raw one"


def test_output_paths_cannot_overwrite_dataset_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.evaluation import cli

    out = tmp_path / "reports"
    out.mkdir()
    source = _write_csv(out / "evaluation_predictions.csv", ["input", "target"], [["raw", "gold"]])
    monkeypatch.setattr(cli, "scorer_provenance", lambda: pytest.fail("model path check came too late"))
    with pytest.raises(SystemExit) as exc:
        cli.main(
            ["evaluate", str(source), "--normalization", "off", "--output-dir", str(out)]
        )
    assert exc.value.code == 2
