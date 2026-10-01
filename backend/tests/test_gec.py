from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

import app.services.gec as gec_module
from app.schemas.schemas import NormalizationItem
from app.services.gec import (
    GECCheckpointError,
    GECInputTooLong,
    GECService,
    MODEL_ID,
    MODEL_REVISION,
    _select_device,
    _read_checkpoint,
    _vocabulary_sha256,
    _validate_checkpoint,
)
from app.services.gec_checkpoint import balarila_registry, transforms
from app.services.gec_checkpoint.tokenizer import tokenize_with_offsets
from app.services.gec_checkpoint.predictor import TokenLabelPrediction


class FakeBatch(dict):
    def __init__(self, input_ids, attention_mask, word_ids):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self._word_ids = word_ids

    def word_ids(self, batch_index=0):
        return self._word_ids[batch_index]


class FakeTokenizer:
    is_fast = True
    add_prefix_space = True

    def __init__(self):
        self.observed: list[tuple[str, ...]] = []
        self._ids = {"x": 10, "y": 11, "z": 12}

    def __call__(self, sentences, **_kwargs):
        rows = []
        word_id_rows = []
        self.observed.extend(tuple(sentence) for sentence in sentences)
        for sentence in sentences:
            rows.append([1, *(self._ids.setdefault(token, 20 + len(self._ids)) for token in sentence), 2])
            word_id_rows.append([None, *range(len(sentence)), None])
        width = max(map(len, rows))
        padded = [row + [0] * (width - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        for word_ids, row in zip(word_id_rows, rows):
            word_ids.extend([None] * (width - len(row)))
        return FakeBatch(
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(masks, dtype=torch.long),
            word_id_rows,
        )


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))
        self.inference_modes: list[bool] = []
        self.seen_ids: list[list[int]] = []

    def forward(self, input_ids, attention_mask=None):
        self.inference_modes.append(torch.is_inference_mode_enabled())
        self.seen_ids.append(input_ids[0].tolist())
        batch, sequence = input_ids.shape
        correction = torch.full((batch, sequence, 3), -8.0, device=input_ids.device)
        detection = torch.full((batch, sequence, 2), -8.0, device=input_ids.device)
        correction[..., 0] = 0.0
        detection[..., 0] = 8.0
        for row in range(batch):
            for position, token_id in enumerate(input_ids[row].tolist()):
                if token_id == 10:
                    correction[row, position, 1] = 8.0
                    detection[row, position, 1] = 8.0
                elif token_id == 11:
                    correction[row, position, 2] = 8.0
                    detection[row, position, 1] = 8.0
        return type("Output", (), {"correction_logits": correction, "detection_logits": detection})()


def make_service() -> tuple[GECService, FakeModel, FakeTokenizer]:
    model = FakeModel()
    tokenizer = FakeTokenizer()
    service = GECService(
        model=model,
        tokenizer=tokenizer,
        label_vocab={"$KEEP": 0, "$REPLACE_y": 1, "$REPLACE_z": 2},
        device="cpu",
    )
    return service, model, tokenizer


def make_checkpoint_payload(
    *,
    stage="stage3",
    scope=None,
    metadata_scope="auto",
    vocab=None,
):
    vocab = vocab or {"$KEEP": 0, "$REPLACE_y": 1, "$REPLACE_z": 2}
    vocab_hash = _vocabulary_sha256(vocab)
    source_hash = "1" * 64
    parent_hash = "2" * 64
    scope = scope or (
        "train_stage2_and_stage3_union" if stage == "stage3" else "train_only"
    )
    config = {
        "stage": stage,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "keep_label_id": 0,
        "vocabulary_support_scope": scope,
        "expected_hashes": {
            "label_vocab": vocab_hash,
            "phase10_manifest": source_hash,
        },
    }
    metadata = {
        "stage": stage,
        "selected_best": True,
        "run_id": "fixture-run",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "label_vocab_hash": vocab_hash,
        "source_manifest_hash": source_hash,
    }
    if metadata_scope == "auto":
        if stage == "stage3" or scope != "train_only":
            metadata["vocabulary_support_scope"] = scope
    elif metadata_scope is not None:
        metadata["vocabulary_support_scope"] = metadata_scope
    hashes = {"label_vocab": vocab_hash, "phase10_manifest": source_hash}
    if stage == "stage3":
        config["expected_hashes"]["stage2_parent_checkpoint"] = parent_hash
        metadata["stage3_parent"] = {
            "run_id": "stage2-fixture-run",
            "checkpoint_sha256": parent_hash,
            "label_vocab_sha256": vocab_hash,
            "source_manifest_sha256": source_hash,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
        hashes["stage2_parent_checkpoint"] = parent_hash
    return {
        "format_version": 1,
        "config": config,
        "metadata": metadata,
        "hashes": hashes,
        "label_vocab": dict(vocab),
        "model_state": {
            "correction_head.weight": torch.zeros((len(vocab), 1024)),
            "correction_head.bias": torch.zeros((len(vocab),)),
            "detection_head.weight": torch.zeros((2, 1024)),
            "detection_head.bias": torch.zeros((2,)),
        },
    }


class SentenceBatchTokenizer(FakeTokenizer):
    def __init__(self):
        super().__init__()
        self._ids = {
            "ako": 100,
            "ay": 101,
            "umuwi": 102,
            ".": 103,
            "kumain": 104,
            "bukas": 105,
            "kalat": 106,
            "sana": 107,
        }


class SentenceBatchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))
        self.seen_batches: list[list[list[int]]] = []
        self.calls = 0

    def forward(self, input_ids, attention_mask=None):
        self.calls += 1
        self.seen_batches.append(input_ids.tolist())
        batch, sequence = input_ids.shape
        correction = torch.full((batch, sequence, 6), -8.0, device=input_ids.device)
        detection = torch.full((batch, sequence, 2), -8.0, device=input_ids.device)
        correction[..., 0] = 8.0
        detection[..., 0] = 8.0
        for row in range(batch):
            first_word_id = int(input_ids[row, 1].item())
            if first_word_id in {100, 104}:
                correction[row, 1, 1] = 12.0
                detection[row, 1, 1] = 12.0
            for position, token_id in enumerate(input_ids[row].tolist()):
                if token_id == 102:
                    correction[row, position, 2] = 12.0
                    detection[row, position, 1] = 12.0
                elif token_id == 100 and position > 1:
                    following_id = int(input_ids[row, position + 1].item())
                    if following_id != 107:
                        correction[row, position, 3] = 12.0
                        detection[row, position, 1] = 12.0
                elif token_id == 106:
                    correction[row, position, 4] = 12.0
                    detection[row, position, 1] = 12.0
                elif self.calls > 1 and token_id == 101 and position >= 4:
                    correction[row, position, 5] = 12.0
                    detection[row, position, 1] = 12.0
        return type("Output", (), {"correction_logits": correction, "detection_logits": detection})()


def test_sentence_batch_inference_preserves_global_offsets_and_original_spacing():
    model = SentenceBatchModel()
    tokenizer = SentenceBatchTokenizer()
    service = GECService(
        model=model,
        tokenizer=tokenizer,
        label_vocab={
            "$KEEP": 0,
            "$TRANSFORM_CASE_CAPITAL": 1,
            "$REPLACE_bumalik": 2,
            "$APPEND_sana": 3,
            "$DELETE": 4,
            "$REPLACE_ng": 5,
        },
        device="cpu",
    )
    text = "  ako ay umuwi. \nkumain ako bukas kalat ay.\t"

    result = service.correct_iteratively(text, iterations=2)
    first_pass, second_pass = result.passes

    assert tokenizer.observed == [
        ("ako", "ay", "umuwi", "."),
        ("kumain", "ako", "bukas", "kalat", "ay", "."),
        ("Ako", "ay", "bumalik", "."),
        ("Kumain", "ako", "sana", "bukas", "ay", "."),
    ]
    assert len(model.seen_batches) == 2
    assert len(model.seen_batches[0]) == 2
    assert [len(batch) for batch in model.seen_batches] == [2, 2]
    assert first_pass.output_text == "  Ako ay bumalik. \nKumain ako sana bukas ay.\t"
    assert second_pass.output_text == "  Ako ay bumalik. \nKumain ako sana bukas ng.\t"
    assert first_pass.input_tokens == (
        "ako", "ay", "umuwi", ".", "kumain", "ako", "bukas", "kalat", "ay", ".",
    )
    assert first_pass.labels == (
        "$TRANSFORM_CASE_CAPITAL",
        "$KEEP",
        "$REPLACE_bumalik",
        "$KEEP",
        "$TRANSFORM_CASE_CAPITAL",
        "$APPEND_sana",
        "$KEEP",
        "$DELETE",
        "$KEEP",
        "$KEEP",
    )
    assert [
        (change.start, change.end, change.original, change.correction, change.label)
        for change in first_pass.changes
    ] == [
        (2, 5, "ako", "Ako", "$TRANSFORM_CASE_CAPITAL"),
        (9, 14, "umuwi", "bumalik", "$REPLACE_bumalik"),
        (17, 23, "kumain", "Kumain", "$TRANSFORM_CASE_CAPITAL"),
        (27, 27, "", " sana", "$APPEND_sana"),
        (33, 39, " kalat", "", "$DELETE"),
    ]
    changed_word_start = first_pass.output_text.index("ay.", first_pass.output_text.index("Kumain"))
    assert [
        (change.start, change.end, change.original, change.correction, change.label)
        for change in second_pass.changes
    ] == [(changed_word_start, changed_word_start + 2, "ay", "ng", "$REPLACE_ng")]


def test_five_passes_chain_each_output_into_the_next_and_use_inference_mode():
    service, model, tokenizer = make_service()

    result = service.correct_iteratively("x", iterations=5)

    assert result.corrected_text == "z"
    assert result.iterations == 5
    assert result.iteration_outputs == ("y", "z", "z", "z", "z")
    assert tokenizer.observed == [("x",), ("y",), ("z",), ("z",), ("z",)]
    assert model.seen_ids == [[1, 10, 2], [1, 11, 2], [1, 12, 2], [1, 12, 2], [1, 12, 2]]
    assert model.inference_modes == [True] * 5
    assert [change.iteration for change in result.changes] == [1, 2]
    assert all(change.start == 0 and change.end == 1 for change in result.changes)


def test_new_period_is_retokenized_but_does_not_create_a_new_sentence_slot():
    service = GECService(
        model=FakeModel(), tokenizer=FakeTokenizer(),
        label_vocab={"$KEEP": 0, "$ADD_PUNC_PERIOD": 1}, device="cpu",
    )
    seen_batches = []

    def predict(token_batches, *, batch_size):
        seen_batches.append(tuple(tuple(row) for row in token_batches))
        predictions = []
        for row in token_batches:
            labels = ["$KEEP"] * len(row)
            if len(seen_batches) == 1:
                labels[row.index("umuwi")] = "$ADD_PUNC_PERIOD"
            predictions.append(TokenLabelPrediction(
                tuple(labels), (0.9,) * len(row), (False,) * len(row),
                any(label != "$KEEP" for label in labels),
            ))
        return predictions

    service.predictor.predict = predict
    result = service.correct_iteratively("ako ay umuwi kumain ako", iterations=2)

    assert result.passes[0].output_text == "ako ay umuwi. kumain ako"
    assert seen_batches[0] == (("ako", "ay", "umuwi", "kumain", "ako"),)
    assert seen_batches[1] == (("ako", "ay", "umuwi", ".", "kumain", "ako"),)
    assert result.passes[1].labels[4] == "$KEEP"


def test_repeated_punctuation_label_is_a_new_prediction_on_each_pass():
    service = GECService(
        model=FakeModel(), tokenizer=FakeTokenizer(),
        label_vocab={"$KEEP": 0, "$ADD_PUNC_EMARK": 1}, device="cpu",
    )
    seen_batches = []

    def predict(token_batches, *, batch_size):
        seen_batches.append(tuple(tuple(row) for row in token_batches))
        predictions = []
        for row in token_batches:
            labels = ["$KEEP"] * len(row)
            labels[-1] = "$ADD_PUNC_EMARK"
            predictions.append(TokenLabelPrediction(
                tuple(labels), (0.9,) * len(row), (True,) * len(row), True,
            ))
        return predictions

    service.predictor.predict = predict
    result = service.correct_iteratively("Umalis siya!", iterations=3)

    assert result.iteration_outputs == (
        "Umalis siya!!", "Umalis siya!!!", "Umalis siya!!!!",
    )
    assert [row[-1] for batch in seen_batches for row in batch] == ["!", "!", "!"]
    assert [len(batch[0]) for batch in seen_batches] == [3, 4, 5]
    assert [change.label for change in result.changes] == ["$ADD_PUNC_EMARK"] * 3


class DeleteSoleTokenModel(FakeModel):
    def forward(self, input_ids, attention_mask=None):
        self.inference_modes.append(torch.is_inference_mode_enabled())
        self.seen_ids.append(input_ids[0].tolist())
        batch, sequence = input_ids.shape
        correction = torch.full((batch, sequence, 2), -8.0, device=input_ids.device)
        detection = torch.full((batch, sequence, 2), -8.0, device=input_ids.device)
        correction[..., 0] = 0.0
        detection[..., 0] = 8.0
        for row in range(batch):
            for position, token_id in enumerate(input_ids[row].tolist()):
                if token_id == 10:
                    correction[row, position, 1] = 8.0
                    detection[row, position, 1] = 8.0
        return type("Output", (), {"correction_logits": correction, "detection_logits": detection})()


def test_deleting_the_only_token_still_records_five_dependent_stage_passes():
    model = DeleteSoleTokenModel()
    tokenizer = FakeTokenizer()
    service = GECService(
        model=model,
        tokenizer=tokenizer,
        label_vocab={"$KEEP": 0, "$DELETE": 1},
        device="cpu",
    )

    result = service.correct_iteratively("x", iterations=5)

    assert result.corrected_text == ""
    assert result.iterations == 5
    assert result.iteration_outputs == ("", "", "", "", "")
    assert [item.input_text for item in result.passes] == ["x", "", "", "", ""]
    assert [item.output_text for item in result.passes] == ["", "", "", "", ""]
    assert [item.model_invoked for item in result.passes] == [True, False, False, False, False]
    assert len(model.inference_modes) == 1
    assert tokenizer.observed == [("x",)]


def test_nonempty_whitespace_has_five_explicit_identity_passes_but_empty_has_none():
    service, model, tokenizer = make_service()

    whitespace = service.correct_iteratively("   \n", iterations=5)
    empty = service.correct_iteratively("", iterations=5)

    assert whitespace.iterations == 5
    assert [item.input_text for item in whitespace.passes] == ["   \n"] * 5
    assert [item.model_invoked for item in whitespace.passes] == [False] * 5
    assert empty.iterations == 0
    assert empty.passes == ()
    assert model.inference_modes == []
    assert tokenizer.observed == []


def test_model_token_limit_is_reported_as_a_clear_gec_input_error():
    service, _model, _tokenizer = make_service()
    service.predictor.max_subword_tokens = 3

    with pytest.raises(GECInputTooLong, match="Shorten or split"):
        service.correct_iteratively("x y", iterations=5)


def test_single_pass_preserves_exact_unicode_surface_and_reports_token_spans():
    service, _model, tokenizer = make_service()
    text = " Kamusta, mundo!  こんにちは。"

    result = service.correct_once(text)

    assert result.input_text == text
    assert result.output_text == text
    assert result.input_tokens == tokenize_with_offsets(text).tokens
    assert result.labels == ("$KEEP",) * len(result.input_tokens)
    assert result.changes == ()
    assert tokenizer.observed == [
        ("Kamusta", ",", "mundo", "!"),
        ("こんにちは", "。"),
    ]
    assert tuple(token for sentence in tokenizer.observed for token in sentence) == result.input_tokens


def test_case_capital_label_is_derived_and_replayed_by_the_active_edit_layer():
    label = transforms.derive_label_for_output("kumain", ("Kumain",))

    assert label == "$TRANSFORM_CASE_CAPITAL"
    assert transforms.apply_label("kumain", label) == ("Kumain",)
    assert transforms.apply_labels(
        ("kumain", "ako", "."),
        (label, "$KEEP", "$KEEP"),
    ) == ("Kumain", "ako", ".")
    assert transforms.apply_labels_to_text(
        "  kumain   ako.\n",
        (label, "$KEEP", "$KEEP"),
    ) == "  Kumain   ako.\n"


@pytest.mark.skipif(
    os.environ.get("FILLY_REAL_GEC_TESTS") != "1",
    reason="set FILLY_REAL_GEC_TESTS=1 to run the supplied checkpoint diagnostic",
)
def test_real_checkpoint_capitalizes_same_sentence_at_paragraph_position():
    from app.services.filly_pipeline import FillyPipeline
    from app.services.normalizer import get_normalizer

    service = GECService()
    pipeline = FillyPipeline(get_normalizer(), service, iterations=5)
    standalone = pipeline.analyze("kumain ako.")
    paragraph = pipeline.analyze("ako ay umuwi. kumain ako.")

    standalone_first_pass = standalone.gec.passes[0]
    paragraph_first_pass = paragraph.gec.passes[0]
    standalone_word = standalone_first_pass.input_tokens.index("kumain")
    paragraph_word = paragraph_first_pass.input_tokens.index("kumain")
    assert standalone_first_pass.labels[standalone_word] == "$TRANSFORM_CASE_CAPITAL"
    assert paragraph_first_pass.labels[paragraph_word] == "$TRANSFORM_CASE_CAPITAL"
    assert standalone_first_pass.output_text == "Kumain ako."
    assert paragraph_first_pass.output_text == "Ako ay umuwi. Kumain ako."
    assert standalone.corrected_text == "Kumain ako."
    assert paragraph.corrected_text == "Ako ay umuwi. Kumain ako."
    assert standalone.corrected_text == standalone.gec.passes[-1].output_text
    assert paragraph.corrected_text == paragraph.gec.passes[-1].output_text


def test_missing_checkpoint_fails_instead_of_using_rule_based_fallback():
    missing = Path(__file__).resolve().parent / "missing-gec-checkpoint.pt"
    with pytest.raises(FileNotFoundError, match="GEC checkpoint does not exist"):
        GECService(checkpoint_path=missing)


def test_checkpoint_fixtures_accept_stage2_scopes_and_require_stage3_union():
    stage2_legacy = make_checkpoint_payload(stage="stage2", scope="train_only")
    stage2_union = make_checkpoint_payload(
        stage="stage2", scope="train_stage2_and_stage3_union",
    )
    stage3_union = make_checkpoint_payload(stage="stage3")

    assert _validate_checkpoint(stage2_legacy)[1]["vocabulary_support_scope"] == "train_only"
    assert _validate_checkpoint(stage2_union)[1]["vocabulary_support_scope"] == "train_stage2_and_stage3_union"
    assert _validate_checkpoint(stage3_union)[1]["stage"] == "stage3"


@pytest.mark.parametrize("stage", ["stage4", "", [], None])
def test_checkpoint_rejects_unknown_or_malformed_stage(stage):
    payload = make_checkpoint_payload()
    payload["config"]["stage"] = stage
    with pytest.raises(GECCheckpointError, match="stage"):
        _validate_checkpoint(payload)


@pytest.mark.parametrize("scope", ["dev_only", "", [], None])
def test_checkpoint_rejects_unknown_or_malformed_vocabulary_scope(scope):
    payload = make_checkpoint_payload()
    payload["config"]["vocabulary_support_scope"] = scope
    with pytest.raises(GECCheckpointError, match="scope"):
        _validate_checkpoint(payload)


def test_stage3_requires_union_scope_and_consistent_config_metadata():
    payload = make_checkpoint_payload()
    payload["config"]["vocabulary_support_scope"] = "train_only"
    payload["metadata"]["vocabulary_support_scope"] = "train_only"
    with pytest.raises(GECCheckpointError, match="Stage 3 checkpoint must declare"):
        _validate_checkpoint(payload)

    payload = make_checkpoint_payload(stage="stage2", scope="train_only")
    payload["metadata"]["vocabulary_support_scope"] = "train_stage2_and_stage3_union"
    with pytest.raises(GECCheckpointError, match="scope differs"):
        _validate_checkpoint(payload)

    payload = make_checkpoint_payload()
    payload["metadata"]["stage"] = "stage2"
    with pytest.raises(GECCheckpointError, match="stage differs"):
        _validate_checkpoint(payload)


def test_checkpoint_rejects_malformed_config_provenance_and_run_identity():
    payload = make_checkpoint_payload()
    payload["config"]["reproducibility_metadata"] = []
    with pytest.raises(GECCheckpointError, match="reproducibility metadata"):
        _validate_checkpoint(payload)
    payload = make_checkpoint_payload()
    payload["config"]["expected_hashes"] = []
    with pytest.raises(GECCheckpointError, match="config hashes"):
        _validate_checkpoint(payload)
    payload = make_checkpoint_payload()
    payload["config"]["run_id"] = "different-run"
    with pytest.raises(GECCheckpointError, match="run identity"):
        _validate_checkpoint(payload)
    payload = make_checkpoint_payload()
    payload["config"]["keep_label_id"] = 1
    with pytest.raises(GECCheckpointError, match="KEEP label id"):
        _validate_checkpoint(payload)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("config", "model_id", "other/encoder"),
        ("metadata", "model_id", "other/encoder"),
        ("config", "model_revision", "0" * 40),
        ("metadata", "model_revision", "0" * 40),
    ],
)
def test_checkpoint_rejects_encoder_identity_mismatch(section, key, value):
    payload = make_checkpoint_payload()
    payload[section][key] = value
    with pytest.raises(GECCheckpointError, match="pinned encoder revision"):
        _validate_checkpoint(payload)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("hashes", "label_vocab", "0" * 64),
        ("metadata", "label_vocab_hash", "0" * 64),
        ("hashes", "phase10_manifest", "x" * 64),
        ("metadata", "source_manifest_hash", "0" * 64),
    ],
)
def test_checkpoint_rejects_inconsistent_vocabulary_or_source_hash(section, key, value):
    payload = make_checkpoint_payload()
    payload[section][key] = value
    with pytest.raises(GECCheckpointError, match="hash"):
        _validate_checkpoint(payload)


@pytest.mark.parametrize("mutation", ["replace", "swap"])
def test_checkpoint_rejects_same_count_vocabulary_identity_or_id_order_mutation(mutation):
    payload = make_checkpoint_payload()
    if mutation == "replace":
        payload["label_vocab"]["$REPLACE_other"] = payload["label_vocab"].pop("$REPLACE_y")
    else:
        payload["label_vocab"]["$REPLACE_y"], payload["label_vocab"]["$REPLACE_z"] = (
            payload["label_vocab"]["$REPLACE_z"], payload["label_vocab"]["$REPLACE_y"],
        )
    assert len(payload["label_vocab"]) == 3
    with pytest.raises(GECCheckpointError, match="vocabulary hash"):
        _validate_checkpoint(payload)


@pytest.mark.parametrize(
    "label",
    [
        "$UNSUPPORTED_CLASS",
        "$REPLACE_%ZZ",
        "$ADD_PUNC_EMARK@gap:%ZZ,",
        "$REPLACE_Ang%20%E2%80%9C@gaps:,%ZZ,",
        "$MERGE_SPACE@garbage@2",
        "$TRANSFORM_INSERT_HYPHEN@bad@2",
    ],
)
def test_checkpoint_rejects_malformed_executable_vocabulary_labels(label):
    vocab = {"$KEEP": 0, label: 1}
    payload = make_checkpoint_payload(stage="stage3", vocab=vocab)
    with pytest.raises(GECCheckpointError, match="invalid edit class"):
        _validate_checkpoint(payload)


def test_checkpoint_rejects_malformed_vocabulary_count_and_head_tensors():
    payload = make_checkpoint_payload()
    payload["label_vocab"]["$REPLACE_y"] = 3
    with pytest.raises(GECCheckpointError, match="contiguous"):
        _validate_checkpoint(payload)

    for name, shape in (
        ("correction_head.weight", (2, 1024)),
        ("correction_head.bias", (2,)),
        ("detection_head.weight", (2, 1023)),
        ("detection_head.bias", (1,)),
    ):
        payload = make_checkpoint_payload()
        payload["model_state"][name] = torch.zeros(shape)
        with pytest.raises(GECCheckpointError, match="model heads"):
            _validate_checkpoint(payload)
    payload = make_checkpoint_payload()
    payload["model_state"]["correction_head.weight"] = object()
    with pytest.raises(GECCheckpointError, match="model heads"):
        _validate_checkpoint(payload)


@pytest.mark.parametrize(
    ("parent_key", "value"),
    [
        ("run_id", ""),
        ("checkpoint_sha256", "x" * 64),
        ("label_vocab_sha256", "0" * 64),
        ("source_manifest_sha256", "0" * 64),
        ("model_id", "other/encoder"),
        ("model_revision", "0" * 40),
    ],
)
def test_stage3_parent_provenance_is_cross_bound(parent_key, value):
    payload = make_checkpoint_payload()
    payload["metadata"]["stage3_parent"][parent_key] = value
    with pytest.raises(GECCheckpointError, match="parent provenance"):
        _validate_checkpoint(payload)


def test_stage3_parent_hash_must_match_checkpoint_hashes_and_config():
    payload = make_checkpoint_payload()
    payload["hashes"]["stage2_parent_checkpoint"] = "3" * 64
    with pytest.raises(GECCheckpointError, match="parent provenance"):
        _validate_checkpoint(payload)
    payload = make_checkpoint_payload()
    payload["config"]["expected_hashes"]["stage2_parent_checkpoint"] = "3" * 64
    with pytest.raises(GECCheckpointError, match="config hashes"):
        _validate_checkpoint(payload)


@pytest.mark.parametrize(
    "hash_name",
    ["label_vocab", "phase10_manifest", "stage2_parent_checkpoint"],
)
def test_stage3_config_requires_each_parent_identity_hash(hash_name):
    payload = make_checkpoint_payload()
    payload["config"]["expected_hashes"][hash_name] = None
    with pytest.raises(GECCheckpointError, match="parent identity"):
        _validate_checkpoint(payload)


def test_checkpoint_rejects_bad_format_and_stage2_parent_declarations():
    payload = make_checkpoint_payload()
    payload["format_version"] = True
    with pytest.raises(GECCheckpointError, match="format"):
        _validate_checkpoint(payload)
    payload = make_checkpoint_payload(stage="stage2")
    payload["hashes"]["stage2_parent_checkpoint"] = "2" * 64
    with pytest.raises(GECCheckpointError, match="unexpectedly declares"):
        _validate_checkpoint(payload)


@pytest.mark.parametrize(
    ("mark", "expected"),
    [("EMARK", "A !C"), ("PERIOD", "A .C"), ("QMARK", "A ?C")],
)
def test_raw_gap_punctuation_replay_and_change_trace_include_suppressed_delete(mark, expected):
    label = f"$ADD_PUNC_{mark}@gap:%20,"
    labels = (label, "$DELETE", "$KEEP")
    text = "A    B C"
    service = GECService(
        model=FakeModel(), tokenizer=FakeTokenizer(),
        label_vocab={"$KEEP": 0, label: 1, "$DELETE": 2}, device="cpu",
    )
    service.predictor.predict = lambda batches, *, batch_size: [
        TokenLabelPrediction(labels, (0.7, 0.9, 0.1), (True, True, False), True)
        for _row in batches
    ]

    result = service.correct_once(text)

    assert transforms.apply_labels_to_text(text, labels) == expected
    assert result.output_text == expected
    assert len(result.changes) == 1
    change = result.changes[0]
    punctuation = {"EMARK": "!", "PERIOD": ".", "QMARK": "?"}[mark]
    assert (change.original, change.correction, change.label) == ("    B ", f" {punctuation}", label)
    assert (change.start, change.end, change.confidence, change.iteration) == (1, 7, 0.9, 1)


@pytest.mark.parametrize(
    ("replacement", "expected"),
    [("Ang", "Ang “CIA\treads"), ("Kung", "Kung “CIA\treads"), ("Sa", "Sa “CIA\treads")],
)
def test_raw_gap_replacement_labels_replay_in_the_integrated_gec_trace(replacement, expected):
    label = f"$REPLACE_{replacement}%20%E2%80%9C@gaps:,%20,"
    labels = (label, "$KEEP", "$KEEP")
    text = "“  CIA\treads"
    service = GECService(
        model=FakeModel(), tokenizer=FakeTokenizer(),
        label_vocab={"$KEEP": 0, label: 1}, device="cpu",
    )
    service.predictor.predict = lambda batches, *, batch_size: [
        TokenLabelPrediction(labels, (0.8, 0.2, 0.1), (True, False, False), True)
        for _row in batches
    ]

    result = service.correct_once(text)

    assert transforms.apply_labels_to_text(text, labels) == expected
    assert result.output_text == expected
    assert len(result.changes) == 1
    change = result.changes[0]
    assert (change.original, change.correction, change.label) == ("“  ", f"{replacement} “", label)
    assert (change.start, change.end, change.confidence, change.iteration) == (0, 3, 0.8, 1)


def test_raw_gap_change_maps_through_filly_pipeline_suggestions():
    from app.services.filly_pipeline import FillyPipeline

    label = "$ADD_PUNC_EMARK@gap:%20,"
    labels = ("$KEEP", label, "$DELETE", "$KEEP")
    normalizer = type("Normalizer", (), {
        "normalize_text": lambda _self, _text: (
            "xx A    B C",
            [NormalizationItem(word="x", suggestion="xx", start=0, end=1)],
        ),
    })()
    service = GECService(
        model=FakeModel(), tokenizer=FakeTokenizer(),
        label_vocab={"$KEEP": 0, label: 1, "$DELETE": 2}, device="cpu",
    )
    service.predictor.predict = lambda batches, *, batch_size: [
        TokenLabelPrediction(labels, (0.1, 0.7, 0.9, 0.1), (False, True, True, False), True)
        for _row in batches
    ]

    result = FillyPipeline(normalizer, service, iterations=1).analyze("x A    B C")

    assert result.normalized_text == "xx A    B C"
    assert result.corrected_text == "xx A !C"
    assert [
        (item.start, item.end, item.original, item.replacement, item.source, item.tag)
        for item in result.suggestions
    ] == [
        (0, 1, "x", "xx", "normalization", "NORMALIZATION"),
        (3, 9, "    B ", " !", "gec", label),
    ]


def test_validated_tiny_checkpoint_initializes_runtime_model_and_tokenizer(monkeypatch, tmp_path):
    payload = make_checkpoint_payload(stage="stage3")
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.touch()
    monkeypatch.setattr(gec_module, "_read_checkpoint", lambda _path: payload)

    class TinyEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def forward(self, input_ids, attention_mask=None):
            hidden = torch.zeros((*input_ids.shape, 1024), device=input_ids.device)
            return type("EncoderOutput", (), {"last_hidden_state": hidden})()

    class PinnedTokenizer(FakeTokenizer):
        def __init__(self):
            super().__init__()
            self.init_kwargs = {
                "vocab_file": (
                    "models--jcblaise--roberta-tagalog-large/snapshots/"
                    f"{MODEL_REVISION}/vocab.json"
                ),
            }

    config = types.SimpleNamespace(
        _commit_hash=MODEL_REVISION,
        hidden_size=1024,
        num_hidden_layers=24,
        vocab_size=30000,
        max_position_embeddings=514,
    )
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = gec_module.TRANSFORMERS_VERSION
    fake_transformers.AutoConfig = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: config,
    )
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: PinnedTokenizer(),
    )
    fake_transformers.AutoModel = types.SimpleNamespace(
        from_config=lambda selected: TinyEncoder(selected),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    service = GECService(checkpoint_path=checkpoint, device="cpu")

    assert service.metadata["stage"] == "stage3"
    assert service.label_vocab == payload["label_vocab"]
    assert service.model.training is False
    assert service.correct_once("x").output_text == "x"


@pytest.mark.skipif(
    os.environ.get("FILLY_REAL_GEC_TESTS") != "1",
    reason="set FILLY_REAL_GEC_TESTS=1 to read the supplied checkpoint diagnostic",
)
def test_optional_real_checkpoint_has_current_stage3_metadata():
    checkpoint = Path(__file__).resolve().parents[2] / "best.pt"
    payload = _read_checkpoint(checkpoint)
    vocab, metadata = _validate_checkpoint(payload)
    assert len(vocab) == 1913
    assert metadata["stage"] == "stage3"
    assert metadata["vocabulary_support_scope"] == "train_stage2_and_stage3_union"
    del payload


def test_vendored_decoder_imports_its_local_canonical_modules():
    assert transforms.__package__ == "app.services.gec_checkpoint"
    assert transforms.TABLE3_TARGET_TAGS is balarila_registry.TABLE3_TARGET_TAGS
    assert "filly_gec" not in transforms.__dict__


def test_device_auto_prefers_cuda_and_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert _select_device(None) == torch.device("cuda")
    assert _select_device("auto") == torch.device("cuda")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _select_device(None) == torch.device("cpu")
    assert _select_device("AUTO") == torch.device("cpu")
    assert _select_device("cpu") == torch.device("cpu")

    with pytest.raises(GECCheckpointError, match="CUDA was requested.*unavailable"):
        _select_device("cuda")


def test_device_rejects_unavailable_explicit_cuda_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(GECCheckpointError, match=r"only 1 CUDA device\(s\)"):
        _select_device("cuda:1")


def test_get_gec_service_forwards_device_and_rejects_conflicting_reuse(monkeypatch):
    class StubService:
        def __init__(self, *, device):
            self.device = _select_device(device)
            self.checkpoint_path = Path("test-checkpoint.pt")

    monkeypatch.setattr(gec_module, "GECService", StubService)
    monkeypatch.setattr(gec_module, "_gec_service", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    service = gec_module.get_gec_service("cpu")
    assert service.device == torch.device("cpu")
    assert gec_module.get_gec_service("auto") is service
    assert gec_module.get_gec_service() is service

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(GECCheckpointError, match="already loaded on cpu"):
        gec_module.get_gec_service("cuda")
