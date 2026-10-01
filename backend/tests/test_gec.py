from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch import nn

import app.services.gec as gec_module
from app.services.gec import (
    GECCheckpointError,
    GECInputTooLong,
    GECService,
    _select_device,
    _read_checkpoint,
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


def test_supplied_checkpoint_metadata_and_train_only_vocabulary_load_safely():
    checkpoint = Path(__file__).resolve().parents[2] / "best.pt"
    if not checkpoint.is_file():
        pytest.skip("the supplied best.pt checkpoint is not present")

    payload = _read_checkpoint(checkpoint)
    vocab, metadata = _validate_checkpoint(payload)

    assert len(vocab) == 892
    assert vocab["$KEEP"] == 0
    assert metadata["model_id"] == "jcblaise/roberta-tagalog-large"
    assert metadata["model_revision"] == "acb6b204dfb1afdd7476eae5da234cbcf8899846"
    assert metadata["stage"] == "stage2"
    assert metadata["readiness_status"] == "PROVISIONAL / NON_PRODUCTION / NOT_THESIS_PERFORMANCE"
    assert vocab["$TRANSFORM_CASE_CAPITAL"] == 70
    assert vocab["$TRANSFORM_CASE_LOWER"] == 71
    assert "$TRANSFORM_CASE_UPPER" not in vocab
    assert "$TRANSFORM_CASE_CAPITAL_1" not in vocab
    del payload


def test_checkpoint_rejects_mismatched_model_identity():
    checkpoint = Path(__file__).resolve().parents[2] / "best.pt"
    if not checkpoint.is_file():
        pytest.skip("the supplied best.pt checkpoint is not present")
    payload = _read_checkpoint(checkpoint)
    payload["metadata"]["model_revision"] = "0" * 40

    with pytest.raises(GECCheckpointError, match="pinned encoder revision"):
        _validate_checkpoint(payload)


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
