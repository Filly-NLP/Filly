from __future__ import annotations

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
    assert tokenizer.observed == [result.input_tokens]


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
