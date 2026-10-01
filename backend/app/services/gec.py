"""Checkpoint-backed FILLY GEC inference.

The supplied checkpoint is the custom two-head Stage-2 model from the
Grammar-Error-Correction training repository.  This service deliberately
fails closed when that checkpoint or its pinned tokenizer is unavailable;
there is no rule-based or randomly initialized fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from app.services.gec_checkpoint.tokenizer import tokenize_with_offsets
from app.services.gec_checkpoint.transforms import (
    KEEP,
    _label_segments,
    apply_labels_to_text,
    render_tokens,
)
from app.services.gec_checkpoint.segmentation import (
    reassemble_sentences,
    segment_sentences,
)

logger = logging.getLogger(__name__)

MODEL_ID = "jcblaise/roberta-tagalog-large"
MODEL_REVISION = "acb6b204dfb1afdd7476eae5da234cbcf8899846"
TRANSFORMERS_VERSION = "5.16.1"
ADDITIONAL_KEEP_CONFIDENCE = 0.2
MIN_ERROR_PROBABILITY = 0.5
DEFAULT_ITERATIONS = 5

if TYPE_CHECKING:
    from app.schemas.schemas import GrammarCorrectionItem


class GECCheckpointError(RuntimeError):
    """The selected checkpoint or one of its required assets is invalid."""


class GECInputTooLong(ValueError):
    """The tokenizer output exceeds the selected encoder's position limit."""


@dataclass(frozen=True, slots=True)
class GECChange:
    """One exact label replay edit, with offsets relative to its pass input."""

    original: str
    correction: str
    start: int
    end: int
    label: str
    confidence: float
    iteration: int


@dataclass(frozen=True, slots=True)
class GECIteration:
    """Inputs, labels, and exact output for one model pass."""

    iteration: int
    input_text: str
    output_text: str
    input_tokens: tuple[str, ...]
    output_tokens: tuple[str, ...]
    labels: tuple[str, ...]
    changes: tuple[GECChange, ...]
    model_invoked: bool = True


@dataclass(frozen=True, slots=True)
class GECResult:
    """Five-pass inference result and the provenance of the loaded model."""

    original_text: str
    corrected_text: str
    iterations: int
    iteration_outputs: tuple[str, ...]
    passes: tuple[GECIteration, ...]
    changes: tuple[GECChange, ...]
    metadata: Mapping[str, Any]


def _default_checkpoint_path() -> Path:
    configured = os.environ.get("GECTOR_MODEL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "best.pt"


def _numpy_checkpoint_allowlist() -> list[Any]:
    """Allow the NumPy RNG tuple saved by the canonical training checkpoint."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - torch normally depends on numpy
        raise GECCheckpointError("checkpoint loading requires NumPy for safe metadata decoding") from exc

    multiarray = getattr(np, "_core", None) or getattr(np, "core", None)
    reconstruct = getattr(getattr(multiarray, "multiarray", None), "_reconstruct", None)
    if reconstruct is None:
        raise GECCheckpointError("this NumPy version cannot safely decode the checkpoint RNG metadata")
    return [np.ndarray, reconstruct, np.dtype, type(np.dtype("uint32"))]


def _read_checkpoint(path: Path) -> dict[str, Any]:
    """Read the checkpoint with mmap and PyTorch's restricted unpickler."""
    if not path.is_file():
        raise FileNotFoundError(f"GEC checkpoint does not exist: {path}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime install
        raise GECCheckpointError("GEC inference requires PyTorch; install backend requirements") from exc
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        raise GECCheckpointError(
            "safe checkpoint loading requires PyTorch 2.6 or newer; refusing unrestricted pickle loading"
        )
    try:
        with safe_globals(_numpy_checkpoint_allowlist()):
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
    except Exception as exc:
        raise GECCheckpointError(f"could not safely load GEC checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise GECCheckpointError("unsupported GEC checkpoint format; expected format_version=1")
    return payload


def _vocabulary_sha256(label_vocab: Mapping[str, int]) -> str:
    labels = [label for label, _index in sorted(label_vocab.items(), key=lambda item: item[1])]
    canonical = json.dumps(
        {"schema_version": 1, "labels": labels},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_checkpoint(payload: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, Any]]:
    config = payload.get("config")
    metadata = payload.get("metadata")
    hashes = payload.get("hashes")
    vocab = payload.get("label_vocab")
    state = payload.get("model_state")
    if not all(isinstance(value, Mapping) for value in (config, metadata, hashes, vocab, state)):
        raise GECCheckpointError("checkpoint is missing config, metadata, hashes, vocabulary, or model_state")

    if (metadata.get("selected_best") is not True
            or config.get("model_id") != MODEL_ID
            or metadata.get("model_id") != MODEL_ID
            or config.get("model_revision") != MODEL_REVISION
            or metadata.get("model_revision") != MODEL_REVISION):
        raise GECCheckpointError("checkpoint is not the selected model at its pinned encoder revision")
    if metadata.get("stage") != config.get("stage"):
        raise GECCheckpointError("checkpoint stage differs between its config and metadata")
    if config.get("keep_label_id") != 0:
        raise GECCheckpointError("checkpoint KEEP label id differs from the inference contract")
    if config.get("vocabulary_support_scope") != "train_only":
        raise GECCheckpointError("checkpoint does not declare the expected train-only label vocabulary")

    label_vocab = dict(vocab)
    if (not label_vocab or any(not isinstance(label, str) or type(index) is not int
                               for label, index in label_vocab.items())):
        raise GECCheckpointError("checkpoint label vocabulary is malformed")
    indexes = sorted(label_vocab.values())
    if indexes != list(range(len(label_vocab))) or label_vocab.get(KEEP) != 0:
        raise GECCheckpointError("checkpoint label ids must be contiguous with $KEEP at id zero")
    actual_vocab_hash = _vocabulary_sha256(label_vocab)
    expected_vocab_hash = hashes.get("label_vocab")
    if (not isinstance(expected_vocab_hash, str)
            or actual_vocab_hash != expected_vocab_hash
            or metadata.get("label_vocab_hash") != expected_vocab_hash):
        raise GECCheckpointError("checkpoint label vocabulary hash is missing or inconsistent")
    if (not isinstance(state, Mapping)
            or "correction_head.weight" not in state
            or "detection_head.weight" not in state
            or tuple(state["correction_head.weight"].shape) != (len(label_vocab), 1024)
            or tuple(state["detection_head.weight"].shape) != (2, 1024)):
        raise GECCheckpointError("checkpoint model heads do not match its label vocabulary and encoder")

    reproducibility = config.get("reproducibility_metadata") or {}
    dataset_status = reproducibility.get("dataset_status", "UNKNOWN")
    research_status = reproducibility.get("research_performance_status", "UNKNOWN")
    is_provisional = dataset_status == "PROVISIONAL" or research_status == "NOT_THESIS_PERFORMANCE"
    readiness = (
        "PROVISIONAL / NON_PRODUCTION / NOT_THESIS_PERFORMANCE"
        if is_provisional else "UNASSESSED"
    )
    safe_metadata = {
        "checkpoint_path": None,
        "checkpoint_format_version": payload.get("format_version"),
        "stage": metadata.get("stage"),
        "selected_best": metadata.get("selected_best"),
        "run_id": metadata.get("run_id"),
        "model_id": metadata.get("model_id"),
        "model_revision": metadata.get("model_revision"),
        "label_vocab_size": len(label_vocab),
        "label_vocab_sha256": expected_vocab_hash,
        "vocabulary_support_scope": config.get("vocabulary_support_scope"),
        "source_manifest_hash": metadata.get("source_manifest_hash"),
        "provisional_preparation_manifest_sha256": hashes.get("provisional_preparation_manifest"),
        "dataset_status": dataset_status,
        "research_performance_status": research_status,
        "readiness_status": readiness,
        "transformers_version": TRANSFORMERS_VERSION,
    }
    return label_vocab, safe_metadata


def _snapshot_revision_from_tokenizer(tokenizer: Any) -> str:
    """Require tokenizer file provenance from the exact Hugging Face snapshot."""
    init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
    revisions: set[str] = set()
    for key, value in init_kwargs.items():
        if not (key.endswith("_file") or key == "tokenizer_file") or value in (None, ""):
            continue
        if not isinstance(value, (str, os.PathLike)):
            raise GECCheckpointError("pinned tokenizer declares a malformed asset path")
        parts = Path(value).parts
        if "snapshots" not in parts:
            raise GECCheckpointError("pinned tokenizer asset has no verifiable Hugging Face snapshot")
        if parts.count("snapshots") != 1:
            raise GECCheckpointError("pinned tokenizer asset has an ambiguous snapshot path")
        index = parts.index("snapshots")
        if index + 2 >= len(parts) or len(parts[index + 1]) != 40:
            raise GECCheckpointError("pinned tokenizer asset has a malformed snapshot revision")
        if parts[index - 1] != "models--jcblaise--roberta-tagalog-large":
            raise GECCheckpointError("pinned tokenizer asset belongs to a different model repository")
        revisions.add(parts[index + 1].lower())
    if revisions != {MODEL_REVISION}:
        raise GECCheckpointError(
            f"tokenizer assets do not prove the pinned model revision {MODEL_REVISION}"
        )
    return next(iter(revisions))


def _select_device(device: str | None) -> Any:
    """Resolve auto mode or validate an explicitly requested CPU/CUDA device."""
    import torch

    if device is None or (isinstance(device, str) and device.strip().lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        selected_device = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise GECCheckpointError(f"invalid GEC inference device {device!r}") from exc
    if selected_device.type not in {"cpu", "cuda"}:
        raise GECCheckpointError(
            f"unsupported GEC inference device {device!r}; choose 'cpu', 'cuda', or 'auto'"
        )
    if selected_device.type == "cuda":
        if not torch.cuda.is_available():
            raise GECCheckpointError("CUDA was requested for GEC inference but is unavailable")
        if selected_device.index is not None and selected_device.index >= torch.cuda.device_count():
            raise GECCheckpointError(
                f"CUDA device index {selected_device.index} was requested, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available"
            )
    return selected_device


def _load_runtime(
    checkpoint_path: Path,
    *,
    device: str | None,
    cache_dir: str | None,
    local_files_only: bool,
) -> tuple[Any, Any, dict[str, int], dict[str, Any], Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"GEC checkpoint does not exist: {checkpoint_path}")
    import torch
    import transformers
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise GECCheckpointError(
            f"checkpoint was trained with transformers=={TRANSFORMERS_VERSION}; "
            f"found {transformers.__version__}"
        )
    payload = _read_checkpoint(checkpoint_path)
    label_vocab, metadata = _validate_checkpoint(payload)
    metadata["checkpoint_path"] = str(checkpoint_path)

    config = AutoConfig.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if getattr(config, "_commit_hash", None) != MODEL_REVISION:
        raise GECCheckpointError("loaded encoder config does not prove the checkpoint's pinned revision")
    if (getattr(config, "hidden_size", None) != 1024
            or getattr(config, "num_hidden_layers", None) != 24
            or getattr(config, "vocab_size", None) != 30000
            or getattr(config, "max_position_embeddings", None) != 514):
        raise GECCheckpointError("loaded encoder config does not match checkpoint tensor dimensions")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        use_fast=True,
        add_prefix_space=True,
    )
    if not getattr(tokenizer, "is_fast", False) or getattr(tokenizer, "add_prefix_space", None) is not True:
        raise GECCheckpointError("checkpoint inference requires the pinned fast prefix-space tokenizer")
    _snapshot_revision_from_tokenizer(tokenizer)

    selected_device = _select_device(device)

    from app.services.gec_checkpoint.model import GECToRTagalogLarge

    # Build the exact architecture with Transformers' normal positional
    # buffers, then assign the memory-mapped checkpoint weights. Strict load
    # prevents these temporary initial values from ever reaching inference.
    encoder = AutoModel.from_config(config)
    model = GECToRTagalogLarge(
        encoder,
        len(label_vocab),
        dropout=float(payload["config"].get("dropout", 0.1)),
        detection_loss_weight=float(payload["config"].get("detection_loss_weight", 1.0)),
        gradient_checkpointing=False,
        keep_label_id=label_vocab[KEEP],
    )
    try:
        model.load_state_dict(payload["model_state"], strict=True, assign=True)
    except Exception as exc:
        raise GECCheckpointError(f"checkpoint weights do not match the reconstructed model: {exc}") from exc
    del payload
    model.to(selected_device)
    model.eval()
    metadata["device"] = str(selected_device)
    return model, tokenizer, label_vocab, metadata, config


def _change_records(
    text: str,
    labels: tuple[str, ...],
    error_probabilities: tuple[float, ...],
    *,
    iteration: int,
) -> tuple[GECChange, ...]:
    tokenized = tokenize_with_offsets(text)
    changes: list[GECChange] = []
    for start_token, end_token, output_tokens, kind in _label_segments(tokenized.tokens, labels):
        if kind == "keep":
            continue
        spans = tokenized.spans
        if kind == "append":
            inserted = output_tokens[1:]
            source_token = tokenized.tokens[start_token]
            rendered_source = render_tokens((source_token,))
            rendered_pair = render_tokens((source_token, *inserted))
            if not rendered_pair.startswith(rendered_source):
                raise ValueError("appended GEC label cannot be separated from its source token")
            start = end = spans[start_token].end
            replacement = rendered_pair[len(rendered_source):]
        elif kind == "delete":
            if start_token == 0:
                start = spans[0].start
                end = spans[end_token].start if end_token < len(spans) else spans[end_token - 1].end
            else:
                start = spans[start_token - 1].end
                end = spans[end_token - 1].end
            replacement = ""
        else:
            start = spans[start_token].start
            end = spans[end_token - 1].end
            replacement = render_tokens(output_tokens)
        confidence = max(error_probabilities[start_token:end_token], default=0.0)
        changes.append(GECChange(
            original=text[start:end],
            correction=replacement,
            start=start,
            end=end,
            label=labels[start_token],
            confidence=confidence,
            iteration=iteration,
        ))
    return tuple(changes)


class GECService:
    """Load the selected checkpoint once and run its exact label decoder."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        device: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        label_vocab: Mapping[str, int] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else _default_checkpoint_path()
        self.iterations = int(os.environ.get("GECTOR_ITERATIONS", DEFAULT_ITERATIONS))
        if self.iterations < 1:
            raise ValueError("GECTOR_ITERATIONS must be at least one")

        injected = (model is not None, tokenizer is not None, label_vocab is not None)
        if any(injected) and not all(injected):
            raise ValueError("model, tokenizer, and label_vocab must be injected together")
        if all(injected):
            self.device = _select_device(device)
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
            self.label_vocab = dict(label_vocab or {})
            if self.label_vocab.get(KEEP) != 0:
                raise ValueError("injected label vocabulary must include $KEEP at id zero")
            self.metadata = dict(metadata or {"readiness_status": "TEST_FIXTURE"})
            max_subword_tokens = None
        else:
            cache_dir = os.environ.get("GECTOR_CACHE_DIR") or None
            local_files_only = os.environ.get("GECTOR_LOCAL_FILES_ONLY", "false").lower() in {"1", "true", "yes"}
            (self.model, self.tokenizer, self.label_vocab,
             self.metadata, encoder_config) = _load_runtime(
                self.checkpoint_path,
                device=device,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            import torch

            self.device = next(self.model.parameters()).device
            max_subword_tokens = int(encoder_config.max_position_embeddings)

        from app.services.gec_checkpoint.predictor import TokenLabelPredictor

        inverse_vocab = {index: label for label, index in self.label_vocab.items()}
        self.predictor = TokenLabelPredictor(
            self.model,
            self.tokenizer,
            inverse_vocab,
            keep_label=KEEP,
            keep_label_id=self.label_vocab[KEEP],
            additional_confidence=ADDITIONAL_KEEP_CONFIDENCE,
            min_error_probability=MIN_ERROR_PROBABILITY,
            max_subword_tokens=max_subword_tokens,
        )
        self.model.eval()

    def _correct_sentence_parts(
        self,
        text: str,
        sentence_texts: tuple[str, ...],
        gaps: tuple[str, ...],
        *,
        iteration: int,
    ) -> tuple[GECIteration, tuple[str, ...]]:
        """Correct fixed sentence slots as a batch and map edits to paragraph offsets."""
        if len(gaps) != len(sentence_texts) + 1:
            raise ValueError("sentence layouts require exactly one more gap than sentence")

        sentence_offsets: list[int] = []
        cursor = len(gaps[0])
        for index, sentence in enumerate(sentence_texts):
            sentence_offsets.append(cursor)
            cursor += len(sentence) + len(gaps[index + 1])

        tokenized_sentences = [
            (index, tokenize_with_offsets(sentence))
            for index, sentence in enumerate(sentence_texts)
        ]
        model_inputs = [
            (index, tokenized.tokens)
            for index, tokenized in tokenized_sentences
            if tokenized.tokens
        ]
        input_tokens = tokenize_with_offsets(text).tokens
        flattened_model_tokens = tuple(
            token for _index, tokens in model_inputs for token in tokens
        )
        if flattened_model_tokens != input_tokens:
            raise RuntimeError("sentence segmentation changed the paragraph token sequence")

        if not model_inputs:
            iteration_result = GECIteration(
                iteration,
                text,
                text,
                (),
                (),
                (),
                (),
                model_invoked=False,
            )
            return iteration_result, sentence_texts

        try:
            predictions = self.predictor.predict(
                [tokens for _index, tokens in model_inputs],
                batch_size=32,
            )
        except ValueError as exc:
            if "input exceeds max_subword_tokens" in str(exc):
                raise GECInputTooLong(
                    "Text exceeds the GEC model's token limit. Shorten or split the text and try again."
                ) from exc
            raise
        if len(predictions) != len(model_inputs):
            raise RuntimeError("GEC predictor did not return one result per sentence")

        output_sentences = list(sentence_texts)
        labels: list[str] = []
        changes: list[GECChange] = []
        for (sentence_index, tokens), prediction in zip(model_inputs, predictions):
            sentence = sentence_texts[sentence_index]
            output_sentences[sentence_index] = apply_labels_to_text(sentence, prediction.labels)
            labels.extend(prediction.labels)
            local_changes = _change_records(
                sentence,
                prediction.labels,
                prediction.detection_error_probabilities,
                iteration=iteration,
            )
            sentence_offset = sentence_offsets[sentence_index]
            changes.extend(
                replace(
                    change,
                    start=sentence_offset + change.start,
                    end=sentence_offset + change.end,
                )
                for change in local_changes
            )

        output_text = reassemble_sentences(output_sentences, gaps)
        output_tokens = tokenize_with_offsets(output_text).tokens
        iteration_result = GECIteration(
            iteration=iteration,
            input_text=text,
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            labels=tuple(labels),
            changes=tuple(changes),
            model_invoked=True,
        )
        return iteration_result, tuple(output_sentences)

    def _correct_once(self, text: str, *, iteration: int) -> GECIteration:
        layout = segment_sentences(text)
        sentence_texts = tuple(sentence.text for sentence in layout.sentences)
        result, _output_sentences = self._correct_sentence_parts(
            text,
            sentence_texts,
            layout.gaps,
            iteration=iteration,
        )
        return result

    def correct_once(self, text: str) -> GECIteration:
        """Run one inference pass and return its exact replay trace."""
        if not isinstance(text, str):
            raise TypeError("GEC input must be a string")
        return self._correct_once(text, iteration=1)

    def correct_iteratively(self, text: str, iterations: int | None = None) -> GECResult:
        """Run sequential passes, feeding each pass output into the next."""
        if not isinstance(text, str):
            raise TypeError("GEC input must be a string")
        pass_count = self.iterations if iterations is None else iterations
        if type(pass_count) is not int or pass_count < 1:
            raise ValueError("iterations must be a positive integer")

        layout = segment_sentences(text)
        sentence_texts = tuple(sentence.text for sentence in layout.sentences)
        gaps = layout.gaps
        current = text
        passes: list[GECIteration] = []
        # Empty input has no inference stages. Once nonempty text is accepted,
        # every scheduled stage is recorded, including identity stages after a deletion.
        if not text:
            return GECResult(
                original_text=text,
                corrected_text=text,
                iterations=0,
                iteration_outputs=(),
                passes=(),
                changes=(),
                metadata=dict(self.metadata),
            )
        for iteration in range(1, pass_count + 1):
            result, sentence_texts = self._correct_sentence_parts(
                current,
                sentence_texts,
                gaps,
                iteration=iteration,
            )
            passes.append(result)
            current = result.output_text

        return GECResult(
            original_text=text,
            corrected_text=current,
            iterations=len(passes),
            iteration_outputs=tuple(result.output_text for result in passes),
            passes=tuple(passes),
            changes=tuple(change for result in passes for change in result.changes),
            metadata=dict(self.metadata),
        )

    def check(self, text: str) -> list[GrammarCorrectionItem]:
        """Return first-pass suggestion items for the existing API surface."""
        from app.schemas.schemas import GrammarCorrectionItem

        result = self.correct_once(text)
        return [
            GrammarCorrectionItem(
                original=change.original,
                correction=change.correction,
                start=change.start,
                end=change.end,
                type="grammar",
                rule=change.label,
                message="",
            )
            for change in result.changes
        ]


_gec_service: GECService | None = None
_gec_service_lock = threading.Lock()


def get_gec_service(device: str | None = None) -> GECService:
    """Return the process-wide model service, initializing it at most once."""
    global _gec_service
    if _gec_service is None:
        with _gec_service_lock:
            if _gec_service is None:
                _gec_service = GECService(device=device)
                logger.info("Loaded GEC checkpoint from %s", _gec_service.checkpoint_path)
    if device is not None:
        requested_device = _select_device(device)
        if _gec_service.device != requested_device:
            raise GECCheckpointError(
                f"GEC service is already loaded on {_gec_service.device}; "
                f"cannot honor requested device {requested_device} without reloading"
            )
    return _gec_service
