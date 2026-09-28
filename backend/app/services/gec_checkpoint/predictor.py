"""Map encoder subword logits back to correction-token labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class TokenLabelPrediction:
    labels: tuple[str, ...]
    detection_error_probabilities: tuple[float, ...]
    error_flags: tuple[bool, ...]
    corrected: bool


class TokenLabelPredictor:
    """Predict one executable vocabulary label per pre-tokenized source token.

    ``apply_edits`` intentionally does not live here: callers provide the
    canonical application function from the tagging/edit layer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: object,
        id_to_label: Mapping[int, str] | Sequence[str],
        *,
        keep_label: str = "$KEEP",
        keep_label_id: int | None = None,
        additional_confidence: float = 0.2,
        min_error_probability: float = 0.5,
        max_subword_tokens: int | None = None,
    ) -> None:
        if not 0.0 <= additional_confidence <= 1.0:
            raise ValueError("additional_confidence must be in [0, 1]")
        if not 0.0 <= min_error_probability <= 1.0:
            raise ValueError("min_error_probability must be in [0, 1]")
        if max_subword_tokens is not None and max_subword_tokens < 1:
            raise ValueError("max_subword_tokens must be positive or None")
        if isinstance(id_to_label, Mapping):
            self.id_to_label = {int(i): label for i, label in id_to_label.items()}
        else:
            self.id_to_label = dict(enumerate(id_to_label))
        labels_by_id = {label: idx for idx, label in self.id_to_label.items()}
        if keep_label not in labels_by_id:
            raise ValueError(f"KEEP label {keep_label!r} is absent from the model vocabulary")
        self.keep_id = labels_by_id[keep_label] if keep_label_id is None else int(keep_label_id)
        if self.id_to_label.get(self.keep_id) != keep_label:
            raise ValueError("keep_label_id and keep_label do not refer to the same class")
        self.model = model
        self.tokenizer = tokenizer
        self.additional_confidence = additional_confidence
        self.min_error_probability = min_error_probability
        self.max_subword_tokens = max_subword_tokens

    def predict(
        self,
        token_batches: Sequence[Sequence[str]],
        *,
        batch_size: int = 32,
    ) -> list[TokenLabelPrediction]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        results: list[TokenLabelPrediction] = []
        self.model.eval()
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        with torch.inference_mode():
            for offset in range(0, len(token_batches), batch_size):
                words = [list(sentence) for sentence in token_batches[offset:offset + batch_size]]
                encoded = self.tokenizer(
                    words,
                    is_split_into_words=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )
                input_ids = encoded["input_ids"]
                if self.max_subword_tokens is not None and input_ids.shape[1] > self.max_subword_tokens:
                    raise ValueError(
                        "input exceeds max_subword_tokens; inference refuses to truncate source text"
                    )
                if not hasattr(encoded, "word_ids"):
                    raise TypeError("a fast tokenizer with word_ids() support is required")
                model_inputs = {
                    key: encoded[key].to(device)
                    for key in ("input_ids", "attention_mask")
                    if key in encoded
                }
                output = self.model(**model_inputs)
                correction_probs = torch.softmax(output.correction_logits, dim=-1)
                correction_probs[..., self.keep_id] += self.additional_confidence
                predicted_ids = correction_probs.argmax(dim=-1)
                error_probs = torch.softmax(output.detection_logits, dim=-1)[..., 1]
                for row, original_words in enumerate(words):
                    word_ids = encoded.word_ids(row)
                    first_piece: dict[int, int] = {}
                    for position, word_id in enumerate(word_ids):
                        if word_id is not None and word_id not in first_piece:
                            first_piece[word_id] = position
                    if sorted(first_piece) != list(range(len(original_words))):
                        raise ValueError("tokenizer word_ids do not map every correction token exactly once")
                    positions = [first_piece[index] for index in range(len(original_words))]
                    row_error_probs = [float(error_probs[row, pos].item()) for pos in positions]
                    row_error_flags = tuple(
                        probability >= self.min_error_probability
                        for probability in row_error_probs
                    )
                    labels: list[str] = []
                    for pos, error_probability in zip(positions, row_error_probs):
                        class_id = (
                            self.keep_id
                            if error_probability < self.min_error_probability
                            else int(predicted_ids[row, pos].item())
                        )
                        try:
                            labels.append(self.id_to_label[class_id])
                        except KeyError as exc:
                            raise ValueError(f"model emitted class id absent from label vocabulary: {class_id}") from exc
                    results.append(
                        TokenLabelPrediction(
                            labels=tuple(labels),
                            detection_error_probabilities=tuple(row_error_probs),
                            error_flags=row_error_flags,
                            corrected=any(label != "$KEEP" for label in labels),
                        )
                    )
        return results
