"""GECToR-style Tagalog RoBERTa encoder with separate prediction heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .losses import compute_multitask_loss

DEFAULT_ENCODER = "jcblaise/roberta-tagalog-large"


@dataclass
class GECToROutput:
    correction_logits: Tensor
    detection_logits: Tensor
    loss: Tensor | None = None
    correction_loss: Tensor | None = None
    detection_loss: Tensor | None = None


class GECToRTagalogLarge(nn.Module):
    """Contextual encoder plus correction-label and KEEP/EDIT classifiers.

    The encoder is injected for tests and local experiments.  The default
    Hugging Face encoder is loaded only through ``from_pretrained``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_edit_labels: int,
        *,
        dropout: float = 0.1,
        detection_loss_weight: float = 1.0,
        gradient_checkpointing: bool = False,
        keep_label_id: int = 0,
    ) -> None:
        super().__init__()
        if num_edit_labels < 2:
            raise ValueError("num_edit_labels must include at least KEEP and one edit")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if detection_loss_weight < 0:
            raise ValueError("detection_loss_weight must be non-negative")
        if not 0 <= keep_label_id < num_edit_labels:
            raise ValueError("keep_label_id is outside the correction label vocabulary")
        hidden_size = getattr(getattr(encoder, "config", None), "hidden_size", None)
        if hidden_size is None:
            raise ValueError("encoder.config.hidden_size is required")
        self.encoder = encoder
        self.num_edit_labels = num_edit_labels
        self.detection_loss_weight = float(detection_loss_weight)
        self.keep_label_id = int(keep_label_id)
        self.dropout = nn.Dropout(dropout)
        self.correction_head = nn.Linear(hidden_size, num_edit_labels)
        self.detection_head = nn.Linear(hidden_size, 2)  # 0 = correct/KEEP, 1 = edit
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.gradient_checkpointing:
            self.enable_gradient_checkpointing()

    @classmethod
    def from_pretrained(
        cls,
        num_edit_labels: int,
        *,
        model_id: str = DEFAULT_ENCODER,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> "GECToRTagalogLarge":
        """Load an ``AutoModel`` encoder without requiring Transformers at import."""
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "Loading the Tagalog encoder requires `transformers`; install "
                "the dependencies from requirements-colab.txt or requirements-runpod.txt."
            ) from exc
        encoder = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        model = cls(encoder, num_edit_labels, **kwargs)
        model.encoder_id = model_id
        model.encoder_revision = revision
        return model

    def enable_gradient_checkpointing(self) -> None:
        enable = getattr(self.encoder, "gradient_checkpointing_enable", None)
        if enable is None:
            raise ValueError("the selected encoder does not support gradient checkpointing")
        enable()
        self.gradient_checkpointing = True

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(trainable)

    @property
    def encoder_trainable(self) -> bool:
        return any(parameter.requires_grad for parameter in self.encoder.parameters())

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        correction_labels: Tensor | None = None,
        detection_labels: Tensor | None = None,
        **encoder_inputs: Any,
    ) -> GECToROutput:
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **encoder_inputs,
        )
        hidden = encoded.last_hidden_state if hasattr(encoded, "last_hidden_state") else encoded[0]
        correction_logits = self.correction_head(self.dropout(hidden))
        detection_logits = self.detection_head(hidden)
        loss = correction_loss = detection_loss = None
        if (correction_labels is None) != (detection_labels is None):
            raise ValueError("provide both correction_labels and detection_labels, or neither")
        if correction_labels is not None and detection_labels is not None:
            losses = compute_multitask_loss(
                correction_logits,
                detection_logits,
                correction_labels,
                detection_labels,
                detection_loss_weight=self.detection_loss_weight,
                attention_mask=attention_mask,
            )
            loss, correction_loss, detection_loss = losses
        return GECToROutput(
            correction_logits=correction_logits,
            detection_logits=detection_logits,
            loss=loss,
            correction_loss=correction_loss,
            detection_loss=detection_loss,
        )
