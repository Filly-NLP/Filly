"""Masked objectives for the two-head GECToR model."""

from __future__ import annotations

from typing import NamedTuple

try:
    import torch
    from torch import Tensor
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised on minimal installs
    raise ImportError(
        "FILLY GEC model components require PyTorch. Install the runtime "
        "dependencies from requirements-colab.txt or requirements-runpod.txt."
    ) from exc


class MultitaskLoss(NamedTuple):
    total: Tensor
    correction: Tensor
    detection: Tensor


def _masked_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.ndim != 3:
        raise ValueError(f"expected [batch, sequence, classes] logits, got {tuple(logits.shape)}")
    if targets.shape != logits.shape[:2]:
        raise ValueError(
            f"target shape {tuple(targets.shape)} does not match logits positions "
            f"{tuple(logits.shape[:2])}"
        )
    valid = targets.ne(-100)
    if not torch.any(valid):
        # CrossEntropyLoss(mean) returns NaN when every position is ignored.
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )


def compute_multitask_loss(
    correction_logits: Tensor,
    detection_logits: Tensor,
    correction_labels: Tensor,
    detection_labels: Tensor,
    *,
    detection_loss_weight: float = 1.0,
    attention_mask: Tensor | None = None,
) -> MultitaskLoss:
    """Compute correction CE plus weighted detection CE.

    Both heads ignore every ``-100`` position.  If supplied, the attention
    mask is applied to both target tensors without modifying caller-owned data.
    """
    if detection_loss_weight < 0:
        raise ValueError("detection_loss_weight must be non-negative")
    if correction_labels.shape != detection_labels.shape:
        raise ValueError("correction and detection targets must have identical shapes")
    if attention_mask is not None:
        if attention_mask.shape != correction_labels.shape:
            raise ValueError("attention_mask must have the same [batch, sequence] shape as labels")
        keep = attention_mask.to(dtype=torch.bool)
        correction_labels = correction_labels.masked_fill(~keep, -100)
        detection_labels = detection_labels.masked_fill(~keep, -100)
    correction = _masked_cross_entropy(correction_logits, correction_labels)
    detection = _masked_cross_entropy(detection_logits, detection_labels)
    return MultitaskLoss(
        total=correction + detection_loss_weight * detection,
        correction=correction,
        detection=detection,
    )
